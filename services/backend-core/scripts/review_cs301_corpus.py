#!/usr/bin/env python3
"""Dump the cs301 corpus with claimed labels beside what retrieval returns.

Built for the human review that the corpus needs before its numbers may be
cited: ``relevant_kp_ids`` decides MRR's numerator, and the labels were
machine-drafted. **The cs301 review was completed on 2026-09-27** (40 rows,
1 change: ``kp-os-sync`` -> ``kp-os-schedule`` on row 17) — see
``benchmark_results/README.md``. This script remains for future corpora and
for re-reviewing cs301 if its labels are ever revised.

For each query it prints the claimed labels, the top-5 returned by each fusion
method, and two automatic red flags:

* **never_retrieved** — a claimed label no method put in its top-5. Either the
  label is wrong, or the retriever has a gap; the reviewer decides which.
* **cross-course hits** — cs201 knowledge points surfacing for a cs301 query.
  Both courses share one Chroma collection and retrieval does not filter by
  course, so semantically nearby items leak in. This does **not** move MRR
  (only the rank of the *correct* KP matters), but it will mislead a reviewer
  skimming the top-5, so it is called out rather than left to be noticed.

根据「声明标注 vs 实际召回」输出**可直接在表格软件里复核**的 CSV：

    no           行号，复核时按此记录进度
    query        查询文本
    claimed      当前声明的相关 KP（空格分隔）——**要改的就是这一列**
    fix          复核结论，留空；填新的 KP 列表（空格分隔），或 `ok` 表示不用改
    never_retrieved  声明了但三种方法都没召回（最可疑）
    foreign_hits 结果里混入的 cs201 知识点（噪声，不影响 MRR）
    top_rrf / top_minmax / top_score
                 各策略实际返回的 top-5，供对照判断漏标

`fix` 列是给你填的：复核完把 CSV 发我，我据此改语料文件并同步 `_meta`。

Usage::

    cd services/backend-core
    export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/review_cs301_corpus.py > /tmp/cs301_review.csv
    python scripts/review_cs301_corpus.py --course cs201   # any course

See ``docs/cs301-corpus-review.md`` for the review procedure.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")
os.environ.setdefault("NO_PROXY", os.environ["no_proxy"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config import settings  # noqa: E402
from src.kg.connection import Neo4jPool  # noqa: E402
from src.kg.repositories.knowledge_point_repo import (  # noqa: E402
    KnowledgePointRepository,
)
from src.kg.vector_index import VectorIndex  # noqa: E402
from src.rag.evaluation.datasets import load_expanded_queries  # noqa: E402
from src.rag.rag_service import RAGRetrievalService  # noqa: E402

METHODS = ("rrf", "minmax", "score")
TOP_K = 5

# Column order the reviewer works through left to right: identity → what to fix
# → why it was flagged → the evidence behind the flag.
COLUMNS = [
    "no", "query", "claimed", "fix", "never_retrieved", "foreign_hits",
    "top_rrf", "top_minmax", "top_score",
]


def _is_foreign(source_id: str, own_prefix: str) -> bool:
    """True for a knowledge point belonging to some *other* course.

    Every cs301 KP id starts with ````kp-os-````; every cs201 one does not.
    Deriving the prefix from the course under review (rather than hardcoding
    ``kp-os-``) is what lets this tool be pointed at a corpus other than the
    one it was written for — with ``kp-os-`` baked in, reviewing cs201 would
    flag *every* cs201 KP as foreign.
    """
    return source_id.startswith("kp-") and not source_id.startswith(own_prefix)


def _own_prefix_for(course_id: str) -> str:
    """The KP id prefix shared by a course's knowledge points.

    Read from the corpus itself: a corpus whose labels all begin with the same
    ``kp-<slug>-`` prefix identifies its own course's KPs unambiguously.
    """
    ids = [
        rid
        for item in load_expanded_queries(course_id)
        for rid in item["relevant_ids"]
        if rid.startswith("kp-")
    ]
    if not ids:
        raise SystemExit(
            f"course {course_id!r} has no kp-* labels — cannot infer its prefix. "
            f"Check the corpus file and its _meta.course_id."
        )
    # Longest common prefix, truncated to the last separator so the result is
    # 'kp-os-' rather than a partial id like 'kp-os-sched'.
    prefix = os.path.commonprefix(ids)
    return prefix[: prefix.rfind("-") + 1] if "-" in prefix else prefix


async def build_review(course_id: str) -> list[dict]:
    own_prefix = _own_prefix_for(course_id)
    pool = Neo4jPool(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password,
    )
    repo = KnowledgePointRepository(pool)
    vector_index = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)

    if getattr(vector_index, "_client", None) is None:
        await pool.close()
        raise SystemExit(
            "ChromaDB client could not be created — VectorIndex fell back to an "
            "in-memory store, so every result would be measured against an EMPTY "
            "index. Export no_proxy=localhost,127.0.0.1,::1 and retry."
        )

    service = RAGRetrievalService(
        vector_index=vector_index,
        kp_repo=repo,
        embedding_model=settings.llm_embedding_model,
    )

    rows: list[dict] = []
    try:
        for item in load_expanded_queries(course_id):
            by_method: dict[str, list[str]] = {}
            for method in METHODS:
                service.fusion_method = method
                results = await service.search(
                    item["query"], top_k=TOP_K, use_cache=False,
                )
                by_method[method] = [r.source_id for r in results]

            union = {sid for ids in by_method.values() for sid in ids}
            rows.append({
                "query": item["query"],
                "claimed": sorted(item["relevant_ids"]),
                "by_method": by_method,
                # A claimed label no method retrieved: worth a human look.
                "never_retrieved": sorted(item["relevant_ids"] - union),
                # Other courses' items surfacing here (noise for the reviewer,
                # not MRR).
                "foreign_hits": sorted(
                    {sid for sid in union if _is_foreign(sid, own_prefix)}
                ),
            })
    finally:
        await pool.close()
    return rows


def _row_for_csv(index: int, row: dict) -> dict[str, str]:
    """Flatten one review row into the CSV columns."""
    def joined(ids: list[str]) -> str:
        return " ".join(ids)

    flat = {
        "no": str(index),
        "query": row["query"],
        "claimed": joined(row["claimed"]),
        # Left blank for the reviewer. `ok` means "labels are right as-is".
        "fix": "",
        "never_retrieved": joined(row["never_retrieved"]),
        "foreign_hits": joined(row["foreign_hits"]),
    }
    for method in METHODS:
        flat[f"top_{method}"] = joined(row["by_method"][method])
    return flat


def write_csv(rows: list[dict], stream) -> None:
    """Write the review worksheet as CSV.

    Written with an explicit dialect so the file opens cleanly in
    Excel/Numbers/Sheets — Chinese text and the space-separated id lists
    survive, and no quoting surprises appear in a spreadsheet.

    **Empty cells are written as ``-``, never as an empty field.**  A row whose
    evidence columns are all empty (``never_retrieved`` and ``foreign_hits``
    both blank is common — most rows are clean) collapses to consecutive commas
    like ``,,,`` when pasted through a spreadsheet, and the columns to its right
    then shift one position left.  That is not hypothetical: the first version
    of this worksheet used empty fields, and a reviewer filling in ``ok``
    produced a file where rows 1 and 2 had ``top_rrf`` sitting in the
    ``foreign_hits`` column and ``top_score`` wiped entirely.  A visible ``-``
    per empty cell keeps every row's field count fixed no matter how it is
    edited.
    """
    writer = csv.DictWriter(
        stream, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n",
    )
    writer.writeheader()
    for i, row in enumerate(rows, 1):
        flat = _row_for_csv(i, row)
        writer.writerow({k: (v if v else "-") for k, v in flat.items()})


def _assert_columns_align(rows: list[dict]) -> None:
    """Fail loudly if any row would not round-trip through the CSV.

    Guards the defect that made the first worksheet unusable: a row whose field
    count differs from the header's, or a row where the ``fix`` column is not
    the fourth field, would silently misalign every column after it for whoever
    opens the file — and the reviewer has no way to notice.
    """
    for i, row in enumerate(rows, 1):
        flat = _row_for_csv(i, row)
        fields = [flat[c] if flat[c] else "-" for c in COLUMNS]
        if len(fields) != len(COLUMNS):
            raise SystemExit(
                f"row {i} produced {len(fields)} fields, expected {len(COLUMNS)} — "
                f"the worksheet would not round-trip."
            )
        # Round-trip through the real CSV reader, not just the field count.
        parsed = next(csv.reader(io.StringIO(
            ",".join(
                '"{}"'.format(f.replace('"', '""')) if ("," in f or '"' in f) else f
                for f in fields
            )
        )))
        if len(parsed) != len(COLUMNS) or parsed[3] != (flat["fix"] or "-"):
            raise SystemExit(
                f"row {i} does not survive a CSV round-trip — the `fix` column "
                f"landed at {parsed[3]!r} instead of {flat['fix'] or '-'!r}."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--course", default="cs301",
        help="Which corpus to review (default: cs301). See "
             "run_fusion_ablation.py --list-courses for what is on disk.",
    )
    args = parser.parse_args()

    rows = asyncio.run(build_review(args.course))
    _assert_columns_align(rows)

    # CSV to stdout so it can be redirected straight into a file or spreadsheet.
    buffer = io.StringIO()
    write_csv(rows, buffer)
    sys.stdout.write(buffer.getvalue())

    # Diagnostics to stderr, so they never contaminate the CSV.
    flagged = [r for r in rows if r["never_retrieved"]]
    foreign = [r for r in rows if r["foreign_hits"]]
    print(
        f"\n{len(rows)} queries written.\n"
        f"  {len(flagged)} with a label no method retrieved — read these first.\n"
        f"  {len(foreign)} with cross-course (cs201) hits — noise, does not move MRR.\n"
        f"\nFill the `fix` column (new KP list, or `ok`), then send the CSV back.",
        file=sys.stderr,
    )
    if flagged:
        print("\nFlagged rows:", file=sys.stderr)
        for i, r in enumerate(rows, 1):
            if r["never_retrieved"]:
                print(f"  row {i}: {r['query']}", file=sys.stderr)
                print(f"           never retrieved: {r['never_retrieved']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
