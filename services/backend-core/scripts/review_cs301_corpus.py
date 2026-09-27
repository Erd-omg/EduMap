#!/usr/bin/env python3
"""Dump the cs301 corpus with claimed labels beside what retrieval returns.

Built for the human review that the corpus needs before its numbers may be
cited: ``relevant_kp_ids`` decides MRR's numerator, and the labels were
machine-drafted (``_meta.review_status == "unreviewed"``).

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

See ``docs/cs301-corpus-review.md`` for the review procedure.
"""

from __future__ import annotations

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
COURSE = "cs301"
TOP_K = 5

# Column order the reviewer works through left to right: identity → what to fix
# → why it was flagged → the evidence behind the flag.
COLUMNS = [
    "no", "query", "claimed", "fix", "never_retrieved", "foreign_hits",
    "top_rrf", "top_minmax", "top_score",
]


def _is_foreign(source_id: str) -> bool:
    """True for a knowledge point that does not belong to cs301."""
    return source_id.startswith("kp-") and not source_id.startswith("kp-os-")


async def build_review() -> list[dict]:
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
        for item in load_expanded_queries(COURSE):
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
                # cs201 items surfacing here (noise for the reviewer, not MRR).
                "foreign_hits": sorted(
                    {sid for sid in union if _is_foreign(sid)}
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

    Written with ``newline=""`` and an explicit dialect so the file opens
    cleanly in Excel/Numbers/Sheets — Chinese text and the space-separated id
    lists survive, and no quoting surprises appear in a spreadsheet.
    """
    writer = csv.DictWriter(
        stream, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n",
    )
    writer.writeheader()
    for i, row in enumerate(rows, 1):
        writer.writerow(_row_for_csv(i, row))


def main() -> int:
    rows = asyncio.run(build_review())

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
