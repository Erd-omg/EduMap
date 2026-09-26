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

Usage::

    cd services/backend-core
    export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python scripts/review_cs301_corpus.py > /tmp/cs301_review.json

See ``docs/cs301-corpus-review.md`` for the review procedure.
"""

from __future__ import annotations

import asyncio
import json
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


def main() -> int:
    rows = asyncio.run(build_review())
    flagged = [r for r in rows if r["never_retrieved"]]
    foreign = [r for r in rows if r["foreign_hits"]]
    print(json.dumps(rows, ensure_ascii=False, indent=2), file=sys.stderr)
    print(
        f"{len(rows)} queries; {len(flagged)} with a label never retrieved; "
        f"{len(foreign)} with cross-course hits.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
