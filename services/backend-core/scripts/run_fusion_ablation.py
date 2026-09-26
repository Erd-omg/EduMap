#!/usr/bin/env python3
"""Fusion-strategy ablation: RRF vs minmax vs raw-score.

EduMap previously defaulted to RRF (``settings.hybrid_fusion_method``) and now
defaults to ``score`` on this corpus's evidence; the TOIS paper (Bruch et al.,
arXiv 2210.11934) finds tuned weighted fusion generally beats RRF — while
BGE-M3's own MIRACL numbers show hybrid beating pure dense by only +0.2
nDCG@10, i.e. fusion's marginal value can be near zero.  Those two facts point
in opposite directions, so the honest move is to **measure on our own corpus**
rather than assume either way.

**Single-corpus caveat:** the result that made ``score`` the default
(``fusion_ablation_expanded_20260923T130739Z.json``) is ``cs201`` only, and on
the n=20 sample set ``rrf`` and ``minmax`` tie.  A second corpus (cs301, seeded
by ``scripts/seed_cs301.py``) changes the ordering — ``minmax`` leads there —
so **no strategy is unconditionally best**.  Use ``--course`` to re-run against
another corpus; ``--list-courses`` shows what is available.

The cs301 corpus's labels are machine-drafted and **not yet human-reviewed**
(see its ``_meta.review_note``), so its specific deltas are preliminary; the
*ordering instability* is the robust part of the finding.

This script answers one question: *on our labelled query sets, which of the
three implemented fusion strategies produces the best retrieval quality, and
at what latency?*

Method notes
------------
* **Cache is bypassed** (``use_cache=False``).  A previous latency claim of
  "hybrid is ×0.64" turned out to be a retrieval-cache artefact; comparing
  latency with the cache on is meaningless.  A test asserts this script
  passes ``use_cache=False``.
* **Refuses to run against an empty index.**  ``VectorIndex`` silently falls
  back to an in-memory dict when ChromaDB is unreachable, which would produce
  metrics measured against nothing.  Same guard as
  ``run_strategy_comparison.py``: a wrong number is worse than no number.
* **Reports per-strategy latency separately**, because the three strategies
  have different cost profiles (RRF and score are O(n log n) sorts; minmax
  adds a normalisation pass per source).

Usage
-----
    cd services/backend-core
    python scripts/run_fusion_ablation.py [--dataset expanded|sample]
        [--course cs201] [--top-k 10] [--k 1,3,5,10]
        [--methods rrf,minmax,score] [--output benchmark_results]
    python scripts/run_fusion_ablation.py --list-courses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must run before any import that can pull in sentence_transformers: it issues
# a HEAD request to huggingface.co on model load even when the model is cached,
# and without outbound access huggingface_hub retries with exponential backoff
# (1s/2s/4s/…×5) — making the benchmark look hung. Diagnosed after a >15-minute
# stall. The cached weights are sufficient for benchmarking.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config import settings  # noqa: E402
from src.kg.connection import Neo4jPool  # noqa: E402
from src.kg.repositories.knowledge_point_repo import (  # noqa: E402
    KnowledgePointRepository,
)
from src.kg.vector_index import VectorIndex  # noqa: E402
from src.rag.evaluation.datasets import (  # noqa: E402
    available_courses,
    load_expanded_queries,
    load_sample_queries,
)
from src.rag.evaluation.metrics import (  # noqa: E402
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.rag.rag_service import RAGRetrievalService  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fusion_ablation")

ALL_METHODS = ("rrf", "minmax", "score")


def _bypass_proxy_for_localhost() -> None:
    """Keep local infra off the system proxy (macOS Clash/Surge trap)."""
    local = "localhost,127.0.0.1,::1"
    existing = os.environ.get("no_proxy", "")
    if "localhost" not in existing:
        os.environ["no_proxy"] = f"{existing},{local}".strip(",") if existing else local
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def build_rag_service() -> tuple[RAGRetrievalService, Neo4jPool | None]:
    """Construct the RAG service exactly as ``src.main`` does."""
    _bypass_proxy_for_localhost()

    pool: Neo4jPool | None = None
    kp_repo: KnowledgePointRepository | None = None
    try:
        pool = Neo4jPool(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
        kp_repo = KnowledgePointRepository(pool)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j unavailable — KG leg disabled: %s", exc)

    vector_index: VectorIndex | None = None
    try:
        vector_index = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ChromaDB unavailable — vector leg disabled: %s", exc)

    if vector_index is not None and getattr(vector_index, "_client", None) is None:
        raise SystemExit(
            "ChromaDB client could not be created — VectorIndex fell back to an "
            "in-memory store, so any metric would be measured against an EMPTY "
            "index.\n"
            f"  host={settings.chroma_host} port={settings.chroma_port}\n"
            "  Fix:  export no_proxy=localhost,127.0.0.1,::1"
        )

    service = RAGRetrievalService(
        vector_index=vector_index,
        kp_repo=kp_repo,
        embedding_model=settings.llm_embedding_model,
    )
    return service, pool


async def evaluate_method(
    service: RAGRetrievalService,
    queries: list[dict],
    method: str,
    *,
    top_k: int,
    k_values: list[int],
) -> dict:
    """Run one fusion strategy over the whole query set.

    Returns aggregated retrieval metrics plus average latency.  ``use_cache``
    is always False — see the module docstring.
    """
    service.fusion_method = method

    max_k = max(k_values)
    precisions: dict[int, list[float]] = {k: [] for k in k_values}
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    ndcgs: dict[int, list[float]] = {k: [] for k in k_values}
    hits: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []
    latencies: list[float] = []

    for item in queries:
        query = item["query"]
        relevant = item.get("relevant_ids") or set()

        started = time.perf_counter()
        results = await service.search(query, top_k=max_k, use_cache=False)
        latencies.append((time.perf_counter() - started) * 1000.0)

        mrrs.append(mean_reciprocal_rank(results, relevant))
        for k in k_values:
            precisions[k].append(precision_at_k(results, relevant, k))
            recalls[k].append(recall_at_k(results, relevant, k))
            ndcgs[k].append(ndcg_at_k(results, relevant, k))
            hits[k].append(hit_rate_at_k(results, relevant, k))

    n = max(len(queries), 1)
    if latencies:
        latencies.sort()
        avg_latency = round(sum(latencies) / n, 2)
        p50 = round(latencies[len(latencies) // 2], 2)
        p95 = round(latencies[min(len(latencies) - 1, int(n * 0.95))], 2)
    else:
        # Empty query set: report 0 rather than indexing an empty list.
        avg_latency = p50 = p95 = 0.0

    return {
        "name": method,
        "n_queries": len(queries),
        "mrr": round(sum(mrrs) / n, 4),
        "precision": {str(k): round(sum(v) / n, 4) for k, v in precisions.items()},
        "recall": {str(k): round(sum(v) / n, 4) for k, v in recalls.items()},
        "ndcg": {str(k): round(sum(v) / n, 4) for k, v in ndcgs.items()},
        "hit_rate": {str(k): round(sum(v) / n, 4) for k, v in hits.items()},
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def _print_table(results: dict[str, dict], k_values: list[int], top_k: int) -> None:
    k1 = k_values[0]
    k3 = 3 if 3 in k_values else k_values[-1]
    header = (
        f"{'method':8} {'MRR':>7} {'P@' + str(k1):>7} {'HR@' + str(k3):>7} "
        f"{'NDCG@' + str(k1):>9} {'P50(ms)':>9} {'P95(ms)':>9}"
    )
    print()
    print(f"融合策略消融 (top_k={top_k}, 缓存绕过)")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:8} {r['mrr']:>7.4f} {r['precision'][str(k1)]:>7.4f} "
            f"{r['hit_rate'][str(k3)]:>7.4f} {r['ndcg'][str(k1)]:>9.4f} "
            f"{r['p50_latency_ms']:>9.2f} {r['p95_latency_ms']:>9.2f}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="expanded", choices=["sample", "expanded"],
        help="评测集：sample(n=20) 或 expanded(n=200)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument(
        "--methods", default=",".join(ALL_METHODS),
        help="参与对比的融合策略，逗号分隔",
    )
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument(
        "--course", default="cs201",
        help=(
            "评测语料所属课程。默认 cs201 —— src/config.py 里 score 成为默认值的"
            "依据只在 cs201 上测过；换课程可检验该结论是否可外推。"
            "可用 --list-courses 查看磁盘上有哪些语料。"
        ),
    )
    parser.add_argument(
        "--list-courses", action="store_true",
        help="列出可用的评测语料后退出",
    )
    args = parser.parse_args()

    if args.list_courses:
        courses = available_courses()
        print("可用评测语料：", ", ".join(courses) if courses else "(无)")
        return 0

    k_values = [int(k) for k in args.k.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        logger.error("未知融合策略 %s（可选：%s）", unknown, ALL_METHODS)
        return 1

    loader = load_expanded_queries if args.dataset == "expanded" else load_sample_queries
    queries = loader(args.course)
    if not queries:
        logger.error(
            "评测集为空 — 课程 %s 没有 %s 语料（可用：%s）",
            args.course, args.dataset, ", ".join(available_courses()) or "(无)",
        )
        return 1
    logger.info("加载评测集: %s/%s, %d 条查询", args.course, args.dataset, len(queries))

    service, pool = build_rag_service()
    logger.info(
        "当前默认融合策略: %s（本脚本会被显式覆盖）", service.default_fusion_method
    )

    try:
        results: dict[str, dict] = {}
        for method in methods:
            logger.info("── 评测融合策略: %s ──", method)
            results[method] = await evaluate_method(
                service, queries, method, top_k=args.top_k, k_values=k_values
            )
    finally:
        if pool is not None:
            try:
                await pool.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("pool close failed: %s", exc)

    _print_table(results, k_values, args.top_k)

    best_mrr = max(results.items(), key=lambda kv: kv[1]["mrr"])[0]
    print()
    print(f"MRR 最优策略: {best_mrr}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"fusion_ablation_{args.dataset}_{stamp}.json"
    payload = {
        "course_id": args.course,
        "dataset": args.dataset,
        "n_queries": len(queries),
        "top_k": args.top_k,
        "k_values": k_values,
        "cache_bypassed": True,
        "results": results,
        "best_mrr_method": best_mrr,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结果已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
