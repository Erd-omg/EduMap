#!/usr/bin/env python3
"""Cross-encoder reranker ablation: off vs on.

Why this exists
---------------
``reranker_enabled`` defaults to False, and every recorded benchmark in the repo
was measured with it off — so the reranker's effect was never established in
either direction. The question is not academic: the NVIDIA evaluation of
cross-encoder rerankers (arXiv 2409.07691) found that a **too-small** reranker
actively *hurts* retrieval (a 33M MiniLM-L-12 cost −3 to −13 NDCG@10 depending
on the base retriever). EduMap's default is
``cross-encoder/ms-marco-MiniLM-L-6-v2`` — *smaller* than the one that study
found harmful.

So "just turn it on" is not a safe default, and this script measures the
actual effect on our own corpus rather than assuming the folklore answer
("+5-15 points") holds.

Method
------
* **Cache bypassed** (``use_cache=False``) — a previous latency claim turned out
  to be a cache artefact; see run_fusion_ablation.py.
* **Candidate fan-out**: the service widens the candidate pool to ``top_k × 3``
  when a reranker is attached. Both arms must therefore be evaluated at the
  *same* retrieval depth so the reranker is the only variable — the "on" arm
  retrieves the wider pool itself (as production does), so this is handled by
  toggling ``_reranker`` on the same service instance.
* **Reports NDCG and MRR plus latency**, because the decision is a trade.

Usage
-----
    cd services/backend-core
    python scripts/run_reranker_ablation.py [--dataset sample|expanded]
        [--top-k 5] [--k 1,3,5] [--output benchmark_results]
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config import settings  # noqa: E402
from src.kg.connection import Neo4jPool  # noqa: E402
from src.kg.repositories.knowledge_point_repo import (  # noqa: E402
    KnowledgePointRepository,
)
from src.kg.vector_index import VectorIndex  # noqa: E402
from src.rag.evaluation.datasets import (  # noqa: E402
    load_expanded_queries,
    load_sample_queries,
)
from src.rag.evaluation.metrics import (  # noqa: E402
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
)
from src.rag.rag_service import RAGRetrievalService  # noqa: E402
from src.rag.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("reranker_ablation")


def _bypass_proxy_for_localhost() -> None:
    local = "localhost,127.0.0.1,::1"
    existing = os.environ.get("no_proxy", "")
    if "localhost" not in existing:
        os.environ["no_proxy"] = f"{existing},{local}".strip(",") if existing else local
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def build_service() -> tuple[RAGRetrievalService, Neo4jPool | None]:
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
            "in-memory store, so any metric would be measured against an EMPTY index.\n"
            f"  host={settings.chroma_host} port={settings.chroma_port}"
        )

    service = RAGRetrievalService(
        vector_index=vector_index,
        kp_repo=kp_repo,
        embedding_model=settings.llm_embedding_model,
    )
    return service, pool


async def evaluate_arm(
    service: RAGRetrievalService,
    queries: list[dict],
    *,
    label: str,
    top_k: int,
    k_values: list[int],
) -> dict:
    """Score one arm (reranker off or on) over the whole query set."""
    max_k = max(k_values)
    mrrs: list[float] = []
    ndcgs: dict[int, list[float]] = {k: [] for k in k_values}
    hits: dict[int, list[float]] = {k: [] for k in k_values}
    precisions: dict[int, list[float]] = {k: [] for k in k_values}
    latencies: list[float] = []
    reranked_flags: list[bool] = []

    for item in queries:
        query = item["query"]
        relevant = item.get("relevant_ids") or set()

        started = time.perf_counter()
        results = await service.search(query, top_k=max_k, use_cache=False)
        latencies.append((time.perf_counter() - started) * 1000.0)
        reranked_flags.append(getattr(service, "_reranker", None) is not None)

        mrrs.append(mean_reciprocal_rank(results, relevant))
        for k in k_values:
            ndcgs[k].append(ndcg_at_k(results, relevant, k))
            hits[k].append(hit_rate_at_k(results, relevant, k))
            precisions[k].append(precision_at_k(results, relevant, k))

    n = max(len(queries), 1)
    latencies.sort()
    return {
        "arm": label,
        "n_queries": len(queries),
        "reranker_active": all(reranked_flags) if reranked_flags else False,
        "mrr": round(sum(mrrs) / n, 4),
        "ndcg": {str(k): round(sum(v) / n, 4) for k, v in ndcgs.items()},
        "hit_rate": {str(k): round(sum(v) / n, 4) for k, v in hits.items()},
        "precision": {str(k): round(sum(v) / n, 4) for k, v in precisions.items()},
        "p50_latency_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
        "p95_latency_ms": round(latencies[min(len(latencies) - 1, int(n * 0.95))], 2) if latencies else 0.0,
    }


def _print_comparison(off: dict, on: dict, k_values: list[int]) -> None:
    k1 = k_values[0]
    header = f"{'arm':22} {'MRR':>7} {'NDCG@' + str(k1):>9} {'HR@' + str(k1):>7} {'P50(ms)':>9} {'P95(ms)':>9}"
    print()
    print(f"Reranker 消融（缓存绕过，模型={settings.reranker_model}）")
    print(header)
    print("-" * len(header))
    for arm in (off, on):
        print(
            f"{arm['arm']:22} {arm['mrr']:>7.4f} {arm['ndcg'][str(k1)]:>9.4f} "
            f"{arm['hit_rate'][str(k1)]:>7.4f} {arm['p50_latency_ms']:>9.2f} "
            f"{arm['p95_latency_ms']:>9.2f}"
        )

    d_mrr = on["mrr"] - off["mrr"]
    d_ndcg = on["ndcg"][str(k1)] - off["ndcg"][str(k1)]
    lat_x = on["p50_latency_ms"] / off["p50_latency_ms"] if off["p50_latency_ms"] else 0.0

    print()
    print(f"· ΔMRR = {d_mrr:+.4f}   ΔNDCG@{k1} = {d_ndcg:+.4f}   延迟 ×{lat_x:.2f}")

    # Three outcomes, not two. A reranker that changes nothing while doubling
    # latency is a *negative* result, and the earlier version of this script
    # reported it as "✅ 开启后指标提升" because it only tested `>= 0` —
    # exactly the kind of unearned conclusion this harness exists to prevent.
    epsilon = 1e-4
    if d_mrr < -epsilon or d_ndcg < -epsilon:
        print("· ⚠️ **开启 reranker 反而变差** —— 与 NVIDIA 的发现一致：过小的")
        print("  cross-encoder 会伤害检索（arXiv 2409.07691）。保持默认关闭是正确的。")
    elif abs(d_mrr) <= epsilon and abs(d_ndcg) <= epsilon:
        print("· ➖ **指标无变化，但延迟显著上升** —— 这是负结果：付出了代价却没有收益。")
        print("  保持默认关闭正确；若将来要启用，应先换更大的模型（如 bge-reranker-v2-m3）再测。")
    else:
        print(f"· ✅ 开启后指标提升（ΔMRR {d_mrr:+.4f}），代价是延迟 ×{lat_x:.2f}。")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="sample", choices=["sample", "expanded"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--k", default="1,3,5")
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument(
        "--model", default=None,
        help="覆盖 settings.reranker_model（需模型已在本地缓存）",
    )
    args = parser.parse_args()

    k_values = [int(k) for k in args.k.split(",")]
    loader = load_expanded_queries if args.dataset == "expanded" else load_sample_queries
    queries = loader()
    if not queries:
        logger.error("评测集为空")
        return 1
    logger.info("加载评测集: %s, %d 条", args.dataset, len(queries))

    service, pool = build_service()
    model_name = args.model or settings.reranker_model

    reranker = CrossEncoderReranker(model_name=model_name)
    reranker.load()
    if not reranker.is_loaded:
        raise SystemExit(
            f"无法加载 reranker 模型 '{model_name}'。本机无外网，只有已缓存的模型可用。"
        )

    try:
        logger.info("── arm 1/2: reranker 关闭 ──")
        service._reranker = None
        off = await evaluate_arm(
            service, queries, label="reranker=off", top_k=args.top_k, k_values=k_values
        )

        logger.info("── arm 2/2: reranker 开启 (%s) ──", model_name)
        service._reranker = reranker
        on = await evaluate_arm(
            service, queries, label=f"reranker=on", top_k=args.top_k, k_values=k_values
        )
    finally:
        if pool is not None:
            try:
                await pool.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("pool close failed: %s", exc)

    _print_comparison(off, on, k_values)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"reranker_ablation_{args.dataset}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_queries": len(queries),
                "top_k": args.top_k,
                "k_values": k_values,
                "reranker_model": model_name,
                "cache_bypassed": True,
                "off": off,
                "on": on,
                "delta_mrr": round(on["mrr"] - off["mrr"], 4),
                "delta_ndcg": round(on["ndcg"][str(k_values[0])] - off["ndcg"][str(k_values[0])], 4),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("结果已写入 %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
