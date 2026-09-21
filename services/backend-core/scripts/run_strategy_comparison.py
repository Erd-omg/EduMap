#!/usr/bin/env python3
"""Run the retrieval strategy comparison benchmark.

Evaluates the sample query dataset against three retrieval strategies
(direct vector-only / hybrid vector+KG / hybrid with LLM query rewrite)
and prints a side-by-side metric comparison.

Usage:
    cd services/backend-core
    python scripts/run_strategy_comparison.py [--dataset sample|expanded]
        [--top-k 10] [--k 1,3,5,10] [--strategies direct,hybrid,rewrite]
        [--output benchmark_results] [--repeat 1]

Requires a running Neo4j + ChromaDB stack (same as run_rag_benchmark.py).
Connection parameters come from ``settings`` so the script works both on the
host and inside Docker. Pass --no-llm to skip the rewrite strategy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from src.rag.evaluation.strategy_compare import StrategyComparator  # noqa: E402
from src.rag.rag_service import RAGRetrievalService  # noqa: E402
from src.utils.llm_adapter import create_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("strategy_compare")


def build_rag_service() -> tuple[RAGRetrievalService, Neo4jPool | None]:
    """Construct the RAG service the same way ``src.main`` does.

    Reads every connection parameter from ``settings`` (no hardcoded
    Docker hostnames) so the script runs from the host as well.

    Note: ``VectorIndex`` silently falls back to an in-memory dict when
    ChromaDB is unreachable, which would produce a benchmark measured
    against an EMPTY index.  We detect that and fail loudly instead —
    a wrong number is worse than no number.
    """
    _bypass_proxy_for_localhost()

    pool: Neo4jPool | None = None
    kp_repo: KnowledgePointRepository | None = None
    try:
        pool = Neo4jPool(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
        kp_repo = KnowledgePointRepository(pool)
    except Exception as exc:
        logger.warning("Neo4j unavailable — KG leg disabled: %s", exc)

    vector_index: VectorIndex | None = None
    try:
        vector_index = VectorIndex(
            host=settings.chroma_host, port=settings.chroma_port
        )
    except Exception as exc:
        logger.warning("ChromaDB unavailable — vector leg disabled: %s", exc)

    if vector_index is not None and getattr(vector_index, "_client", None) is None:
        raise SystemExit(
            "ChromaDB client could not be created — VectorIndex fell back to an "
            "in-memory store, so any benchmark would be measured against an EMPTY "
            "index.\n"
            f"  host={settings.chroma_host} port={settings.chroma_port}\n"
            "  Most likely cause on this machine: the macOS system proxy "
            "(127.0.0.1:7890) is intercepting httpx requests to localhost.\n"
            "  Fix:  export no_proxy=localhost,127.0.0.1,::1"
        )

    rag_service = RAGRetrievalService(
        vector_index=vector_index,
        kp_repo=kp_repo,
        embedding_model=settings.llm_embedding_model,
    )
    rag_service.fusion_method = settings.hybrid_fusion_method
    rag_service.fusion_k = settings.hybrid_rrf_k
    return rag_service, pool


def _require_real_llm() -> None:
    """Refuse to benchmark an LLM strategy against the mock adapter.

    With an empty ``llm_api_key`` the adapter returns a canned
    "演示模式" string for every call.  The rewrite strategy then fails on
    100% of queries and the comparison table shows three identical rows —
    a plausible-looking but entirely fake result.
    """
    if not settings.llm_api_key or settings.llm_model == "spark":
        raise SystemExit(
            "LLM_API_KEY is not configured (llm_model=%r) — the adapter would "
            "run in mock mode and every rewrite would silently fail, producing "
            "a fake comparison.\n"
            "  Check that the repo-root .env is present and readable.\n"
            "  Or drop the rewrite strategy: --strategies direct,hybrid"
            % (settings.llm_model,)
        )


def _bypass_proxy_for_localhost() -> None:
    """Ensure local infra is never routed through the system proxy.

    httpx honours ``trust_env`` and will send localhost traffic to a proxy
    if one is configured (common on macOS with Clash/Surge).  Unlike curl it
    does not read the system proxy exception list, so we set the no_proxy
    env vars ourselves before any client is constructed.
    """
    import os

    local = "localhost,127.0.0.1,::1"
    existing = os.environ.get("no_proxy", "")
    if "localhost" not in existing:
        os.environ["no_proxy"] = f"{existing},{local}".strip(",") if existing else local
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10, help="检索深度")
    parser.add_argument("--k", type=str, default="1,3,5,10", help="评测 K 值列表")
    parser.add_argument(
        "--strategies", type=str, default="direct,hybrid,rewrite",
        help="参与对比的策略，逗号分隔",
    )
    parser.add_argument("--no-llm", action="store_true", help="跳过 rewrite 策略")
    parser.add_argument(
        "--dataset", type=str, default="expanded", choices=["sample", "expanded"],
        help="评测集：sample(n=20) 或 expanded(n=200)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="重复跑同一评测集的次数（用于测量检索结果缓存的收益）",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results",
        help="结果输出目录",
    )
    args = parser.parse_args()

    k_values = [int(k) for k in args.k.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.no_llm and "rewrite" in strategies:
        strategies.remove("rewrite")

    loader = load_expanded_queries if args.dataset == "expanded" else load_sample_queries
    queries = loader()
    if not queries:
        logger.error("评测集为空 — 检查 datasets/%s_queries.json", args.dataset)
        return 1
    logger.info("加载评测集: %s, %d 条查询", args.dataset, len(queries))

    rag_service, pool = build_rag_service()

    llm = None
    if "rewrite" in strategies:
        _require_real_llm()
        llm = create_llm(settings)
        logger.info("LLM 已启用: 用于 query rewrite (%s)", settings.llm_model)

    comparator = StrategyComparator(rag_service, llm=llm, k_values=k_values)

    # Multiple passes: pass N>1 measures the retrieval-result cache.  Each
    # pass gets a fresh StrategyMetrics slot keyed f"{strategy}#pass{n}".
    passes: list[dict] = []
    result: dict = {}
    try:
        for pass_no in range(1, args.repeat + 1):
            logger.info("── 第 %d/%d 轮 ──", pass_no, args.repeat)
            result = await comparator.run(
                queries, top_k=args.top_k, strategies=strategies
            )
            passes.append(
                {
                    "pass": pass_no,
                    "latency_ms": {
                        name: m["avg_latency_ms"]
                        for name, m in result["strategies"].items()
                    },
                }
            )
    finally:
        if pool is not None:
            await pool.close()

    StrategyComparator.print_comparison(result)

    if len(passes) > 1:
        _print_repeat_summary(passes)

    result["passes"] = passes
    result["dataset"] = args.dataset

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"strategy_comparison_{timestamp}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("结果已保存: %s", out_path)
    return 0


def _print_repeat_summary(passes: list[dict]) -> None:
    """Print pass-1 vs pass-N latency to quantify the result cache."""
    first, last = passes[0], passes[-1]
    print("\n  🔁 多轮延迟对比（第 1 轮 vs 第 %d 轮）:" % last["pass"])
    for name in first["latency_ms"]:
        p1 = first["latency_ms"][name]
        pn = last["latency_ms"].get(name, 0.0)
        speedup = (p1 / pn) if pn > 0 else 0.0
        print(
            f"     {name:<10} 第1轮 {p1:>9.2f}ms → 第{last['pass']}轮 {pn:>9.2f}ms "
            f"(×{speedup:.2f})"
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
