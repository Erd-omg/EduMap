#!/usr/bin/env python3
"""EduMap RAG 混合检索消融实验 (ablation) 基准脚本.

对比三种融合策略 (score / minmax / rrf) 在真实 ChromaDB + Neo4j 数据上的
检索质量差异，输出 precision / recall / MRR / NDCG / hit_rate 对比表，
可选开启 Cross-Encoder 重排与查询扩展做进一步消融。

前置条件（依赖运行中的 Docker 服务 postgres / neo4j / chromadb）:
    ./start-edumap.command

用法示例:
    # 默认: 依次跑 score / minmax / rrf 三组并输出对比表
    python scripts/run_rag_benchmark.py

    # 只跑一种融合策略
    python scripts/run_rag_benchmark.py --fusion rrf

    # 调整 RRF k 参数 / 查询条数 / 评测集 / 开启重排与查询扩展
    python scripts/run_rag_benchmark.py --fusion rrf --rrf-k 60 \\
        --dataset expanded --limit 20 --reranker --expand-query

    # 结果落盘为 JSON
    python scripts/run_rag_benchmark.py --json-out docs/benchmark/ablation.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "backend-core"))

from src.config import settings  # noqa: E402
from src.kg.connection import Neo4jPool  # noqa: E402
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository  # noqa: E402
from src.kg.vector_index import VectorIndex  # noqa: E402
from src.rag.evaluation.datasets import (  # noqa: E402
    load_expanded_queries,
    load_sample_queries,
)
from src.rag.evaluation.evaluator import RAGEvaluator  # noqa: E402
from src.rag.evaluation.metrics import hit_rate_at_k  # noqa: E402
from src.rag.rag_service import RAGRetrievalService  # noqa: E402

logger = logging.getLogger("rag_benchmark")

FUSION_METHODS = ("score", "minmax", "rrf")
DATASET_LOADERS = {
    "sample": load_sample_queries,
    "expanded": load_expanded_queries,
}


def _bypass_system_proxy_for_local() -> None:
    """豁免本机地址，绕过 macOS 系统代理。

    httpx (trust_env) 会读取 macOS 系统代理设置（scutil --proxy，如 Clash
    127.0.0.1:7890），导致发往本机 ChromaDB 的请求被代理拦截并返回 502。
    显式设置 no_proxy 让本机地址直连。
    """
    bypass_hosts = {settings.chroma_host, "localhost", "127.0.0.1", "::1"}
    existing = os.environ.get("no_proxy", "")
    merged = ",".join(sorted({*filter(None, existing.split(",")), *bypass_hosts}))
    os.environ["no_proxy"] = merged
    os.environ["NO_PROXY"] = merged
    # 模型已缓存在本地时强制离线，避免 huggingface.co 连通性检查卡死。
    # 需在线下载新模型时：HF_HUB_OFFLINE=0 且 HF_ENDPOINT=https://hf-mirror.com（国内镜像）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG 混合检索融合策略消融实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fusion",
        default="all",
        choices=[*FUSION_METHODS, "all"],
        help="要评测的融合策略；'all' 依次评测三种并输出对比表",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数 k")
    parser.add_argument(
        "--dataset",
        default="sample",
        choices=sorted(DATASET_LOADERS),
        help="评测数据集",
    )
    parser.add_argument("--limit", type=int, default=0, help="限制查询条数（0=全部）")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回条数")
    parser.add_argument("--reranker", action="store_true", help="开启 Cross-Encoder 重排")
    parser.add_argument("--expand-query", action="store_true", help="开启查询扩展")
    parser.add_argument("--json-out", type=Path, default=None, help="结果 JSON 落盘路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出 DEBUG 日志")
    return parser.parse_args()


def _check_services() -> None:
    """确保 ChromaDB / Neo4j 可达，否则给出友好提示后退出。"""
    try:
        vector_index = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
        if vector_index._client is None:
            raise RuntimeError("ChromaDB client unavailable")
        vector_index._client.heartbeat()
    except Exception as exc:  # pragma: no cover - 运维提示
        print(
            f"[错误] ChromaDB ({settings.chroma_host}:{settings.chroma_port}) 不可达: {exc}\n"
            "请先启动 Docker 服务: ./start-edumap.command",
            file=sys.stderr,
        )
        sys.exit(1)


async def _run_ablation(args: argparse.Namespace) -> dict:
    pool = Neo4jPool(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    try:
        kp_repo = KnowledgePointRepository(pool)
        vector_index = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
        rag_service = RAGRetrievalService(
            vector_index=vector_index,
            kp_repo=kp_repo,
            embedding_model=settings.llm_embedding_model,
        )

        # 消融开关：查询扩展 / 重排（默认关闭，保证只对比融合策略本身）
        rag_service._expand_query_enabled = args.expand_query
        if args.reranker:
            from src.rag.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name=settings.reranker_model)
            reranker.load()
            rag_service._reranker = reranker
            logger.info("Cross-Encoder reranker loaded: %s", settings.reranker_model)

        # 预热 embedding 模型，避免首次加载计入首轮评测
        logger.info("Pre-warming embedding model: %s", settings.llm_embedding_model)
        await rag_service.search("预热查询", top_k=1)

        queries = DATASET_LOADERS[args.dataset]()
        if args.limit and args.limit < len(queries):
            queries = queries[: args.limit]
        if not queries:
            raise RuntimeError(f"评测数据集 '{args.dataset}' 为空，无法运行消融实验")
        logger.info("Loaded %d queries from dataset '%s'", len(queries), args.dataset)

        evaluator = RAGEvaluator(k_values=[1, 3, 5])
        methods = FUSION_METHODS if args.fusion == "all" else (args.fusion,)
        max_k = max(evaluator._k_values)

        results: dict[str, dict] = {}
        for method in methods:
            rag_service.fusion_method = method
            rag_service.fusion_k = args.rrf_k
            logger.info("=== 融合策略: %s (rrf_k=%d) ===", method, args.rrf_k)

            started = time.perf_counter()
            eval_result = await evaluator.evaluate_retrieval(
                queries=queries,
                retrieval_fn=rag_service.search,
            )
            elapsed = time.perf_counter() - started

            hit_rates: dict[str, float] = {}
            for k in evaluator._k_values:
                hits = []
                for q_data in queries:
                    relevant = q_data["relevant_ids"]
                    if isinstance(relevant, list):
                        relevant = set(relevant)
                    search_results = await rag_service.search(q_data["query"], top_k=max_k)
                    hits.append(hit_rate_at_k(search_results, relevant, k))
                hit_rates[str(k)] = round(sum(hits) / max(len(hits), 1), 4)

            metrics = eval_result.to_dict()
            metrics["hit_rate"] = hit_rates
            metrics["elapsed_seconds"] = round(elapsed, 2)
            results[method] = metrics
        return {
            "dataset": args.dataset,
            "n_queries": len(queries),
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "reranker": args.reranker,
            "expand_query": args.expand_query,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }
    finally:
        await pool.close()


def _print_table(report: dict) -> None:
    """以表格形式打印消融结果对比。"""
    results = report["results"]
    ks = ["1", "3", "5"]

    header = f"{'fusion':<8}{'overall':>9}{'MRR':>8}"
    for k in ks:
        header += f"{'P@' + k:>7}{'R@' + k:>7}{'NDCG@' + k:>9}{'HR@' + k:>7}"
    header += f"{'time(s)':>9}"

    lines = [header, "-" * len(header)]
    for method, m in results.items():
        row = f"{method:<8}{m['overall']:>9}{m['mrr']:>8}"
        for k in ks:
            row += (
                f"{m['precision'].get(k, 0):>7}"
                f"{m['recall'].get(k, 0):>7}"
                f"{m['ndcg'].get(k, 0):>9}"
                f"{m['hit_rate'].get(k, 0):>7}"
            )
        row += f"{m['elapsed_seconds']:>9}"
        lines.append(row)

    print("\nRAG 混合检索融合策略消融实验")
    print(f"数据集: {report['dataset']}  查询数: {report['n_queries']}  "
          f"rrf_k: {report['rrf_k']}  "
          f"reranker: {report['reranker']}  query_expansion: {report['expand_query']}")
    print("\n".join(lines))
    print()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _bypass_system_proxy_for_local()
    _check_services()
    report = asyncio.run(_run_ablation(args))
    _print_table(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入: {args.json_out}")


if __name__ == "__main__":
    main()
