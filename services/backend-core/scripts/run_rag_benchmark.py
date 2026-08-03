"""Comprehensive RAG Benchmark Runner.

Usage:
    cd services/backend-core && python scripts/run_rag_benchmark.py

Runs retrieval + generation evaluation against the sample query dataset
and KG-generated queries, saves results to benchmark_results/.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configure paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE_DIR))

from src.kg.connection import Neo4jPool
from src.kg.vector_index import VectorIndex
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
from src.rag.rag_service import RAGRetrievalService
from src.rag.reranking.cross_encoder import CrossEncoderReranker
from src.config import settings
from src.rag.evaluation.metrics import (
    precision_at_k, recall_at_k, mean_reciprocal_rank,
    ndcg_at_k, hit_rate_at_k, faithfulness, answer_relevancy,
    semantic_faithfulness,
)
from src.rag.evaluation.datasets import load_sample_queries, load_expanded_queries
from src.rag.models import RAGResult


# ═══════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

class RAGBenchmarkRunner:
    """Full RAG evaluation runner with result persistence."""

    K_VALUES = [1, 3, 5]

    def __init__(self) -> None:
        self._pool: Neo4jPool | None = None
        self._rag: RAGRetrievalService | None = None
        self._results_cache: dict[str, list[RAGResult]] = {}
        self._llm_adapter = None

    async def connect(self) -> None:
        """Connect to Neo4j + VectorIndex and set up RAG service."""
        self._pool = Neo4jPool("bolt://neo4j:7687", "neo4j", "edumap_dev")
        kp_repo = KnowledgePointRepository(self._pool)
        vi = VectorIndex(host="chromadb", port=8000)
        self._rag = RAGRetrievalService(
            vector_index=vi,
            kp_repo=kp_repo,
            embedding_model=settings.llm_embedding_model,
        )

        # ── Cross-Encoder Reranker (optional) ──────────────────────────────
        if settings.reranker_enabled:
            try:
                reranker = CrossEncoderReranker(model_name=settings.reranker_model)
                reranker.load()
                if reranker.is_loaded:
                    self._rag._reranker = reranker
                    logger.info("Reranker loaded and wired: %s", settings.reranker_model)
            except Exception as exc:
                logger.warning("Reranker failed to load (continuing without): %s", exc)

        # ── KG-based Query Expansion (optional) ─────────────────────────────
        if settings.expand_query_enabled:
            self._rag._expand_query_enabled = True
            self._rag._expand_query_max_terms = settings.expand_query_max_terms
            logger.info("KG query expansion enabled (max %d terms)", settings.expand_query_max_terms)

        # Try to load LLM adapter for generation evaluation
        try:
            from src.utils.llm_adapter import create_llm
            if getattr(settings, "llm_api_key", None):
                self._llm_adapter = create_llm(settings)
                logger.info("LLM adapter loaded for generation evaluation")
            else:
                logger.info("No LLM API key — skipping generation evaluation")
        except Exception as exc:
            logger.info("LLM adapter unavailable (%s) — skipping generation eval", exc)

        logger.info("RAG service ready")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── KP Coverage ─────────────────────────────────────────────

    async def check_kp_coverage(self, queries: list[dict]) -> dict:
        """Check which dataset KPs exist in Neo4j."""
        pool = self._pool
        all_ids = set()
        for q in queries:
            all_ids.update(q.get("relevant_ids", []))
        # Fetch all KP IDs from Neo4j
        rows = await pool.execute_read("MATCH (k:KnowledgePoint) RETURN k.id AS id")
        existing_ids = {r["id"] for r in rows}
        missing = sorted(all_ids - existing_ids)
        return {
            "total_dataset_kps": len(all_ids),
            "found_in_neo4j": len(all_ids & existing_ids),
            "missing": missing,
            "coverage_pct": round(len(all_ids & existing_ids) / max(len(all_ids), 1) * 100, 1),
        }

    # ── Search with caching ──────────────────────────────────────

    async def search_cached(self, query: str, top_k: int = 10) -> list[RAGResult]:
        """Search with result caching to avoid duplicate calls."""
        cache_key = f"{query}:{top_k}"
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]
        results = await self._rag.search(query, top_k=top_k)
        self._results_cache[cache_key] = results
        return results

    # ── Latency Benchmark ─────────────────────────────────────────

    async def benchmark_latency(
        self, queries: list[dict], iterations: int = 100,
    ) -> dict:
        """Run latency benchmark — measures full pipeline P50/P95/P99.

        Times only the full ``RAGRetrievalService.search()`` call (the
        public API surface), not internal components, to avoid interfering
        with the service's internal state. Each iteration is a fresh
        search against the running backends.
        """
        import time as time_module
        import statistics

        if not queries:
            return {"error": "no queries"}

        n_queries = len(queries)
        # Round-robin across all unique queries
        query_cycle = [queries[i % n_queries] for i in range(iterations)]

        # Warmup: run 3 searches to load the embedding model (excluded from stats)
        logger.info("  Warming up (3 calls, excluded from stats)...")
        for q_data in queries[:3]:
            _ = await self._rag.search(q_data["query"], top_k=5)

        latencies: list[float] = []

        for run_idx, q_data in enumerate(query_cycle):
            query = q_data["query"]
            t0 = time_module.perf_counter()
            _ = await self._rag.search(query, top_k=10)
            latencies.append((time_module.perf_counter() - t0) * 1000)

            if (run_idx + 1) % 20 == 0:
                logger.info("  Latency run %d/%d", run_idx + 1, iterations)

        def percentile(data: list[float], pct: int) -> float:
            if not data:
                return 0.0
            s = sorted(data)
            idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
            return round(s[idx], 2)

        return {
            "iterations": iterations,
            "n_unique_queries": n_queries,
            "full_pipeline_ms": {
                "n": len(latencies),
                "min": round(min(latencies), 2),
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": round(max(latencies), 2),
                "avg": round(sum(latencies) / len(latencies), 2),
            },
        }

    def print_latency_report(self, latency: dict) -> None:
        """Print formatted latency benchmark report."""
        sep = "─" * 55
        print(f"\n⏱️  端到端延迟评测（{latency['iterations']} 次调用，{latency['n_unique_queries']} 条不同查询）")
        print(sep)
        print(f"  {'阶段':<22} {'P50':<8} {'P95':<8} {'P99':<8} {'平均':<8}")
        print(sep)

        for label, key in [
            ("Full Pipeline", "full_pipeline_ms"),
            ("  ChromaDB 搜索", "chroma_search_ms"),
            ("  Neo4j 搜索", "neo4j_search_ms"),
            ("  Embedding", "embedding_ms"),
        ]:
            stats = latency.get(key, {})
            if stats.get("n", 0) > 0:
                print(f"  {label:<22} {stats['p50']:<8} {stats['p95']:<8} "
                      f"{stats['p99']:<8} {stats['avg']:<8}")

        # Overall summary
        total = latency.get("full_pipeline_ms", {})
        print(sep)
        print(f"  ✅ P50={total.get('p50', '-')}ms  |  P95={total.get('p95', '-')}ms  "
              f"|  P99={total.get('p99', '-')}ms  |  max={total.get('max', '-')}ms")

    # ── Retrieval Evaluation ─────────────────────────────────────

    async def evaluate_retrieval(
        self, queries: list[dict], top_k: int = 10,
    ) -> dict:
        """Run retrieval evaluation with difficulty breakdown."""
        max_k = max(self.K_VALUES)

        # Per-query results
        per_query: list[dict] = []
        for q_data in queries:
            query = q_data["query"]
            relevant = set(q_data.get("relevant_ids", []))
            difficulty = q_data.get("difficulty", "basic")
            results = await self.search_cached(query, top_k=max_k)

            per_query.append({
                "query": query,
                "difficulty": difficulty,
                "relevant_ids": sorted(relevant),
                "top_k_ids": [r.source_id for r in results],
                "top_k_names": [r.source_name for r in results],
                "top_k_scores": [round(r.score, 4) for r in results],
                "precision": {str(k): round(precision_at_k(results, relevant, k), 4) for k in self.K_VALUES},
                "recall": {str(k): round(recall_at_k(results, relevant, k), 4) for k in self.K_VALUES},
                "mrr": round(mean_reciprocal_rank(results, relevant), 4),
                "ndcg": {str(k): round(ndcg_at_k(results, relevant, k), 4) for k in self.K_VALUES},
                "hit_rate": {str(k): round(hit_rate_at_k(results, relevant, k), 4) for k in self.K_VALUES},
            })

        # Aggregate overall
        overall = self._aggregate(per_query)

        # Difficulty breakdown
        difficulties = {}
        for diff in ("basic", "intermediate", "advanced"):
            subset = [pq for pq in per_query if pq["difficulty"] == diff]
            if subset:
                difficulties[diff] = self._aggregate(subset)

        return {
            "overall": overall,
            "by_difficulty": difficulties,
            "per_query": per_query,
        }

    @staticmethod
    def _aggregate(results: list[dict]) -> dict:
        """Aggregate per-query results into overall metrics."""
        n = len(results)
        if n == 0:
            return {}
        return {
            "n_queries": n,
            "precision": {
                k: round(sum(r["precision"][k] for r in results) / n, 4)
                for k in results[0]["precision"]
            },
            "recall": {
                k: round(sum(r["recall"][k] for r in results) / n, 4)
                for k in results[0]["recall"]
            },
            "mrr": round(sum(r["mrr"] for r in results) / n, 4),
            "ndcg": {
                k: round(sum(r["ndcg"][k] for r in results) / n, 4)
                for k in results[0]["ndcg"]
            },
            "hit_rate": {
                k: round(sum(r["hit_rate"][k] for r in results) / n, 4)
                for k in results[0]["hit_rate"]
            },
        }

    # ── Generation Evaluation ────────────────────────────────────

    async def evaluate_generation(self, queries: list[dict]) -> dict | None:
        """Use LLM to answer queries, then evaluate faithfulness + relevancy.

        Only runs if LLM adapter is available.
        """
        if self._llm_adapter is None:
            return None

        llm = self._llm_adapter
        max_k = max(self.K_VALUES)

        generation_results = []
        for q_data in queries:
            query = q_data["query"]
            relevant = set(q_data.get("relevant_ids", []))
            difficulty = q_data.get("difficulty", "basic")
            expected_points = q_data.get("expected_answer_points", [])

            # Retrieve context (search + KG expansion)
            results = await self.search_cached(query, top_k=max_k)
            context = await self._rag.assemble_context(results, max_chars=2000)

            # Expand context with prerequisite/related KP info from KG
            kg_extra = await self._expand_context_with_kg(results, max_chars=1000)
            if kg_extra:
                enriched_context_str = context.context_str + "\n\n" + kg_extra
                enriched_context = context  # keep same object, just update str
                enriched_context.context_str = enriched_context_str
                context = enriched_context

            # Generate answer using LLM
            answer = await self._generate_answer(llm, query, context)
            if not answer:
                continue

            # Evaluate (both token-overlap and semantic)
            f_result = faithfulness(answer, [context.context_str])
            sem_result = semantic_faithfulness(
                answer, [context.context_str],
                embedding_model=getattr(self._rag, "_model", None),
            )
            relevancy = answer_relevancy(answer, query)

            generation_results.append({
                "query": query,
                "difficulty": difficulty,
                "answer": answer,
                "context_length": len(context.context_str),
                "faithfulness": f_result,
                "semantic_faithfulness": sem_result["faithfulness"],
                "avg_semantic_similarity": sem_result["avg_similarity"],
                "answer_relevancy": round(relevancy, 4),
                "expected_answer_points": expected_points,
            })

        if not generation_results:
            return None

        # Aggregate
        n = len(generation_results)
        avg_faithfulness = round(
            sum(r["faithfulness"]["faithfulness"] for r in generation_results) / n, 4
        )
        avg_semantic_faithfulness = round(
            sum(r["semantic_faithfulness"] for r in generation_results) / n, 4
        )
        avg_semantic_sim = round(
            sum(r["avg_semantic_similarity"] for r in generation_results) / n, 4
        )
        avg_relevancy = round(
            sum(r["answer_relevancy"] for r in generation_results) / n, 4
        )
        avg_context_length = round(
            sum(r["context_length"] for r in generation_results) / n, 1
        )

        aggregated = {
            "n_queries": n,
            "avg_faithfulness": avg_faithfulness,
            "avg_semantic_faithfulness": avg_semantic_faithfulness,
            "avg_semantic_similarity": avg_semantic_sim,
            "avg_answer_relevancy": avg_relevancy,
            "avg_context_length": avg_context_length,
            "per_query": generation_results,
        }

        # Optional LLM-as-Judge for a subset
        try:
            from src.rag.evaluation.llm_judge import llm_faithfulness as llm_f_judge
            llm_scores = []
            for sample in generation_results[:3]:
                judge = await llm_f_judge(
                    answer=sample["answer"],
                    context=context.context_str if "context" in locals() else "",
                    llm=llm,
                )
                llm_scores.append(judge.get("faithfulness_score", 0))
            if llm_scores:
                aggregated["llm_faithfulness_avg"] = round(sum(llm_scores) / len(llm_scores), 4)
        except Exception as exc:
            logger.warning("LLM-as-judge failed: %s", exc)

        return aggregated

    async def _expand_context_with_kg(
        self, results: list[RAGResult], max_chars: int = 2000,
    ) -> str:
        """Expand search context with prerequisite/related KP descriptions from Neo4j."""
        if self._pool is None:
            return ""

        # Collect unique KP IDs from search results
        kp_ids = set()
        for r in results:
            if r.source_id.startswith("kp-") and r.source_type in ("chroma", "neo4j"):
                kp_ids.add(r.source_id)

        if not kp_ids:
            return ""

        extra_parts: list[str] = []
        char_count = 0

        for kp_id in list(kp_ids)[:5]:  # Limit to 5 KPs
            try:
                # Fetch prerequisites
                prereq_rows = await self._pool.execute_read(
                    "MATCH (a:KnowledgePoint)-[:PREREQUISITE_OF]->(b:KnowledgePoint) "
                    "WHERE b.id = $kp_id RETURN a.name, a.description, a.difficulty, a.category",
                    {"kp_id": kp_id},
                )
                for row in prereq_rows:
                    text = f"[前置] {row['name']}: {row.get('description', '')} (难度: {row.get('difficulty', '')})"
                    if char_count + len(text) > max_chars:
                        break
                    extra_parts.append(text)
                    char_count += len(text)

                # Fetch related KPs
                related_rows = await self._pool.execute_read(
                    "MATCH (a:KnowledgePoint)-[:RELATED_TO]->(b:KnowledgePoint) "
                    "WHERE a.id = $kp_id RETURN b.name, b.description, b.difficulty",
                    {"kp_id": kp_id},
                )
                for row in related_rows:
                    text = f"[相关] {row['name']}: {row.get('description', '')} (难度: {row.get('difficulty', '')})"
                    if char_count + len(text) > max_chars:
                        break
                    extra_parts.append(text)
                    char_count += len(text)

            except Exception:
                continue

        return "\n".join(extra_parts)

    async def _generate_answer(
        self, llm, query: str, context,
    ) -> str:
        """Generate an answer using the LLM based on retrieved context."""
        prompt = f"""你是一个数据结构与算法的辅导老师。请基于以下参考资料回答问题。

参考资料：
{context.context_str}

学生问题：{query}

要求：
- 只使用参考资料中的信息回答
- 如果参考资料不足以回答问题，请明确说明
- 使用中文回答
- 用 [N] 标注信息来源"""

        try:
            result = await llm.generate(prompt)
            text = result.content.strip() if hasattr(result, "content") else str(result).strip()
            return text[:2000]
        except Exception as exc:
            logger.warning("LLM generation failed for query '%s': %s", query[:30], exc)
            return f"（LLM 生成失败: {exc}）"

    # ── Report ───────────────────────────────────────────────────

    def print_report(self, coverage: dict, retrieval: dict, generation: dict | None) -> None:
        """Print formatted benchmark report to terminal."""
        sep = "=" * 60
        print(f"\n{sep}")
        print("  RAG 评测报告")
        print(f"  时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"{sep}")

        # Active features
        features = []
        if self._rag and self._rag._reranker is not None:
            features.append("Reranker")
        if self._rag and self._rag._expand_query_enabled:
            features.append("Query Expansion")
        print(f"  Active Features: {' + '.join(features) if features else '(none — baseline)'}")
        print(f"{sep}")

        # KP Coverage
        print(f"\n📊 知识图谱覆盖率")
        print(f"  - 数据集引用 KPs: {coverage['total_dataset_kps']}")
        print(f"  - Neo4j 中存在:   {coverage['found_in_neo4j']}")
        print(f"  - 覆盖率:         {coverage['coverage_pct']}%")
        if coverage["missing"]:
            print(f"  ❌ 缺失 KPs: {', '.join(coverage['missing'])}")
        else:
            print(f"  ✅ 完整覆盖")

        # Retrieval metrics
        overall = retrieval["overall"]
        print(f"\n📈 检索质量指标（整体，n={overall['n_queries']}）")
        print(f"  {'Metric':<20} {'@1':<10} {'@3':<10} {'@5':<10}")
        print(f"  {'─'*50}")
        for name, key in [("Precision", "precision"), ("Recall", "recall"),
                           ("NDCG", "ndcg"), ("Hit Rate", "hit_rate")]:
            vals = overall.get(key, {})
            print(f"  {name:<20} {vals.get('1', '-'):<10} {vals.get('3', '-'):<10} {vals.get('5', '-'):<10}")
        print(f"  {'MRR':<20} {'─':<10} {overall.get('mrr', '-'):<10} {'─':<10}")

        # Difficulty breakdown
        by_diff = retrieval.get("by_difficulty", {})
        if by_diff:
            print(f"\n📊 按难度分解")
            print(f"  {'难度':<15} {'n':<5} {'P@1':<8} {'P@3':<8} {'R@3':<8} {'MRR':<8} {'HR@3':<8}")
            print(f"  {'─'*57}")
            for diff in ("basic", "intermediate", "advanced"):
                d = by_diff.get(diff, {})
                if not d:
                    continue
                print(f"  {diff:<15} {d.get('n_queries', 0):<5} "
                      f"{d.get('precision', {}).get('1', '-'):<8} "
                      f"{d.get('precision', {}).get('3', '-'):<8} "
                      f"{d.get('recall', {}).get('3', '-'):<8} "
                      f"{d.get('mrr', '-'):<8} "
                      f"{d.get('hit_rate', {}).get('3', '-'):<8}")

        # Generation metrics
        if generation:
            print(f"\n🤖 生成质量指标（LLM-as-Judge, n={generation['n_queries']}）")
            print(f"  {'指标':<30} {'分数':<10}")
            print(f"  {'─'*40}")
            print(f"  {'Faithfulness (词重叠)':<30} {generation.get('avg_faithfulness', '-'):<10}")
            print(f"  {'Faithfulness (语义)':<30} {generation.get('avg_semantic_faithfulness', '-'):<10}")
            print(f"  {'语义相似度 (平均)':<30} {generation.get('avg_semantic_similarity', '-'):<10}")
            print(f"  {'Answer Relevancy':<30} {generation.get('avg_answer_relevancy', '-'):<10}")
            if "llm_faithfulness_avg" in generation:
                print(f"  {'Faithfulness (LLM Judge)':<30} {generation['llm_faithfulness_avg']:<10}")
            print(f"  {'平均上下文长度':<30} {generation.get('avg_context_length', '-'):<10} chars")

        print(f"\n{'─'*60}\n")

    def save_report(
        self, name: str, coverage: dict, retrieval: dict,
        generation: dict | None, extra: dict | None = None,
    ) -> str:
        """Save benchmark report to JSON file."""
        report = {
            "report_name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "k_values": self.K_VALUES,
                "chromadb_host": "chromadb",
                "chromadb_port": 8000,
                "embedding_model": "BAAI/bge-small-zh-v1.5",
                "reranker_enabled": self._rag._reranker is not None if self._rag else False,
                "reranker_model": settings.reranker_model if (self._rag and self._rag._reranker is not None) else None,
                "expand_query_enabled": self._rag._expand_query_enabled if self._rag else False,
                "expand_query_max_terms": self._rag._expand_query_max_terms if self._rag else 5,
            },
            "kp_coverage": coverage,
            "retrieval_metrics": {
                "overall": retrieval["overall"],
                "by_difficulty": retrieval["by_difficulty"],
            },
            "per_query": retrieval.get("per_query", []),
        }
        if generation is not None:
            report["generation_metrics"] = {k: v for k, v in generation.items() if k != "per_query"}
        if extra:
            report.update(extra)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{name}_{timestamp}.json"
        path = RESULTS_DIR / filename
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Report saved to %s", path)
        return str(path)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--dataset", choices=["sample", "expanded", "all"], default="all",
                        help="Dataset to evaluate (default: all)")
    parser.add_argument("--no-latency", action="store_true",
                        help="Skip latency benchmark")
    parser.add_argument("--no-generation", action="store_true",
                        help="Skip generation evaluation")
    parser.add_argument("--reranker", action="store_true", default=None,
                        help="Enable cross-encoder reranker (overrides config)")
    parser.add_argument("--expand-query", action="store_true", default=None,
                        help="Enable KG-based query expansion (overrides config)")
    args = parser.parse_args()

    t_start = time.time()
    runner = RAGBenchmarkRunner()
    await runner.connect()

    # ── Apply CLI overrides for feature flags ────────────────────────────
    if args.reranker is True:
        try:
            reranker = CrossEncoderReranker(model_name=settings.reranker_model)
            reranker.load()
            if reranker.is_loaded:
                runner._rag._reranker = reranker
                logger.info("Reranker enabled via --reranker flag")
        except Exception as exc:
            logger.warning("Reranker failed to load: %s", exc)
    elif args.reranker is False and runner._rag is not None:
        runner._rag._reranker = None
        logger.info("Reranker disabled via config default")

    if args.expand_query is True and runner._rag is not None:
        runner._rag._expand_query_enabled = True
        runner._rag._expand_query_max_terms = settings.expand_query_max_terms
        logger.info("KG expansion enabled via --expand-query flag")
    elif args.expand_query is False and runner._rag is not None:
        runner._rag._expand_query_enabled = False
        logger.info("KG expansion disabled via config default")

    datasets_to_run = ["sample", "expanded"] if args.dataset == "all" else [args.dataset]

    for dataset_name in datasets_to_run:
        logger.info("=" * 50)
        logger.info("Running %s dataset benchmark...", dataset_name)

        if dataset_name == "expanded":
            queries = load_expanded_queries()
        else:
            queries = load_sample_queries()

        logger.info("Loaded %d queries from %s dataset", len(queries), dataset_name)

        coverage = await runner.check_kp_coverage(queries)
        logger.info("KP coverage: %d/%d (%.1f%%)",
                    coverage["found_in_neo4j"], coverage["total_dataset_kps"],
                    coverage["coverage_pct"])

        if coverage["missing"]:
            logger.warning("Missing KPs: %s", coverage["missing"])

        retrieval = await runner.evaluate_retrieval(queries, top_k=10)
        logger.info("Sample retrieval: P@1=%.3f, P@3=%.3f, MRR=%.3f",
                    retrieval["overall"]["precision"].get("1", 0),
                    retrieval["overall"]["precision"].get("3", 0),
                    retrieval["overall"].get("mrr", 0))

        generation = None if args.no_generation else await runner.evaluate_generation(queries)

        runner.print_report(coverage, retrieval, generation)

        # ── 2. Per-Difficulty Summary ────────────────────────────
        by_diff = retrieval.get("by_difficulty", {})
        for diff in ("basic", "intermediate", "advanced"):
            if diff in by_diff:
                d = by_diff[diff]
                logger.info("%s (n=%d): P@1=%.3f, MRR=%.3f, HR@3=%.3f",
                            diff, d.get("n_queries", 0),
                            d.get("precision", {}).get("1", 0),
                            d.get("mrr", 0),
                            d.get("hit_rate", {}).get("3", 0))

        # ── 3. Save report ──────────────────────────────────────
        path = runner.save_report(dataset_name, coverage, retrieval, generation)

        # ── 4. Latency Benchmark (run once, on sample dataset) ───
        if dataset_name == "sample" and not args.no_latency:
            logger.info("Running latency benchmark (100 iterations)...")
            latency = await runner.benchmark_latency(queries, iterations=100)
            runner.print_latency_report(latency)
            extra = {"latency_benchmark": latency}
            _ = runner.save_report(dataset_name, coverage, retrieval, generation, extra=extra)

        elapsed = time.time() - t_start
        logger.info("Total time so far: %.1fs", elapsed)

    t_total = time.time() - t_start
    logger.info("All benchmarks completed in %.1fs", t_total)

    await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
