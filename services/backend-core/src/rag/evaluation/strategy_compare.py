"""Retrieval strategy comparison: direct vs hybrid vs rewrite.

Runs the same evaluation dataset through three retrieval strategies and
produces a side-by-side metric comparison (Recall / Precision / MRR /
NDCG / HitRate @k, plus per-strategy latency):

- **direct**  — pure vector recall (ChromaDB only): the baseline.
- **hybrid**  — the production pipeline (vector + KG keyword recall,
  RRF fusion, dedup, optional reranker).
- **rewrite** — LLM query rewrite first (colloquial → textbook terminology,
  pronoun/context resolution), then the hybrid pipeline.

This answers questions like "is hybrid worth the extra latency?" and
"how much does query rewriting add on top?" with numbers instead of
intuition.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from src.rag.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.rag.query_rewrite import (  # noqa: F401  (re-exported for compatibility)
    QueryRewriter,
    RewrittenQuery,
    _REWRITE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# ``QueryRewriter`` / ``RewrittenQuery`` / the rewrite prompt now live in
# src/rag/query_rewrite.py (production code) and are imported above; this
# module only drives them for measurement.




@dataclass
class StrategyMetrics:
    """Metrics for one strategy over the whole dataset."""

    name: str
    n_queries: int = 0
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    n_rewrites_used: int = 0  # only meaningful for the rewrite strategy
    n_rewrites_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            "precision": {str(k): v for k, v in self.precision.items()},
            "recall": {str(k): v for k, v in self.recall.items()},
            "ndcg": {str(k): v for k, v in self.ndcg.items()},
            "hit_rate": {str(k): v for k, v in self.hit_rate.items()},
            "mrr": self.mrr,
            "avg_latency_ms": self.avg_latency_ms,
            "n_rewrites_used": self.n_rewrites_used,
            "n_rewrites_failed": self.n_rewrites_failed,
        }


class StrategyComparator:
    """Run direct / hybrid / rewrite retrieval strategies on one dataset."""

    def __init__(
        self,
        rag_service: Any,
        llm: Any = None,
        k_values: list[int] | None = None,
    ) -> None:
        self._rag = rag_service
        self._llm = llm
        self._k_values = k_values or [1, 3, 5, 10]
        self._rewriter = QueryRewriter(llm) if llm is not None else None

    # ── Strategy implementations ─────────────────────────────────────

    async def _search_direct(self, query: str, top_k: int) -> list:
        """Pure vector recall — ChromaDB only, no KG, no fusion, no rerank.

        Note this bypasses ``RAGRetrievalService.search()`` and therefore the
        result cache.  For a fair latency comparison the hybrid path below must
        also bypass it — see ``use_cache=False`` there.
        """
        return await self._rag._chroma_search(query, top_k)

    async def _search_hybrid(self, query: str, top_k: int) -> list:
        """Production pipeline: vector + KG + RRF fusion (+ reranker).

        ``use_cache=False`` is passed explicitly: ``direct`` cannot use the
        cache (it calls ``_chroma_search`` directly), so allowing hybrid to hit
        it would make hybrid's measured latency partly a cache artifact rather
        than a property of the pipeline.
        """
        return await self._rag.search(query, top_k=top_k, use_cache=False)

    async def _search_rewrite(self, query: str, top_k: int) -> tuple[list, bool]:
        """LLM rewrite, then hybrid retrieval."""
        if self._rewriter is None:
            return await self._search_hybrid(query, top_k), False
        rewritten = await self._rewriter.rewrite(query)
        used = rewritten != query
        results = await self._search_hybrid(rewritten, top_k)
        return results, used

    # ── Main entry ───────────────────────────────────────────────────

    async def run(
        self,
        queries: list[dict],
        top_k: int | None = None,
        strategies: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate the selected strategies on the dataset.

        Args:
            queries: Dataset entries with ``query`` and ``relevant_ids``.
            top_k: Retrieval depth (defaults to max k value).
            strategies: Subset of ``["direct", "hybrid", "rewrite"]``.

        Returns:
            Dict with per-strategy metrics, comparison summary and
            per-query details.
        """
        strategies = strategies or ["direct", "hybrid", "rewrite"]
        max_k = top_k or max(self._k_values)

        results: dict[str, StrategyMetrics] = {}
        per_query: dict[str, list[dict]] = {s: [] for s in strategies}

        for strategy in strategies:
            metrics = await self._run_single(strategy, queries, max_k, per_query)
            results[strategy] = metrics

        summary = self._build_summary(results)
        return {
            "n_queries": len(queries),
            "k_values": self._k_values,
            "strategies": {name: m.to_dict() for name, m in results.items()},
            "summary": summary,
            "per_query": per_query,
        }

    async def _run_single(
        self,
        strategy: str,
        queries: list[dict],
        max_k: int,
        per_query: dict[str, list[dict]],
    ) -> StrategyMetrics:
        metrics = StrategyMetrics(name=strategy, n_queries=len(queries))
        prec: dict[int, list[float]] = {k: [] for k in self._k_values}
        rec: dict[int, list[float]] = {k: [] for k in self._k_values}
        ndcg_l: dict[int, list[float]] = {k: [] for k in self._k_values}
        hr: dict[int, list[float]] = {k: [] for k in self._k_values}
        mrr_l: list[float] = []
        latencies: list[float] = []

        search_fn = {
            "direct": self._search_direct,
            "hybrid": self._search_hybrid,
        }

        for q_data in queries:
            query = q_data["query"]
            relevant = q_data.get("relevant_ids", set())
            if isinstance(relevant, list):
                relevant = set(relevant)

            t0 = time.perf_counter()
            if strategy == "rewrite":
                res, used = await self._search_rewrite(query, max_k)
                if used:
                    metrics.n_rewrites_used += 1
                else:
                    metrics.n_rewrites_failed += 1
            else:
                res = await search_fn[strategy](query, max_k)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            for k in self._k_values:
                prec[k].append(precision_at_k(res, relevant, k))
                rec[k].append(recall_at_k(res, relevant, k))
                ndcg_l[k].append(ndcg_at_k(res, relevant, k))
                hr[k].append(hit_rate_at_k(res, relevant, k))
            mrr_l.append(mean_reciprocal_rank(res, relevant))

            per_query[strategy].append({
                "query": query,
                "relevant_ids": sorted(relevant),
                "retrieved_ids": [r.source_id for r in res],
                "mrr": round(mean_reciprocal_rank(res, relevant), 4),
                "recall@5": round(recall_at_k(res, relevant, 5), 4),
            })

        metrics.precision = {
            k: round(sum(v) / max(len(v), 1), 4) for k, v in prec.items()
        }
        metrics.recall = {
            k: round(sum(v) / max(len(v), 1), 4) for k, v in rec.items()
        }
        metrics.ndcg = {
            k: round(sum(v) / max(len(v), 1), 4) for k, v in ndcg_l.items()
        }
        metrics.hit_rate = {
            k: round(sum(v) / max(len(v), 1), 4) for k, v in hr.items()
        }
        metrics.mrr = round(sum(mrr_l) / max(len(mrr_l), 1), 4)
        metrics.avg_latency_ms = round(
            sum(latencies) / max(len(latencies), 1), 2
        )
        return metrics

    def _build_summary(self, results: dict[str, StrategyMetrics]) -> dict:
        """Pick the best strategy per headline metric and compute deltas."""
        if not results:
            return {}

        headline_k = 5
        best: dict[str, str] = {}

        def _best(metric_fn) -> str:
            return max(results, key=lambda n: metric_fn(results[n]))

        best["recall@5"] = _best(lambda m: m.recall.get(headline_k, 0.0))
        best["precision@5"] = _best(lambda m: m.precision.get(headline_k, 0.0))
        best["mrr"] = _best(lambda m: m.mrr)
        best["ndcg@5"] = _best(lambda m: m.ndcg.get(headline_k, 0.0))
        best["hit_rate@5"] = _best(lambda m: m.hit_rate.get(headline_k, 0.0))
        best["latency"] = _best(lambda m: -m.avg_latency_ms)

        baseline = results.get("direct")
        deltas: dict[str, dict[str, float]] = {}
        if baseline is not None:
            for name, m in results.items():
                if name == "direct":
                    continue
                deltas[name] = {
                    "recall@5_delta": round(
                        m.recall.get(headline_k, 0.0)
                        - baseline.recall.get(headline_k, 0.0),
                        4,
                    ),
                    "mrr_delta": round(m.mrr - baseline.mrr, 4),
                    "latency_x": round(
                        m.avg_latency_ms / max(baseline.avg_latency_ms, 0.001), 2
                    ),
                }

        return {
            "best_per_metric": best,
            "improvement_vs_direct": deltas,
        }

    # ── Reporting ────────────────────────────────────────────────────

    @staticmethod
    def print_comparison(result: dict[str, Any]) -> None:
        """Print a formatted strategy comparison table to the terminal."""
        sep = "=" * 72
        print(f"\n{sep}")
        print("  检索策略对比评测 (direct vs hybrid vs rewrite)")
        print(f"  查询数: {result['n_queries']}   K 值: {result['k_values']}")
        print(sep)

        strategies = result["strategies"]
        ks = [str(k) for k in result["k_values"]]

        print(f"\n  {'策略':<10} ", end="")
        for metric in ("R@5", "P@5", "MRR", "NDCG@5", "HR@5", "延迟ms"):
            print(f"{metric:<12}", end="")
        print()
        print("  " + "─" * 82)

        labels = {
            "direct": "direct",
            "hybrid": "hybrid",
            "rewrite": "rewrite",
        }
        for name, m in strategies.items():
            print(f"  {labels.get(name, name):<10} ", end="")
            print(f"{m['recall'].get('5', '-'):<12.4}", end="")
            print(f"{m['precision'].get('5', '-'):<12}", end="")
            print(f"{m['mrr']:<12}", end="")
            print(f"{m['ndcg'].get('5', '-'):<12}", end="")
            print(f"{m['hit_rate'].get('5', '-'):<12}", end="")
            print(f"{m['avg_latency_ms']:<12}")

        if "rewrite" in strategies:
            m = strategies["rewrite"]
            print(
                f"\n  rewrite: LLM 改写生效 {m['n_rewrites_used']} 条, "
                f"失败/跳过 {m['n_rewrites_failed']} 条"
            )

        summary = result.get("summary", {})
        if summary.get("best_per_metric"):
            print(f"\n  🏆 各指标最优策略:")
            for metric, strategy in summary["best_per_metric"].items():
                print(f"     {metric:<16} → {strategy}")

        if summary.get("improvement_vs_direct"):
            print(f"\n  📈 相对 direct 基线的提升:")
            for name, delta in summary["improvement_vs_direct"].items():
                print(
                    f"     {name:<10} Recall@5 {delta['recall@5_delta']:+.4f}  "
                    f"MRR {delta['mrr_delta']:+.4f}  "
                    f"延迟 ×{delta['latency_x']}"
                )
        print(f"\n{sep}\n")
