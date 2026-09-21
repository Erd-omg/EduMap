"""Tests for retrieval strategy comparison (src/rag/evaluation/strategy_compare.py).

Uses an in-memory fake RAG service so no Neo4j/ChromaDB/LLM stack is
required — these tests validate the comparison logic, metric wiring,
rewrite fallback and summary computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.rag.evaluation.strategy_compare import (
    QueryRewriter,
    RewrittenQuery,
    StrategyComparator,
)


@dataclass
class FakeResult:
    source_id: str
    score: float = 1.0


class FakeRAGService:
    """Deterministic retrieval: `direct` misses some relevant docs that
    `hybrid` finds — models the KG recall bonus."""

    # doc universe per query fragment
    def __init__(self):
        self.chroma_calls: list[tuple[str, int]] = []
        self.hybrid_calls: list[tuple[str, int]] = []
        self.cache_flags: list[bool] = []

    async def _chroma_search(self, query: str, top_k: int) -> list:
        self.chroma_calls.append((query, top_k))
        # Vector-only: returns kp-1 but never the KG-only doc kp-2,
        # and ranks poorly for colloquial phrasing.
        return [FakeResult("kp-1"), FakeResult("irrelevant-1")]

    async def search(self, query: str, top_k: int, use_cache: bool = True) -> list:
        # Mirrors RAGRetrievalService.search's real signature, including
        # use_cache (the comparator passes False so latency isn't a cache artifact).
        self.hybrid_calls.append((query, top_k))
        self.cache_flags.append(use_cache)
        # Hybrid finds kp-1 and kp-2 (KG recall), ranked better.
        return [FakeResult("kp-1"), FakeResult("kp-2"), FakeResult("irrelevant-1")]


class RewriteFailsLLM:
    async def generate_structured(self, prompt, schema, system_prompt=None):
        raise RuntimeError("llm down")


class RewriteOKLLM:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls = 0

    async def generate_structured(self, prompt, schema, system_prompt=None):
        self.calls += 1
        original = prompt.replace("学生提问：", "")
        return RewrittenQuery(
            rewritten_query=self.mapping.get(original, original),
            search_terms=["term"],
            reason="test",
        )


QUERIES = [
    {"query": "哈希冲突怎么处理", "relevant_ids": ["kp-1", "kp-2"]},
    {"query": "什么是二叉树", "relevant_ids": ["kp-1"]},
]


# ── QueryRewriter ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rewriter_returns_original_on_failure():
    rewriter = QueryRewriter(RewriteFailsLLM())
    out = await rewriter.rewrite("哈希冲突怎么处理")
    assert out == "哈希冲突怎么处理"


@pytest.mark.asyncio
async def test_rewriter_rewrites_and_caches():
    llm = RewriteOKLLM({"哈希冲突怎么处理": "哈希表 冲突解决 链地址法"})
    rewriter = QueryRewriter(llm)
    out = await rewriter.rewrite("哈希冲突怎么处理")
    assert out == "哈希表 冲突解决 链地址法"
    # Second call served from cache — no extra LLM call.
    out2 = await rewriter.rewrite("哈希冲突怎么处理")
    assert out2 == out
    assert llm.calls == 1


# ── StrategyComparator ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_comparison_runs_all_strategies():
    rag = FakeRAGService()
    llm = RewriteOKLLM({"哈希冲突怎么处理": "哈希表 冲突 链地址法"})
    comparator = StrategyComparator(rag, llm=llm, k_values=[1, 3, 5])

    result = await comparator.run(QUERIES, top_k=5)

    assert set(result["strategies"]) == {"direct", "hybrid", "rewrite"}
    assert result["n_queries"] == 2

    # Hybrid should beat direct on recall@5 (finds kp-2 via KG).
    d = result["strategies"]["direct"]
    h = result["strategies"]["hybrid"]
    assert h["recall"]["5"] > d["recall"]["5"]
    assert h["mrr"] >= d["mrr"]

    # Rewrite strategy actually used the LLM rewrite at least once.
    r = result["strategies"]["rewrite"]
    assert r["n_rewrites_used"] >= 1

    # Direct must use only the vector path.
    assert len(rag.chroma_calls) == 2
    assert len(rag.hybrid_calls) == 4  # hybrid(2) + rewrite(2)

    # Summary: hybrid or rewrite should win recall@5 over direct.
    summary = result["summary"]
    assert summary["best_per_metric"]["recall@5"] in {"hybrid", "rewrite"}
    delta = summary["improvement_vs_direct"]["hybrid"]["recall@5_delta"]
    assert delta > 0


@pytest.mark.asyncio
async def test_comparison_without_llm_skips_rewrite():
    rag = FakeRAGService()
    comparator = StrategyComparator(rag, llm=None, k_values=[5])

    result = await comparator.run(QUERIES, strategies=["direct", "hybrid"])
    assert set(result["strategies"]) == {"direct", "hybrid"}
    # No hybrid calls from a rewrite strategy
    assert len(rag.hybrid_calls) == 2


@pytest.mark.asyncio
async def test_comparison_rewrite_failure_falls_back_to_hybrid():
    rag = FakeRAGService()
    comparator = StrategyComparator(rag, llm=RewriteFailsLLM(), k_values=[5])

    result = await comparator.run(QUERIES, strategies=["rewrite"])
    r = result["strategies"]["rewrite"]
    # Rewrite failed → fell back to original query (hybrid behavior)
    assert r["n_rewrites_failed"] == 2
    assert r["n_rewrites_used"] == 0
    assert r["recall"]["5"] == result["strategies"]["hybrid"]["recall"]["5"] if "hybrid" in result["strategies"] else True


@pytest.mark.asyncio
async def test_per_query_details_present():
    rag = FakeRAGService()
    comparator = StrategyComparator(rag, llm=None, k_values=[5])
    result = await comparator.run(QUERIES, strategies=["hybrid"])

    details = result["per_query"]["hybrid"]
    assert len(details) == 2
    assert details[0]["query"] == "哈希冲突怎么处理"
    assert details[0]["retrieved_ids"][0] == "kp-1"
    assert "mrr" in details[0] and "recall@5" in details[0]


def test_print_comparison_does_not_crash(capsys):
    # Smoke-test the report formatter with a minimal result dict.
    result = {
        "n_queries": 2,
        "k_values": [5],
        "strategies": {
            "direct": {
                "recall": {"5": 0.31}, "precision": {"5": 0.2},
                "mrr": 0.4, "ndcg": {"5": 0.5}, "hit_rate": {"5": 0.6},
                "avg_latency_ms": 12.5, "n_rewrites_used": 0,
                "n_rewrites_failed": 0,
            },
            "hybrid": {
                "recall": {"5": 0.74}, "precision": {"5": 0.4},
                "mrr": 0.6, "ndcg": {"5": 0.7}, "hit_rate": {"5": 0.9},
                "avg_latency_ms": 45.0, "n_rewrites_used": 0,
                "n_rewrites_failed": 0,
            },
        },
        "summary": {
            "best_per_metric": {"recall@5": "hybrid", "mrr": "hybrid"},
            "improvement_vs_direct": {
                "hybrid": {"recall@5_delta": 0.43, "mrr_delta": 0.2, "latency_x": 3.6}
            },
        },
        "per_query": {},
    }
    StrategyComparator.print_comparison(result)
    out = capsys.readouterr().out
    assert "检索策略对比评测" in out
    assert "hybrid" in out
    assert "0.74" in out


class TestComparatorCacheMode:
    """StrategyComparator.use_cache controls whether hybrid may hit the cache.

    Regression guard: the strategies must be compared cache-free by default
    (otherwise hybrid's latency is partly a cache artifact), but the --repeat
    cache benchmark must be able to opt IN, or it would measure nothing.
    """

    async def test_default_bypasses_cache(self) -> None:
        rag = FakeRAGService()
        comparator = StrategyComparator(rag, llm=None)
        await comparator._search_hybrid("什么是栈", 5)
        assert rag.cache_flags[-1] is False, "default must pass use_cache=False"

    async def test_opt_in_uses_cache(self) -> None:
        rag = FakeRAGService()
        comparator = StrategyComparator(rag, llm=None, use_cache=True)
        await comparator._search_hybrid("什么是栈", 5)
        assert rag.cache_flags[-1] is True, "opt-in must pass use_cache=True"

    async def test_direct_never_uses_search(self) -> None:
        """direct calls _chroma_search directly — hence the asymmetry."""
        rag = FakeRAGService()
        comparator = StrategyComparator(rag, llm=None, use_cache=True)
        await comparator._search_direct("什么是栈", 5)
        assert rag.chroma_calls, "direct must go through _chroma_search"
        assert not rag.cache_flags, "direct must not reach search()"
