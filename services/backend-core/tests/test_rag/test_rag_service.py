"""Tests for RAGRetrievalService — merge, dedup, context assembly."""

from __future__ import annotations

from src.rag.models import RAGResult
from src.rag.rag_service import RAGRetrievalService


def _make_result(
    source_id: str,
    source_type: str = "chroma",
    score: float = 0.8,
    content: str = "测试内容",
    source_name: str = "测试来源",
) -> RAGResult:
    return RAGResult(
        content=content,
        source_type=source_type,  # type: ignore[arg-type]
        source_id=source_id,
        source_name=source_name,
        score=score,
    )


class TestMergeAndRank:
    """_merge_and_rank — dedup + sorting logic."""

    def test_dedup_by_composite_key(self) -> None:
        """Duplicate source_type:source_id pairs are removed, highest score wins."""
        a1 = _make_result("kp-1", "chroma", score=0.9)
        a2 = _make_result("kp-1", "chroma", score=0.5)  # same composite key
        b = _make_result("kp-1", "neo4j", score=0.7)  # different source_type

        merged = RAGRetrievalService._merge_and_rank([a1, a2, b], "test")
        assert len(merged) == 2
        # a1 should win over a2 (higher score)
        assert merged[0].source_type == "chroma"
        assert merged[0].score == 0.9

    def test_sorted_by_score_desc(self) -> None:
        """Results are sorted by score descending."""
        results = [
            _make_result("a", score=0.5),
            _make_result("b", score=0.9),
            _make_result("c", score=0.7),
        ]
        merged = RAGRetrievalService._merge_and_rank(results, "test")
        scores = [r.score for r in merged]
        assert scores == [0.9, 0.7, 0.5]

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert RAGRetrievalService._merge_and_rank([], "test") == []


class TestAssembleContext:
    """assemble_context — context string assembly from results."""

    async def test_basic_assembly(self) -> None:
        """Context includes source headers + content snippets."""
        results = [
            _make_result("kp-1", content="数组插入O(n)", source_name="数组"),
            _make_result("kp-2", content="链表插入O(1)", source_name="链表"),
        ]
        ctx = await RAGRetrievalService().assemble_context(results, max_chars=500)
        assert "[1]" in ctx.context_str
        assert "[2]" in ctx.context_str
        assert "数组" in ctx.context_str
        assert len(ctx.sources) == 2

    async def test_max_chars_truncation(self) -> None:
        """Context is truncated at max_chars boundary."""
        results = [
            _make_result("kp-1", content="A" * 300, source_name="长内容"),
            _make_result("kp-2", content="B" * 300, source_name="更多内容"),
        ]
        ctx = await RAGRetrievalService().assemble_context(results, max_chars=100)
        assert len(ctx.context_str) <= 300  # 100 chars + overhead for headers

    async def test_no_results(self) -> None:
        """Empty results produce fallback message."""
        ctx = await RAGRetrievalService().assemble_context([], max_chars=500)
        assert "未找到相关参考资料" in ctx.context_str
        assert len(ctx.sources) == 0
