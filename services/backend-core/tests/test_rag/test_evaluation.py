"""Tests for RAG evaluation modules — EvalResult, RAGEvaluator, RAGEvalBenchmark, llm_judge."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.evaluation.benchmark import RAGEvalBenchmark
from src.rag.evaluation.evaluator import EvalResult, RAGEvaluator
from src.rag.evaluation.llm_judge import llm_answer_relevancy, llm_faithfulness
from src.rag.models import RAGResult


# ── EvalResult ────────────────────────────────────────────────────


class TestEvalResult:
    """EvalResult data class and derived properties."""

    def test_overall_score_default(self) -> None:
        """Overall score is 0 when all metrics are 0."""
        result = EvalResult(
            precision={1: 0.0, 3: 0.0},
            recall={1: 0.0, 3: 0.0},
            mrr=0.0,
            ndcg={1: 0.0, 3: 0.0},
            faithfulness_score=0.0,
            relevancy_score=0.0,
            n_queries=5,
        )
        assert result.overall_score == 0.0

    def test_overall_score_perfect(self) -> None:
        """Overall score is 1.0 when all metrics are perfect."""
        result = EvalResult(
            precision={1: 1.0, 3: 1.0},
            recall={1: 1.0, 3: 1.0},
            mrr=1.0,
            ndcg={1: 1.0, 3: 1.0},
            faithfulness_score=1.0,
            relevancy_score=1.0,
            n_queries=5,
        )
        # 0.25 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0 + 0.10 * 1.0 + 0.10 * 1.0
        assert result.overall_score == 1.0

    def test_overall_score_weighted(self) -> None:
        """Overall score is a weighted combination of all metrics."""
        result = EvalResult(
            precision={1: 0.5},
            recall={1: 0.4},
            mrr=0.3,
            ndcg={1: 0.6},
            faithfulness_score=0.2,
            relevancy_score=0.1,
            n_queries=2,
        )
        expected = (
            0.25 * 0.5
            + 0.25 * 0.4
            + 0.15 * 0.3
            + 0.15 * 0.6
            + 0.10 * 0.2
            + 0.10 * 0.1
        )
        assert result.overall_score == round(expected, 4)

    def test_overall_score_empty_dicts(self) -> None:
        """Empty precision/recall/ndcg dicts do not cause division by zero."""
        result = EvalResult(
            precision={},
            recall={},
            mrr=0.5,
            ndcg={},
            faithfulness_score=0.8,
            relevancy_score=0.7,
            n_queries=3,
        )
        # avg of empty is 0.0
        expected = 0.15 * 0.5 + 0.10 * 0.8 + 0.10 * 0.7
        assert result.overall_score == round(expected, 4)

    def test_to_dict_keys(self) -> None:
        """to_dict() contains all expected keys."""
        result = EvalResult(
            precision={1: 0.5},
            recall={1: 0.4},
            mrr=0.3,
            ndcg={1: 0.6},
            faithfulness_score=0.8,
            relevancy_score=0.7,
            n_queries=10,
        )
        d = result.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "mrr" in d
        assert "ndcg" in d
        assert "faithfulness" in d
        assert "relevancy" in d
        assert "n_queries" in d
        assert "overall" in d

    def test_to_dict_keys_are_strings(self) -> None:
        """Dict keys for precision/recall/ndcg are strings (JSON-friendly)."""
        result = EvalResult(
            precision={1: 0.5, 5: 0.3},
            recall={1: 0.4},
            mrr=0.3,
            ndcg={3: 0.6},
            faithfulness_score=0.8,
            relevancy_score=0.7,
            n_queries=5,
        )
        d = result.to_dict()
        for k in d["precision"]:
            assert isinstance(k, str)
        for k in d["recall"]:
            assert isinstance(k, str)
        for k in d["ndcg"]:
            assert isinstance(k, str)

    def test_n_queries_stored(self) -> None:
        """n_queries is stored and retrievable."""
        result = EvalResult(
            precision={}, recall={}, mrr=0.0, ndcg={},
            faithfulness_score=0.0, relevancy_score=0.0, n_queries=42,
        )
        assert result.n_queries == 42


# ── RAGEvaluator ──────────────────────────────────────────────────


class TestRAGEvaluatorRetrieval:
    """RAGEvaluator.evaluate_retrieval tests."""

    async def test_basic_retrieval_evaluation(self) -> None:
        """Evaluate retrieval with a mock retrieval function."""
        async def mock_retrieval(query: str, top_k: int) -> list[RAGResult]:
            return [
                RAGResult(content="doc_a", source_type="chroma", source_id="kp-1", source_name="A", score=0.9),
                RAGResult(content="doc_b", source_type="chroma", source_id="kp-2", source_name="B", score=0.8),
            ]

        queries = [
            {"query": "数组插入", "relevant_ids": {"kp-1"}},
        ]

        evaluator = RAGEvaluator(k_values=[1, 2])
        result = await evaluator.evaluate_retrieval(queries, mock_retrieval)
        assert result.n_queries == 1
        assert 1 in result.precision
        assert 2 in result.recall
        assert result.mrr >= 0
        assert result.faithfulness_score == 0.0
        assert result.relevancy_score == 0.0

    async def test_retrieval_relevant_ids_list(self) -> None:
        """relevant_ids as a list (not set) is handled correctly."""
        async def mock_retrieval(query: str, top_k: int) -> list[RAGResult]:
            return [
                RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9),
            ]

        queries = [
            {"query": "测试", "relevant_ids": ["kp-1"]},  # list instead of set
        ]

        evaluator = RAGEvaluator(k_values=[1])
        result = await evaluator.evaluate_retrieval(queries, mock_retrieval)
        assert result.precision[1] > 0

    async def test_no_queries(self) -> None:
        """Empty queries list returns zeroed results."""
        async def mock_retrieval(query: str, top_k: int) -> list[RAGResult]:
            return []

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_retrieval([], mock_retrieval)
        assert result.n_queries == 0
        assert all(v == 0.0 for v in result.precision.values())
        assert all(v == 0.0 for v in result.recall.values())
        assert result.mrr == 0.0

    async def test_default_k_values(self) -> None:
        """Default k_values is [1, 3, 5, 10]."""
        evaluator = RAGEvaluator()
        assert evaluator._k_values == [1, 3, 5, 10]

    async def test_custom_k_values(self) -> None:
        """Custom k_values are used."""
        async def mock_retrieval(query: str, top_k: int) -> list[RAGResult]:
            return [RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9)]

        queries = [{"query": "测试", "relevant_ids": {"kp-1"}}]
        evaluator = RAGEvaluator(k_values=[5])
        result = await evaluator.evaluate_retrieval(queries, mock_retrieval)
        assert list(result.precision.keys()) == [5]
        assert list(result.recall.keys()) == [5]
        assert list(result.ndcg.keys()) == [5]


class TestRAGEvaluatorGeneration:
    """RAGEvaluator.evaluate_generation tests."""

    async def test_basic_generation_evaluation(self) -> None:
        """Evaluate generation with faithfulness and relevancy scores."""
        qa_pairs = [
            {
                "query": "插入 复杂度",
                "answer": "Array 插入 复杂度 O(n)",
                "context_sentences": ["Array 插入 复杂度 O(n)", "Linked 链表 O(1)"],
            },
        ]

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_generation(qa_pairs)
        assert result.n_queries == 1
        # faithfulness: "Array 插入 复杂度 O(n)" fully overlaps with context
        assert result.faithfulness_score > 0
        # answer_relevancy: "插入 复杂度" tokens "插入","复杂度" appear in answer "Array 插入 复杂度 O(n)"
        assert result.relevancy_score > 0
        assert result.precision == {}
        assert result.mrr == 0.0
        assert result.ndcg == {}

    async def test_generation_empty_qa_pairs(self) -> None:
        """Empty qa_pairs returns zeroed results."""
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_generation([])
        assert result.n_queries == 0
        assert result.faithfulness_score == 0.0
        assert result.relevancy_score == 0.0

    async def test_generation_no_context(self) -> None:
        """Missing context_sentences results in lower faithfulness."""
        qa_pairs = [
            {
                "query": "测试",
                "answer": "Some answer content here",
                # no context_sentences key
            },
        ]

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_generation(qa_pairs)
        assert result.n_queries == 1
        assert result.faithfulness_score == 0.0  # no context to support anything

    async def test_multiple_qa_pairs(self) -> None:
        """Scores are averaged across multiple QA pairs."""
        qa_pairs = [
            {"query": "q1", "answer": "Common tokens here", "context_sentences": ["Common tokens here"]},
            {"query": "q2", "answer": "Different answer content", "context_sentences": ["Different answer content"]},
        ]

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate_generation(qa_pairs)
        assert result.n_queries == 2
        assert result.faithfulness_score > 0


# ── RAGEvalBenchmark ──────────────────────────────────────────────


class TestRAGEvalBenchmark:
    """RAGEvalBenchmark.run_benchmark tests."""

    async def test_benchmark_with_sample_queries(self) -> None:
        """Benchmark using sample queries returns completed status."""
        mock_rag = AsyncMock()
        mock_rag.search.return_value = [
            RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9),
        ]

        benchmark = RAGEvalBenchmark(
            rag_service=mock_rag,
            kp_repo=MagicMock(),
        )

        result = await benchmark.run_benchmark(
            n_queries=3,
            use_sample_queries=True,
        )
        assert result["status"] == "completed"
        assert result["dataset_source"] == "sample_queries"
        assert result["n_queries"] > 0
        assert "metrics" in result
        assert "hit_rate" in result["metrics"]
        assert "generated_at" in result

    async def test_benchmark_no_queries(self) -> None:
        """When there are no queries, benchmark is skipped."""
        mock_kp_repo = AsyncMock()
        mock_kp_repo.list_all.return_value = []

        benchmark = RAGEvalBenchmark(
            rag_service=MagicMock(),
            kp_repo=mock_kp_repo,
        )

        result = await benchmark.run_benchmark(n_queries=5)
        assert result["status"] == "skipped"
        assert result["reason"] == "No test queries available"
        assert result["n_queries"] == 0

    async def test_benchmark_with_course_id(self) -> None:
        """course_id is passed to _generate_test_queries and returned in result."""
        mock_kp_repo = AsyncMock()
        mock_kp_repo.get_by_course.return_value = []
        mock_rag = AsyncMock()

        benchmark = RAGEvalBenchmark(
            rag_service=mock_rag,
            kp_repo=mock_kp_repo,
        )

        result = await benchmark.run_benchmark(
            n_queries=5,
            course_id="course-123",
            use_sample_queries=False,
        )
        # With no KPs, should be skipped
        assert result["status"] == "skipped"

    async def test_benchmark_llm_faithfulness(self) -> None:
        """LLM faithfulness evaluation runs when llm is provided."""
        mock_rag = AsyncMock()
        mock_rag.search.return_value = [
            RAGResult(content="内容", source_type="chroma", source_id="kp-1", source_name="知识", score=0.9),
        ]
        mock_rag.assemble_context = AsyncMock()
        mock_rag.assemble_context.return_value = MagicMock(
            context_str="测试上下文",
            sources=[RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9)],
        )

        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            faithfulness_score=0.85,
            supported_sentences=["supported"],
            unsupported_sentences=[],
            reasoning="good",
        )

        benchmark = RAGEvalBenchmark(
            rag_service=mock_rag,
            kp_repo=MagicMock(),
        )

        result = await benchmark.run_benchmark(
            n_queries=2,
            use_sample_queries=True,
            llm=mock_llm,
        )

        assert "llm_faithfulness" in result["metrics"]
        assert result["metrics"]["llm_faithfulness"] > 0

    async def test_benchmark_llm_faithfulness_failure(self) -> None:
        """LLM faithfulness failure doesn't crash the benchmark (falls back to heuristic)."""
        mock_rag = AsyncMock()
        mock_rag.search.return_value = [
            RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9),
        ]
        mock_rag.assemble_context = AsyncMock()
        mock_rag.assemble_context.return_value = MagicMock(
            context_str="ctx",
            sources=[RAGResult(content="doc", source_type="chroma", source_id="kp-1", source_name="A", score=0.9)],
        )

        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("LLM failed"))

        benchmark = RAGEvalBenchmark(
            rag_service=mock_rag,
            kp_repo=MagicMock(),
        )

        result = await benchmark.run_benchmark(
            n_queries=1,
            use_sample_queries=True,
            llm=mock_llm,
        )

        assert result["status"] == "completed"
        # Fallback heuristic runs, so llm_faithfulness IS present
        assert "llm_faithfulness" in result.get("metrics", {})


# ── LLM Judge ─────────────────────────────────────────────────────


class TestLLMFaithfulness:
    """llm_faithfulness function tests."""

    async def test_basic_faithfulness(self) -> None:
        """LLM faithfulness returns score and sentences."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            faithfulness_score=0.75,
            supported_sentences=["句子一被支持", "句子二被支持"],
            unsupported_sentences=["句子三不支持"],
            reasoning="大部分支持",
        )

        result = await llm_faithfulness(
            answer="这是一个测试回答。",
            context="这是参考上下文。",
            llm=mock_llm,
        )
        assert result["faithfulness_score"] == 0.75
        assert len(result["supported_sentences"]) == 2
        assert len(result["unsupported_sentences"]) == 1
        assert result["reasoning"] == "大部分支持"
        mock_llm.generate_structured.assert_called_once()

    async def test_empty_context(self) -> None:
        """Empty context is still passed to the LLM (with placeholder)."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            faithfulness_score=0.0,
            supported_sentences=[],
            unsupported_sentences=["句子"],
            reasoning="无参考资料",
        )

        result = await llm_faithfulness(
            answer="测试回答。",
            context="",
            llm=mock_llm,
        )
        assert result["faithfulness_score"] == 0.0
        # Verify the prompt had the "无参考资料" placeholder
        prompt_text = mock_llm.generate_structured.call_args[1]["prompt"]
        assert "无参考资料" in prompt_text

    async def test_fallback_on_exception(self) -> None:
        """When LLM call fails, fall back to heuristic faithfulness."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("API error"))

        result = await llm_faithfulness(
            answer="Test answer content.",
            context="Some context content.",
            llm=mock_llm,
        )
        # Fallback returns heuristic result
        assert "faithfulness_score" in result
        assert result["reasoning"] == "fallback_heuristic"

    async def test_long_context_truncated(self) -> None:
        """Context longer than 3000 chars is truncated."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            faithfulness_score=1.0,
            supported_sentences=[],
            unsupported_sentences=[],
            reasoning="ok",
        )

        long_context = "A" * 5000
        await llm_faithfulness(
            answer="Short answer.",
            context=long_context,
            llm=mock_llm,
        )

        prompt_text = mock_llm.generate_structured.call_args[1]["prompt"]
        # Context should be <= 3000 chars
        context_start = prompt_text.index("参考资料：\n") + len("参考资料：\n")
        context_end = prompt_text.index("\n\n回答：")
        assert context_end - context_start <= 3000

    async def test_long_answer_truncated(self) -> None:
        """Answer longer than 2000 chars is truncated."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            faithfulness_score=1.0,
            supported_sentences=[],
            unsupported_sentences=[],
            reasoning="ok",
        )

        long_answer = "B" * 3000
        await llm_faithfulness(
            answer=long_answer,
            context="Some context.",
            llm=mock_llm,
        )

        prompt_text = mock_llm.generate_structured.call_args[1]["prompt"]
        answer_start = prompt_text.index("回答：\n") + len("回答：\n")
        # Find where the answer section ends (before the JSON schema block)
        answer_end = prompt_text.index("\n\n逐句分析")
        answer_len = answer_end - answer_start
        assert answer_len <= 2000  # answer truncated to 2000 chars


class TestLLMAnswerRelevancy:
    """llm_answer_relevancy function tests."""

    async def test_basic_relevancy(self) -> None:
        """LLM answer relevancy returns score and reasons."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            relevancy_score=0.9,
            reasons=["回答直接针对问题", "包含了关键信息"],
            missing_points=[],
        )

        result = await llm_answer_relevancy(
            query="数组插入的复杂度是多少？",
            answer="数组插入的复杂度是O(n)。",
            llm=mock_llm,
        )
        assert result["relevancy_score"] == 0.9
        assert len(result["reasons"]) == 2
        assert result["missing_points"] == []
        mock_llm.generate_structured.assert_called_once()

    async def test_fallback_on_exception(self) -> None:
        """When LLM call fails, fall back to heuristic relevancy."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=RuntimeError("API error"))

        result = await llm_answer_relevancy(
            query="test query",
            answer="test answer",
            llm=mock_llm,
        )
        assert "relevancy_score" in result
        assert result["reasons"] == []
        assert result["missing_points"] == []

    async def test_empty_query(self) -> None:
        """Empty query is still passed to the LLM."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            relevancy_score=0.5,
            reasons=["neutral"],
            missing_points=[],
        )

        result = await llm_answer_relevancy(
            query="",
            answer="Some answer.",
            llm=mock_llm,
        )
        assert result["relevancy_score"] == 0.5

    async def test_long_query_truncated(self) -> None:
        """Query longer than 500 chars is truncated."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock()
        mock_llm.generate_structured.return_value = MagicMock(
            relevancy_score=0.5,
            reasons=[],
            missing_points=[],
        )

        await llm_answer_relevancy(
            query="X" * 1000,
            answer="Short answer.",
            llm=mock_llm,
        )

        prompt_text = mock_llm.generate_structured.call_args[1]["prompt"]
        # Verify truncation — the query appears after "问题："
        query_start = prompt_text.index("问题：") + len("问题：")
        query_end = prompt_text.index("\n\n回答：")
        assert query_end - query_start <= 500


class TestLLMFallbackIntegration:
    """Verify fallback paths use the correct heuristic functions."""

    async def test_faithfulness_fallback_imports_metrics(self) -> None:
        """Faithfulness fallback uses the metrics module's faithfulness function."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=ValueError("fail"))

        with patch("src.rag.evaluation.metrics.faithfulness") as mock_f:
            mock_f.return_value = {
                "faithfulness": 0.42,
                "supported_sentences": 0,
                "total_sentences": 2,
                "unsupported": ["unsupported sentence"],
            }

            result = await llm_faithfulness(
                answer="Some answer. Another sentence.",
                context="Some context.",
                llm=mock_llm,
            )
            assert result["faithfulness_score"] == 0.42
            assert result["reasoning"] == "fallback_heuristic"
            mock_f.assert_called_once()

    async def test_relevancy_fallback_imports_metrics(self) -> None:
        """Relevancy fallback uses the metrics module's answer_relevancy function."""
        mock_llm = MagicMock()
        mock_llm.generate_structured = AsyncMock(side_effect=ValueError("fail"))

        with patch("src.rag.evaluation.metrics.answer_relevancy") as mock_ar:
            mock_ar.return_value = 0.33

            result = await llm_answer_relevancy(
                query="test query",
                answer="test answer",
                llm=mock_llm,
            )
            assert result["relevancy_score"] == 0.33
            mock_ar.assert_called_once()
