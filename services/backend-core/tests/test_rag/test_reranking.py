"""Tests for CrossEncoderReranker — cross-encoder re-ranking logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.rag.models import RAGResult
from src.rag.reranking.cross_encoder import CrossEncoderReranker


def _make_result(
    source_id: str,
    score: float = 0.8,
    content: str = "测试内容",
    source_name: str = "测试来源",
) -> RAGResult:
    """Helper to create a RAGResult for tests."""
    return RAGResult(
        content=content,
        source_type="chroma",  # type: ignore[arg-type]
        source_id=source_id,
        source_name=source_name,
        score=score,
    )


# ── Construction ─────────────────────────────────────────────────


class TestCrossEncoderRerankerInit:
    """CrossEncoderReranker construction."""

    def test_default_params(self) -> None:
        """Default constructor uses expected defaults."""
        reranker = CrossEncoderReranker()
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._batch_size == 16
        assert reranker._max_length == 512
        assert reranker._model is None
        assert reranker.is_loaded is False

    def test_custom_params(self) -> None:
        """Custom parameters are stored correctly."""
        reranker = CrossEncoderReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            batch_size=32,
            max_length=256,
        )
        assert reranker._model_name == "BAAI/bge-reranker-v2-m3"
        assert reranker._batch_size == 32
        assert reranker._max_length == 256


# ── Model Loading ────────────────────────────────────────────────


class TestCrossEncoderRerankerLoad:
    """Model loading logic."""

    def test_load_success(self) -> None:
        """load() sets _model on success."""
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_instance = MagicMock()
            mock_ce.return_value = mock_instance
            reranker.load()
            assert reranker._model is mock_instance
            assert reranker.is_loaded is True
            mock_ce.assert_called_once_with(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )

    def test_load_twice_idempotent(self) -> None:
        """Calling load() twice only creates the model once."""
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            reranker.load()
            reranker.load()
            mock_ce.assert_called_once()

    def test_load_failure_sets_none(self) -> None:
        """When CrossEncoder construction fails, _model stays None."""
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder", side_effect=RuntimeError("fail")):
            reranker.load()
            assert reranker._model is None
            assert reranker.is_loaded is False

    def test_is_loaded_false_before_load(self) -> None:
        """is_loaded property returns False before model is loaded."""
        reranker = CrossEncoderReranker()
        assert reranker.is_loaded is False

    def test_is_loaded_true_after_load(self) -> None:
        """is_loaded property returns True after successful load."""
        reranker = CrossEncoderReranker()
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker.load()
            assert reranker.is_loaded is True


# ── Re-ranking Logic ─────────────────────────────────────────────


class TestCrossEncoderRerankerRerank:
    """Core reranking logic."""

    async def test_empty_results(self) -> None:
        """Empty results list returns empty list."""
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_empty_query(self) -> None:
        """Empty query returns top_k results as-is."""
        results = [_make_result("a"), _make_result("b")]
        reranker = CrossEncoderReranker()
        result = await reranker.rerank("", results, top_k=5)
        assert len(result) == 2

    async def test_model_not_loaded_uses_original_order(self) -> None:
        """When model is None, original list order is preserved."""
        results = [
            _make_result("a", score=0.3),
            _make_result("b", score=0.9),
        ]
        reranker = CrossEncoderReranker()
        reranker._model = None  # explicitly no model
        result = await reranker.rerank("query", results, top_k=5)
        # The method returns results[:top_k] without re-sorting
        assert result[0].source_id == "a"
        assert result[1].source_id == "b"

    async def test_rerank_changes_order(self) -> None:
        """Re-ranking with mock model changes result order by cross-encoder scores."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Return scores that invert the original order
        # dtype=float32 to match what a real cross-encoder returns.
        # A default np.array gives float64, and np.float64 IS a subclass of
        # float while np.float32 is NOT — so a float64 mock silently exercises
        # a different code path than production and hides type-branch bugs.
        mock_model.predict.return_value = np.array([0.2, 0.9, 0.5], dtype=np.float32)
        reranker._model = mock_model

        results = [
            _make_result("a", score=0.9, content="文档A的内容"),
            _make_result("b", score=0.8, content="文档B的内容"),
            _make_result("c", score=0.7, content="文档C的内容"),
        ]
        top = await reranker.rerank("test query", results, top_k=3)
        assert len(top) == 3
        # b has highest cross-encoder score (0.9), should be first
        assert top[0].source_id == "b"
        assert top[0].score == pytest.approx(0.7109, abs=1e-3)  # sigmoid(0.9)
        # a has 0.2, should be last
        assert top[-1].source_id == "a"

    async def test_rerank_updates_scores(self) -> None:
        """Scores on RAGResult are updated with cross-encoder scores."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.75, 0.25], dtype=np.float32)
        reranker._model = mock_model

        results = [
            _make_result("a", score=0.5, content="aaa"),
            _make_result("b", score=0.6, content="bbb"),
        ]
        top = await reranker.rerank("query", results, top_k=5)
        assert top[0].score == pytest.approx(0.6792, abs=1e-3)  # sigmoid(0.75)
        assert top[1].score == pytest.approx(0.5622, abs=1e-3)  # sigmoid(0.25)

    async def test_top_k_limits_results(self) -> None:
        """top_k parameter limits the number of returned results."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32
        )
        reranker._model = mock_model

        results = [_make_result(str(i), content=f"doc_{i}") for i in range(5)]
        top = await reranker.rerank("query", results, top_k=2)
        assert len(top) == 2

    async def test_predict_exception_triggers_fallback(self) -> None:
        """When predict() raises, results in original order are returned."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("评分失败")
        reranker._model = mock_model

        results = [
            _make_result("a", score=0.5),
            _make_result("b", score=0.9),
        ]
        result = await reranker.rerank("query", results, top_k=5)
        # results[:top_k] returns original list order
        assert result[0].source_id == "a"

    async def test_model_returns_list_scores(self) -> None:
        """Handle models that return [neg_score, pos_score] lists."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Some cross-encoders return a list of [neg_logit, pos_logit] per pair
        mock_model.predict.return_value = [
            [0.1, 0.8],
            [0.4, 0.3],
        ]
        reranker._model = mock_model

        results = [
            _make_result("a", content="doc_a"),
            _make_result("b", content="doc_b"),
        ]
        top = await reranker.rerank("query", results, top_k=5)
        assert len(top) == 2
        # Should use the positive class score (index 1)
        assert top[0].source_id == "a"  # pos score = 0.8
        assert top[0].score == pytest.approx(0.6900, abs=1e-3)  # sigmoid(0.8)
        assert top[1].score == pytest.approx(0.5744, abs=1e-3)  # sigmoid(0.3)

    async def test_model_returns_single_element_list(self) -> None:
        """Handle models that return [score] per pair."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [[0.9], [0.1]]
        reranker._model = mock_model

        results = [
            _make_result("a", content="doc_a"),
            _make_result("b", content="doc_b"),
        ]
        top = await reranker.rerank("query", results, top_k=5)
        assert top[0].score == pytest.approx(0.7109, abs=1e-3)  # sigmoid(0.9)

    async def test_content_truncated_to_max_length(self) -> None:
        """Content is truncated to max_length * 3 characters for scoring."""
        reranker = CrossEncoderReranker(max_length=10)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5, 0.5], dtype=np.float32)
        reranker._model = mock_model

        long_content = "X" * 100
        results = [
            _make_result("a", content=long_content),
            _make_result("b", content="short"),
        ]
        await reranker.rerank("query", results, top_k=5)
        # Check that the pairs were truncated
        call_args = mock_model.predict.call_args
        assert call_args is not None
        pairs = call_args[0][0]
        assert len(pairs[0][1]) <= 30  # max_length * 3 = 30
        assert pairs[1][1] == "short"  # short enough, unchanged

    async def test_top_k_larger_than_results(self) -> None:
        """top_k larger than the number of results returns all results."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.8], dtype=np.float32)
        reranker._model = mock_model

        results = [_make_result("a"), _make_result("b")]
        top = await reranker.rerank("query", results, top_k=100)
        assert len(top) == 2

    async def test_single_result(self) -> None:
        """Single result is returned as-is (with score update)."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.95], dtype=np.float32)
        reranker._model = mock_model

        result = await reranker.rerank("query", [_make_result("a")], top_k=5)
        assert len(result) == 1
        # The value is the *normalised* score: sigmoid(0.95) = 0.7211. Raw
        # logits are never surfaced — see test_scores_are_unit_interval.
        assert result[0].score == pytest.approx(0.7211, abs=1e-3)


class TestScoreScaleContract:
    """``RAGResult.score`` must stay in [0, 1].

    Every other score producer in this codebase (RRF/score fusion ~0.016,
    cosine similarity) yields [0,1], and the frontend renders ``score`` as a
    percentage with a bar width of ``score * 100``. A raw cross-encoder logit
    (e.g. 7.83) would therefore display as "783%" and overflow its container —
    a bug that was dormant only because the reranker never actually ran.
    """

    @pytest.mark.asyncio
    async def test_scores_are_unit_interval(self) -> None:
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        # Logits well outside [0,1] — the real model's typical output range.
        mock_model.predict.return_value = np.array([7.83, -4.2, 0.0], dtype=np.float32)
        reranker._model = mock_model

        results = [_make_result(x, score=0.5, content=f"doc_{x}") for x in "abc"]
        top = await reranker.rerank("query", results, top_k=3)

        for r in top:
            assert 0.0 <= r.score <= 1.0, f"score {r.score} escaped [0,1]"
        # The largest logit must still rank first.
        assert top[0].source_id == "a"
        assert top[2].source_id == "b"

    @pytest.mark.asyncio
    async def test_ordering_follows_the_raw_logit(self) -> None:
        """Normalisation must be monotone — it may not reorder."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-1.0, 5.0, 2.0], dtype=np.float32)
        reranker._model = mock_model

        results = [_make_result(x, score=0.5, content=f"doc_{x}") for x in "abc"]
        top = await reranker.rerank("query", results, top_k=3)

        # logits: b(5.0) > c(2.0) > a(-1.0)
        assert [r.source_id for r in top] == ["b", "c", "a"]

    def test_extreme_logits_do_not_overflow(self) -> None:
        """math.exp(1000) raises OverflowError; the helper must clamp."""
        assert CrossEncoderReranker._to_unit_score(1000.0) == pytest.approx(1.0)
        assert CrossEncoderReranker._to_unit_score(-1000.0) == pytest.approx(0.0)

    def test_zero_logit_is_one_half(self) -> None:
        assert CrossEncoderReranker._to_unit_score(0.0) == pytest.approx(0.5)
