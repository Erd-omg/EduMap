"""Tests for ColdStartEvaluator — verdict thresholds, score calculations, mocked DB.

The evaluator queries Neo4j for node/edge counts then computes a weighted
confidence score.  All DB interactions are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.kg.cold_start_eval import ColdStartEvaluator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def evaluator(mock_conn: AsyncMock) -> ColdStartEvaluator:
    return ColdStartEvaluator(conn=mock_conn)


# ---------------------------------------------------------------------------
# _cross_validate_count — pure static method
# ---------------------------------------------------------------------------

class TestCrossValidateCount:
    """Cross-validation score is static in the prototype."""

    def test_returns_constant_score(self) -> None:
        result = ColdStartEvaluator._cross_validate_count()
        assert result["score"] == 0.5
        assert "validated_sources" in result["detail"]
        assert result["detail"]["validated_sources"] == 0


# ---------------------------------------------------------------------------
# _verdict — pure static method, threshold mapping
# ---------------------------------------------------------------------------

class TestVerdict:
    """Confidence score → human-readable verdict mapping."""

    def test_ready_threshold(self) -> None:
        """Confidence >= 0.7 → ready."""
        assert "ready" in ColdStartEvaluator._verdict(0.7)
        assert "ready" in ColdStartEvaluator._verdict(0.85)
        assert "ready" in ColdStartEvaluator._verdict(1.0)

    def test_limited_threshold(self) -> None:
        """0.4 <= confidence < 0.7 → limited."""
        assert "limited" in ColdStartEvaluator._verdict(0.4)
        assert "limited" in ColdStartEvaluator._verdict(0.5)
        assert "limited" in ColdStartEvaluator._verdict(0.69)

    def test_insufficient_threshold(self) -> None:
        """Confidence < 0.4 → insufficient."""
        assert "insufficient" in ColdStartEvaluator._verdict(0.0)
        assert "insufficient" in ColdStartEvaluator._verdict(0.1)
        assert "insufficient" in ColdStartEvaluator._verdict(0.39)

    def test_boundary_values(self) -> None:
        """Boundary values are tested explicitly."""
        assert "ready" in ColdStartEvaluator._verdict(0.7)
        assert "ready" in ColdStartEvaluator._verdict(0.7 + 1e-9)
        assert "limited" in ColdStartEvaluator._verdict(0.4)
        assert "limited" in ColdStartEvaluator._verdict(0.69999)
        assert "insufficient" in ColdStartEvaluator._verdict(0.39999)


# ---------------------------------------------------------------------------
# _corpus_density — mocked DB
# ---------------------------------------------------------------------------

class TestCorpusDensity:
    """Corpus density score based on KnowledgePoint count."""

    @pytest.mark.asyncio
    async def test_empty_graph(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 0}])
        result = await evaluator._corpus_density()
        assert result["score"] == 0.0
        assert result["detail"]["topic_count"] == 0

    @pytest.mark.asyncio
    async def test_full_target(self, evaluator: ColdStartEvaluator) -> None:
        """Topic count == target minimum → score 1.0."""
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 5}])
        result = await evaluator._corpus_density()
        assert result["score"] == 1.0
        assert result["detail"]["topic_count"] == 5

    @pytest.mark.asyncio
    async def test_above_target_is_capped(self, evaluator: ColdStartEvaluator) -> None:
        """Topic count above target minimum → score still 1.0 (capped)."""
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 100}])
        result = await evaluator._corpus_density()
        assert result["score"] == 1.0  # min(1.0, 100/5)

    @pytest.mark.asyncio
    async def test_partial_density(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 3}])
        result = await evaluator._corpus_density()
        assert result["score"] == 3.0 / 5  # 0.6
        assert result["detail"]["topic_count"] == 3
        assert result["detail"]["target_minimum"] == 5

    @pytest.mark.asyncio
    async def test_db_error_returns_zero(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(side_effect=Exception("DB timeout"))
        result = await evaluator._corpus_density()
        assert result["score"] == 0.0
        assert "error" in result["detail"]


# ---------------------------------------------------------------------------
# _kg_coverage — mocked DB
# ---------------------------------------------------------------------------

class TestKgCoverage:
    """KG coverage score based on node + edge counts."""

    @pytest.mark.asyncio
    async def test_empty_graph(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 0}])
        result = await evaluator._kg_coverage()
        assert result["score"] == 0.0
        assert result["detail"]["node_count"] == 0
        assert result["detail"]["edge_count"] == 0

    @pytest.mark.asyncio
    async def test_partial_coverage(self, evaluator: ColdStartEvaluator) -> None:
        """Both queries return results sequentially."""
        evaluator.conn.execute_read = AsyncMock(side_effect=[
            [{"cnt": 4}],   # nodes  - 4/8 = 0.5
            [{"cnt": 5}],   # edges  - 5/10 = 0.5
        ])
        result = await evaluator._kg_coverage()
        # score = 0.6 * 0.5 + 0.4 * 0.5 = 0.5
        assert result["score"] == 0.5
        assert result["detail"]["node_count"] == 4
        assert result["detail"]["edge_count"] == 5

    @pytest.mark.asyncio
    async def test_full_coverage(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(side_effect=[
            [{"cnt": 8}],    # nodes → 1.0
            [{"cnt": 10}],   # edges → 1.0
        ])
        result = await evaluator._kg_coverage()
        # score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_capped_at_target(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(side_effect=[
            [{"cnt": 20}],   # 20/8 = 2.5 → min(1.0, 2.5) = 1.0
            [{"cnt": 30}],   # 30/10 = 3.0 → min(1.0, 3.0) = 1.0
        ])
        result = await evaluator._kg_coverage()
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_db_error_returns_zero(self, evaluator: ColdStartEvaluator) -> None:
        evaluator.conn.execute_read = AsyncMock(side_effect=Exception("Connection refused"))
        result = await evaluator._kg_coverage()
        assert result["score"] == 0.0
        assert "error" in result["detail"]


# ---------------------------------------------------------------------------
# evaluate — full integration with mocked DB
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Full evaluate() pipeline — combines all sub-scores with weights."""

    @pytest.mark.asyncio
    async def test_evaluate_empty_graph(self, evaluator: ColdStartEvaluator) -> None:
        """An empty graph should yield low confidence."""
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 0}])

        result = await evaluator.evaluate()

        assert "confidence" in result
        assert "factors" in result
        assert "verdict" in result
        # Empty graph: corpus_density=0, kg_coverage=0, cross_validate=0.5
        # confidence = 0.4*0 + 0.4*0 + 0.2*0.5 = 0.1
        assert result["confidence"] == 0.1
        assert "insufficient" in result["verdict"]

    @pytest.mark.asyncio
    async def test_evaluate_populated_graph(self, evaluator: ColdStartEvaluator) -> None:
        """A well-populated graph should yield high confidence."""
        evaluator.conn.execute_read = AsyncMock(side_effect=[
            [{"cnt": 10}],   # corpus_density: 10/5 = 1.0 (capped)
            [{"cnt": 20}],   # kg_coverage: 20 nodes → 1.0
            [{"cnt": 25}],   # kg_coverage: 25 edges → 1.0
        ])

        result = await evaluator.evaluate()

        # corpus_density = 1.0 (weight 0.4)
        # kg_coverage = 0.6*1.0 + 0.4*1.0 = 1.0 (weight 0.4)
        # cross_validate = 0.5 (weight 0.2)
        # confidence = 0.4*1.0 + 0.4*1.0 + 0.2*0.5 = 0.4 + 0.4 + 0.1 = 0.9
        assert result["confidence"] == 0.9
        assert "ready" in result["verdict"]

    @pytest.mark.asyncio
    async def test_evaluate_mid_range(self, evaluator: ColdStartEvaluator) -> None:
        """A partially populated graph yields mid-range confidence."""
        evaluator.conn.execute_read = AsyncMock(side_effect=[
            [{"cnt": 2}],    # corpus_density: 2/5 = 0.4
            [{"cnt": 4}],    # kg_coverage: 4/8 = 0.5 (nodes)
            [{"cnt": 5}],    # kg_coverage: 5/10 = 0.5 (edges)
        ])

        result = await evaluator.evaluate()

        # corpus_density = 0.4 (weight 0.4)
        # kg_coverage = 0.6*0.5 + 0.4*0.5 = 0.5 (weight 0.4)
        # cross_validate = 0.5 (weight 0.2)
        # confidence = 0.4*0.4 + 0.4*0.5 + 0.2*0.5 = 0.16 + 0.20 + 0.10 = 0.46
        assert 0.4 <= result["confidence"] <= 0.7
        assert "limited" in result["verdict"]

    @pytest.mark.asyncio
    async def test_evaluate_confidence_clamped(self, evaluator: ColdStartEvaluator) -> None:
        """Confidence is clamped to [0.0, 1.0]."""
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 999}])

        # Very high counts → corpus=1.0, kg_coverage=1.0, cross_val=0.5
        # confidence = 0.4+0.4+0.1 = 0.9 (within bounds anyway)
        result = await evaluator.evaluate()
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_custom_weights(self, mock_conn: AsyncMock) -> None:
        """Confidence changes when custom weights are supplied."""
        evaluator = ColdStartEvaluator(conn=mock_conn, w1=0.0, w2=0.0, w3=1.0)
        mock_conn.execute_read = AsyncMock(return_value=[{"cnt": 0}])

        result = await evaluator.evaluate()
        # Only cross_validate matters: 1.0 * 0.5 = 0.5
        assert result["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_factors_structure(self, evaluator: ColdStartEvaluator) -> None:
        """The factors dict contains expected keys for each sub-factor."""
        evaluator.conn.execute_read = AsyncMock(return_value=[{"cnt": 3}])

        result = await evaluator.evaluate()
        factors = result["factors"]
        assert set(factors.keys()) == {"corpus_density", "kg_coverage", "cross_validate"}
        for key in factors:
            assert "weight" in factors[key]
            assert "score" in factors[key]
            assert "detail" in factors[key]
