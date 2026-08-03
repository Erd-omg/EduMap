"""Tests for ForgettingCurveService — Ebbinghaus + Bayesian update.

Pure algorithm tests. Uses time.time() manipulation via contextlib.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.learning_path.forgetting_curve import (
    ALERT_RECALL_THRESHOLD,
    PRIOR_ALPHA,
    PRIOR_BETA,
    S_BASE,
    URGENT_RECALL_THRESHOLD,
    ForgettingCurveService,
    ForgettingState,
)


class TestForgettingState:
    """ForgettingState — individual KP memory state."""

    def test_initial_state(self) -> None:
        """New state has default prior parameters."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        assert state.alpha == PRIOR_ALPHA
        assert state.beta == PRIOR_BETA
        assert state.strength == S_BASE
        assert state.review_count == 0

    def test_posterior_mean_default(self) -> None:
        """Default prior gives posterior mean of 0.5."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        assert state.posterior_mean == 0.5

    def test_predict_recall_unseen_returns_zero(self) -> None:
        """Never-reviewed KP returns recall probability 0.0."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        assert state.predict_recall() == 0.0

    def test_predict_recall_immediate_after_review(self) -> None:
        """Just-reviewed KP returns recall probability ~1.0."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        state.last_review_time = time.time()
        recall = state.predict_recall()
        assert recall == pytest.approx(1.0, rel=0.01)

    def test_predict_recall_decreases_over_time(self) -> None:
        """Recall probability decreases as time passes."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        state.last_review_time = time.time() - 3600  # 1 hour ago
        recall_1h = state.predict_recall()

        state.last_review_time = time.time() - 7200  # 2 hours ago
        recall_2h = state.predict_recall()

        assert recall_2h < recall_1h

    def test_update_after_quiz_perfect_score(self) -> None:
        """Perfect quiz score increases alpha and sets reasonable strength."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1", alpha=2, beta=2, strength=S_BASE)
        old_alpha = state.alpha

        state.update_after_quiz(1.0)

        assert state.alpha == old_alpha + 1.0  # alpha increased by score
        assert state.review_count == 1
        # Strength = S_BASE * posterior_mean * log2(review_count+1)
        # = 24 * (3/5) * 1.0 = 14.4
        assert state.strength > 0
        assert state.strength < S_BASE  # conservative estimate after first review

    def test_update_after_quiz_zero_score(self) -> None:
        """Zero quiz score increases beta (failure count)."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1", alpha=2, beta=2)
        old_beta = state.beta

        state.update_after_quiz(0.0)

        assert state.beta > old_beta
        assert state.review_count == 1

    def test_update_after_quiz_partial_score(self) -> None:
        """Partial score updates both alpha and beta."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1", alpha=2, beta=2)

        state.update_after_quiz(0.75)

        assert state.alpha == 2.75
        assert state.beta == 2.25
        assert state.review_count == 1

    def test_posterior_mean_after_updates(self) -> None:
        """Posterior mean reflects cumulative scores."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        state.update_after_quiz(1.0)  # perfect
        state.update_after_quiz(0.5)  # half
        state.update_after_quiz(1.0)  # perfect

        # Alpha = 2 + 1 + 0.5 + 1 = 4.5
        # Beta  = 2 + 0 + 0.5 + 0 = 2.5
        # posterior_mean = 4.5 / 7.0 ≈ 0.643
        assert state.posterior_mean == pytest.approx(4.5 / 7.0, rel=0.01)

    def test_strength_increases_with_reviews(self) -> None:
        """Strength grows sub-linearly with repeated reviews."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        strengths = []
        for _ in range(5):
            state.update_after_quiz(0.9)
            strengths.append(state.strength)

        # Each review should increase strength (with diminishing returns)
        for i in range(1, len(strengths)):
            assert strengths[i] >= strengths[i - 1]

    def test_to_dict_roundtrip(self) -> None:
        """to_dict() → from_dict() roundtrip preserves state."""
        state = ForgettingState(kp_id="kp-a", user_id="user-1")
        state.update_after_quiz(0.85)
        data = state.to_dict()

        restored = ForgettingState.from_dict(data)
        assert restored.kp_id == state.kp_id
        assert restored.alpha == pytest.approx(state.alpha)
        assert restored.beta == pytest.approx(state.beta)
        assert restored.strength == pytest.approx(state.strength)


class TestForgettingCurveService:
    """ForgettingCurveService — service-level operations."""

    @pytest.fixture
    def svc(self) -> ForgettingCurveService:
        return ForgettingCurveService()

    @pytest.mark.asyncio
    async def test_predict_recall_no_state(self, svc: ForgettingCurveService) -> None:
        """No state for KP returns 0.0."""
        recall = await svc.predict_recall("kp-unknown", "user-1")
        assert recall == 0.0

    @pytest.mark.asyncio
    async def test_update_creates_state(self, svc: ForgettingCurveService) -> None:
        """First update creates a new ForgettingState."""
        state = await svc.update_after_quiz("kp-a", "user-1", score=0.9)
        assert state.kp_id == "kp-a"
        assert state.user_id == "user-1"
        assert state.review_count == 1

    @pytest.mark.asyncio
    async def test_update_then_predict(self, svc: ForgettingCurveService) -> None:
        """After quiz update, predict_recall returns non-zero."""
        await svc.update_after_quiz("kp-a", "user-1", score=0.9)
        # Immediately after review, recall should be near 1.0
        recall = await svc.predict_recall("kp-a", "user-1")
        assert recall == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_get_state_after_update(self, svc: ForgettingCurveService) -> None:
        """get_state returns the state after update."""
        updated = await svc.update_after_quiz("kp-a", "user-1", score=0.8)
        fetched = await svc.get_state("kp-a", "user-1")
        assert fetched is not None
        assert fetched.alpha == updated.alpha

    @pytest.mark.asyncio
    async def test_get_state_nonexistent(self, svc: ForgettingCurveService) -> None:
        """get_state for unknown KP returns None."""
        state = await svc.get_state("kp-nonexistent", "user-1")
        assert state is None

    @pytest.mark.asyncio
    async def test_record_review(self, svc: ForgettingCurveService) -> None:
        """record_review increments review_count and strengthens memory."""
        await svc.update_after_quiz("kp-a", "user-1", score=0.9)
        old_strength = (await svc.get_state("kp-a", "user-1")).strength

        await svc.record_review("kp-a", "user-1")
        state = await svc.get_state("kp-a", "user-1")
        assert state.review_count == 2
        assert state.strength >= old_strength

    @pytest.mark.asyncio
    async def test_get_alerts_empty(self, svc: ForgettingCurveService) -> None:
        """get_alerts returns empty list when no states exist."""
        alerts = await svc.get_alerts("user-1")
        assert alerts == []

    @pytest.mark.asyncio
    async def test_get_alerts_classification(self, svc: ForgettingCurveService) -> None:
        """get_alerts classifies by urgency level."""
        # Create a "forgotten" KP (very old review)
        forgotten = ForgettingState(kp_id="kp-old", user_id="user-1")
        forgotten.last_review_time = time.time() - 3600 * 24 * 30  # 30 days ago
        forgotten.review_count = 1
        svc._states[("user-1", "kp-old")] = forgotten

        # Create a "recent" KP
        recent = ForgettingState(kp_id="kp-recent", user_id="user-1")
        recent.last_review_time = time.time() - 60  # 1 min ago
        recent.review_count = 1
        svc._states[("user-1", "kp-recent")] = recent

        alerts = await svc.get_alerts("user-1")

        # Should have both
        assert len(alerts) == 2
        # Sorted: urgent first, then warning, then ok
        level_order = [a["alert_level"] for a in alerts]
        assert level_order == sorted(level_order, key=lambda x: {"urgent": 0, "warning": 1, "ok": 2}[x])

    @pytest.mark.asyncio
    async def test_get_all_states(self, svc: ForgettingCurveService) -> None:
        """get_all_states returns all states for a user."""
        await svc.update_after_quiz("kp-a", "user-1", score=0.9)
        await svc.update_after_quiz("kp-b", "user-1", score=0.5)
        await svc.update_after_quiz("kp-c", "user-2", score=1.0)  # different user

        states = await svc.get_all_states("user-1")
        assert len(states) == 2  # only user-1's states
        kp_ids = {s["kp_id"] for s in states}
        assert kp_ids == {"kp-a", "kp-b"}

    def test_serialize_roundtrip(self) -> None:
        """get_state_dict → load_state_dict roundtrip."""
        svc = ForgettingCurveService()

        # Add some states directly
        svc._states[("user-1", "kp-a")] = ForgettingState(
            kp_id="kp-a", user_id="user-1", alpha=5, beta=2,
        )
        svc._states[("user-1", "kp-b")] = ForgettingState(
            kp_id="kp-b", user_id="user-1", alpha=3, beta=4,
        )

        data = svc.get_state_dict()
        assert len(data) == 2

        # Load into a new service
        svc2 = ForgettingCurveService()
        svc2.load_state_dict(data)

        state_a = svc2._states.get(("user-1", "kp-a"))
        assert state_a is not None
        assert state_a.alpha == 5

    @pytest.mark.asyncio
    async def test_skip_db_load(self, svc: ForgettingCurveService) -> None:
        """Service works without db_pool."""
        svc._db_pool = None
        result = await svc.predict_recall("kp-a", "user-1")
        assert result == 0.0  # no state, returns 0.0
