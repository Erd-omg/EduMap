"""Tests for the forgetting-curve review log (data pipeline for I-3).

Why this file exists
--------------------
``forgetting_curve_state`` keeps only the latest fitted parameters and
``learning_progress`` is an upsert, so the (time, score) *series* was being
discarded on every update.  That is why the forgetting curve could not be
fitted or evaluated on real data — not an algorithm problem but a missing
pipeline.

The two properties worth protecting:

1. **The review log is append-only evidence.** A failure to append must not
   break the live state update (the feature must keep working even if the
   audit trail has a gap).
2. **The read-back shape matches what the evaluator consumes.** If ``load_review_history``
   returned gaps measured from the wrong reference point, every downstream
   metric would silently be wrong.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.learning_path.forgetting_curve import ForgettingCurveService


def _svc_with_conn() -> tuple[ForgettingCurveService, AsyncMock]:
    """Service backed by a mock pool.

    ``fetchrow`` returns None so ``_load_state`` sees "no persisted state" and
    starts from the Beta prior — otherwise the auto-generated MagicMock rows
    leak in and arithmetic on them explodes with confusing TypeErrors.
    """
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetchrow = AsyncMock(return_value=None)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=cm)
    return ForgettingCurveService(db_pool=pool), conn


class TestReviewLogging:
    @pytest.mark.asyncio
    async def test_update_after_quiz_appends_a_row(self) -> None:
        svc, conn = _svc_with_conn()

        await svc.update_after_quiz("kp1", "u1", 0.9)

        sql_calls = [
            str(c.args[0]) for c in conn.execute.call_args_list if c.args
        ]
        assert any("INSERT INTO forgetting_review_log" in s for s in sql_calls)

    @pytest.mark.asyncio
    async def test_score_is_clamped_into_unit_range(self) -> None:
        """A caller passing 1.5 or -0.2 must not poison the history."""
        svc, conn = _svc_with_conn()

        await svc.update_after_quiz("kp1", "u1", 1.5)
        await svc.update_after_quiz("kp2", "u1", -0.2)

        logged = [
            c.args[3] for c in conn.execute.call_args_list
            if c.args and "forgetting_review_log" in str(c.args[0])
        ]
        assert logged == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_first_review_logged_as_learn(self) -> None:
        """First exposure and later reviews are distinguishable in the log."""
        svc, conn = _svc_with_conn()

        await svc.update_after_quiz("kp1", "u1", 0.7)  # first -> review_count 1
        first_types = [
            c.args[4] for c in conn.execute.call_args_list
            if c.args and "forgetting_review_log" in str(c.args[0])
        ]
        assert first_types == ["learn"]

        await svc.update_after_quiz("kp1", "u1", 0.8)  # second -> review
        all_types = [
            c.args[4] for c in conn.execute.call_args_list
            if c.args and "forgetting_review_log" in str(c.args[0])
        ]
        assert all_types == ["learn", "review"]

    @pytest.mark.asyncio
    async def test_record_review_also_appends(self) -> None:
        svc, conn = _svc_with_conn()

        await svc.record_review("kp1", "u1")

        logged = [
            c.args for c in conn.execute.call_args_list
            if c.args and "forgetting_review_log" in str(c.args[0])
        ]
        assert len(logged) == 1
        # (sql, user_id, kp_id, score, event_type, source)
        assert logged[0][1] == "u1"
        assert logged[0][2] == "kp1"
        assert logged[0][4] == "review"
        assert logged[0][5] == "path"

    @pytest.mark.asyncio
    async def test_logging_failure_does_not_break_state_update(self) -> None:
        """The audit trail is evidence; the state update is the product."""
        svc, conn = _svc_with_conn()
        conn.execute = AsyncMock(side_effect=RuntimeError("log table missing"))

        state = await svc.update_after_quiz("kp1", "u1", 0.9)

        assert state.review_count == 1, "state must still have been updated"
        assert state.posterior_mean > 0.5

    @pytest.mark.asyncio
    async def test_no_db_configured_is_a_noop(self) -> None:
        svc = ForgettingCurveService(db_pool=None)
        state = await svc.update_after_quiz("kp1", "u1", 0.9)
        assert state.review_count == 1


class TestLoadReviewHistory:
    @pytest.mark.asyncio
    async def test_first_entry_has_zero_elapsed(self) -> None:
        """The initial exposure has no prior interval — the scorer skips it."""
        base = datetime(2026, 9, 1, tzinfo=timezone.utc)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"kp_id": "kp1", "score": 0.5, "reviewed_at": base},
            {"kp_id": "kp1", "score": 0.8, "reviewed_at": base + timedelta(hours=24)},
        ])
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        svc = ForgettingCurveService(db_pool=pool)

        histories = await svc.load_review_history("u1")

        assert histories["kp1"][0] == (0.0, 0.5)
        assert histories["kp1"][1] == (24.0, 0.8)

    @pytest.mark.asyncio
    async def test_gaps_are_relative_to_previous_not_epoch(self) -> None:
        """Three reviews 24h apart must yield gaps 0, 24, 24 — not 0, 24, 48."""
        base = datetime(2026, 9, 1, tzinfo=timezone.utc)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"kp_id": "kp1", "score": 0.5, "reviewed_at": base},
            {"kp_id": "kp1", "score": 0.6, "reviewed_at": base + timedelta(hours=24)},
            {"kp_id": "kp1", "score": 0.7, "reviewed_at": base + timedelta(hours=48)},
        ])
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        svc = ForgettingCurveService(db_pool=pool)

        histories = await svc.load_review_history("u1")

        assert [gap for gap, _ in histories["kp1"]] == [0.0, 24.0, 24.0]

    @pytest.mark.asyncio
    async def test_multiple_kps_are_tracked_separately(self) -> None:
        """A gap must not be computed across different knowledge points."""
        base = datetime(2026, 9, 1, tzinfo=timezone.utc)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"kp_id": "kp1", "score": 0.5, "reviewed_at": base},
            {"kp_id": "kp2", "score": 0.6, "reviewed_at": base + timedelta(hours=100)},
            {"kp_id": "kp1", "score": 0.7, "reviewed_at": base + timedelta(hours=5)},
        ])
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        svc = ForgettingCurveService(db_pool=pool)

        histories = await svc.load_review_history("u1")

        assert [g for g, _ in histories["kp1"]] == [0.0, 5.0]
        assert [g for g, _ in histories["kp2"]] == [0.0]

    @pytest.mark.asyncio
    async def test_query_failure_returns_empty_not_raises(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(side_effect=RuntimeError("table missing"))
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        svc = ForgettingCurveService(db_pool=pool)

        assert await svc.load_review_history("u1") == {}

    @pytest.mark.asyncio
    async def test_no_db_returns_empty(self) -> None:
        svc = ForgettingCurveService(db_pool=None)
        assert await svc.load_review_history("u1") == {}


class TestHistoryFeedsEvaluator:
    """The read-back must be directly consumable by the evaluation harness."""

    @pytest.mark.asyncio
    async def test_output_shape_matches_predict_all_contract(self) -> None:
        from src.learning_path.forgetting_eval import predict_all

        base = datetime(2026, 9, 1, tzinfo=timezone.utc)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {"kp_id": "kp1", "score": 0.5, "reviewed_at": base},
            {"kp_id": "kp1", "score": 0.9, "reviewed_at": base + timedelta(hours=10)},
            {"kp_id": "kp1", "score": 0.8, "reviewed_at": base + timedelta(hours=30)},
        ])
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)
        svc = ForgettingCurveService(db_pool=pool)

        histories = await svc.load_review_history("u1")
        observations = predict_all(lambda e, n, m: 0.6, list(histories.values()))

        # 3 reviews -> 2 scored observations (initial exposure skipped)
        assert len(observations) == 2
        assert observations[0].elapsed_hours == 10.0
        assert observations[1].elapsed_hours == 20.0
