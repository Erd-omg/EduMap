"""Tests for Orchestrator graph routing logic.

Pure deterministic tests for conditional routing functions.
No mocks, no async, no LangGraph compilation required.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from src.agents.orchestrator.graph import (
    route_after_designer,
    route_after_failure,
    route_after_guardian,
    route_after_review,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_state(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a minimal EduMapState-like dict with defaults."""
    base: dict[str, Any] = {
        "task_input": "",
        "task_type": "generate",
        "current_phase": "EXTRACT",
        "overall_status": "running",
        "user_id": "test-user",
        "session_id": "test-session",
        "agent_results": {},
        "generation_plan": None,
        "knowledge_units": [],
        "generated_resources": [],
        "audit_results": [],
        "audit_log": [],
        "assessment_result": None,
        "generation_retry_count": 0,
        "max_retries": 2,
        "errors": [],
    }
    if overrides:
        base.update(overrides)
    return base


def make_passed_entry(id_: str = "kp-a") -> dict[str, Any]:
    """Return an audit-results entry that passes."""
    return {"id": id_, "passed": True, "detail": "ok"}


def make_failed_entry(id_: str = "kp-a", detail: str = "Failed quality check") -> dict[str, Any]:
    """Return an audit-results entry that fails."""
    return {"id": id_, "passed": False, "detail": detail}


# ── route_after_guardian ─────────────────────────────────────────────────────


class TestRouteAfterGuardian:
    """route_after_guardian: after VALIDATE phase."""

    def test_valid_returns_designer(self) -> None:
        """When guardian passes, route to designer."""
        state = make_state()
        assert route_after_guardian(state) == "designer"

    def test_failed_returns_end(self) -> None:
        """When guardian fails, abort the pipeline."""
        state = make_state({"overall_status": "failed"})
        assert route_after_guardian(state) == END

    def test_empty_state_defaults_to_designer(self) -> None:
        """Missing overall_status defaults to designer (no failure key)."""
        state = make_state({"overall_status": ""})
        assert route_after_guardian(state) == "designer"

    def test_degraded_returns_designer(self) -> None:
        """Degraded status does NOT abort — only 'failed' triggers END."""
        state = make_state({"overall_status": "degraded"})
        assert route_after_guardian(state) == "designer"

    def test_completed_returns_designer(self) -> None:
        """Already-completed status also routes to designer."""
        state = make_state({"overall_status": "completed"})
        assert route_after_guardian(state) == "designer"


# ── route_after_failure ──────────────────────────────────────────────────────


class TestRouteAfterFailure:
    """route_after_failure: generic failure check used by coder."""

    def test_ok_returns_merge(self) -> None:
        """When no failure, route to merge."""
        state = make_state()
        assert route_after_failure(state) == "merge"

    def test_failed_returns_end(self) -> None:
        """When failed, abort the pipeline."""
        state = make_state({"overall_status": "failed"})
        assert route_after_failure(state) == END

    def test_empty_status_returns_merge(self) -> None:
        """Missing or empty overall_status continues to merge."""
        state = make_state({"overall_status": ""})
        assert route_after_failure(state) == "merge"

    def test_degraded_returns_merge(self) -> None:
        """Degraded status is not a hard failure."""
        state = make_state({"overall_status": "degraded"})
        assert route_after_failure(state) == "merge"


# ── route_after_designer ─────────────────────────────────────────────────────


class TestRouteAfterDesigner:
    """route_after_designer: after GENERATE phase (designer branch)."""

    def test_ok_returns_coder(self) -> None:
        """On success, route to coder."""
        state = make_state()
        assert route_after_designer(state) == "coder"

    def test_failed_returns_end(self) -> None:
        """When failed, abort the pipeline."""
        state = make_state({"overall_status": "failed"})
        assert route_after_designer(state) == END

    def test_degraded_status_continues_to_coder(self) -> None:
        """Only 'failed' status triggers abort; degraded continues."""
        state = make_state({"overall_status": "degraded"})
        assert route_after_designer(state) == "coder"


# ── route_after_review ───────────────────────────────────────────────────────


class TestRouteAfterReview:
    """route_after_review: after REVIEW phase — retry logic."""

    # ── All pass ─────────────────────────────────────────────────────────

    def test_all_pass_returns_assessment(self) -> None:
        """When all audit entries pass, proceed to assessment."""
        state = make_state({
            "audit_results": [make_passed_entry("kp-a"), make_passed_entry("kp-b")],
        })
        assert route_after_review(state) == "assessment"

    def test_empty_audit_results_returns_assessment(self) -> None:
        """No audit results at all means nothing failed."""
        state = make_state({"audit_results": []})
        assert route_after_review(state) == "assessment"

    def test_no_passed_field_assumed_true(self) -> None:
        """Entries without a 'passed' key are treated as passing."""
        state = make_state({
            "audit_results": [
                {"id": "kp-a"},           # no passed key
                {"id": "kp-b", "passed": True},
            ],
        })
        assert route_after_review(state) == "assessment"

    # ── All fail — retry path ────────────────────────────────────────────

    def test_all_fail_triggers_retry(self) -> None:
        """When all entries fail and retries remain, route to retry_generate."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })
        assert route_after_review(state) == "retry_generate"

    def test_retry_count_incremented(self) -> None:
        """Route_after_review increments generation_retry_count when failures exist."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })
        route_after_review(state)  # Don't care about return value here
        assert state["generation_retry_count"] == 1

    def test_retry_count_increments_on_subsequent_calls(self) -> None:
        """Each sequential retry call increments the counter."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })

        route_after_review(state)
        assert state["generation_retry_count"] == 1

        route_after_review(state)
        assert state["generation_retry_count"] == 2

    # ── Mixed pass/fail ──────────────────────────────────────────────────

    def test_some_fail_triggers_retry(self) -> None:
        """Even a single failure triggers retry when retries remain."""
        state = make_state({
            "audit_results": [
                make_passed_entry("kp-a"),
                make_failed_entry("kp-b"),
                make_passed_entry("kp-c"),
            ],
            "generation_retry_count": 0,
        })
        assert route_after_review(state) == "retry_generate"
        assert state["generation_retry_count"] == 1

    # ── Max retries reached → degraded ───────────────────────────────────

    def test_retry_exhausted_returns_assess_degraded(self) -> None:
        """After max retries, route to assess_degraded."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 2,
            "max_retries": 2,
        })
        assert route_after_review(state) == "assess_degraded"

    def test_retry_exhausted_increments_beyond_max(self) -> None:
        """Counter still increments even when exhausted (for observability)."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 2,
            "max_retries": 2,
        })
        route_after_review(state)
        assert state["generation_retry_count"] == 3  # still incremented

    # ── Boundary: exactly max_retries - 1 → retry ────────────────────────

    def test_retry_at_boundary_last_allowed_retry(self) -> None:
        """At generation_retry_count=0 with max_retries=2, one retry is allowed.

        The counter is incremented BEFORE the check, so:
        0 -> 1, 1 < 2 -> retry_generate.
        """
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 2,
        })
        assert route_after_review(state) == "retry_generate"
        assert state["generation_retry_count"] == 1

    def test_retry_exactly_at_max_returns_degraded(self) -> None:
        """When generation_retry_count reaches max_retries, route to assess_degraded.

        Counter is incremented first: 1 -> 2, then 2 < 2 is False.
        """
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 1,
            "max_retries": 2,
        })
        assert route_after_review(state) == "assess_degraded"

    # ── Default max_retries ──────────────────────────────────────────────

    def test_default_max_retries_is_two(self) -> None:
        """When max_retries is not set, defaults to 2 (allowing 1 retry).

        Counter is incremented before the check:
        count=0 -> 1, 1 < 2 -> retry
        count=1 -> 2, 2 < 2 -> degraded
        """
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })
        # Remove max_retries to test default
        del state["max_retries"]

        # First failure -> retry allowed
        r1 = route_after_review(state)
        assert r1 == "retry_generate"
        assert state["generation_retry_count"] == 1

        # Second failure -> degraded (max_retries reached)
        r2 = route_after_review(state)
        assert r2 == "assess_degraded"
        assert state["generation_retry_count"] == 2

    # ── Custom max_retries ───────────────────────────────────────────────

    def test_custom_max_retries_honored(self) -> None:
        """A custom max_retries setting is respected (max_retries=5 -> 4 retries)."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 5,
        })

        for _ in range(4):
            result = route_after_review(state)
            assert result == "retry_generate"

        # 5th time → exhausted (count goes from 4 to 5, 5 < 5 is False)
        result = route_after_review(state)
        assert result == "assess_degraded"
        assert state["generation_retry_count"] == 5

    # ── Edge: single failure after many retries ──────────────────────────

    def test_zero_max_retries_immediately_degraded(self) -> None:
        """When max_retries is 0, any failure immediately goes degraded."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 0,
        })
        assert route_after_review(state) == "assess_degraded"
        assert state["generation_retry_count"] == 1


# ── Cross-routing: failure status respected everywhere ───────────────────────


class TestRoutingOnFailureStatus:
    """All routing functions respect the 'failed' overall_status."""

    @pytest.mark.parametrize(
        "routing_fn",
        [
            route_after_guardian,
            route_after_failure,
            route_after_designer,
        ],
    )
    def test_all_routes_abort_on_failure(self, routing_fn) -> None:
        """Every routing function returns END when overall_status is 'failed'."""
        state = make_state({"overall_status": "failed"})
        assert routing_fn(state) == END

    @pytest.mark.parametrize(
        "routing_fn",
        [
            route_after_guardian,
            route_after_failure,
            route_after_designer,
        ],
    )
    def test_all_routes_continue_on_running(self, routing_fn) -> None:
        """Every routing function continues when overall_status is 'running'."""
        state = make_state({"overall_status": "running"})
        result = routing_fn(state)
        assert result != END


class TestRouteAfterReviewFailureRespect:
    """route_after_review does NOT check overall_status — relies on audit."""

    def test_failed_status_still_routes_based_on_audit(self) -> None:
        """route_after_review ignores overall_status and decides via audit results."""
        state = make_state({
            "overall_status": "failed",
            "audit_results": [make_passed_entry("kp-a")],
        })
        assert route_after_review(state) == "assessment"
