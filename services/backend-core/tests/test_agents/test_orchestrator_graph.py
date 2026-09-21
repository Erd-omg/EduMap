"""Tests for Orchestrator graph routing logic.

Pure deterministic tests for conditional routing functions.
No mocks, no async, no LangGraph compilation required.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.graph import END

from src.agents.orchestrator.graph import (
    _graph_ctx,
    create_graph,
    retry_prep_node,
    route_after_failure,
    route_after_generation,
    route_after_guardian,
    route_after_review,
)
from src.agents.orchestrator.state import (
    EduMapState,
    _accumulate_resources,
    _merge_agent_results,
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
    """route_after_guardian: after VALIDATE phase — fans out to both branches."""

    def test_valid_fans_out_to_both_branches(self) -> None:
        """When guardian passes, route to designer AND coder (parallel fan-out)."""
        state = make_state()
        assert route_after_guardian(state) == ["designer", "coder"]

    def test_failed_returns_end(self) -> None:
        """When guardian fails, abort the pipeline."""
        state = make_state({"overall_status": "failed"})
        assert route_after_guardian(state) == END

    def test_empty_state_defaults_to_both_branches(self) -> None:
        """Missing overall_status fans out (no failure key)."""
        state = make_state({"overall_status": ""})
        assert route_after_guardian(state) == ["designer", "coder"]

    def test_degraded_fans_out(self) -> None:
        """Degraded status does NOT abort — only 'failed' triggers END."""
        state = make_state({"overall_status": "degraded"})
        assert route_after_guardian(state) == ["designer", "coder"]

    def test_completed_fans_out(self) -> None:
        """Already-completed status also fans out."""
        state = make_state({"overall_status": "completed"})
        assert route_after_guardian(state) == ["designer", "coder"]


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


# ── route_after_generation (parallel fan-in) ─────────────────────────────────


class TestRouteAfterGeneration:
    """route_after_generation: shared fan-in for the designer + coder branches."""

    def test_ok_returns_merge(self) -> None:
        """On success, join the sibling branch at merge."""
        state = make_state()
        assert route_after_generation(state) == "merge"

    def test_failed_returns_end(self) -> None:
        """When failed, abort before the merge barrier."""
        state = make_state({"overall_status": "failed"})
        assert route_after_generation(state) == END

    def test_degraded_status_continues_to_merge(self) -> None:
        """Only 'failed' status triggers abort; degraded continues."""
        state = make_state({"overall_status": "degraded"})
        assert route_after_generation(state) == "merge"


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

    def test_router_does_not_mutate_retry_count(self) -> None:
        """route_after_review must NOT increment the counter itself.

        A conditional-edge router only returns an edge label; LangGraph commits
        state from *node* returns, so mutating ``state[...]`` here is discarded
        between supersteps.  When the router did the increment, the counter
        stayed 0 forever and a permanently-failing audit looped until the
        recursion limit.  The increment belongs to retry_prep_node.
        """
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })
        route_after_review(state)
        assert state["generation_retry_count"] == 0  # untouched by the router

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

    # ── Max retries reached → degraded ───────────────────────────────────

    def test_retry_exhausted_returns_assess_degraded(self) -> None:
        """After max retries, route to assess_degraded."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 2,
            "max_retries": 2,
        })
        assert route_after_review(state) == "assess_degraded"

    # ── Boundary: retries completed so far vs. the cap ───────────────────

    def test_retry_at_boundary_last_allowed_retry(self) -> None:
        """With 0 completed retries and max_retries=2, another retry is allowed."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 2,
        })
        assert route_after_review(state) == "retry_generate"

    def test_retry_exactly_at_max_returns_degraded(self) -> None:
        """At generation_retry_count == max_retries the budget is spent."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 2,
            "max_retries": 2,
        })
        assert route_after_review(state) == "assess_degraded"

    # ── Default max_retries ──────────────────────────────────────────────

    def test_default_max_retries_is_two(self) -> None:
        """With max_retries unset (default 2), 2 completed retries exhaust it."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
        })
        del state["max_retries"]

        assert route_after_review(state) == "retry_generate"  # 0 completed

        state["generation_retry_count"] = 2
        assert route_after_review(state) == "assess_degraded"  # 2 == default cap

    # ── Custom max_retries ───────────────────────────────────────────────

    def test_custom_max_retries_honored(self) -> None:
        """A custom cap is respected: max_retries=5 allows 5 completed rounds."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 5,
        })

        for completed in range(5):
            state["generation_retry_count"] = completed
            assert route_after_review(state) == "retry_generate"

        state["generation_retry_count"] = 5
        assert route_after_review(state) == "assess_degraded"

    # ── Edge: single failure after many retries ──────────────────────────

    def test_zero_max_retries_immediately_degraded(self) -> None:
        """When max_retries is 0, any failure immediately goes degraded."""
        state = make_state({
            "audit_results": [make_failed_entry("kp-a")],
            "generation_retry_count": 0,
            "max_retries": 0,
        })
        assert route_after_review(state) == "assess_degraded"


# ── Cross-routing: failure status respected everywhere ───────────────────────


class TestRoutingOnFailureStatus:
    """All routing functions respect the 'failed' overall_status."""

    @pytest.mark.parametrize(
        "routing_fn",
        [
            route_after_guardian,
            route_after_failure,
            route_after_generation,
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
            route_after_generation,
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


# ── State reducers (parallel fan-out legality) ───────────────────────────────


class TestAccumulateResourcesReducer:
    """_accumulate_resources: the append/reset reducer for parallel branches.

    Designer and coder write this channel in the same superstep, so it needs a
    reducer; ``retry_prep`` clears it, which is why empty means "reset".
    """

    def test_append_concatenates_concurrent_branch_writes(self) -> None:
        """Both branches' resources survive the merge."""
        designer = [{"type": "explanation"}]
        coder = [{"type": "code"}]
        assert _accumulate_resources(designer, coder) == [
            {"type": "explanation"}, {"type": "code"},
        ]

    def test_append_order_is_left_then_right(self) -> None:
        assert _accumulate_resources([1], [2, 3]) == [1, 2, 3]

    def test_empty_right_resets_the_channel(self) -> None:
        """retry_prep returns [] to clear — this is what makes retry idempotent."""
        assert _accumulate_resources([{"type": "explanation"}], []) == []

    def test_reset_from_empty_is_empty(self) -> None:
        assert _accumulate_resources([], []) == []

    def test_none_inputs_tolerated(self) -> None:
        """Missing channel values must not raise (first write of a run)."""
        assert _accumulate_resources(None, [1]) == [1]
        assert _accumulate_resources([1], None) == []

    def test_does_not_mutate_left_operand(self) -> None:
        """Reducers must return a new list — LangGraph may reuse the input."""
        left = [1]
        _accumulate_resources(left, [2])
        assert left == [1]


class TestMergeAgentResultsReducer:
    """_merge_agent_results: shallow merge so both branches keep their key."""

    def test_disjoint_keys_both_survive(self) -> None:
        merged = _merge_agent_results({"designer": {"n": 1}}, {"coder": {"n": 2}})
        assert merged == {"designer": {"n": 1}, "coder": {"n": 2}}

    def test_none_inputs_tolerated(self) -> None:
        assert _merge_agent_results(None, {"coder": {}}) == {"coder": {}}
        assert _merge_agent_results({"planner": {}}, None) == {"planner": {}}
        assert _merge_agent_results(None, None) == {}

    def test_does_not_mutate_left_operand(self) -> None:
        left = {"planner": {}}
        _merge_agent_results(left, {"designer": {}})
        assert left == {"planner": {}}


class TestParallelFanOutIsLegal:
    """The designer/coder fan-out must not raise InvalidUpdateError.

    Regression guard for the state channels both branches write.  LangGraph
    does not reflect ``Annotated`` reducers into importable type hints (see the
    note on the test below), so this asserts the *behaviour* — concurrent
    updates to the shared channels are accepted and merged — rather than
    inspecting metadata.
    """

    @staticmethod
    def _build_probe_graph():
        """Two concurrent branches over EduMapState, mirroring the real graph."""
        from langgraph.graph import END, StateGraph

        async def start(_state):
            return {}

        async def designer(_state):
            return {
                "current_phase": "GENERATE",
                "generated_resources": [{"type": "explanation"}],
                "agent_results": {"designer": {"n": 1}},
            }

        async def coder(_state):
            return {
                "current_phase": "GENERATE",
                "generated_resources": [{"type": "code"}],
                "agent_results": {"coder": {"n": 2}},
            }

        async def join(_state):
            return {}

        builder = StateGraph(EduMapState)
        for name, node in (
            ("start", start), ("designer", designer),
            ("coder", coder), ("join", join),
        ):
            builder.add_node(name, node)
        builder.set_entry_point("start")
        builder.add_conditional_edges(
            "start", lambda s: ["designer", "coder"],
            {"designer": "designer", "coder": "coder"},
        )
        builder.add_conditional_edges("designer", lambda s: "join", {"join": "join"})
        builder.add_conditional_edges("coder", lambda s: "join", {"join": "join"})
        builder.add_edge("join", END)
        return builder.compile()

    async def test_concurrent_branches_do_not_raise(self) -> None:
        """A missing reducer surfaces here as InvalidUpdateError."""
        graph = self._build_probe_graph()
        result = await graph.ainvoke(
            {"generated_resources": [], "agent_results": {}}
        )
        assert result["current_phase"] == "GENERATE"

    async def test_both_branches_resources_survive(self) -> None:
        graph = self._build_probe_graph()
        result = await graph.ainvoke(
            {"generated_resources": [], "agent_results": {}}
        )
        assert sorted(r["type"] for r in result["generated_resources"]) == [
            "code", "explanation",
        ]

    async def test_both_branches_agent_results_survive(self) -> None:
        graph = self._build_probe_graph()
        result = await graph.ainvoke(
            {"generated_resources": [], "agent_results": {}}
        )
        assert result["agent_results"] == {"designer": {"n": 1}, "coder": {"n": 2}}


class TestRetryPrepNode:
    """retry_prep_node: clears resources AND owns the retry counter.

    The counter lives here rather than in route_after_review because only node
    return values are committed to state — router mutations are discarded.
    """

    async def test_clears_resources(self) -> None:
        result = await retry_prep_node(make_state({
            "generated_resources": [{"type": "explanation"}],
        }))
        assert result["generated_resources"] == []

    async def test_increments_retry_count(self) -> None:
        result = await retry_prep_node(make_state({"generation_retry_count": 0}))
        assert result["generation_retry_count"] == 1

    async def test_increments_from_existing_value(self) -> None:
        result = await retry_prep_node(make_state({"generation_retry_count": 3}))
        assert result["generation_retry_count"] == 4

    async def test_missing_counter_defaults_to_one(self) -> None:
        state = make_state()
        del state["generation_retry_count"]
        result = await retry_prep_node(state)
        assert result["generation_retry_count"] == 1

    async def test_empty_return_actually_clears_under_reducer(self) -> None:
        """The [] must reset the channel — a no-op here duplicates on retry."""
        prior = [{"type": "explanation"}, {"type": "code"}]
        result = await retry_prep_node(make_state({"generated_resources": prior}))
        assert _accumulate_resources(prior, result["generated_resources"]) == []


class TestRetryLoopTerminates:
    """End-to-end guard: a permanently-failing audit must not loop forever.

    This is the regression test for the router-mutation bug.  The unit test
    ``test_router_does_not_mutate_retry_count`` cannot catch it alone — calling
    the router on a plain dict makes the mutation *appear* to persist.  Only a
    real graph run proves the counter advances between supersteps.
    """

    @staticmethod
    def _graph_with_permanently_failing_audit():
        """Build the real graph whose auditor always fails."""
        from src.agents.models import (
            AssessmentOutput, AuditEntry, AuditorOutput, CoderOutput,
            DesignerOutput, GeneratedResource, GenerationPlan, GuardianOutput,
            KnowledgeUnit, PlannerOutput,
        )
        from src.harness.types import ExecutionReport

        ku = KnowledgeUnit(id="kp-1", name="k", description="d", difficulty=2)

        def rep(name, output):
            return ExecutionReport(
                agent_name=name, success=True, duration_ms=1.0, output=output
            )

        class _Planner:
            async def execute(self, _i):
                return rep("planner", PlannerOutput(
                    plan=GenerationPlan(primary_kp_id="kp-1", knowledge_units=[ku]),
                    summary="s",
                ))

        class _Guardian:
            async def run(self, _u):
                return GuardianOutput(is_valid=True)

        class _Designer:
            async def execute(self, _i):
                return rep("designer", DesignerOutput(resources=[GeneratedResource(
                    type="explanation", title="t", content="c",
                    kp_id="kp-1", difficulty=2,
                )]))

        class _Coder:
            async def execute(self, _i):
                return rep("coder", CoderOutput(
                    code="p", ast_valid=True, execution_success=True,
                ))

        class _Auditor:
            async def execute(self, _i):
                return rep("content_auditor", AuditorOutput(
                    entries=[AuditEntry(resource_index=0, passed=False,
                                        similarity_score=0.1)],
                    all_passed=False,
                ))

        class _Assessment:
            async def execute(self, _i):
                return rep("assessment", AssessmentOutput(confidence=0.8))

            async def run(self, knowledge_unit=None, **_kw):
                # The degraded node calls .run() directly (legacy path).
                return AssessmentOutput(confidence=0.5)

        _graph_ctx.planner = _Planner()
        _graph_ctx.guardian = _Guardian()
        _graph_ctx.designer = _Designer()
        _graph_ctx.coder = _Coder()
        _graph_ctx.content_auditor = _Auditor()
        _graph_ctx.assessment = _Assessment()
        return create_graph()

    async def test_failing_audit_reaches_degraded_within_budget(self) -> None:
        """The loop terminates and lands on "degraded", not a recursion error."""
        from src.agents.orchestrator.state import create_initial_state

        graph = self._graph_with_permanently_failing_audit()
        result = await graph.ainvoke(
            create_initial_state("x", "u", "s", knowledge_point_id="kp-1"),
            config={"recursion_limit": 100},  # would blow up if the loop ran away
        )
        assert result["overall_status"] == "degraded"

    async def test_retry_budget_is_respected(self) -> None:
        """max_retries=2 means exactly 2 retry rounds before degrading."""
        from src.agents.orchestrator.state import create_initial_state

        graph = self._graph_with_permanently_failing_audit()
        result = await graph.ainvoke(
            create_initial_state("x", "u", "s", knowledge_point_id="kp-1"),
            config={"recursion_limit": 100},
        )
        assert result["generation_retry_count"] == 2

    async def test_no_resource_duplication_across_retries(self) -> None:
        """Each retry regenerates cleanly — the accumulator must not stack up."""
        from src.agents.orchestrator.state import create_initial_state

        graph = self._graph_with_permanently_failing_audit()
        result = await graph.ainvoke(
            create_initial_state("x", "u", "s", knowledge_point_id="kp-1"),
            config={"recursion_limit": 100},
        )
        types = [r["type"] for r in result["generated_resources"]]
        # 2 branches x 1 resource, from the final round only.
        assert sorted(types) == ["code", "explanation"], types
