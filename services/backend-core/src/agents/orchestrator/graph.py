"""LangGraph state graph for EduMap multi-agent orchestration.

Pipeline::

    START → planner → guardian ─┬→ designer ─┐
                                └→ coder ────┤
                                             ↓
                                           merge → content_auditor → assessment → END
                                                    │                  │
                                                    └← retry (≤2)──────┘

Designer and coder run as **parallel branches** off guardian: both read the same
``knowledge_units`` and each appends its own resources to ``generated_resources``
before merge joins them.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.assessment.agent import AssessmentAgent
from src.agents.coder.agent import CoderAgent
from src.agents.content_auditor.agent import ContentAuditorAgent
from src.agents.designer.agent import DesignerAgent
from src.agents.guardian.agent import GuardianAgent
from src.agents.models import (
    AssessmentOutput,
    AuditorOutput,
    CoderOutput,
    DesignerOutput,
    GuardianOutput,
    PlannerOutput,
)
from src.agents.orchestrator.state import EduMapState
from src.agents.planner.agent import PlannerAgent

logger = logging.getLogger(__name__)


# ── Graph context (populated at creation time) ──────────────────────────


class _GraphContext:
    """Wiring for agent dependencies — injected once when ``create_graph`` runs."""

    def __init__(self) -> None:
        self.planner: PlannerAgent | None = None
        self.guardian: GuardianAgent | None = None
        self.designer: DesignerAgent | None = None
        self.coder: CoderAgent | None = None
        self.content_auditor: ContentAuditorAgent | None = None
        self.assessment: AssessmentAgent | None = None
        # Durable state store (I-6). When None the graph runs with ephemeral
        # state and progress reporting falls back to the Redis snapshot the
        # router writes — see src/agents/orchestrator/checkpointing.py.
        self.checkpointer: object | None = None


_graph_ctx = _GraphContext()


# ── Node functions ──────────────────────────────────────────────────────


async def planner_node(state: EduMapState) -> dict:
    """EXTRACT phase: Planner Agent (via harness)."""
    logger.info("Planner EXTRACT phase")
    agent = _graph_ctx.planner
    if not agent:
        return _error_state("planner", "PlannerAgent not configured")

    from src.harness.types import AgentInput

    report = await agent.execute(AgentInput(
        task_input=state.get("task_input", ""),
        user_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
    ))

    if not report.success:
        return _error_state("planner", report.error or "Unknown planner error")

    result: PlannerOutput = report.output
    # NOTE: ``_report`` must be NESTED inside the agent's own entry.  It used to
    # be a sibling key of ``planner``, so it was clobbered by whichever agent
    # ran next and could never be read back as ``agent_results.planner._report``
    # (which is what /status consumers — and the trace panel — expect).
    return {
        "current_phase": "EXTRACT",
        "agent_results": {
            "planner": {
                **result.model_dump(),
                "_report": {
                    "duration_ms": report.duration_ms,
                    "retries": report.retries,
                    "memory_context_loaded": report.memory_context_loaded,
                    "tool_calls": report.tool_calls,
                },
            },
        },
        "generation_plan": result.plan.model_dump() if result.plan else None,
        "knowledge_units": [ku.model_dump() for ku in result.plan.knowledge_units],
    }


async def guardian_node(state: EduMapState) -> dict:
    """VALIDATE phase: Guardian Agent."""
    logger.info("Guardian VALIDATE phase")
    agent = _graph_ctx.guardian
    if not agent:
        return _error_state("guardian", "GuardianAgent not configured")

    try:
        kus = state.get("knowledge_units", [])
        # Deserialize from dicts if needed
        from src.agents.models import KnowledgeUnit
        units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]

        result: GuardianOutput = await agent.run(units)

        update: dict[str, Any] = {
            "current_phase": "VALIDATE",
            "agent_results": {"guardian": result.model_dump()},
        }

        if not result.is_valid:
            details = []
            if result.cycle_details:
                details.extend(result.cycle_details)
            if result.dangling_references:
                details.extend(result.dangling_references)
            update["errors"] = state.get("errors", []) + [
                {"agent": "guardian", "error": d, "phase": "VALIDATE"}
                for d in details
            ]
            update["overall_status"] = "failed"

        return update
    except Exception as exc:
        logger.exception("Guardian failed")
        return _error_state("guardian", str(exc))


async def designer_node(state: EduMapState) -> dict:
    """GENERATE phase (designer branch): Designer Agent (via harness)."""
    logger.info("Designer GENERATE phase")
    agent = _graph_ctx.designer
    if not agent:
        return _error_state("designer", "DesignerAgent not configured")

    kus = state.get("knowledge_units", [])
    from src.agents.models import KnowledgeUnit
    units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]
    from src.harness.types import AgentInput

    # Return ONLY this branch's delta.  ``generated_resources`` carries an
    # append/reset reducer (``_accumulate_resources``, see state.py) because
    # designer and coder append to it concurrently, so echoing back the
    # accumulated list would re-append every prior round's resources.
    # ``retry_prep`` clears the channel before each retry, so the reducer
    # rebuilds the list from the two branch deltas alone.
    designer_resources: list[dict] = []
    designer_reports = []

    for ku in units:
        report = await agent.execute(AgentInput(
            task_input=f"Generate content for {ku.name}",
            user_id=state.get("user_id", ""),
            session_id=state.get("session_id", ""),
            extra={"knowledge_unit": ku.model_dump(), "content_types": ["explanation", "exercise", "visualization"]},
        ))
        designer_reports.append({
            "kp": ku.id,
            "duration_ms": report.duration_ms,
            "retries": report.retries,
            "success": report.success,
        })
        if report.success and report.output:
            result: DesignerOutput = report.output
            designer_resources.extend(r.model_dump() for r in result.resources)

    return {
        "current_phase": "GENERATE",
        "generated_resources": designer_resources,
        "agent_results": {
            "designer": {"resources_count": len(designer_resources), "_reports": designer_reports},
        },
    }


async def coder_node(state: EduMapState) -> dict:
    """GENERATE phase (coder branch): Coder Agent (via harness)."""
    logger.info("Coder GENERATE phase")
    agent = _graph_ctx.coder
    if not agent:
        return _error_state("coder", "CoderAgent not configured")

    kus = state.get("knowledge_units", [])
    from src.agents.models import KnowledgeUnit
    units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]
    from src.harness.types import AgentInput

    # Return ONLY this branch's delta — see the matching note in designer_node.
    coder_resources: list[dict] = []
    coder_reports: list[dict] = []

    for ku in units:
        report = await agent.execute(AgentInput(
            task_input=f"Generate code for {ku.name}",
            user_id=state.get("user_id", ""),
            session_id=state.get("session_id", ""),
            extra={"knowledge_unit": ku.model_dump(), "language": "python"},
        ))
        coder_reports.append({
            "kp": ku.id,
            "duration_ms": report.duration_ms,
            "retries": report.retries,
            "success": report.success,
        })

        if report.success and report.output:
            result: CoderOutput = report.output
            if result.code:
                from src.agents.models import GeneratedResource
                res = GeneratedResource(
                    type="code",
                    title=f"{ku.name} — 代码示例",
                    content=result.code,
                    kp_id=ku.id,
                    difficulty=ku.difficulty,
                    metadata={"execution_success": result.execution_success},
                )
                coder_resources.append(res.model_dump())

                if not result.execution_success:
                    logger.warning(
                        "Coder output for %s: execution_success=False (AST: %s)",
                        ku.name, result.ast_valid,
                    )

    agent_update: dict[str, Any] = {
        "_reports": coder_reports,
        "count": len(coder_reports),
    }

    return {
        "current_phase": "GENERATE",
        "generated_resources": coder_resources,
        "agent_results": {"coder": agent_update},
    }


async def merge_node(state: EduMapState) -> dict:
    """Collect generated resources from designer + coder."""
    logger.info("Merge GENERATE outputs")
    return {
        "current_phase": "GENERATE",
        "agent_results": {
            **state.get("agent_results", {}),
            "merge": {"status": "completed"},
        },
    }


async def content_auditor_node(state: EduMapState) -> dict:
    """REVIEW phase: Content Auditor Agent (via harness)."""
    logger.info("Content Auditor REVIEW phase")
    agent = _graph_ctx.content_auditor
    if not agent:
        return _error_state("content_auditor", "ContentAuditorAgent not configured")

    resources = state.get("generated_resources", [])
    kus = state.get("knowledge_units", [])
    from src.agents.models import GeneratedResource, KnowledgeUnit

    parsed_resources = [
        GeneratedResource(**r) if isinstance(r, dict) else r
        for r in resources
    ]
    units = [
        KnowledgeUnit(**ku) if isinstance(ku, dict) else ku
        for ku in kus
    ]
    primary_kp = units[0] if units else KnowledgeUnit(id="", name="", description="", difficulty=1)
    from src.harness.types import AgentInput

    report = await agent.execute(AgentInput(
        task_input=f"Audit content for {primary_kp.name}",
        user_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
        extra={
            "resources": [r.model_dump() if hasattr(r, 'model_dump') else r for r in parsed_resources],
            "knowledge_unit": primary_kp.model_dump(),
        },
    ))

    if not report.success:
        return _error_state("content_auditor", report.error or "Auditor failed")

    result: AuditorOutput = report.output
    return {
        "current_phase": "REVIEW",
        "audit_results": [e.model_dump() for e in result.entries],
        "agent_results": {
            **state.get("agent_results", {}),
            "content_auditor": {"all_passed": result.all_passed, "_report": {
                "duration_ms": report.duration_ms, "retries": report.retries,
            }},
        },
    }


async def assessment_node(state: EduMapState) -> dict:
    """ASSESS phase: Assessment Agent (via harness)."""
    logger.info("Assessment ASSESS phase")
    agent = _graph_ctx.assessment
    if not agent:
        return _error_state("assessment", "AssessmentAgent not configured")

    kus = state.get("knowledge_units", [])
    from src.agents.models import KnowledgeUnit
    units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]
    primary_kp = units[0] if units else KnowledgeUnit(id="", name="", description="", difficulty=1)

    from src.harness.types import AgentInput

    report = await agent.execute(AgentInput(
        task_input=f"Generate quiz for {primary_kp.name}",
        user_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
        extra={"knowledge_unit": primary_kp.model_dump()},
    ))

    if not report.success:
        return _error_state("assessment", report.error or "Unknown assessment error")

    result: AssessmentOutput = report.output

    # Mark the run completed unless an earlier stage already set a terminal
    # status.  The live value here is "running" (see create_initial_state), not
    # "processing" — that string is an HTTP response field in the /generate
    # endpoint and never appears in graph state, so whitelisting it meant the
    # success path kept the initial "running" and every consumer (SSE
    # workflow_complete, GET /status) reported a finished run as in-progress.
    # Only genuinely terminal statuses are preserved; anything transient
    # (running / empty / unknown) resolves to "completed".
    prior_status = state.get("overall_status", "")
    new_status = (
        prior_status
        if prior_status in ("failed", "degraded", "cancelled")
        else "completed"
    )
    return {
        "current_phase": "ASSESS",
        "assessment_result": result.model_dump(),
        "overall_status": new_status,
        "agent_results": {
            **state.get("agent_results", {}),
            "assessment": {"confidence": result.confidence, "_report": {
                "duration_ms": report.duration_ms,
                "retries": report.retries,
            }},
        },
    }


# ── Conditional routing ─────────────────────────────────────────────────


def route_after_guardian(state: EduMapState) -> str:
    """After guardian: if valid, fan out to BOTH generation branches.

    LangGraph sends every value returned from a conditional-edge router along
    the matching edge, so returning both node names here is what makes the
    designer and coder branches run concurrently off the same validated plan.
    """
    if state.get("overall_status") == "failed":
        return END
    return ["designer", "coder"]


def route_after_failure(state: EduMapState) -> str:
    """Abort the pipeline if an agent has already failed."""
    if state.get("overall_status") == "failed":
        return END
    # Continue to next node (edge mapping determines actual target per node)
    return "merge"


def route_after_generation(state: EduMapState) -> str:
    """Fan-in for the parallel generation branches — routes to ``merge``.

    Designer and coder share ``route_after_failure``'s failure check; neither
    has a branch-specific target of its own any more, since both feed merge.

    Note: with two concurrent branches a failure in one does *not* stop the
    other — LangGraph runs both to completion.  The merge barrier joins them
    only after both finish, and the failed branch still sets
    ``overall_status="failed"``, so review/assessment are skipped.  The cost is
    that the sibling branch's LLM calls are already spent by then.
    """
    return route_after_failure(state)


def route_after_review(state: EduMapState) -> str:
    """After content auditor: retry, proceed to assessment, or degrade.

    The retry counter is incremented by the **``retry_prep`` node**, not here.
    A conditional-edge router's return value is only the edge label — LangGraph
    commits state from node returns, so mutating ``state[...]`` in this function
    is discarded between supersteps.  Incrementing here (the previous design)
    meant the counter reset to 0 on every pass and a permanently-failing audit
    looped until the graph's recursion limit / the caller's timeout, re-running
    both expensive generation branches each round.

    Routing only *reads* it now: 0 retries so far → retry; already at the cap
    (``max_retries``) → degrade.
    """
    audit_results = state.get("audit_results", [])
    failed = [a for a in audit_results if not a.get("passed", True)]

    if not failed:
        return "assessment"

    retries = state.get("generation_retry_count", 0)
    max_retries = state.get("max_retries", 2)
    logger.info(
        "Content Auditor: %d/%d resources failed, retry %d/%d",
        len(failed), len(audit_results), retries, max_retries,
    )
    # ``retries`` counts completed retry rounds, so a value of 0 means the
    # original generation attempt just failed and one retry is still allowed.
    if retries < max_retries:
        return "retry_generate"

    # Degraded: still proceed to assessment but mark overall as degraded
    return "assess_degraded"


async def assessment_degraded_node(state: EduMapState) -> dict:
    """Run assessment on degraded path and mark overall as degraded."""
    logger.info("Assessment ASSESS phase (degraded)")
    agent = _graph_ctx.assessment
    if not agent:
        return _error_state("assessment", "AssessmentAgent not configured")

    try:
        kus = state.get("knowledge_units", [])
        from src.agents.models import KnowledgeUnit
        units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]

        primary_kp = units[0] if units else KnowledgeUnit(id="", name="", description="", difficulty=1)
        result: AssessmentOutput = await agent.run(knowledge_unit=primary_kp)

        # Same whitelist fix as assessment_node: "running" (the live value) is
        # not terminal, so it must resolve to "degraded" rather than being
        # preserved and reported as still in-progress.
        prior_status = state.get("overall_status", "")
        new_status = (
            prior_status
            if prior_status in ("failed", "degraded", "cancelled")
            else "degraded"
        )
        return {
            "current_phase": "ASSESS",
            "assessment_result": result.model_dump(),
            "overall_status": new_status,
            "agent_results": {
                **state.get("agent_results", {}),
                "assessment": {"confidence": result.confidence},
            },
        }
    except Exception as exc:
        logger.exception("Assessment failed (degraded)")
        return _error_state("assessment", str(exc))


# ── Retry prep node ─────────────────────────────────────────────────────


async def retry_prep_node(state: EduMapState) -> dict:
    """Clear generated resources before retry to avoid duplication.

    Also owns the retry counter.  It has to live in a **node**, because only
    node return values are committed to state — a router mutating ``state[...]``
    has that write discarded between supersteps (see ``route_after_review``).

    The ``generated_resources`` reset only works because that channel's reducer
    treats an empty list as "reset" (see ``_accumulate_resources``): under a
    plain concatenating reducer an empty list would be a no-op, so retries would
    accumulate duplicate resources instead of regenerating cleanly.
    """
    retries = state.get("generation_retry_count", 0) + 1
    logger.info(
        "Retry prep: clearing generated_resources (was %d items), retry %d/%d",
        len(state.get("generated_resources", [])),
        retries, state.get("max_retries", 2),
    )
    return {
        "generated_resources": [],
        "generation_retry_count": retries,
    }


# ── Graph builder ───────────────────────────────────────────────────────


def configure_graph(
    *,
    planner: PlannerAgent,
    guardian: GuardianAgent,
    designer: DesignerAgent,
    coder: CoderAgent,
    content_auditor: ContentAuditorAgent,
    assessment: AssessmentAgent,
    checkpointer: object | None = None,
) -> None:
    """Inject agent instances into the graph context.

    Call this during app startup **once** before ``create_graph()``.

    Args:
        checkpointer: optional durable state store (an
            ``AsyncPostgresSaver`` from
            :class:`src.agents.orchestrator.checkpointing.CheckpointerHolder`).
            When provided, ``create_graph`` compiles the graph with it, so
            LangGraph persists every super-step and applies the declared
            reducers itself. When omitted, state is ephemeral.
    """
    _graph_ctx.planner = planner
    _graph_ctx.guardian = guardian
    _graph_ctx.designer = designer
    _graph_ctx.coder = coder
    _graph_ctx.content_auditor = content_auditor
    _graph_ctx.assessment = assessment
    _graph_ctx.checkpointer = checkpointer


def graph_has_checkpointer() -> bool:
    """Whether the graph will be compiled with a durable checkpointer.

    Callers must know this because the two cases have different invocation
    contracts: a checkpointed graph **requires** a ``thread_id`` in its config,
    while an ephemeral one must be invoked **without** one. Passing a thread
    config to a graph that has no checkpointer is not merely redundant — it
    changed runtime behaviour in testing (the astream call stalled), so the
    decision is made from this single source of truth rather than guessed at
    each call site.
    """
    return _graph_ctx.checkpointer is not None


def create_graph() -> StateGraph:
    """Build and return the compiled LangGraph state graph.

    The graph must be configured via ``configure_graph()`` before first use.

    When a checkpointer was supplied, the graph is compiled with it: LangGraph
    then persists a checkpoint at every super-step and owns reducer
    application, so state survives a restart and can be inspected/travelled
    afterwards. Without one, behaviour is unchanged (ephemeral state).

    Callers that use a checkpointer **must** pass a ``config`` carrying a
    ``thread_id`` on every invocation — ``thread_config(session_id)`` builds
    one. Invoking without it raises, which is LangGraph's own contract and is
    left to surface rather than being papered over with a generated id: a
    silently-invented thread would give each call its own state and defeat the
    purpose.
    """
    builder = StateGraph(EduMapState)

    # Register nodes
    builder.add_node("planner", planner_node)
    builder.add_node("guardian", guardian_node)
    builder.add_node("designer", designer_node)
    builder.add_node("coder", coder_node)
    builder.add_node("merge", merge_node)
    builder.add_node("content_auditor", content_auditor_node)
    builder.add_node("assessment", assessment_node)
    builder.add_node("assess_degraded", assessment_degraded_node)
    builder.add_node("retry_prep", retry_prep_node)

    # Define edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "guardian")
    # Guardian fans out to BOTH generation branches — they run in parallel.
    builder.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {END: END, "designer": "designer", "coder": "coder"},
    )
    # Both branches fan back in at merge.  A shared router keeps the abort
    # boundary identical on each branch: on failure this returns END, so the
    # merge barrier is never entered with a poisoned state.
    builder.add_conditional_edges(
        "designer",
        route_after_generation,
        {END: END, "merge": "merge"},
    )
    builder.add_conditional_edges(
        "coder",
        route_after_generation,
        {END: END, "merge": "merge"},
    )
    builder.add_edge("merge", "content_auditor")
    builder.add_conditional_edges(
        "content_auditor",
        route_after_review,
        {
            "assessment": "assessment",
            "retry_generate": "retry_prep",
            "assess_degraded": "assess_degraded",
        },
    )
    # retry_prep re-enters BOTH generation branches, mirroring the initial fan-out.
    builder.add_edge("retry_prep", "designer")
    builder.add_edge("retry_prep", "coder")
    builder.add_edge("assessment", END)
    builder.add_edge("assess_degraded", END)

    checkpointer = _graph_ctx.checkpointer
    if checkpointer is not None:
        logger.info("Compiling orchestrator graph with a durable checkpointer")
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


# ── Shared helpers ──────────────────────────────────────────────────────


def _error_state(agent: str, message: str, existing_errors: list | None = None) -> dict:
    """Return a partial state update for an agent failure.

    Appends to existing errors so no prior failure history is lost.
    """
    logger.error("Agent '%s' error: %s", agent, message)
    return {
        "errors": (existing_errors or []) + [{"agent": agent, "error": message, "phase": "UNKNOWN"}],
        "overall_status": "failed",
    }
