"""LangGraph state graph for EduMap multi-agent orchestration.

Pipeline::

    START → planner → guardian → designer ─┐
                             └→ coder ─────┤
                                            ↓
                                          merge → content_auditor → assessment → END
                                                   │                  │
                                                   └← retry (≤2)──────┘
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
    return {
        "current_phase": "EXTRACT",
        "agent_results": {"planner": result.model_dump(), "_report": {
            "duration_ms": report.duration_ms,
            "retries": report.retries,
            "memory_context_loaded": report.memory_context_loaded,
        }},
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

    all_resources: list[dict] = list(state.get("generated_resources", []))
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
            all_resources.extend(r.model_dump() for r in result.resources)

    return {
        "current_phase": "GENERATE",
        "generated_resources": all_resources,
        "agent_results": {
            **state.get("agent_results", {}),
            "designer": {"resources_count": len(all_resources), "_reports": designer_reports},
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

    all_resources: list[dict] = list(state.get("generated_resources", []))
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
                all_resources.append(res.model_dump())

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
        "generated_resources": all_resources,
        "agent_results": {
            **state.get("agent_results", {}),
            "coder": agent_update,
        },
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

    # Preserve existing failure/degraded status
    prior_status = state.get("overall_status", "")
    new_status = "completed" if prior_status in ("", "processing") else prior_status
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
    """After guardian: if valid, proceed to generation; otherwise END."""
    if state.get("overall_status") == "failed":
        return END
    return "designer"


def route_after_failure(state: EduMapState) -> str:
    """Abort the pipeline if an agent has already failed."""
    if state.get("overall_status") == "failed":
        return END
    # Continue to next node (edge mapping determines actual target per node)
    return "merge"


def route_after_designer(state: EduMapState) -> str:
    """After designer: route to coder on success, or abort on failure."""
    if state.get("overall_status") == "failed":
        return END
    return "coder"


def route_after_review(state: EduMapState) -> str:
    """After content auditor: retry, proceed to assessment, or degrade."""
    audit_results = state.get("audit_results", [])
    failed = [a for a in audit_results if not a.get("passed", True)]

    if not failed:
        return "assessment"

    # Critical: increment retry counter before routing to retry
    state["generation_retry_count"] = state.get("generation_retry_count", 0) + 1
    retries = state["generation_retry_count"]
    max_retries = state.get("max_retries", 2)
    logger.info(
        "Content Auditor: %d/%d resources failed, retry %d/%d",
        len(failed), len(audit_results), retries, max_retries,
    )
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

        # Preserve existing failure status (don't overwrite with "degraded")
        prior_status = state.get("overall_status", "")
        new_status = "degraded" if prior_status in ("", "processing") else prior_status
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
    """Clear generated resources before retry to avoid duplication."""
    logger.info(
        "Retry prep: clearing generated_resources (was %d items)",
        len(state.get("generated_resources", [])),
    )
    return {"generated_resources": []}


# ── Graph builder ───────────────────────────────────────────────────────


def configure_graph(
    *,
    planner: PlannerAgent,
    guardian: GuardianAgent,
    designer: DesignerAgent,
    coder: CoderAgent,
    content_auditor: ContentAuditorAgent,
    assessment: AssessmentAgent,
) -> None:
    """Inject agent instances into the graph context.

    Call this during app startup **once** before ``create_graph()``.
    """
    _graph_ctx.planner = planner
    _graph_ctx.guardian = guardian
    _graph_ctx.designer = designer
    _graph_ctx.coder = coder
    _graph_ctx.content_auditor = content_auditor
    _graph_ctx.assessment = assessment


def create_graph() -> StateGraph:
    """Build and return the compiled LangGraph state graph.

    The graph must be configured via ``configure_graph()`` before first use.
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
    builder.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {END: END, "designer": "designer"},
    )
    # If designer or coder fails, abort the pipeline
    builder.add_conditional_edges(
        "designer",
        route_after_designer,
        {END: END, "coder": "coder"},
    )
    builder.add_conditional_edges(
        "coder",
        route_after_failure,
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
    builder.add_edge("retry_prep", "designer")
    builder.add_edge("assessment", END)
    builder.add_edge("assess_degraded", END)

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
