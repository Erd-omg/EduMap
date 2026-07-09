"""LangGraph state graph for EduMap multi-agent orchestration.

Pipeline::

    START → planner → guardian → [designer, coder] → merge
        → content_auditor → assessment → END
              │                  │
              └← retry (≤2)──────┘
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

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
    """EXTRACT phase: Planner Agent."""
    logger.info("Planner EXTRACT phase")
    agent = _graph_ctx.planner
    if not agent:
        return _error_state("planner", "PlannerAgent not configured")

    try:
        result: PlannerOutput = await agent.run(
            task_input=state.get("task_input", ""),
            user_id=state.get("user_id", ""),
        )
        return {
            "current_phase": "EXTRACT",
            "agent_results": {"planner": result.model_dump()},
            "generation_plan": result.plan.model_dump() if result.plan else None,
            "knowledge_units": [ku.model_dump() for ku in result.plan.knowledge_units],
        }
    except Exception as exc:
        logger.exception("Planner failed")
        return _error_state("planner", str(exc))


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
    """GENERATE phase (designer branch): Designer Agent."""
    logger.info("Designer GENERATE phase")
    agent = _graph_ctx.designer
    if not agent:
        return _error_state("designer", "DesignerAgent not configured")

    try:
        kus = state.get("knowledge_units", [])
        from src.agents.models import KnowledgeUnit
        units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]

        all_resources: list[dict] = list(state.get("generated_resources", []))
        for ku in units:
            result: DesignerOutput = await agent.run(
                knowledge_unit=ku,
                content_types=["explanation", "exercise", "visualization"],
            )
            all_resources.extend(r.model_dump() for r in result.resources)

        return {
            "current_phase": "GENERATE",
            "generated_resources": all_resources,
            "agent_results": {
                **state.get("agent_results", {}),
                "designer": {"resources_count": len(all_resources)},
            },
        }
    except Exception as exc:
        logger.exception("Designer failed")
        return _error_state("designer", str(exc))


async def coder_node(state: EduMapState) -> dict:
    """GENERATE phase (coder branch): Coder Agent."""
    logger.info("Coder GENERATE phase")
    agent = _graph_ctx.coder
    if not agent:
        return _error_state("coder", "CoderAgent not configured")

    try:
        kus = state.get("knowledge_units", [])
        from src.agents.models import KnowledgeUnit
        units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]

        all_resources: list[dict] = list(state.get("generated_resources", []))
        coder_results: list[dict] = []

        for ku in units:
            result: CoderOutput = await agent.run(knowledge_unit=ku)
            coder_results.append(result.model_dump())
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

        return {
            "current_phase": "GENERATE",
            "generated_resources": all_resources,
            "agent_results": {
                **state.get("agent_results", {}),
                "coder": {"results": coder_results},
            },
        }
    except Exception as exc:
        logger.exception("Coder failed")
        return _error_state("coder", str(exc))


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
    """REVIEW phase: Content Auditor Agent."""
    logger.info("Content Auditor REVIEW phase")
    agent = _graph_ctx.content_auditor
    if not agent:
        return _error_state("content_auditor", "ContentAuditorAgent not configured")

    try:
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

        # Audit all resources against the primary KP
        primary_kp = units[0] if units else KnowledgeUnit(id="", name="", description="", difficulty=1)
        result: AuditorOutput = await agent.run(parsed_resources, primary_kp)

        return {
            "current_phase": "REVIEW",
            "audit_results": [e.model_dump() for e in result.entries],
            "agent_results": {
                **state.get("agent_results", {}),
                "content_auditor": {"all_passed": result.all_passed},
            },
        }
    except Exception as exc:
        logger.exception("Content Auditor failed")
        return _error_state("content_auditor", str(exc))


async def assessment_node(state: EduMapState) -> dict:
    """ASSESS phase: Assessment Agent."""
    logger.info("Assessment ASSESS phase")
    agent = _graph_ctx.assessment
    if not agent:
        return _error_state("assessment", "AssessmentAgent not configured")

    try:
        kus = state.get("knowledge_units", [])
        from src.agents.models import KnowledgeUnit
        units = [KnowledgeUnit(**ku) if isinstance(ku, dict) else ku for ku in kus]

        primary_kp = units[0] if units else KnowledgeUnit(id="", name="", description="", difficulty=1)
        result: AssessmentOutput = await agent.run(knowledge_unit=primary_kp)

        return {
            "current_phase": "ASSESS",
            "assessment_result": result.model_dump(),
            "overall_status": "completed",
            "agent_results": {
                **state.get("agent_results", {}),
                "assessment": {"confidence": result.confidence},
            },
        }
    except Exception as exc:
        logger.exception("Assessment failed")
        return _error_state("assessment", str(exc))


# ── Conditional routing ─────────────────────────────────────────────────


def route_after_guardian(state: EduMapState) -> str:
    """After guardian: if valid, proceed to generation; otherwise END."""
    if state.get("overall_status") == "failed":
        return END
    return "designer"  # designer_node runs first, then fan-out to coder


def route_after_review(state: EduMapState) -> str:
    """After content auditor: retry, proceed to assessment, or degrade."""
    audit_results = state.get("audit_results", [])
    failed = [a for a in audit_results if not a.get("passed", True)]

    if not failed:
        return "assessment"

    retries = state.get("generation_retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retries < max_retries:
        return "retry_generate"

    # Degraded: still proceed to assessment but mark overall as degraded
    return "assessment_degraded"


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

    # Define edges
    builder.set_entry_point("planner")
    builder.add_edge("planner", "guardian")
    builder.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {END: END, "designer": "designer"},
    )
    # Fan-out: designer → merge, coder → merge
    builder.add_edge("designer", "coder")
    builder.add_edge("coder", "merge")
    builder.add_edge("merge", "content_auditor")
    builder.add_conditional_edges(
        "content_auditor",
        route_after_review,
        {
            "assessment": "assessment",
            "retry_generate": "designer",
            "assessment_degraded": "assessment",
        },
    )

    # Retry needs to be handled in assessment_degraded path: mark degraded
    async def _assessment_degraded(state: EduMapState) -> dict:
        return {"overall_status": "degraded"}

    # Add a node for the degraded path
    builder.add_node("assessment_enter", assessment_node)
    builder.add_node("assess_degraded", _assessment_degraded)
    builder.add_edge("assessment", END)
    builder.add_edge("assess_degraded", END)

    return builder.compile()


# ── Shared helpers ──────────────────────────────────────────────────────


def _error_state(agent: str, message: str) -> dict:
    """Return a partial state update for an agent failure."""
    logger.error("Agent '%s' error: %s", agent, message)
    return {
        "errors": [{"agent": agent, "error": message, "phase": "UNKNOWN"}],
        "overall_status": "failed",
    }
