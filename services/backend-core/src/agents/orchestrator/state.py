"""LangGraph state type for EduMap orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class EduMapState(TypedDict, total=False):
    """LangGraph state type for EduMap orchestration.

    Existing (Phase 2) fields:
        task_input, task_type, current_phase, agent_results,
        errors, audit_log, user_id, session_id, knowledge_point_id

    Phase 3 additions:
        generation_plan, knowledge_units, generated_resources,
        audit_results, assessment_result, generation_retry_count,
        max_retries, llm_config, overall_status
    """

    # ── Core task fields ────────────────────────────────────────────────
    task_input: str
    task_type: str
    current_phase: str  # EXTRACT | VALIDATE | GENERATE | REVIEW | ASSESS
    overall_status: str  # running | completed | failed | degraded

    # ── Identity ────────────────────────────────────────────────────────
    user_id: str
    session_id: str
    knowledge_point_id: Optional[str]

    # ── Agent results ───────────────────────────────────────────────────
    agent_results: Dict[str, Any]

    # ── Phase 3: Plan ───────────────────────────────────────────────────
    generation_plan: Optional[dict]  # serialized GenerationPlan
    knowledge_units: list  # serialized KnowledgeUnit list

    # ── Phase 3: Generated content ──────────────────────────────────────
    generated_resources: list  # serialized GeneratedResource list

    # ── Phase 3: Audit ──────────────────────────────────────────────────
    audit_results: list  # serialized AuditEntry list
    audit_log: List[Dict[str, Any]]  # retained from Phase 2

    # ── Phase 3: Assessment ─────────────────────────────────────────────
    assessment_result: Optional[dict]  # serialized AssessmentOutput

    # ── Phase 3: Retry ──────────────────────────────────────────────────
    generation_retry_count: int
    max_retries: int

    # ── Phase 3: LLM config ─────────────────────────────────────────────
    llm_config: Optional[dict]  # model, temperature, etc.

    # ── Errors ──────────────────────────────────────────────────────────
    errors: List[Dict[str, str]]


def create_initial_state(
    task_input: str,
    user_id: str,
    session_id: str,
    task_type: str = "generate",
    knowledge_point_id: Optional[str] = None,
    max_retries: int = 2,
) -> dict:
    """Return a fresh EduMapState with defaults."""
    return {
        "task_input": task_input,
        "task_type": task_type,
        "current_phase": "EXTRACT",
        "overall_status": "running",
        "user_id": user_id,
        "session_id": session_id,
        "knowledge_point_id": knowledge_point_id,
        "agent_results": {},
        "generation_plan": None,
        "knowledge_units": [],
        "generated_resources": [],
        "audit_results": [],
        "audit_log": [],
        "assessment_result": None,
        "generation_retry_count": 0,
        "max_retries": max_retries,
        "llm_config": None,
        "errors": [],
    }
