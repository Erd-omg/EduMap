"""LangGraph state type for EduMap orchestration."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_agent_results(left: Dict[str, Any] | None, right: Dict[str, Any] | None) -> Dict[str, Any]:
    """Reducer for ``agent_results``: shallow-merge the two concurrent writes.

    The designer and coder branches run in the same superstep and each returns
    ``agent_results`` containing only its own key.  Without a reducer LangGraph
    raises ``InvalidUpdateError`` ("can receive only one value per step") — the
    reducer is not optional, it is required for the fan-out to be legal.

    Shallow merge is correct here because the two branches write disjoint
    top-level keys (``designer`` / ``coder``); a shared key would be
    last-write-wins, which no current node relies on.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def _accumulate_resources(left: list | None, right: list | None) -> list:
    """Reducer for ``generated_resources``.

    Designer and coder append concurrently, so their writes must combine rather
    than collide.  Two distinct intents share this channel, which is why it is
    not plain ``operator.add``:

    - **append** — a branch reporting the resources it just produced (non-empty
      ``right``) concatenates onto what is already there.
    - **reset** — ``retry_prep_node`` clears the channel before a regeneration
      round by returning ``[]``.  Under ``operator.add`` an empty list is a
      no-op, so retries would accumulate duplicates; treating empty as "reset"
      is what makes the retry loop idempotent.

    The trade-off: a node cannot report "produced nothing" without also erasing
    the sibling's contribution.  That is acceptable only because the sole empty
    writer is ``retry_prep`` — which runs alone in its superstep (content
    auditor → retry_prep → both branches), never concurrently with a branch.
    Any future node that returns ``[]`` here would silently wipe the channel.
    """
    if not right:
        return []
    return list(left or []) + list(right)


class EduMapState(TypedDict, total=False):
    """LangGraph state type for EduMap orchestration.

    Existing (Phase 2) fields:
        task_input, task_type, current_phase, agent_results,
        errors, audit_log, user_id, session_id, knowledge_point_id

    Phase 3 additions:
        generation_plan, knowledge_units, generated_resources,
        audit_results, assessment_result, generation_retry_count,
        max_retries, llm_config, overall_status

    Concurrency note: the planner→guardian stage is linear, but generation
    fans out into parallel ``designer`` and ``coder`` branches that run in the
    same superstep.  Every field BOTH branches write must carry a reducer, or
    LangGraph rejects the concurrent update.  The annotated fields below are
    exactly that set:

    - ``current_phase`` — written by both; both write the same literal
      ("GENERATE"), so last-wins is harmless.
    - ``generated_resources`` — append/reset reducer, see
      :func:`_accumulate_resources`.
    - ``agent_results`` — each branch contributes its own key; shallow merge.
    """

    # ── Core task fields ────────────────────────────────────────────────
    task_input: str
    task_type: str
    # Both generation branches write "GENERATE" concurrently.
    current_phase: Annotated[str, lambda _l, r: r]
    overall_status: str  # running | completed | failed | degraded | cancelled

    # ── Identity ────────────────────────────────────────────────────────
    user_id: str
    session_id: str
    knowledge_point_id: Optional[str]

    # ── Agent results ───────────────────────────────────────────────────
    # Merged across concurrent designer/coder writes.
    agent_results: Annotated[Dict[str, Any], _merge_agent_results]

    # ── Phase 3: Plan ───────────────────────────────────────────────────
    generation_plan: Optional[dict]  # serialized GenerationPlan
    knowledge_units: list  # serialized KnowledgeUnit list

    # ── Phase 3: Generated content ──────────────────────────────────────
    # Both branches append; ``retry_prep`` resets it by returning [].
    generated_resources: Annotated[list, _accumulate_resources]

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

    # ── Memory & Tools (Phase 6+) ───────────────────────────────
    memory_context: Optional[dict]  # loaded from memory system
    tool_results: List[dict]  # results from tool calls

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
