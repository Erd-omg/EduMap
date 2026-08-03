"""Orchestrator API router — endpoints for triggering and monitoring
multi-agent generation workflows.

Endpoints::

    POST   /api/v1/orchestrator/generate   — start a new generation workflow
    GET    /api/v1/orchestrator/status/{session_id} — poll progress
    GET    /api/v1/orchestrator/stream/{session_id}  — SSE progress stream
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.orchestrator.graph import create_graph
from src.agents.orchestrator.state import EduMapState, create_initial_state
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

# In-memory SSE event queues (process-local by nature — cannot be serialized to Redis)
_sse_queues: dict[str, asyncio.Queue] = {}


def _get_short_term(request: Request) -> ShortTermMemory | None:
    """Get the ShortTermMemory from app state."""
    memory_ops = getattr(request.app.state, "memory_ops", None)
    if memory_ops:
        return memory_ops.short_term
    return None


# ── Request / Response models ─────────────────────────────────────────


class GenerateRequest(BaseModel):
    task_input: str
    task_type: str = "generate"
    user_id: str = "anonymous"
    session_id: str | None = None
    knowledge_point_id: str | None = None


class GenerateResponse(BaseModel):
    session_id: str
    status: str  # processing | completed | failed


# ── Endpoints ──────────────────────────────────────────────────────────


@router.post("/generate", response_model=GenerateResponse)
async def start_generation(body: GenerateRequest, request: Request):
    """Start a multi-agent generation workflow.

    The endpoint returns immediately with a ``session_id``.  The frontend
    can then poll ``GET /status/{session_id}`` or connect to the SSE
    stream at ``GET /stream/{session_id}``.
    """
    session_id = body.session_id or str(uuid.uuid4())

    state = create_initial_state(
        task_input=body.task_input,
        user_id=body.user_id,
        session_id=session_id,
        task_type=body.task_type,
        knowledge_point_id=body.knowledge_point_id,
    )

    # Store session state in ShortTermMemory (Redis or in-memory fallback)
    short_term = _get_short_term(request)
    if short_term:
        await short_term.create_session(
            session_id=session_id,
            user_id=body.user_id,
            metadata={
                "orchestrator_state": state,
                "orchestrator_result": None,
            },
        )
    else:
        # Fallback: store in local dict (should not happen — ShortTermMemory has _in_memory_fallback)
        logger.warning("ShortTermMemory not available — falling back to local storage")

    # SSE events queue stays in-memory (process-local by nature)
    _sse_queues[session_id] = asyncio.Queue()

    # Fire generation in the background
    asyncio.create_task(_run_generation(session_id, request))

    return GenerateResponse(session_id=session_id, status="processing")


@router.get("/status/{session_id}")
async def get_status(session_id: str, request: Request):
    """Return the current state snapshot for a generation session."""
    # Load from ShortTermMemory (or fallback)
    short_term = _get_short_term(request)
    state_dict = None

    if short_term:
        session_mem = await short_term.get_session(session_id)
        if session_mem:
            state_dict = session_mem.metadata.get("orchestrator_state")

    if not state_dict:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "current_phase": state_dict.get("current_phase", "UNKNOWN"),
        "overall_status": state_dict.get("overall_status", "unknown"),
        "knowledge_point_id": state_dict.get("knowledge_point_id"),
        "task_input": state_dict.get("task_input", ""),
        "errors": state_dict.get("errors", []),
        "agent_results": {
            k: v for k, v in state_dict.get("agent_results", {}).items()
            if not isinstance(v, dict) or "error" not in v
        },
        "generated_resources": [
            {"type": r.get("type"), "title": r.get("title"), "kp_id": r.get("kp_id")}
            for r in state_dict.get("generated_resources", [])
        ],
        "assessment": state_dict.get("assessment_result"),
    }


@router.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """SSE endpoint — streams agent progress events as they happen.

    Events::

        event: phase_change
        data: {"phase": "VALIDATE", "agent": "guardian"}

        event: agent_complete
        data: {"agent": "planner", "result": ...}

        event: workflow_complete
        data: {"status": "completed"}

        event: workflow_error
        data: {"agent": "planner", "error": "..."}
    """
    queue: asyncio.Queue | None = _sse_queues.get(session_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=300)
                if event.get("type") == "workflow_complete":
                    yield f"event: workflow_complete\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    break
                if event.get("type") == "workflow_error":
                    yield f"event: workflow_error\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    break
                yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event.get('data', event), ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield f"event: heartbeat\ndata: {json.dumps({'type': 'heartbeat', 'reason': 'timeout', 'timestamp': time.time()})}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Background runner ──────────────────────────────────────────────────


async def _run_generation(session_id: str, request: Request) -> None:
    """Execute the LangGraph state graph and stream events."""
    short_term = _get_short_term(request)

    # Load state from ShortTermMemory
    state: dict = {}
    if short_term:
        session_mem = await short_term.get_session(session_id)
        if session_mem:
            state = session_mem.metadata.get("orchestrator_state", {})
    if not state:
        logger.error("Session %s not found in ShortTermMemory — aborting", session_id)
        return

    queue: asyncio.Queue = _sse_queues.get(session_id)
    if not queue:
        logger.error("SSE queue for session %s not found — aborting", session_id)
        return

    try:
        graph = _get_graph()

        # Push initial phase events for each agent
        phase_labels = {
            "planner": "EXTRACT",
            "guardian": "VALIDATE",
            "designer": "GENERATE",
            "coder": "GENERATE",
            "content_auditor": "REVIEW",
            "assessment": "ASSESS",
            "assess_degraded": "ASSESS",
        }
        for agent_name, phase in phase_labels.items():
            await queue.put({
                "type": "phase_change",
                "data": {"phase": phase, "agent": agent_name},
            })

        # Run graph via astream — yields state after each node completes,
        # giving us per-agent progress events.  Wrapped in a hard timeout so
        # a hung agent/LLM call cannot block the session forever; on timeout
        # the session is failed so the frontend can offer a retry.
        final_state: dict | None = None

        async def _graph_runner() -> dict | None:
            last: dict | None = None
            # Use stream_mode="updates" so step keys are actual node names
            async for step in graph.astream(state, stream_mode="updates"):
                if step is None:
                    continue
                last = step
                # Write every step's update back to the session state so
                # the GET /status endpoint returns live agent progress
                # instead of always showing the initial "running" state.
                for node_name, update in step.items():
                    if isinstance(update, dict):
                        state.update(update)
                for node_name in step:
                    if node_name in phase_labels:
                        await queue.put({
                            "type": "agent_complete",
                            "data": {
                                "agent": node_name,
                                "phase": phase_labels[node_name],
                            },
                        })
                # Persist live progress so GET /status can restore completed
                # agents after a page refresh (the full final state is only
                # written once at the end of the run).
                if short_term:
                    try:
                        await short_term.update_metadata(session_id, {
                            "orchestrator_state": {
                                "current_phase": state.get("current_phase", ""),
                                "overall_status": "running",
                                "agent_results": {
                                    n: {"status": "completed"}
                                    for n in state.get("agent_results", {})
                                },
                            },
                        })
                    except Exception:
                        # Best-effort; a failed progress write must not kill
                        # the generation.
                        pass
            return last

        try:
            try:
                # Real 6-agent generation with LLM can take 5+ minutes
                final_state = await asyncio.wait_for(_graph_runner(), timeout=600)
            except AttributeError:
                # Fallback: astream() not available in this LangGraph version
                final_state = await graph.ainvoke(state)
        except TimeoutError:
            logger.error("Generation %s timed out after 600s — failing session", session_id)
            state["overall_status"] = "failed"
            state["errors"] = state.get("errors", []) + [
                {"agent": "orchestrator", "error": "Generation timed out after 600s", "phase": "UNKNOWN"}
            ]

        # Use the fully-accumulated state, not the last streamed step — the
        # final step is only that node's partial update and would lose the
        # overall_status (e.g. a planner failure reported as "completed").
        result = state
        if final_state and isinstance(final_state, dict):
            result = {**state, **final_state}

        # Store result in ShortTermMemory
        if short_term:
            await short_term.update_metadata(session_id, {
                "orchestrator_state": result,
                "orchestrator_result": result,
            })

        # ── Sync generated resources to the resource store ────────────────
        generated = result.get("generated_resources", [])
        if generated:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    sync_payload = [
                        {
                            # GeneratedResource has no stable id and the
                            # resources.id column is a UUID — always mint one.
                            "id": str(uuid.uuid4()),
                            "user_id": result.get("user_id", "anonymous"),
                            "name": r.get("title", r.get("type", "resource")),
                            "type": r.get("type", "explanation"),
                            "source": "system_generated",
                            "kp_id": r.get("kp_id"),
                            "kp_name": r.get("kp_name"),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "parse_status": "parsed",
                        }
                        for i, r in enumerate(generated)
                    ]
                    await client.post(
                        f"http://localhost:8000/api/v1/resources/sync-generated",
                        json=sync_payload,
                    )
                    logger.info("Synced %d generated resources to resource store", len(generated))
            except Exception as sync_exc:
                logger.warning("Failed to sync generated resources: %s", sync_exc)

        # Push completion
        status = result.get("overall_status", "completed")
        await queue.put({"type": "workflow_complete", "data": {"status": status}})

        logger.info("Generation %s completed with status '%s'", session_id, status)

    except Exception as exc:
        logger.exception("Generation %s failed", session_id)
        state["overall_status"] = "failed"
        state["errors"] = state.get("errors", []) + [
            {"agent": "orchestrator", "error": str(exc), "phase": "UNKNOWN"}
        ]
        if short_term:
            await short_term.update_metadata(session_id, {
                "orchestrator_state": state,
            })
        await queue.put({
            "type": "workflow_error",
            "data": {"agent": "orchestrator", "error": str(exc)},
        })


def _get_graph():
    """Get the compiled graph (lazy init)."""
    from src.agents.orchestrator.graph import create_graph
    return create_graph()
