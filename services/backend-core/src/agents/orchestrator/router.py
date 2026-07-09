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
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.orchestrator.graph import create_graph
from src.agents.orchestrator.state import EduMapState, create_initial_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

# In-memory session store (replace with Redis for production)
_sessions: dict[str, dict[str, Any]] = {}


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
    _sessions[session_id] = {"state": state, "result": None, "events": asyncio.Queue()}

    # Fire generation in the background
    asyncio.create_task(_run_generation(session_id, request))

    return GenerateResponse(session_id=session_id, status="processing")


@router.get("/status/{session_id}")
async def get_status(session_id: str):
    """Return the current state snapshot for a generation session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    state = session["state"]
    return {
        "session_id": session_id,
        "current_phase": state.get("current_phase", "UNKNOWN"),
        "overall_status": state.get("overall_status", "unknown"),
        "errors": state.get("errors", []),
        "agent_results": {
            k: v for k, v in state.get("agent_results", {}).items()
            if not isinstance(v, dict) or "error" not in v
        },
        "generated_resources": [
            {"type": r.get("type"), "title": r.get("title"), "kp_id": r.get("kp_id")}
            for r in state.get("generated_resources", [])
        ],
        "assessment": state.get("assessment_result"),
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
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    queue: asyncio.Queue = session["events"]

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
                yield f"event: heartbeat\ndata: {json.dumps({'ts': 'timeout'})}\n\n"
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
    session = _sessions.get(session_id)
    if not session:
        return

    queue: asyncio.Queue = session["events"]
    state: dict = session["state"]

    try:
        graph = _get_graph()

        # Push initial event
        await queue.put({"type": "phase_change", "data": {"phase": "EXTRACT", "agent": "planner"}})

        # Run graph (langgraph handles the full pipeline)
        result = await graph.ainvoke(state)

        # Store result
        session["state"] = result
        session["result"] = result

        # Push completion
        status = result.get("overall_status", "completed")
        await queue.put({"type": "workflow_complete", "data": {"status": status}})

        logger.info("Generation %s completed with status '%s'", session_id, status)

    except Exception as exc:
        logger.exception("Generation %s failed", session_id)
        session["state"]["overall_status"] = "failed"
        session["state"]["errors"] = session["state"].get("errors", []) + [
            {"agent": "orchestrator", "error": str(exc), "phase": "UNKNOWN"}
        ]
        await queue.put({
            "type": "workflow_error",
            "data": {"agent": "orchestrator", "error": str(exc)},
        })


def _get_graph():
    """Get the compiled graph (lazy init)."""
    from src.agents.orchestrator.graph import create_graph
    return create_graph()
