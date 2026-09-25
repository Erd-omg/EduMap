"""Orchestrator API router — endpoints for triggering and monitoring
multi-agent generation workflows.

Endpoints::

    POST   /api/v1/orchestrator/generate   — start a new generation workflow
    GET    /api/v1/orchestrator/status/{session_id} — poll progress
    GET    /api/v1/orchestrator/stream/{session_id}  — SSE progress stream
    POST   /api/v1/orchestrator/cancel/{session_id} — request hard cancellation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import httpx

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.orchestrator.state import _STATE_REDUCERS, create_initial_state
from src.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

# In-memory SSE event queues (process-local by nature — cannot be serialized to Redis)
_sse_queues: dict[str, asyncio.Queue] = {}

# Process-local task registry so the cancel endpoint can find (and cancel)
# the running generation Task.  Like _sse_queues, this is per-process.
_tasks: dict[str, asyncio.Task] = {}

# SSE read timeout: when no event arrives for this long we emit a heartbeat
# and KEEP the stream alive instead of tearing it down.  Real generations
# (6-agent LLM pipeline) can easily exceed the old hard 300s event gap.
_SSE_TIMEOUT = 300


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


class CancelResponse(BaseModel):
    session_id: str
    status: str  # cancelling | completed | failed | cancelled


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

    # Fire generation in the background.  Keep the Task handle in _tasks so
    # the cancel endpoint can request hard cancellation; _run_generation's
    # finally block pops it (and the SSE queue) when the run terminates.
    task = asyncio.create_task(_run_generation(session_id, request))
    _tasks[session_id] = task

    return GenerateResponse(session_id=session_id, status="processing")


@router.post("/cancel/{session_id}", response_model=CancelResponse)
async def cancel_generation(session_id: str, request: Request):
    """Request hard cancellation of an in-flight generation.

    Cancels the background ``asyncio.Task`` running the LangGraph workflow.
    Idempotent: a second POST while cancellation is pending returns the same
    ``cancelling`` status with 200.  A finished session returns its persisted
    terminal status.  409 when Redis says the session is ``running`` but no
    Task is tracked in this process (e.g. a different uvicorn worker owns it);
    404 for unknown sessions.
    """
    task = _tasks.get(session_id)
    short_term = _get_short_term(request)

    if task is None or task.done():
        # No tracked (running) task — fall back to the persisted state.
        if short_term:
            session_mem = await short_term.get_session(session_id)
            if session_mem:
                state = session_mem.metadata.get("orchestrator_state") or {}
                status = state.get("overall_status", "unknown")
                if status in ("completed", "failed", "cancelled"):
                    return CancelResponse(session_id=session_id, status=status)
                if status == "running":
                    raise HTTPException(
                        status_code=409,
                        detail="No active generation task tracked for session",
                    )
        raise HTTPException(status_code=404, detail="Session not found or already finished")

    if task.cancelling():
        # Cancellation already pending — idempotent no-op.
        return CancelResponse(session_id=session_id, status="cancelling")

    # Immediate marker so GET /status reflects cancellation right away; the
    # authoritative write happens in the task's CancelledError handler.
    await _mark_session_cancelled(short_term, session_id)

    task.cancel()
    logger.info("Cancellation requested for generation %s", session_id)
    return CancelResponse(session_id=session_id, status="cancelling")


async def _state_from_checkpointer(session_id: str) -> dict | None:
    """Read a session's graph state back from the durable checkpointer.

    Fallback for ``GET /status`` when Redis has no snapshot. The two stores are
    independent: Redis holds the live progress snapshot the run writes as it
    goes, while the checkpointer holds LangGraph's own per-super-step state in
    Postgres. If Redis was flushed, evicted (its TTL is an hour), or the
    snapshot write failed, the checkpoint is still there — and unlike the Redis
    copy it is written by LangGraph with the graph's own reducers applied, so it
    is the more authoritative of the two.

    Returns the state dict, or None when no checkpointer is configured or the
    thread is unknown. Never raises — a status lookup must not fail the request.
    """
    try:
        from src.agents.orchestrator.graph import graph_has_checkpointer

        if not graph_has_checkpointer():
            return None

        from src.agents.orchestrator.checkpointing import thread_config

        graph = _get_graph()
        snapshot = await graph.aget_state(thread_config(session_id))
        values = getattr(snapshot, "values", None)
        return dict(values) if values else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Checkpointer fallback failed for %s: %s", session_id, exc)
        return None


@router.get("/status/{session_id}")
async def get_status(session_id: str, request: Request):
    """Return the current state snapshot for a generation session.

    Reads Redis first (the live snapshot the run maintains), falling back to
    the durable checkpointer when Redis has nothing — see
    :func:`_state_from_checkpointer` for why those are separate stores.
    """
    # Load from ShortTermMemory (or fallback)
    short_term = _get_short_term(request)
    state_dict = None

    if short_term:
        session_mem = await short_term.get_session(session_id)
        if session_mem:
            state_dict = session_mem.metadata.get("orchestrator_state")

    if not state_dict:
        state_dict = await _state_from_checkpointer(session_id)

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

    return StreamingResponse(
        _sse_event_generator(session_id, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_event_generator(session_id: str, queue: asyncio.Queue):
    """Yield SSE frames for a session until it terminates.

    Ends on: a terminal event (``workflow_complete`` / ``workflow_error``),
    client disconnect, or the session's queue being popped (see below).
    Module-level so it can be consumed directly in tests on one event loop
    (cross-thread ``queue.put_nowait`` cannot reliably wake the generator
    inside a TestClient thread).
    """
    while True:
        # Session finished and its queue was popped by _run_generation's
        # finally → stop.  This bounds multi-client zombies: a single
        # queue.get() delivers the terminal event to one waiter, and the
        # rest would otherwise heartbeat forever.
        if _sse_queues.get(session_id) is not queue:
            break
        try:
            event = await asyncio.wait_for(queue.get(), timeout=_SSE_TIMEOUT)
            if event.get("type") == "workflow_complete":
                yield f"event: workflow_complete\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                break
            if event.get("type") == "workflow_error":
                yield f"event: workflow_error\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                break
            payload = json.dumps(event.get('data', event), ensure_ascii=False)
            yield f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"
        except asyncio.TimeoutError:
            # No event for _SSE_TIMEOUTs — emit a heartbeat and KEEP
            # looping.  Real generations (5+ min) exceed this gap.  The
            # loop only ends on a terminal event, client disconnect, or
            # the pop-guard above.
            hb = json.dumps({
                'type': 'heartbeat', 'reason': 'timeout', 'timestamp': time.time(),
            })
            yield f"event: heartbeat\ndata: {hb}\n\n"
            continue


# ── Background runner ──────────────────────────────────────────────────


async def _mark_session_cancelled(
    short_term: ShortTermMemory | None, session_id: str
) -> None:
    """Best-effort immediate cancellation marker.

    Read-modify-write of the FULL ``orchestrator_state`` snapshot (not just
    the status field) because ``update_metadata`` is a shallow merge — passing
    ``{"overall_status": "cancelled"}`` alone would wipe ``agent_results`` and
    ``generated_resources``.  The task's CancelledError handler overwrites
    this marker with the authoritative in-memory snapshot.
    """
    if not short_term:
        return
    session_mem = await short_term.get_session(session_id)
    if not session_mem:
        return
    state = dict(session_mem.metadata.get("orchestrator_state") or {})
    state["overall_status"] = "cancelled"
    state["errors"] = list(state.get("errors") or []) + [
        {"agent": "orchestrator", "error": "Generation cancelled by user", "phase": "UNKNOWN"}
    ]
    await short_term.update_metadata(session_id, {
        "orchestrator_state": state,
        "orchestrator_result": state,
    })


async def _sync_generated_resources(state: dict) -> None:
    """Best-effort POST of ``generated_resources`` to the resource store.

    Used on the success path and on cancellation (to persist partial
    resources).  Never raises — failures are logged.
    """
    generated = state.get("generated_resources", [])
    if not generated:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Bind every generated resource to the node the user selected on
            # /generate.  The planner invents sub-KP ids (e.g.
            # "kp-linkedlist-basic") that do not exist in the knowledge graph,
            # so the graph panel could never match them; the selected node id
            # is a real KG node.
            selected_kp_id = state.get("knowledge_point_id")
            sync_payload = [
                {
                    # GeneratedResource has no stable id and the resources.id
                    # column is a UUID — always mint one.
                    "id": str(uuid.uuid4()),
                    "user_id": state.get("user_id", "anonymous"),
                    "name": r.get("title", r.get("type", "resource")),
                    "type": r.get("type", "explanation"),
                    "source": "system_generated",
                    "kp_id": selected_kp_id or r.get("kp_id"),
                    "kp_name": r.get("kp_name"),
                    # Persist the generated markdown body so the library /
                    # graph can open it (the resources table has no separate
                    # content column).
                    "description": (r.get("content") or "")[:20000],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parse_status": "parsed",
                }
                for i, r in enumerate(generated)
            ]
            await client.post(
                "http://localhost:8000/api/v1/resources/sync-generated",
                json=sync_payload,
            )
            logger.info("Synced %d generated resources to resource store", len(generated))
    except Exception as sync_exc:
        logger.warning("Failed to sync generated resources: %s", sync_exc)


async def _run_generation(session_id: str, request: Request) -> None:
    """Execute the LangGraph state graph and stream events.

    ``try`` starts at function entry so a cancellation delivered during the
    initial ``get_session`` await is still caught and cleaned up, and the
    early-return paths also run the ``finally``.
    """
    short_term = _get_short_term(request)
    state: dict = {}
    queue: asyncio.Queue | None = None

    try:
        # Load state from ShortTermMemory
        if short_term:
            session_mem = await short_term.get_session(session_id)
            if session_mem:
                state = session_mem.metadata.get("orchestrator_state", {})
        queue = _sse_queues.get(session_id)

        if not state:
            logger.error("Session %s not found in ShortTermMemory — aborting", session_id)
            if queue is not None:
                # Heartbeat-continue invariant: every path that owns a live
                # queue must push a terminal event before the finally pops it,
                # or a connected SSE client would heartbeat forever.
                await queue.put({
                    "type": "workflow_error",
                    "data": {"agent": "orchestrator", "error": "Session state not found"},
                })
            return
        if queue is None:
            logger.error("SSE queue for session %s not found — aborting", session_id)
            return

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
            # Invocation contract differs between the two graph kinds:
            #
            #   * checkpointed → a thread_id is REQUIRED; LangGraph persists
            #     each super-step under that thread, which is what lets
            #     progress survive a restart.
            #   * ephemeral → must be invoked WITHOUT a config at all. Passing
            #     `config=None` explicitly is not equivalent to omitting it —
            #     it broke invocation in testing — so kwargs are built
            #     conditionally instead of always passing the parameter.
            from src.agents.orchestrator.checkpointing import thread_config
            from src.agents.orchestrator.graph import graph_has_checkpointer

            # ``stream_mode="updates"`` yields per-node deltas, which is what
            # both the SSE events and the live progress snapshot need (step
            # keys are real node names).
            #
            # NOTE on the reducer replay below. It is tempting to replace it
            # with ``stream_mode=["updates", "values"]`` and take the assembled
            # state LangGraph hands back. That was tried and **it hangs** the
            # invocation (verified: adding "values" made the API test suite
            # stall, while the single mode completes in ~10s). So the deltas are
            # folded here instead.
            #
            # That folding is NOT a duplicate of the graph's logic:
            # ``_STATE_REDUCERS`` is *derived from* EduMapState's ``Annotated``
            # metadata (see state.py), so it is the same source of truth the
            # graph uses. What must not be reintroduced is a hand-written merge
            # — a previous last-write-wins version was correct only while the
            # generation branches were sequential, and silently dropped one
            # parallel branch's resources once each branch returned only its
            # own delta.
            stream_kwargs: dict = {"stream_mode": "updates"}
            if graph_has_checkpointer():
                stream_kwargs["config"] = thread_config(session_id)

            async for step in graph.astream(state, **stream_kwargs):
                if step is None:
                    continue
                last = step
                for node_name, update in step.items():
                    if not isinstance(update, dict):
                        continue
                    for key, value in update.items():
                        reducer = _STATE_REDUCERS.get(key)
                        if reducer is None:
                            state[key] = value
                        else:
                            state[key] = reducer(state.get(key), value)
                for node_name in step:
                    if node_name in phase_labels:
                        # Carry the node's ``_report`` (duration, retries, tool
                        # calls) so the client can render a per-agent trace
                        # without a second /status round-trip.
                        node_report = (
                            (state.get("agent_results") or {})
                            .get(node_name, {})
                            .get("_report")
                        )
                        await queue.put({
                            "type": "agent_complete",
                            "data": {
                                "agent": node_name,
                                "phase": phase_labels[node_name],
                                "report": node_report or {},
                            },
                        })
                # Persist live progress so GET /status can restore completed
                # agents after a page refresh (the full final state is only
                # written once at the end of the run).
                #
                # NOTE: ``update_metadata`` is a *shallow* merge, so
                # ``orchestrator_state`` must be written as a COMPLETE snapshot.
                # An earlier version wrote only {current_phase, overall_status,
                # agent_results}, which wiped task_input / generated_resources /
                # errors and replaced the real agent_results with bare
                # {"status": "completed"} placeholders — so a refresh saw
                # gutted state.
                if short_term:
                    try:
                        snapshot = dict(state)
                        snapshot["overall_status"] = "running"
                        await short_term.update_metadata(session_id, {
                            "orchestrator_state": snapshot,
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
                # Fallback: astream() not available in this LangGraph version.
                # asyncio.CancelledError (BaseException) passes through this
                # handler and the TimeoutError handler into the outer
                # CancelledError handler untouched.
                # Same conditional-kwargs rule as the astream path above.
                from src.agents.orchestrator.checkpointing import thread_config as _tc
                from src.agents.orchestrator.graph import graph_has_checkpointer as _has_ck
                if _has_ck():
                    final_state = await graph.ainvoke(state, config=_tc(session_id))
                else:
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

        # Sync whatever was generated — success AND timeout both reach here.
        await _sync_generated_resources(result)

        # Push completion
        status = result.get("overall_status", "completed")
        await queue.put({"type": "workflow_complete", "data": {"status": status}})

        logger.info("Generation %s completed with status '%s'", session_id, status)

    except asyncio.CancelledError:
        # Hard cancel.  Single-cancel contract: the pending cancel was consumed
        # by this very raise, so the awaits below run normally.  A *second*
        # task.cancel() mid-handler could interrupt them — prevented in practice
        # by the endpoint's task.cancelling() guard and the frontend's
        # double-click guard.  asyncio.shield(...) around the cleanup awaits is
        # optional hardening, not required.
        logger.info("Generation %s cancelled by user", session_id)
        if state:
            state["overall_status"] = "cancelled"
            state["errors"] = list(state.get("errors") or []) + [
                {"agent": "orchestrator", "error": "Generation cancelled by user", "phase": "UNKNOWN"}
            ]
            if short_term:
                try:
                    # Authoritative FULL snapshot — never a partial status-only
                    # dict, or update_metadata's shallow merge wipes
                    # agent_results / generated_resources.
                    await short_term.update_metadata(session_id, {
                        "orchestrator_state": state,
                        "orchestrator_result": state,
                    })
                except Exception:
                    logger.exception("Failed to persist cancelled state for %s", session_id)
            # Best-effort: sync partial resources produced so far.
            await _sync_generated_resources(state)

        q = queue if queue is not None else _sse_queues.get(session_id)
        if q is not None:
            try:
                await q.put({"type": "workflow_complete", "data": {"status": "cancelled"}})
            except Exception:
                logger.debug("Failed to push cancelled event for %s", session_id)
        raise  # finish the task in the cancelled state

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
        q = queue if queue is not None else _sse_queues.get(session_id)
        if q is not None:
            await q.put({
                "type": "workflow_error",
                "data": {"agent": "orchestrator", "error": str(exc)},
            })

    finally:
        # Only after a terminal event has been pushed: a queue removed here
        # means new SSE clients get 404 and the pop-guard in stream_events
        # terminates any straggler stream.  Never before.
        _sse_queues.pop(session_id, None)
        _tasks.pop(session_id, None)


def _get_graph():
    """Get the compiled graph (lazy init)."""
    from src.agents.orchestrator.graph import create_graph
    return create_graph()
