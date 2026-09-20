"""Tests for orchestrator hard-cancellation + task/SSE-queue cleanup.

Covers:
    - ``_run_generation``'s CancelledError handler (authoritative state write,
      partial-resource sync, terminal SSE event, finally cleanup)
    - the ``POST /api/v1/orchestrator/cancel/{session_id}`` endpoint
      (tracked / already-cancelling / done / unknown / running-without-task)
    - ``_mark_session_cancelled`` full-snapshot write
    - ``_sync_generated_resources`` (payload build, empty no-op, never-raises)
    - SSE heartbeat-continue + terminal-event end + pop-guard
    - early-return terminal-event invariant
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastapi.testclient import TestClient

from src.agents.orchestrator import router
from src.agents.orchestrator.state import create_initial_state


@pytest.fixture(autouse=True)
def _clean_orchestrator_registry():
    """The module dicts are process-global singletons — keep them clean."""
    router._sse_queues.clear()
    router._tasks.clear()
    yield
    router._sse_queues.clear()
    router._tasks.clear()


# ── Fakes ──────────────────────────────────────────────────────────────


class FakeShortTerm:
    """Async stand-in for ShortTermMemory.

    ``get_session`` returns a shallow copy so in-place mutation of the state
    dict inside ``_run_generation`` does not reflect back until
    ``update_metadata`` writes it — letting tests assert the authoritative
    write actually happened.
    """

    def __init__(self, initial_state: dict):
        self.metadata = {
            "orchestrator_state": dict(initial_state),
            "orchestrator_result": None,
        }

    async def get_session(self, session_id):
        return SimpleNamespace(metadata={
            "orchestrator_state": dict(self.metadata.get("orchestrator_state") or {}),
            "orchestrator_result": self.metadata.get("orchestrator_result"),
        })

    async def update_metadata(self, session_id, metadata):
        self.metadata.update(metadata)

    async def create_session(self, *a, **k):
        pass


class BlockingGraph:
    """astream yields one update then blocks forever (until cancelled)."""

    def __init__(self, entered: asyncio.Event):
        self._entered = entered

    async def astream(self, state, stream_mode="updates"):
        yield {"planner": {
            "current_phase": "EXTRACT",
            "agent_results": {"planner": {"n": 1}},
        }}
        self._entered.set()
        await asyncio.Event().wait()  # block until the task is cancelled
        yield {}

    async def ainvoke(self, state):
        raise NotImplementedError


# ── _run_generation: cancel path ───────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_writes_authoritative_state_and_cleans_up(monkeypatch):
    initial = create_initial_state(
        task_input="explain linked list",
        user_id="u1",
        session_id="s1",
        knowledge_point_id="kp-real",
    )
    initial["generated_resources"] = [
        {"type": "explanation", "title": "T", "content": "body"}
    ]

    entered = asyncio.Event()
    monkeypatch.setattr(router, "_get_graph", lambda: BlockingGraph(entered))
    st = FakeShortTerm(initial)
    monkeypatch.setattr(router, "_get_short_term", lambda request: st)
    synced: list[dict] = []

    async def fake_sync(state):
        synced.append(state)

    monkeypatch.setattr(router, "_sync_generated_resources", fake_sync)

    q = asyncio.Queue()
    router._sse_queues["s1"] = q
    task = asyncio.create_task(router._run_generation("s1", object()))
    router._tasks["s1"] = task

    await entered.wait()          # generation is mid-flight (blocked in graph)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # finally popped both registries
    assert "s1" not in router._sse_queues
    assert "s1" not in router._tasks

    # terminal event pushed before the queue was popped (drain past the
    # 7 leading phase_change events — no SSE client consumed them)
    pushed: list[dict] = []
    while True:
        try:
            pushed.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    terminal = [e for e in pushed if e["type"] == "workflow_complete"]
    assert terminal and terminal[0]["data"]["status"] == "cancelled"

    # authoritative FULL snapshot persisted (not a status-only dict)
    written = st.metadata["orchestrator_state"]
    assert written["overall_status"] == "cancelled"
    assert written["agent_results"]["planner"] == {"n": 1}      # preserved
    assert len(written["generated_resources"]) == 1             # preserved
    assert any("cancelled by user" in e["error"] for e in written["errors"])

    # partial resources were synced
    assert len(synced) == 1
    assert synced[0]["overall_status"] == "cancelled"


@pytest.mark.asyncio
async def test_run_generation_state_missing_pushes_terminal_event(monkeypatch):
    """Regression guard for the heartbeat invariant: even the early-return
    path must push a terminal event before the finally pops the queue."""
    q = asyncio.Queue()
    router._sse_queues["s1"] = q

    st = FakeShortTerm({})          # empty orchestrator_state → early return
    monkeypatch.setattr(router, "_get_short_term", lambda request: st)

    task = asyncio.create_task(router._run_generation("s1", object()))
    router._tasks["s1"] = task
    await task                       # completes normally, no exception

    evt = q.get_nowait()
    assert evt["type"] == "workflow_error"
    assert evt["data"]["error"] == "Session state not found"
    assert "s1" not in router._sse_queues
    assert "s1" not in router._tasks


# ── Cancel endpoint ────────────────────────────────────────────────────


class TestCancelEndpoint:
    def test_cancel_tracked_task_requests_cancel(self, client):
        fake = MagicMock()
        fake.done.return_value = False
        fake.cancelling.return_value = 0
        router._sse_queues["s1"] = asyncio.Queue()
        router._tasks["s1"] = fake

        resp = client.post("/api/v1/orchestrator/cancel/s1")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelling"
        fake.cancel.assert_called_once_with()

    def test_cancel_already_cancelling_is_idempotent(self, client):
        fake = MagicMock()
        fake.done.return_value = False
        fake.cancelling.return_value = 1
        router._tasks["s1"] = fake

        resp = client.post("/api/v1/orchestrator/cancel/s1")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelling"
        fake.cancel.assert_not_called()

    def test_cancel_unknown_session_returns_404(self, client):
        resp = client.post("/api/v1/orchestrator/cancel/nope")
        assert resp.status_code == 404

    def test_cancel_done_task_returns_persisted_terminal_status(self):
        short_term = AsyncMock()
        short_term.get_session = AsyncMock(return_value=SimpleNamespace(
            metadata={"orchestrator_state": {
                "overall_status": "cancelled",
                "agent_results": {"planner": {"n": 1}},
            }}
        ))
        memory_ops = AsyncMock()
        memory_ops.short_term = short_term

        from tests.test_api.conftest import _create_test_app
        app = _create_test_app(memory_ops=memory_ops)
        with TestClient(app) as c:
            resp = c.post("/api/v1/orchestrator/cancel/s1")

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_running_without_tracked_task_returns_409(self):
        short_term = AsyncMock()
        short_term.get_session = AsyncMock(return_value=SimpleNamespace(
            metadata={"orchestrator_state": {"overall_status": "running"}}
        ))
        memory_ops = AsyncMock()
        memory_ops.short_term = short_term

        from tests.test_api.conftest import _create_test_app
        app = _create_test_app(memory_ops=memory_ops)
        with TestClient(app) as c:
            resp = c.post("/api/v1/orchestrator/cancel/s1")

        assert resp.status_code == 409


# ── Helpers ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_session_cancelled_writes_full_snapshot():
    state = create_initial_state(task_input="t", user_id="u", session_id="s1")
    state["agent_results"] = {"planner": {"status": "completed"}}
    state["generated_resources"] = [{"type": "explanation", "title": "T"}]
    st = FakeShortTerm(state)

    await router._mark_session_cancelled(st, "s1")

    written = st.metadata["orchestrator_state"]
    assert written["overall_status"] == "cancelled"
    assert written["agent_results"] == {"planner": {"status": "completed"}}
    assert len(written["generated_resources"]) == 1
    assert any("cancelled by user" in e["error"] for e in written["errors"])


@pytest.mark.asyncio
async def test_mark_session_cancelled_missing_session_is_noop():
    st = AsyncMock()
    st.get_session = AsyncMock(return_value=None)
    await router._mark_session_cancelled(st, "s1")
    st.update_metadata.assert_not_called()


class FakeHttpxClient:
    """Context-manager httpx.AsyncClient stand-in that records posts."""

    def __init__(self, *, fail: bool = False):
        self.posted: list[tuple[str, list]] = []
        self._fail = fail

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        if self._fail:
            raise RuntimeError("sync endpoint unavailable")
        self.posted.append((url, json))


@pytest.mark.asyncio
async def test_sync_generated_resources_posts_payload(monkeypatch):
    fake = FakeHttpxClient()
    monkeypatch.setattr(router.httpx, "AsyncClient", lambda *a, **k: fake)

    state = {
        "user_id": "u1",
        "knowledge_point_id": "kp-selected",
        "generated_resources": [
            {"type": "explanation", "title": "Intro", "content": "body", "kp_id": "kp-invented"},
        ],
    }
    await router._sync_generated_resources(state)

    assert len(fake.posted) == 1
    url, payload = fake.posted[0]
    assert url.endswith("/api/v1/resources/sync-generated")
    assert payload[0]["kp_id"] == "kp-selected"      # bound to selected KG node
    assert payload[0]["name"] == "Intro"
    assert payload[0]["type"] == "explanation"
    assert payload[0]["source"] == "system_generated"


@pytest.mark.asyncio
async def test_sync_generated_resources_empty_is_noop(monkeypatch):
    fake = FakeHttpxClient()
    monkeypatch.setattr(router.httpx, "AsyncClient", lambda *a, **k: fake)

    await router._sync_generated_resources({})

    assert fake.posted == []


@pytest.mark.asyncio
async def test_sync_generated_resources_never_raises(monkeypatch):
    monkeypatch.setattr(
        router.httpx, "AsyncClient", lambda *a, **k: FakeHttpxClient(fail=True)
    )

    # Must not raise — failures are logged and swallowed.
    await router._sync_generated_resources({
        "generated_resources": [{"type": "x", "title": "y"}],
    })


# ── SSE stream: heartbeat + terminal event + pop-guard ─────────────────


class TestSSEStream:
    @pytest.mark.asyncio
    async def test_heartbeat_continues_and_terminal_event_ends_stream(self, monkeypatch):
        monkeypatch.setattr(router, "_SSE_TIMEOUT", 0.02)
        q = asyncio.Queue()
        router._sse_queues["s1"] = q
        frames: list[str] = []

        async def consume():
            async for frame in router._sse_event_generator("s1", q):
                frames.append(frame)
                if sum(1 for f in frames if "heartbeat" in f) >= 2:
                    await q.put({"type": "workflow_complete", "data": {"status": "completed"}})
                if any("workflow_complete" in f for f in frames):
                    break

        await asyncio.wait_for(consume(), timeout=5)

        # Heartbeat does NOT break the stream after one timeout...
        assert sum(1 for f in frames if "heartbeat" in f) >= 2
        # ...but a terminal event still ends it.
        assert any("workflow_complete" in f for f in frames)

    @pytest.mark.asyncio
    async def test_stream_pop_guard_terminates_without_terminal_event(self, monkeypatch):
        """Once _run_generation's finally pops the queue, the stream ends
        even though no terminal event is ever delivered to this client."""
        monkeypatch.setattr(router, "_SSE_TIMEOUT", 0.02)
        q = asyncio.Queue()
        router._sse_queues["s1"] = q
        frames: list[str] = []
        popped = False

        async def consume():
            nonlocal popped
            async for frame in router._sse_event_generator("s1", q):
                frames.append(frame)
                if not popped and "heartbeat" in frame:
                    router._sse_queues.pop("s1", None)   # simulate finally-cleanup
                    popped = True
                # Do NOT break here — the pop-guard itself must end the loop.

        await asyncio.wait_for(consume(), timeout=5)

        # The stream terminated via the pop-guard (no deadlock / no infinite
        # heartbeats after the queue is gone).
        assert popped
        assert any("heartbeat" in f for f in frames)
