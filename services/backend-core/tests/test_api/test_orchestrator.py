"""API tests for Orchestrator endpoints.

These drive the real endpoints with real assertions.  An earlier version sent a
payload that did not match ``GenerateRequest`` at all (``course_id``/``kp_id``/
``content_types`` instead of ``task_input``/``knowledge_point_id``) and asserted
only ``status_code in (200, 422, 500)`` — which accepts every outcome, so the
tests passed regardless of whether the endpoint worked.
"""
from __future__ import annotations

from src.agents.orchestrator.router import GenerateRequest


class TestGenerateRequestContract:
    """The schema the endpoints are built on."""

    def test_required_field_is_task_input(self):
        """``task_input`` is the required field — not ``course_id``."""
        req = GenerateRequest(task_input="解释二分查找")
        assert req.task_input == "解释二分查找"
        assert req.task_type == "generate"
        assert req.user_id == "anonymous"
        assert req.knowledge_point_id is None

    def test_missing_task_input_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest(course_id="cs101", kp_id="kp-1")


class TestOrchestratorGenerate:
    """POST /api/v1/orchestrator/generate — start generation."""

    def test_generate_accepts_valid_payload(self, client):
        """A well-formed request returns a session id and processing status."""
        resp = client.post("/api/v1/orchestrator/generate", json={
            "task_input": "解释二分查找",
            "user_id": "u-test",
            "knowledge_point_id": "kp-test",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Contract the frontend depends on to open the SSE stream.
        assert body["session_id"], "session_id must be non-empty"
        assert body["status"] == "processing"

    def test_generate_with_client_supplied_session_id_echoes_it(self, client):
        """A caller-supplied session_id is honoured (resume/reconnect case)."""
        resp = client.post("/api/v1/orchestrator/generate", json={
            "task_input": "x", "session_id": "my-session-123",
        })
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session-123"

    def test_generate_without_task_input_returns_422(self, client):
        """``task_input`` is required — omitting it is a validation error."""
        resp = client.post("/api/v1/orchestrator/generate", json={})
        assert resp.status_code == 422, resp.text

    def test_generate_rejects_unknown_extra_fields(self, client):
        """The old fictional payload shape must not be silently accepted.

        ``course_id``/``kp_id`` are not fields on GenerateRequest; Pydantic's
        default ignores extras, so the request still succeeds — but without a
        task_input it must 422 rather than quietly generate nothing.
        """
        resp = client.post("/api/v1/orchestrator/generate", json={
            "course_id": "cs101", "kp_id": "kp-test",
        })
        assert resp.status_code == 422


class TestOrchestratorStatus:
    """GET /api/v1/orchestrator/status/{session_id}."""

    def test_unknown_session_returns_404(self, client):
        """An unknown session is a 404 — not a 500 and not a silent 200."""
        resp = client.get("/api/v1/orchestrator/status/nonexistent-session")
        assert resp.status_code == 404, resp.text
        assert "not found" in resp.json()["detail"].lower()

    def test_known_session_returns_state_snapshot(self, client):
        """A started session is retrievable and carries the documented fields."""
        created = client.post("/api/v1/orchestrator/generate", json={
            "task_input": "解释二分查找",
            "user_id": "u-test",
            "knowledge_point_id": "kp-test",
        })
        session_id = created.json()["session_id"]

        resp = client.get(f"/api/v1/orchestrator/status/{session_id}")
        # The background task may have already finished (mocked graph), but the
        # session must exist and expose the snapshot contract.
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for field in (
            "session_id", "current_phase", "overall_status",
            "knowledge_point_id", "agent_results", "generated_resources",
        ):
            assert field in body, f"missing {field!r} in status snapshot"
        assert body["session_id"] == session_id
        assert body["knowledge_point_id"] == "kp-test"

    def test_started_session_reaches_a_terminal_status(self, client):
        """A started run must not stay at "running" forever.

        NOTE ON SCOPE: this test app never calls ``configure_graph``, so the
        compiled graph has no agents wired and the run fails fast with
        "Agent not configured".  That still exercises the property we care
        about here — the status resolves rather than hanging at the initial
        ``"running"`` — but it does NOT exercise ``assessment_node``'s
        completion logic (the ``overall_status`` whitelist bug).  That path is
        covered properly by the graph-level tests in
        ``tests/test_agents/test_orchestrator_graph.py``, which wire fake
        agents via ``_graph_ctx`` and run the real compiled graph.
        """
        import time

        created = client.post("/api/v1/orchestrator/generate", json={
            "task_input": "x", "user_id": "u-test",
        })
        session_id = created.json()["session_id"]

        body = {}
        for _ in range(50):
            body = client.get(f"/api/v1/orchestrator/status/{session_id}").json()
            if body["overall_status"] not in ("running", "unknown"):
                break
            time.sleep(0.1)

        assert body["overall_status"] in (
            "completed", "degraded", "failed", "cancelled",
        ), f"run never reached a terminal status: {body['overall_status']!r}"
        assert body["overall_status"] != "running"


class TestOrchestratorCancel:
    """POST /api/v1/orchestrator/cancel/{session_id}."""

    def test_cancel_unknown_session_returns_404(self, client):
        resp = client.post("/api/v1/orchestrator/cancel/nonexistent-session")
        assert resp.status_code == 404, resp.text

    def test_cancel_known_session_is_accepted(self, client):
        """Cancelling a live session returns a recognised status."""
        created = client.post("/api/v1/orchestrator/generate", json={
            "task_input": "x", "user_id": "u-test",
        })
        session_id = created.json()["session_id"]
        resp = client.post(f"/api/v1/orchestrator/cancel/{session_id}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] in (
            "cancelling", "cancelled", "completed", "failed",
        )


class TestStatusCheckpointerFallback:
    """GET /status falls back to the durable checkpointer when Redis is empty.

    The two stores are independent: Redis holds the live snapshot (TTL 1h),
    the checkpointer holds LangGraph's own per-super-step state in Postgres.
    A flushed/expired Redis must not make a live session look nonexistent.
    """

    def test_status_404_when_neither_store_has_the_session(self, client):
        resp = client.get("/api/v1/orchestrator/status/does-not-exist")
        assert resp.status_code == 404

    def test_fallback_is_attempted_and_does_not_crash(self, client, monkeypatch):
        """With no checkpointer configured the endpoint still returns 404.

        This pins the important property: the fallback path must be *safe* when
        unavailable, not that it produces data — the checkpointer is absent in
        the test app by design.
        """
        import src.agents.orchestrator.router as r

        called: list[str] = []

        async def _spy(session_id: str):
            called.append(session_id)
            return None

        monkeypatch.setattr(r, "_state_from_checkpointer", _spy)

        resp = client.get("/api/v1/orchestrator/status/no-such-session")
        assert resp.status_code == 404
        assert called == ["no-such-session"], (
            "the checkpointer fallback must be consulted before 404-ing"
        )

    def test_state_from_checkpointer_serves_a_state_dict(self, client, monkeypatch):
        """When the fallback yields state, /status returns it."""
        import src.agents.orchestrator.router as r

        async def _stub(session_id: str):
            return {
                "current_phase": "GENERATE",
                "overall_status": "running",
                "task_input": "from checkpoint",
                "agent_results": {},
                "generated_resources": [],
            }

        monkeypatch.setattr(r, "_state_from_checkpointer", _stub)

        resp = client.get("/api/v1/orchestrator/status/recovered-session")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["current_phase"] == "GENERATE"
        assert body["task_input"] == "from checkpoint"

    def test_fallback_returns_none_without_a_checkpointer(self):
        """No checkpointer configured → None, not an exception."""
        import asyncio
        from src.agents.orchestrator.router import _state_from_checkpointer

        assert asyncio.run(_state_from_checkpointer("any")) is None


class TestStatusDoesNotLeakQuizAnswers:
    """GET /status must not hand out the quiz answer key.

    The assessment node stores the whole ``AssessmentOutput`` dump in graph
    state, so ``assessment_result.quiz[*].correct_answer`` travels with it.
    This endpoint trims its two neighbouring keys (``agent_results``,
    ``generated_resources``) but used to pass ``assessment`` through raw —
    which made ``/quiz/generate``'s answer-key hiding bypassable by anyone who
    knew a ``session_id``, since the same key was reachable from here instead.

    Real HTTP via the ``client`` fixture.  The session is seeded through the
    same ``memory_ops.short_term`` store the endpoint reads back, mirroring how
    the orchestrator run populates it.
    """

    _SESSION = "sess-with-assessment"

    def _seed(self, client, assessment: dict) -> None:
        import asyncio

        state = {"assessment_result": assessment}
        asyncio.get_event_loop_policy()
        # The memory_ops mock exposes ``create_session`` as an AsyncMock, so
        # seed it synchronously by driving the coroutine.
        asyncio.run(
            client.app.state.memory_ops.short_term.create_session(
                self._SESSION, "u-test", {"orchestrator_state": state}
            )
        )

    _ASSESSMENT = {
        "quiz": [
            {
                "id": "q1", "type": "choice", "content": "1+1=?",
                "options": ["1", "2"], "correct_answer": "2",
                "knowledge_point_id": "kp-array",
            },
        ],
        "confidence": 0.8,
    }

    def test_answer_key_is_stripped_from_assessment(self, client):
        self._seed(client, self._ASSESSMENT)

        resp = client.get(f"/api/v1/orchestrator/status/{self._SESSION}")
        assert resp.status_code == 200, resp.text
        quiz = resp.json()["assessment"]["quiz"]
        assert len(quiz) == 1
        assert "correct_answer" not in quiz[0], "answer key leaked via /status"
        # The rest of the question still reaches the client.
        assert quiz[0]["id"] == "q1"
        assert quiz[0]["content"] == "1+1=?"

    def test_other_assessment_fields_survive(self, client):
        """Trimming must be surgical — not a blanket drop of the payload."""
        self._seed(client, {"quiz": [], "confidence": 0.8})
        resp = client.get(f"/api/v1/orchestrator/status/{self._SESSION}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["assessment"]["confidence"] == 0.8
