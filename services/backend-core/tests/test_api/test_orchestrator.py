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
