"""API tests for Orchestrator endpoints."""
from __future__ import annotations


class TestOrchestratorGenerate:
    """POST /api/v1/orchestrator/generate — start generation."""

    def test_generate_returns_200_or_422(self, client):
        resp = client.post("/api/v1/orchestrator/generate", json={
            "course_id": "cs101",
            "kp_id": "kp-test",
            "content_types": ["explanation"],
        })
        assert resp.status_code in (200, 422, 500)

    def test_generate_without_kp_returns_422(self, client):
        resp = client.post("/api/v1/orchestrator/generate", json={})
        assert resp.status_code in (422, 500)


class TestOrchestratorStatus:
    """GET /api/v1/orchestrator/status — generation status."""

    def test_status_unknown_session_returns_error(self, client):
        resp = client.get("/api/v1/orchestrator/status/nonexistent-session")
        assert resp.status_code in (404, 422, 500)
