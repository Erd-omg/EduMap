"""API integration tests for health endpoints."""
from __future__ import annotations


class TestHealth:
    """GET /health — basic liveness check."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "backend-core"


class TestHealthReady:
    """GET /health/ready — readiness check."""

    def test_ready_returns_200(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200

    def test_ready_returns_dict(self, client):
        resp = client.get("/health/ready")
        assert isinstance(resp.json(), dict)
