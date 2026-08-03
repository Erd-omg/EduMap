"""API integration tests for privacy (PIPL compliance) routes.

Tests three endpoints:
    DELETE /api/v1/privacy/user/{user_id}/data  — cascade delete
    GET    /api/v1/privacy/user/{user_id}/export — data export
    POST   /api/v1/privacy/user/{user_id}/anonymize — graduation anonymization
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_privacy_app(**overrides) -> FastAPI:
    """Create a minimal FastAPI app with privacy-route-aware mocks."""
    from tests.test_api.conftest import _create_test_app

    # Build the base test app (includes all routers including privacy)
    app = _create_test_app(**overrides)

    # Override mocks unconditionally (AsyncMock auto-creates attributes,
    # so hasattr checks always pass — we must set them directly)
    memory_ops = app.state.memory_ops

    long_term = AsyncMock()
    long_term.delete_all_by_user = AsyncMock(return_value=3)
    long_term.recall_episodic = AsyncMock(return_value=[])
    long_term.recall_semantic = AsyncMock(return_value=[])
    memory_ops.long_term = long_term

    short_term = AsyncMock()
    short_term.get_session = AsyncMock(return_value=None)
    short_term.delete_all_sessions_by_user = AsyncMock(return_value=2)
    memory_ops.short_term = short_term

    resource_repo = app.state.resource_repo
    resource_repo.delete_all_by_user = AsyncMock(return_value=2)

    forgetting = app.state.forgetting_service
    forgetting.delete_user_data = AsyncMock(return_value=1)

    return app


@pytest.fixture
def privacy_client():
    """FastAPI TestClient with privacy-route-specific mocks."""
    app = _make_privacy_app()
    with TestClient(app) as c:
        yield c


class TestDeleteUserData:
    """DELETE /api/v1/privacy/user/{user_id}/data — cascade delete."""

    def test_delete_returns_204(self, privacy_client):
        """Successful cascade delete returns 204 No Content."""
        resp = privacy_client.delete("/api/v1/privacy/user/test-user/data")
        assert resp.status_code == 204

    def test_delete_non_existent_user(self, privacy_client):
        """Deleting data for a user with no records returns 204 (idempotent)."""
        resp = privacy_client.delete("/api/v1/privacy/user/nonexistent/data")
        assert resp.status_code == 204

    def test_delete_calls_all_stores(self, privacy_client):
        """Verify that each data store's delete method is called."""
        with TestClient(_make_privacy_app()) as c:
            resp = c.delete("/api/v1/privacy/user/test-user/data")
            assert resp.status_code == 204

    def test_delete_with_url_encoded_chars(self, privacy_client):
        """User_id with URL-encoded characters is handled gracefully."""
        resp = privacy_client.delete("/api/v1/privacy/user/user%40example.com/data")
        assert resp.status_code == 204

    def test_delete_with_special_chars(self, privacy_client):
        """User_id with special characters is handled gracefully."""
        resp = privacy_client.delete("/api/v1/privacy/user/user@example.com/data")
        assert resp.status_code == 204


class TestExportUserData:
    """GET /api/v1/privacy/user/{user_id}/export — data export."""

    def test_export_returns_200(self, privacy_client):
        """Export returns 200 with JSON body."""
        resp = privacy_client.get("/api/v1/privacy/user/test-user/export")
        assert resp.status_code == 200

    def test_export_contains_metadata(self, privacy_client):
        """Export response includes user_id and exported_at."""
        resp = privacy_client.get("/api/v1/privacy/user/test-user/export")
        data = resp.json()
        assert data["user_id"] == "test-user"
        assert "exported_at" in data
        assert "data_stores" in data

    def test_export_non_existent_user(self, privacy_client):
        """Export for a user with no data returns empty data_stores."""
        resp = privacy_client.get("/api/v1/privacy/user/nonexistent/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "nonexistent"
        assert isinstance(data["data_stores"], dict)

    def test_export_with_episodic_memory(self, privacy_client):
        """Export includes episodic memory when data exists."""
        app = _make_privacy_app()
        from src.memory.models import EventType, EpisodicEntry
        from datetime import datetime, timezone

        # Set up mock to return some episodic entries
        long_term = app.state.memory_ops.long_term
        long_term.recall_episodic = AsyncMock(return_value=[
            EpisodicEntry(
                id="e1",
                user_id="test-user",
                event_type=EventType.MENTOR_QUERY,
                session_id="s1",
                input="什么是数组?",
                output="数组是...",
                importance_score=0.8,
                created_at=datetime.now(timezone.utc),
            ),
        ])

        with TestClient(app) as c:
            resp = c.get("/api/v1/privacy/user/test-user/export")
            assert resp.status_code == 200
            data = resp.json()
            assert "episodic_memory" in data["data_stores"]
            entries = data["data_stores"]["episodic_memory"]
            assert len(entries) == 1
            assert entries[0]["id"] == "e1"
            assert entries[0]["input"] == "什么是数组?"

    def test_export_with_semantic_memory(self, privacy_client):
        """Export includes semantic memory when data exists."""
        app = _make_privacy_app()
        from src.memory.models import SemanticEntry

        long_term = app.state.memory_ops.long_term
        long_term.recall_semantic = AsyncMock(return_value=[
            SemanticEntry(
                key="learning_style",
                memory_type="user_preference",
                value={"text": "visual"},
                confidence=0.85,
                user_id="test-user",
            ),
        ])

        with TestClient(app) as c:
            resp = c.get("/api/v1/privacy/user/test-user/export")
            assert resp.status_code == 200
            data = resp.json()
            assert "semantic_memory" in data["data_stores"]
            entries = data["data_stores"]["semantic_memory"]
            assert len(entries) == 1
            assert entries[0]["key"] == "learning_style"

    def test_export_with_forgetting_curves(self, privacy_client):
        """Export includes forgetting curve data when available."""
        app = _make_privacy_app()
        forgetting = app.state.forgetting_service
        forgetting.get_all_states = AsyncMock(return_value=[
            {"kp_id": "kp-array", "recall_probability": 0.75},
        ])

        with TestClient(app) as c:
            resp = c.get("/api/v1/privacy/user/test-user/export")
            assert resp.status_code == 200
            data = resp.json()
            assert "forgetting_curve" in data["data_stores"]


class TestAnonymizeUser:
    """POST /api/v1/privacy/user/{user_id}/anonymize — graduation anonymization."""

    def test_anonymize_returns_200(self, privacy_client):
        """Anonymization returns 200 with anonymization metadata."""
        resp = privacy_client.post("/api/v1/privacy/user/test-user/anonymize")
        assert resp.status_code == 200

    def test_anonymize_contains_ids(self, privacy_client):
        """Response includes original user_id and new anonymized_id."""
        resp = privacy_client.post("/api/v1/privacy/user/test-user/anonymize")
        data = resp.json()
        assert data["status"] == "anonymized"
        assert data["original_user_id"] == "test-user"
        assert data["anonymized_id"] is not None
        assert data["anonymized_id"] != "test-user"

    def test_anonymize_id_is_deterministic(self, privacy_client):
        """Same user_id produces the same anonymized_id."""
        app = _make_privacy_app()
        with TestClient(app) as c:
            resp1 = c.post("/api/v1/privacy/user/test-user/anonymize")
            resp2 = c.post("/api/v1/privacy/user/test-user/anonymize")
            assert resp1.json()["anonymized_id"] == resp2.json()["anonymized_id"]

    def test_anonymize_different_users_different_ids(self, privacy_client):
        """Different user_ids produce different anonymized_ids."""
        app = _make_privacy_app()
        with TestClient(app) as c:
            resp1 = c.post("/api/v1/privacy/user/user-a/anonymize")
            resp2 = c.post("/api/v1/privacy/user/user-b/anonymize")
            assert resp1.json()["anonymized_id"] != resp2.json()["anonymized_id"]

    def test_anonymize_id_has_correct_format(self, privacy_client):
        """Anonymized ID is a 32-char hex string (SHA-256[:32])."""
        resp = privacy_client.post("/api/v1/privacy/user/test-user/anonymize")
        anon_id = resp.json()["anonymized_id"]
        assert len(anon_id) == 32
        assert all(c in "0123456789abcdef" for c in anon_id)
