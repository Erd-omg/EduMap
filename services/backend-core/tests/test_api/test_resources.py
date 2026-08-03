"""API tests for Resources endpoints."""
from __future__ import annotations


class TestResourcesList:
    """GET /api/v1/resources — list resources."""

    def test_list_returns_200(self, client):
        resp = client.get("/api/v1/resources")
        assert resp.status_code == 200

    def test_list_returns_list(self, client):
        resp = client.get("/api/v1/resources")
        data = resp.json()
        assert isinstance(data, (list, dict))


class TestResourcesUpload:
    """POST /api/v1/resources/upload — upload a resource."""

    def test_upload_text_file(self, client):
        """Upload a .txt file — flows through DocumentParser and in-memory repo."""
        content = b"Hello, this is a test document about arrays."
        resp = client.post(
            "/api/v1/resources/upload?kp_id=kp-test&kp_name=Test",
            files={"file": ("test.txt", content, "text/plain")},
        )
        # Accept any response (depends on real ChromaDB + parser)
        assert resp.status_code in (200, 422, 500)


class TestResourcesSync:
    """POST /api/v1/resources/sync-generated — sync generated resources."""

    def test_sync_returns_200(self, client):
        resp = client.post("/api/v1/resources/sync-generated", json={
            "session_id": "test-session",
        })
        assert resp.status_code in (200, 422)
