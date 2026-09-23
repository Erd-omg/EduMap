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
        """Upload a .txt file — flows through DocumentParser and in-memory repo.

        Previously asserted only ``status_code in (200, 422, 500)`` — which
        accepts any outcome — and hung the suite: with no ``_embedding_model``
        on app state, ``DocumentParser`` loaded a real ``SentenceTransformer``
        in-process.  The fixture now supplies a stub model, and the assertions
        below pin the actual response contract.
        """
        content = b"Hello, this is a test document about arrays."
        # kp_id/kp_name are Form fields on the upload route (router.py:46-47),
        # not query params — the old test passed them in the URL, where they
        # were silently ignored (its `in (200, 422, 500)` assertion hid that).
        resp = client.post(
            "/api/v1/resources/upload",
            files={"file": ("test.txt", content, "text/plain")},
            data={"kp_id": "kp-test", "kp_name": "Test"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Contract the library view depends on.
        assert body["name"] == "test.txt"
        assert body["kp_id"] == "kp-test", (
            "kp_id must round-trip — a query param here would be ignored"
        )
        assert body["id"], "uploaded resource must have an id"


class TestResourcesSync:
    """POST /api/v1/resources/sync-generated — sync generated resources."""

    def test_sync_returns_200(self, client):
        resp = client.post("/api/v1/resources/sync-generated", json={
            "session_id": "test-session",
        })
        assert resp.status_code in (200, 422)
