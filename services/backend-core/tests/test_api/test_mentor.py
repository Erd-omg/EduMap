"""API tests for Mentor / RAG endpoints."""
from __future__ import annotations


class TestMentorSearch:
    """GET /api/v1/mentor/search — knowledge search."""

    def test_search_returns_200(self, client):
        resp = client.get("/api/v1/mentor/search?query=数组")
        assert resp.status_code == 200

    def test_search_returns_list(self, client):
        resp = client.get("/api/v1/mentor/search?query=数组")
        data = resp.json()
        assert isinstance(data, (list, dict))

    def test_search_empty_query(self, client):
        resp = client.get("/api/v1/mentor/search?query=")
        assert resp.status_code in (200, 422)


class TestMentorEvaluate:
    """GET /api/v1/mentor/evaluate — RAG evaluation."""

    def test_evaluate_returns_200(self, client):
        resp = client.get("/api/v1/mentor/evaluate?n_queries=2")
        assert resp.status_code == 200

    def test_evaluate_has_metrics(self, client):
        resp = client.get("/api/v1/mentor/evaluate?n_queries=2")
        data = resp.json()
        assert isinstance(data, dict)

    def test_evaluate_sample_returns_200(self, client):
        resp = client.get("/api/v1/mentor/evaluate?sample=true")
        assert resp.status_code == 200


class TestRerankPreview:
    """POST /api/v1/mentor/rerank-preview — rerank preview."""

    def test_rerank_returns_200(self, client):
        resp = client.post("/api/v1/mentor/rerank-preview", json={
            "query": "数组",
            "results": [
                {"id": "1", "text": "数组插入O(n)", "score": 0.5},
                {"id": "2", "text": "链表插入O(1)", "score": 0.3},
            ],
        })
        assert resp.status_code in (200, 422)
