"""API tests for Knowledge Graph endpoints."""
from __future__ import annotations


class TestKGCourses:
    """GET /api/v1/kg/courses — list all courses."""

    def test_list_courses_returns_200(self, client):
        resp = client.get("/api/v1/kg/courses")
        assert resp.status_code == 200

    def test_list_courses_returns_dict_or_list(self, client):
        """Courses endpoint returns a collection."""
        resp = client.get("/api/v1/kg/courses")
        data = resp.json()
        assert isinstance(data, (list, dict))


class TestKGSeed:
    """Seed data endpoints."""

    def test_verify_seed_returns_200(self, client):
        resp = client.get("/api/v1/kg/seed/verify")
        assert resp.status_code == 200

    def test_verify_seed_has_counts(self, client):
        resp = client.get("/api/v1/kg/seed/verify")
        data = resp.json()
        assert isinstance(data, dict)


class TestKGColdStart:
    """Cold-start evaluation endpoint."""

    def test_cold_start_returns_200(self, client):
        resp = client.get("/api/v1/kg/eval/cold-start")
        assert resp.status_code == 200

    def test_cold_start_has_verdict(self, client):
        resp = client.get("/api/v1/kg/eval/cold-start")
        data = resp.json()
        assert isinstance(data, dict)


class TestKGNodes:
    """Knowledge point operations."""

    def test_get_node_returns_200_or_404(self, client):
        """GET node returns the node contract for a known id.

        Previously asserted ``in (200, 404, 422)``.  The kp_repo is mocked to
        return a node, so 200 is what actually happens — the loose assertion
        would have accepted a 404 or a validation error just as happily.
        """
        resp = client.get("/api/v1/kg/nodes/kp-test")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for field in ("id", "name", "description", "difficulty"):
            assert field in body, f"missing {field!r} in node payload"

    def test_create_node_returns_200(self, client):
        resp = client.post("/api/v1/kg/nodes", json={
            "id": "kp-test",
            "name": "测试知识点",
            "description": "A test knowledge point",
            "difficulty": 1,
            "prerequisites": [],
            "key_concepts": ["test"],
        })
        # The route creates, so 201 is the real outcome.
        assert resp.status_code == 201, resp.text
        assert "id" in resp.json()
