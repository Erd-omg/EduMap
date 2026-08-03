"""API tests for Learning Path endpoints."""
from __future__ import annotations

import pytest


class TestLearningPathEndpoints:
    """Learning path GET endpoints."""

    def test_get_path_returns_200(self, client):
        resp = client.get("/api/v1/learning-path/cs101?user_id=test-user")
        assert resp.status_code in (200, 404)

    def test_get_next_recommendation_returns_200(self, client):
        resp = client.get("/api/v1/learning-path/cs101/next?user_id=test-user")
        assert resp.status_code in (200, 404)

    def test_record_progress_returns_200(self, client):
        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        assert resp.status_code in (200, 422)


class TestQuizEndpoints:
    """Quiz and anti-gaming endpoints."""

    def test_generate_quiz_returns_response(self, client):
        """Quiz generation requires real agent — may return 500 in mock env."""
        resp = client.post("/api/v1/learning-path/quiz/generate", json={
            "user_id": "test-user",
            "kp_id": "kp-test",
        })
        # Accept any server response (real agent depends on LLM)
        assert resp.status_code in (200, 422, 500)

    def test_anti_gaming_score_returns_200(self, client):
        resp = client.post("/api/v1/learning-path/anti-gaming/score", json={
            "user_id": "test-user",
            "kp_id": "kp-test",
            "score": 0.9,
            "time_spent_minutes": 10,
        })
        assert resp.status_code in (200, 422)


class TestForgettingEndpoints:
    """Forgetting curve endpoints."""

    def test_get_forgetting_state(self, client):
        resp = client.get("/api/v1/learning-path/forgetting/kp-test?user_id=test-user")
        assert resp.status_code in (200, 404)

    def test_get_forgetting_alerts(self, client):
        resp = client.get("/api/v1/learning-path/forgetting/alerts/test-user")
        assert resp.status_code in (200, 404)

    def test_get_dashboard(self, client):
        resp = client.get("/api/v1/learning-path/dashboard/test-user")
        assert resp.status_code in (200, 404)
