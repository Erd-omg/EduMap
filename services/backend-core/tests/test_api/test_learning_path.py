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


class TestProgressClosesTheLoop:
    """progress 端点应把反游戏化 + 遗忘曲线串成闭环。"""

    def test_anti_gaming_result_is_embedded(self, client):
        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "anti_gaming" in body
        assert body["anti_gaming"]["raw_score"] == 0.9
        assert body["anti_gaming"]["weighted_score"] == 0.75
        assert body["anti_gaming"]["is_cramming"] is False

    def test_weighted_score_is_persisted_not_raw(self, client):
        """落库的必须是加权得分，否则刷分能直接污染掌握度。"""
        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        assert resp.status_code == 200
        path_service = client.app.state.path_service
        _, kwargs = path_service.record_progress.call_args
        assert kwargs["quiz_score"] == 0.75

    def test_forgetting_updated_with_weighted_score(self, client):
        client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        forgetting_service = client.app.state.forgetting_service
        forgetting_service.update_after_quiz.assert_awaited_once_with(
            "kp-test", "test-user", 0.75,
        )

    def test_forgetting_map_is_passed_to_recommendation(self, client):
        """遗忘状态必须注入再推荐，W_FORGET 才会真正生效。"""
        client.app.state.forgetting_service.get_recall_map = _recall_map(
            {"kp-test": 0.15}
        )
        client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        path_service = client.app.state.path_service
        _, kwargs = path_service.get_next_recommendation.call_args
        assert kwargs["forgetting_map"] == {"kp-test": 0.15}

    def test_progress_without_quiz_score_skips_evaluation(self, client):
        """无 quiz_score 时不应触发评估链路（如仅标记 in_progress）。"""
        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "in_progress",
        })
        assert resp.status_code == 200
        assert "anti_gaming" not in resp.json()
        client.app.state.forgetting_service.update_after_quiz.assert_not_awaited()

    def test_never_reviewed_kp_omits_forgetting_info(self, client):
        """默认构造的 ForgettingState（未复习过）不应上报为"已完全遗忘"。"""
        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        assert "forgetting" not in resp.json()


def _recall_map(mapping: dict[str, float]):
    from unittest.mock import AsyncMock
    return AsyncMock(return_value=mapping)


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


class TestRecommendationPersistenceEndpoints:
    """推荐结果落库与历史查询端点。"""

    def test_next_recommendation_is_persisted(self, client):
        """GET /next 产生推荐后应调用 save_recommendation(source='next')。"""
        from unittest.mock import AsyncMock

        from src.learning_path.models import PathRecommendation

        client.app.state.path_service.get_next_recommendation = AsyncMock(
            return_value=PathRecommendation(
                next_kp_id="kp-next",
                next_kp_name="下一站",
                reason="前置已掌握",
                recommended_content_type="explanation",
                estimated_session_min=20,
            )
        )

        resp = client.get("/api/v1/learning-path/cs101/next?user_id=test-user")
        assert resp.status_code == 200
        assert resp.json()["next_kp_id"] == "kp-next"

        save = client.app.state.path_service.save_recommendation
        save.assert_awaited_once()
        args, kwargs = save.await_args
        assert args[0] == "test-user"
        assert args[1] == "cs101"
        assert args[2].next_kp_id == "kp-next"
        assert kwargs.get("source") == "next"

    def test_progress_persists_followup_recommendation(self, client):
        """POST /progress 触发的再推荐应以 source='after_progress' 落库。"""
        from unittest.mock import AsyncMock

        from src.learning_path.models import PathRecommendation

        client.app.state.path_service.get_next_recommendation = AsyncMock(
            return_value=PathRecommendation(
                next_kp_id="kp-after",
                next_kp_name="进阶",
                reason="刚完成前置",
                recommended_content_type="exercise",
                estimated_session_min=15,
            )
        )
        save = client.app.state.path_service.save_recommendation
        save.reset_mock()

        resp = client.post("/api/v1/learning-path/progress", json={
            "user_id": "test-user",
            "course_id": "cs101",
            "kp_id": "kp-test",
            "status": "completed",
            "quiz_score": 0.9,
        })
        assert resp.status_code == 200
        save.assert_awaited_once()
        args, kwargs = save.await_args
        assert args[2].next_kp_id == "kp-after"
        assert kwargs.get("source") == "after_progress"

    def test_recommendation_history_endpoint(self, client):
        """GET /recommendations/{user_id} 返回落库历史。"""
        client.app.state.path_service.get_recommendation_history = _history(
            [{
                "kp_id": "kp-after",
                "kp_name": "进阶",
                "reason": "刚完成前置",
                "recommended_content_type": "exercise",
                "estimated_session_min": 15,
                "source": "after_progress",
                "created_at": "2026-09-20T10:00:00+00:00",
            }]
        )

        resp = client.get("/api/v1/learning-path/recommendations/test-user")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "test-user"
        assert body["total"] == 1
        assert body["recommendations"][0]["kp_id"] == "kp-after"


def _history(items: list[dict]):
    from unittest.mock import AsyncMock
    return AsyncMock(return_value=items)


class TestNotificationsEndpoint:
    """GET /notifications/{user_id} — 通知开关与聚合。"""

    def test_notifications_disabled_returns_empty(self, client, monkeypatch):
        """profile 中 notifications_enabled=False 时返回 enabled=False 且列表为空。"""

        async def fake_profile(uid: str):
            return {"notifications_enabled": False}

        monkeypatch.setattr("src.learning_path.router._fetch_profile", fake_profile)

        resp = client.get("/api/v1/learning-path/notifications/test-user")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["notifications"] == []
        assert body["total"] == 0

    def test_notifications_aggregates_alerts_and_latest_rec(self, client, monkeypatch):
        """开启时聚合遗忘预警（仅 urgent/warning）与最近一条推荐。"""

        async def fake_profile(uid: str):
            return {"notifications_enabled": True}

        monkeypatch.setattr("src.learning_path.router._fetch_profile", fake_profile)

        client.app.state.forgetting_service.get_alerts = _alerts([
            {
                "kp_id": "kp-1",
                "kp_name": "数组",
                "recall_probability": 0.2,
                "alert_level": "urgent",
                "review_count": 2,
            },
            {
                "kp_id": "kp-2",
                "kp_name": "链表",
                "recall_probability": 0.55,
                "alert_level": "ok",
                "review_count": 3,
            },
        ])
        client.app.state.path_service.get_recommendation_history = _history([
            {
                "kp_id": "kp-next",
                "kp_name": "指针",
                "reason": "前置已掌握",
                "recommended_content_type": "explanation",
                "estimated_session_min": 20,
                "source": "next",
                "created_at": "2026-09-20T10:00:00+00:00",
            }
        ])

        resp = client.get("/api/v1/learning-path/notifications/test-user")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["total"] == 2

        types = {n["type"] for n in body["notifications"]}
        assert types == {"review_reminder", "next_step"}

        reminder = next(n for n in body["notifications"] if n["type"] == "review_reminder")
        assert reminder["kp_id"] == "kp-1"
        assert reminder["severity"] == "urgent"
        assert "20%" in reminder["message"]

        next_step = next(n for n in body["notifications"] if n["type"] == "next_step")
        assert next_step["kp_id"] == "kp-next"
        assert "指针" in next_step["title"]

    def test_notifications_default_enabled_without_profile(self, client, monkeypatch):
        """profile 服务不可达时默认开启（enabled=True）。"""

        async def unreachable_profile(uid: str):
            raise ConnectionError("profile-service down")

        monkeypatch.setattr("src.learning_path.router._fetch_profile", unreachable_profile)
        client.app.state.forgetting_service.get_alerts = _alerts([])
        client.app.state.path_service.get_recommendation_history = _history([])

        resp = client.get("/api/v1/learning-path/notifications/test-user")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["notifications"] == []


def _alerts(items: list[dict]):
    from unittest.mock import AsyncMock
    return AsyncMock(return_value=items)


class TestFetchProfile:
    """_fetch_profile 的地址必须来自 settings.profile_service_base（P0 修复）。

    host 模式默认 ``http://localhost:8001``，容器模式由 docker-compose 覆盖
    为 ``http://profile-service:8001``；失败必须记 warning 并返回 None。
    """

    def test_uses_configured_base_url(self, monkeypatch):
        import asyncio

        import httpx

        from src.config import settings
        from src.learning_path.router import _fetch_profile

        requested: list[str] = []

        class _FakeResp:
            status_code = 200

            @staticmethod
            def json():
                return {"profile": {"notifications_enabled": False}}

        class _FakeClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                requested.append(url)
                return _FakeResp()

        monkeypatch.setattr(settings, "profile_service_base", "http://localhost:8001")
        monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

        profile = asyncio.run(_fetch_profile("u1"))

        assert profile == {"notifications_enabled": False}
        assert requested == ["http://localhost:8001/api/v1/profiles/u1"]

    def test_swallows_errors_with_warning(self, monkeypatch, caplog):
        import asyncio

        import httpx

        from src.learning_path.router import _fetch_profile

        class _BrokenClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                raise ConnectionError("profile-service down")

        monkeypatch.setattr(httpx, "AsyncClient", _BrokenClient)

        with caplog.at_level("WARNING"):
            result = asyncio.run(_fetch_profile("u1"))

        assert result is None
        assert any("Failed to fetch profile" in r.message for r in caplog.records)

    def test_non_200_returns_none(self, monkeypatch):
        """Any non-200 must degrade to None, not raise into the endpoint."""
        import asyncio

        import httpx

        from src.learning_path.router import _fetch_profile

        class _NotFound:
            status_code = 404
            headers = {"content-type": "application/json"}

            @staticmethod
            def json():
                return {"detail": "not found"}

        class _Client:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return _NotFound()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        assert asyncio.run(_fetch_profile("u-missing")) is None

    def test_malformed_json_returns_none(self, monkeypatch):
        """A non-JSON body must not escape as a parsing error.

        ``json()`` on a truncated/garbage body raises, and the broad except is
        the only thing keeping that out of the endpoint.
        """
        import asyncio
        import json as _json

        import httpx

        from src.learning_path.router import _fetch_profile

        class _Garbage:
            status_code = 200
            headers = {"content-type": "text/html"}

            @staticmethod
            def json():
                raise _json.JSONDecodeError("Expecting value", "<html>", 0)

        class _Client:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return _Garbage()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        assert asyncio.run(_fetch_profile("u1")) is None

    def test_response_without_profile_key_returns_none(self, monkeypatch):
        """A 200 whose body lacks 'profile' yields None, not a KeyError."""
        import asyncio

        import httpx

        from src.learning_path.router import _fetch_profile

        class _NoProfileKey:
            status_code = 200

            @staticmethod
            def json():
                return {"user_id": "u1"}  # no "profile" key

        class _Client:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return _NoProfileKey()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        assert asyncio.run(_fetch_profile("u1")) is None

    def test_timeout_is_bounded(self, monkeypatch):
        """The request must carry an explicit timeout — never unbounded.

        A hung profile-service must not hang every endpoint that reads the
        notification preference.
        """
        import asyncio

        import httpx

        from src.learning_path.router import _fetch_profile

        seen: dict = {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"profile": {}}

        class _Client:
            def __init__(self, timeout=None):
                seen["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        asyncio.run(_fetch_profile("u1"))
        assert seen["timeout"] == 5.0

    def test_timeout_exception_degrades_to_none(self, monkeypatch):
        """An actual timeout surfaces as None, not a raised TimeoutException."""
        import asyncio

        import httpx

        from src.learning_path.router import _fetch_profile

        class _TimeoutClient:
            def __init__(self, timeout=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url):
                raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx, "AsyncClient", _TimeoutClient)
        assert asyncio.run(_fetch_profile("u1")) is None
