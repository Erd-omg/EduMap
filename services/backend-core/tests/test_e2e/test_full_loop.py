"""端到端链路测试：检索 → 推荐 → 答题（反游戏化加权）→ 遗忘更新 → 再推荐 → 通知聚合。

与单模块 API 测试不同，这里使用 **有状态的 fake 服务** 驱动完整业务闭环，
验证各环节之间的数据流（加权得分写入遗忘曲线、遗忘状态驱动再推荐、
推荐落库驱动通知聚合）真实贯通，而非各自孤立地返回 mock 值。
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from tests.test_api.conftest import _create_test_app

from src.learning_path.models import PathRecommendation, PersonalizedPath
from src.rag.models import RAGResult

USER = "e2e-user"
COURSE = "cs101"


# ── Stateful fakes ─────────────────────────────────────────────────────


class FakePathService:
    """有状态路径服务：记录进度/推荐落库，进度后切换推荐内容。"""

    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.progress_records: list[dict] = []
        self.last_forgetting_map: dict | None = None
        self.recs = {
            "next": PathRecommendation(
                next_kp_id="kp-arrays",
                next_kp_name="数组",
                reason="数据结构入门的第一站",
                recommended_content_type="explanation",
                estimated_session_min=25,
            ),
            "after_progress": PathRecommendation(
                next_kp_id="kp-linked-list",
                next_kp_name="链表",
                reason="数组已掌握，推荐学习线性结构的进阶",
                recommended_content_type="exercise",
                estimated_session_min=30,
            ),
        }

    async def get_next_recommendation(
        self, course_id, user_id, last_kp_id=None, profile=None, *, forgetting_map=None,
    ):
        self.last_forgetting_map = forgetting_map
        if self.progress_records:
            return self.recs["after_progress"]
        return self.recs["next"]

    async def save_recommendation(self, user_id, course_id, rec, *, source):
        self.saved.append({
            "user_id": user_id,
            "course_id": course_id,
            "kp_id": rec.next_kp_id,
            "kp_name": rec.next_kp_name,
            "reason": rec.reason,
            "recommended_content_type": str(rec.recommended_content_type),
            "estimated_session_min": rec.estimated_session_min,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    async def get_recommendation_history(self, user_id, limit=20):
        return list(reversed(self.saved))[:limit]

    async def record_progress(self, *, user_id, course_id, kp_id, status, quiz_score):
        self.progress_records.append({
            "user_id": user_id,
            "course_id": course_id,
            "kp_id": kp_id,
            "status": status,
            "quiz_score": quiz_score,
        })
        return PersonalizedPath(
            course_id=course_id,
            user_id=user_id,
            nodes=[],
            total_count=1,
            mastered_count=1,
            completed_count=1,
            progress_percent=100.0,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


class FakeForgettingService:
    """有状态遗忘服务：答题更新会反映到 recall map 与 alerts。"""

    def __init__(self) -> None:
        self.quiz_updates: list[tuple[str, str, float]] = []
        self.recall_map: dict[str, float] = {}
        self.reviews: list[tuple[str, str]] = []

    async def update_after_quiz(self, kp_id, user_id, score):
        self.quiz_updates.append((kp_id, user_id, score))
        # 答题（加权得分）会提升该 KP 的回忆概率
        self.recall_map[kp_id] = max(self.recall_map.get(kp_id, 0.0), score * 0.95)

    async def record_review(self, *, kp_id, user_id):
        self.reviews.append((kp_id, user_id))
        self.recall_map[kp_id] = self.recall_map.get(kp_id, 0.5) + 0.2
        from src.learning_path.forgetting_curve import ForgettingState

        state = ForgettingState(kp_id=kp_id, user_id=user_id)
        state.strength = 2.0
        state.review_count = 1
        return state

    async def get_recall_map(self, user_id):
        return dict(self.recall_map)

    async def get_state(self, kp_id, user_id):
        import time

        from src.learning_path.forgetting_curve import ForgettingState

        if kp_id not in self.recall_map:
            return None
        state = ForgettingState(kp_id=kp_id, user_id=user_id)
        state.strength = 2.0
        state.review_count = 1
        state.last_review_time = time.time()  # 已学过，否则路由视为无遗忘数据
        return state

    async def get_alerts(self, user_id):
        from src.learning_path.forgetting_curve import ALERT_RECALL_THRESHOLD

        alerts = []
        for kp_id, recall in self.recall_map.items():
            if recall < ALERT_RECALL_THRESHOLD:
                alerts.append({
                    "kp_id": kp_id,
                    "kp_name": kp_id.replace("kp-", ""),
                    "alert_level": "urgent" if recall < 0.3 else "warning",
                    "recall_probability": recall,
                    "review_count": 1,
                })
        return alerts


def _make_rag_service():
    """有状态 RAG mock：返回两组来源（kg + resource）的检索结果。"""
    rag = AsyncMock()
    rag.search = AsyncMock(return_value=[
        RAGResult(
            content="数组是线性表的基本实现，插入复杂度 O(n)。",
            source_type="neo4j",
            source_id="kp-arrays",
            source_name="数组",
            score=0.92,
        ),
        RAGResult(
            content="讲义：数组与链表对比，插入删除性能分析。",
            source_type="chroma",
            source_id="res-001",
            source_name="数据结构讲义.pdf",
            score=0.85,
        ),
    ])
    return rag


def _make_e2e_app():
    path = FakePathService()
    forgetting = FakeForgettingService()
    app = _create_test_app(
        rag_service=_make_rag_service(),
        path_service=path,
        forgetting_service=forgetting,
    )
    app.state._e2e_path = path
    app.state._e2e_forgetting = forgetting
    return app


# ── Tests ──────────────────────────────────────────────────────────────


class TestFullLearningLoop:
    """完整学习闭环：检索 → 推荐 → 答题 → 遗忘 → 再推荐 → 通知。"""

    def test_full_loop(self):
        """主链路：一次 HTTP 序列驱动全部环节并校验数据流贯通。"""
        app = _make_e2e_app()
        path: FakePathService = app.state._e2e_path
        fc: FakeForgettingService = app.state._e2e_forgetting

        with TestClient(app) as client:
            # ── 1. 知识检索（mentor search）─────────────────────────
            resp = client.get("/api/v1/mentor/search", params={"query": "数组 插入", "top_k": 5})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 2
            assert {r["id"] for r in data["results"]} == {"kp-arrays", "res-001"}

            # ── 2. 获取下一步推荐 ────────────────────────────────────
            resp = client.get(f"/api/v1/learning-path/{COURSE}/next", params={"user_id": USER})
            assert resp.status_code == 200
            rec = resp.json()
            assert rec["next_kp_id"] == "kp-arrays"
            # 推荐已落库，source="next"
            assert len(path.saved) == 1
            assert path.saved[0]["kp_id"] == "kp-arrays"
            assert path.saved[0]["source"] == "next"

            # ── 3. 答题提交（反游戏化加权 + 遗忘曲线更新 + 再推荐）──
            resp = client.post("/api/v1/learning-path/progress", json={
                "user_id": USER,
                "course_id": COURSE,
                "kp_id": "kp-arrays",
                "status": "completed",
                "quiz_score": 0.9,
                "difficulty": 3,
            })
            assert resp.status_code == 200
            body = resp.json()

            # 3a. 反游戏化：原始 0.9 → 加权 0.75（conftest mock 的因子）
            assert body["anti_gaming"]["raw_score"] == 0.9
            assert body["anti_gaming"]["weighted_score"] == 0.75

            # 3b. 遗忘曲线收到的是 **加权后** 的得分
            assert fc.quiz_updates == [("kp-arrays", USER, 0.75)]

            # 3c. 进度落库用的也是加权得分
            assert path.progress_records[0]["quiz_score"] == 0.75

            # 3d. 闭环产生了再推荐（after_progress），且注入了更新后的遗忘 map
            assert body["next_recommendation"]["next_kp_id"] == "kp-linked-list"
            assert path.last_forgetting_map == {"kp-arrays": 0.75 * 0.95}
            assert path.saved[-1]["source"] == "after_progress"

            # 3e. 响应携带遗忘状态摘要（元认知透明度）
            assert body["forgetting"]["kp_id"] == "kp-arrays"

            # ── 4. 推荐历史（最新在前）──────────────────────────────
            resp = client.get(f"/api/v1/learning-path/recommendations/{USER}")
            assert resp.status_code == 200
            history = resp.json()["recommendations"]
            assert len(history) == 2
            assert history[0]["kp_id"] == "kp-linked-list"  # 最新
            assert history[1]["kp_id"] == "kp-arrays"

            # ── 5. 通知聚合（review_reminder + next_step）───────────
            # 人为压低回忆概率，制造 urgent 预警
            fc.recall_map["kp-arrays"] = 0.25
            resp = client.get(f"/api/v1/learning-path/notifications/{USER}")
            assert resp.status_code == 200
            notes = resp.json()
            assert notes["enabled"] is True
            types = {n["type"] for n in notes["notifications"]}
            assert types == {"review_reminder", "next_step"}
            urgent = next(n for n in notes["notifications"] if n["type"] == "review_reminder")
            assert urgent["severity"] == "urgent"
            assert urgent["kp_id"] == "kp-arrays"

    def test_notifications_disabled_via_profile(self, monkeypatch):
        """profile 的 notifications_enabled=False 时通知聚合直接关闭。"""

        async def fake_fetch_profile(user_id: str):
            return {"notifications_enabled": False}

        monkeypatch.setattr(
            "src.learning_path.router._fetch_profile", fake_fetch_profile
        )
        app = _make_e2e_app()
        fc: FakeForgettingService = app.state._e2e_forgetting
        fc.recall_map["kp-arrays"] = 0.25  # 有 urgent 预警也不应推送

        with TestClient(app) as client:
            resp = client.get(f"/api/v1/learning-path/notifications/{USER}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["enabled"] is False
            assert data["notifications"] == []
            assert data["total"] == 0

    def test_progress_without_quiz_score_skips_forgetting(self):
        """无 quiz_score 的进度提交不触发反游戏化/遗忘更新。"""
        app = _make_e2e_app()
        path: FakePathService = app.state._e2e_path
        fc: FakeForgettingService = app.state._e2e_forgetting

        with TestClient(app) as client:
            resp = client.post("/api/v1/learning-path/progress", json={
                "user_id": USER,
                "course_id": COURSE,
                "kp_id": "kp-stack",
                "status": "in_progress",
            })
            assert resp.status_code == 200
            body = resp.json()

            assert "anti_gaming" not in body
            assert fc.quiz_updates == []
            assert path.progress_records[0]["quiz_score"] is None
            # 仍会产生再推荐（学习闭环不断）
            assert body["next_recommendation"]["next_kp_id"] == "kp-linked-list"
