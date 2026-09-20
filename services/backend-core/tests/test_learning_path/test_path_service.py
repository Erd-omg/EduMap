"""Tests for PathService — topological sort and personalized path computation.

Uses mocked KnowledgePointRepository and EdgeRepository.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.learning_path.path_service import PathService


def _mock_graph(nodes: list, edges: list) -> tuple[AsyncMock, AsyncMock]:
    """Create mock repositories that return the given graph."""
    kp_repo = AsyncMock()
    kp_repo.get_course_graph = AsyncMock(
        return_value=MagicMock(nodes=nodes, edges=edges)
    )

    edge_repo = AsyncMock()
    return kp_repo, edge_repo


def _mk_node(
    id: str, name: str = "",
    difficulty: int = 1, description: str = "",
) -> MagicMock:
    node = MagicMock()
    node.id = id
    node.name = name or id
    node.difficulty = difficulty
    node.description = description or f"Description for {id}"
    return node


def _mk_edge(source: str, target: str, relation: str = "PREREQUISITE_OF") -> MagicMock:
    edge = MagicMock()
    edge.source = source
    edge.target = target
    edge.relation_type = relation
    return edge


class TestTopoSort:
    """Kahn topological sort — core deterministic algorithm."""

    @pytest.mark.asyncio
    async def test_linear_chain(self) -> None:
        """Linear A→B→C produces [A, B, C]."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A"), _mk_node("B"), _mk_node("C")],
            edges=[_mk_edge("A", "B"), _mk_edge("B", "C")],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        ids = [n.kp_id for n in result.nodes]
        assert ids == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_branching_dag(self) -> None:
        """Diamond shape: A→(B,C)→D."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A"), _mk_node("B"), _mk_node("C"), _mk_node("D")],
            edges=[_mk_edge("A", "B"), _mk_edge("A", "C"), _mk_edge("B", "D"), _mk_edge("C", "D")],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        ids = [n.kp_id for n in result.nodes]
        # A must be first, D must be last, B and C in any order
        assert ids[0] == "A"
        assert ids[-1] == "D"
        assert set(ids[1:-1]) == {"B", "C"}

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_path(self) -> None:
        """Empty course graph returns empty path."""
        kp_repo, edge_repo = _mock_graph(nodes=[], edges=[])
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        assert result.nodes == []

    @pytest.mark.asyncio
    async def test_single_node(self) -> None:
        """Single node with no edges."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A")],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        assert len(result.nodes) == 1
        assert result.nodes[0].kp_id == "A"

    @pytest.mark.asyncio
    async def test_disconnected_nodes(self) -> None:
        """Disconnected nodes appear after connected ones in topo sort."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A"), _mk_node("B"), _mk_node("C")],
            edges=[_mk_edge("A", "B")],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        ids = [n.kp_id for n in result.nodes]
        # A must come before B; C can be anywhere
        assert ids.index("A") < ids.index("B")


class TestProgressTracking:
    """Progress recording and status determination."""

    @pytest.mark.asyncio
    async def test_record_progress(self) -> None:
        """Recording progress creates an entry."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A")],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.record_progress("user-1", "course-1", "A", "completed", 0.9)
        assert "user-1" in svc._progress
        assert len(svc._progress["user-1"]) == 1
        assert svc._progress["user-1"][0].kp_id == "A"

    @pytest.mark.asyncio
    async def test_progress_affects_path_status(self) -> None:
        """Completed nodes show as 'completed' in the path."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A"), _mk_node("B")],
            edges=[_mk_edge("A", "B")],
        )
        svc = PathService(kp_repo, edge_repo)

        await svc.record_progress("user-1", "course-1", "A", "completed", 1.0)

        result = await svc.get_personalized_path("course-1", "user-1")
        statuses = {n.kp_id: n.status for n in result.nodes}
        assert statuses["A"] == "completed"

    @pytest.mark.asyncio
    async def test_progress_not_shared_across_users(self) -> None:
        """Progress for user-1 doesn't affect user-2's path."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A")],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        await svc.record_progress("user-1", "course-1", "A", True, 1.0)

        result = await svc.get_personalized_path("course-1", "user-2")
        statuses = {n.kp_id: n.status for n in result.nodes}
        assert "A" not in statuses or statuses["A"] != "completed"


class TestPersonalizedPath:
    """Full personalized path computation."""

    @pytest.mark.asyncio
    async def test_basic_path_structure(self) -> None:
        """get_personalized_path returns correct structure."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A", difficulty=1), _mk_node("B", difficulty=2)],
            edges=[_mk_edge("A", "B")],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        assert result.course_id == "course-1"
        assert result.user_id == "user-1"
        assert result.total_count == 2
        assert result.progress_percent == 0.0

    @pytest.mark.asyncio
    async def test_path_with_prerequisites_met(self) -> None:
        """Prerequisites_met flag is correct for each node."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A"), _mk_node("B", "B")],
            edges=[_mk_edge("A", "B")],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        prereqs = {n.kp_id: n.prerequisites_met for n in result.nodes}
        assert prereqs["A"] is True  # no prereqs, trivially met
        # B has A as prereq, but A is not completed
        # (In the mock, no progress → prerequisites not met)
        assert prereqs["B"] is False

    @pytest.mark.asyncio
    async def test_path_node_fields(self) -> None:
        """Each node has the expected fields populated."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A", difficulty=3)],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        result = await svc.get_personalized_path("course-1", "user-1")
        node = result.nodes[0]
        assert node.kp_id == "A"
        assert node.difficulty == 3
        assert isinstance(node.prerequisites, list)
        assert isinstance(node.recommended_content_types, list)


class TestGetNextRecommendation:
    """Next-step recommendation logic."""

    @pytest.mark.asyncio
    async def test_next_recommendation_from_ready_nodes(self) -> None:
        """get_next_recommendation returns a ready node."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A")],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        rec = await svc.get_next_recommendation("course-1", "user-1")
        assert rec is not None
        assert rec.next_kp_id == "A"
        assert rec.recommended_content_type in ("explanation", "exercise")

    @pytest.mark.asyncio
    async def test_no_recommendation_when_all_done(self) -> None:
        """No recommendation when all nodes are completed."""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A")],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)
        await svc.record_progress("user-1", "course-1", "A", "completed", 1.0)

        rec = await svc.get_next_recommendation("course-1", "user-1")
        assert rec is None

    @pytest.mark.asyncio
    async def test_next_recommendation_empty_graph(self) -> None:
        """Empty graph returns no recommendation."""
        kp_repo, edge_repo = _mock_graph(nodes=[], edges=[])
        svc = PathService(kp_repo, edge_repo)

        rec = await svc.get_next_recommendation("course-1", "user-1")
        assert rec is None


class TestForgettingDrivenRecommendation:
    """遗忘曲线接入再推荐 —— 闭合"评估 → 遗忘预测 → 再推荐"链路。"""

    @pytest.mark.asyncio
    async def test_forgotten_completed_node_resurfaces(self) -> None:
        """已完成但回忆概率极低的知识点应重新进入候选集。"""
        kp_repo, edge_repo = _mock_graph(nodes=[_mk_node("A")], edges=[])
        svc = PathService(kp_repo, edge_repo)
        await svc.record_progress("user-1", "course-1", "A", "completed", 1.0)

        # 无遗忘信息 → 全部完成，无推荐
        assert await svc.get_next_recommendation("course-1", "user-1") is None

        # 注入低回忆概率 → 该节点重新被推荐
        rec = await svc.get_next_recommendation(
            "course-1", "user-1", forgetting_map={"A": 0.2},
        )
        assert rec is not None
        assert rec.next_kp_id == "A"
        assert "尽快复习" in rec.reason

    @pytest.mark.asyncio
    async def test_forgetting_outranks_plain_candidate(self) -> None:
        """紧急遗忘的节点应压过同等条件下的普通候选。"""
        kp_repo, edge_repo = _mock_graph(
            nodes=[_mk_node("A", difficulty=3), _mk_node("B", difficulty=3)],
            edges=[],
        )
        svc = PathService(kp_repo, edge_repo)

        rec = await svc.get_next_recommendation(
            "course-1", "user-1", forgetting_map={"B": 0.1},
        )
        assert rec is not None
        assert rec.next_kp_id == "B"

    @pytest.mark.asyncio
    async def test_healthy_recall_adds_no_boost(self) -> None:
        """回忆概率健康时不应产生遗忘加成或复习话术。"""
        kp_repo, edge_repo = _mock_graph(nodes=[_mk_node("A")], edges=[])
        svc = PathService(kp_repo, edge_repo)

        rec = await svc.get_next_recommendation(
            "course-1", "user-1", forgetting_map={"A": 0.95},
        )
        assert rec is not None
        assert "复习" not in rec.reason

    @pytest.mark.asyncio
    async def test_last_kp_excluded_even_if_forgotten(self) -> None:
        """刚学完的知识点不应立即被推荐复习（避免原地打转）。"""
        kp_repo, edge_repo = _mock_graph(nodes=[_mk_node("A")], edges=[])
        svc = PathService(kp_repo, edge_repo)

        rec = await svc.get_next_recommendation(
            "course-1", "user-1", last_kp_id="A", forgetting_map={"A": 0.1},
        )
        assert rec is None

    def test_forget_factor_thresholds(self) -> None:
        """紧迫度映射覆盖未知态与三个召回区间。"""
        assert PathService._forget_factor(None) == 0.0     # 从未学习
        assert PathService._forget_factor(0.1) == 1.0      # < 0.3 紧急
        assert PathService._forget_factor(0.5) == 0.7      # < 0.6 告警
        assert PathService._forget_factor(0.9) == 0.0      # 健康

    def test_priority_score_stays_bounded(self) -> None:
        """权重归一化后优先级分数仍落在 [0, 1]。"""
        svc = PathService(AsyncMock(), AsyncMock())
        node = MagicMock()
        node.kp_id = "A"
        node.name = "A"
        node.status = "ready"
        node.difficulty = 3
        node.prerequisites_met = True

        worst = svc._priority_score(node, 3.0, set(), {})
        best = svc._priority_score(node, 3.0, {"A"}, {"A": 0.1})
        assert 0.0 <= worst <= 1.0
        assert 0.0 <= best <= 1.0
        assert best == pytest.approx(1.0)
        assert best > worst


class TestGetContentStyleSuggestion:
    """Content style suggestions based on user profile."""

    def test_visual_style_suggests_visualization(self) -> None:
        """Visual learners get visualization content type.

        ``get_content_type_suggestion()`` is a synchronous static method
        that does not need a service instance or repository mocks.
        """
        suggestion = PathService.get_content_type_suggestion(
            profile={"interaction_style": {"visual": 0.8, "textual": 0.2}},
        )
        assert suggestion is not None
        assert "visualization" in suggestion.content_types
        assert suggestion.dominant_style == "visual"

    def test_interactive_style_suggests_exercise(self) -> None:
        """Interactive learners get exercise content type."""
        suggestion = PathService.get_content_type_suggestion(
            profile={"interaction_style": {"interactive": 0.9}},
        )
        assert suggestion is not None
        assert "exercise" in suggestion.content_types
        assert suggestion.dominant_style == "interactive"

    def test_no_profile_uses_explanation(self) -> None:
        """No profile returns default explanation type."""
        suggestion = PathService.get_content_type_suggestion(profile=None)
        assert suggestion is not None
        assert "explanation" in suggestion.content_types


class TestRecommendationPersistence:
    """save_recommendation / get_recommendation_history 落库行为。"""

    @pytest.mark.asyncio
    async def test_save_without_db_pool_is_noop(self) -> None:
        """无数据库连接池时保存静默跳过，不抛错。"""
        kp_repo, edge_repo = _mock_graph(nodes=[], edges=[])
        svc = PathService(kp_repo, edge_repo)

        rec = MagicMock()
        rec.next_kp_id = "A"
        await svc.save_recommendation("u1", "c1", rec)  # 不应抛错

    @pytest.mark.asyncio
    async def test_save_executes_ddl_then_insert(self) -> None:
        """有连接池时：先建表（幂等 DDL）再插入推荐记录。"""
        from contextlib import asynccontextmanager

        conn = AsyncMock()

        @asynccontextmanager
        async def fake_acquire():
            yield conn

        pool = MagicMock()
        pool.acquire = fake_acquire
        svc = PathService(AsyncMock(), AsyncMock(), db_pool=pool)

        rec = MagicMock()
        rec.next_kp_id = "kp-9"
        rec.next_kp_name = "九、指针进阶"
        rec.reason = "前置已掌握"
        rec.recommended_content_type = "explanation"
        rec.estimated_session_min = 25

        await svc.save_recommendation("u1", "c1", rec, source="next")

        assert conn.execute.await_count == 2
        first_sql = conn.execute.await_args_list[0].args[0]
        second_sql = conn.execute.await_args_list[1].args[0]
        assert "CREATE TABLE IF NOT EXISTS path_recommendations" in first_sql
        assert "INSERT INTO path_recommendations" in second_sql
        insert_args = conn.execute.await_args_list[1].args
        assert insert_args[1] == "u1"
        assert insert_args[2] == "c1"
        assert insert_args[3] == "kp-9"
        assert insert_args[8] == "next"

    @pytest.mark.asyncio
    async def test_ddl_executed_only_once(self) -> None:
        """建表 DDL 只执行一次：第二次保存不再发 DDL，只发 INSERT。"""
        from contextlib import asynccontextmanager

        conn = AsyncMock()

        @asynccontextmanager
        async def fake_acquire():
            yield conn

        pool = MagicMock()
        pool.acquire = fake_acquire
        svc = PathService(AsyncMock(), AsyncMock(), db_pool=pool)
        rec = MagicMock()
        rec.next_kp_id = "kp-9"

        await svc.save_recommendation("u1", "c1", rec, source="next")
        await svc.save_recommendation("u1", "c1", rec, source="after_progress")

        # 2 次保存 = 1 次 DDL + 2 次 INSERT
        assert conn.execute.await_count == 3
        assert "CREATE TABLE IF NOT EXISTS" in conn.execute.await_args_list[0].args[0]
        for call in conn.execute.await_args_list[1:]:
            assert "INSERT INTO path_recommendations" in call.args[0]

    @pytest.mark.asyncio
    async def test_history_serializes_created_at(self) -> None:
        """历史查询返回字典列表，created_at 序列化为 ISO 字符串。"""
        from contextlib import asynccontextmanager
        from datetime import datetime, timezone

        row = {
            "kp_id": "kp-9",
            "kp_name": "指针进阶",
            "reason": "前置已掌握",
            "recommended_content_type": "explanation",
            "estimated_session_min": 25,
            "source": "next",
            "created_at": datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc),
        }

        @asynccontextmanager
        async def fake_acquire():
            yield AsyncMock(fetch=AsyncMock(return_value=[row]))

        pool = MagicMock()
        pool.acquire = fake_acquire
        svc = PathService(AsyncMock(), AsyncMock(), db_pool=pool)

        history = await svc.get_recommendation_history("u1", limit=5)
        assert len(history) == 1
        assert history[0]["kp_id"] == "kp-9"
        assert isinstance(history[0]["created_at"], str)
        assert history[0]["created_at"].startswith("2026-09-20T10:00")

    @pytest.mark.asyncio
    async def test_history_without_db_pool_returns_empty(self) -> None:
        """无连接池时历史查询返回空列表。"""
        kp_repo, edge_repo = _mock_graph(nodes=[], edges=[])
        svc = PathService(kp_repo, edge_repo)

        assert await svc.get_recommendation_history("u1") == []
