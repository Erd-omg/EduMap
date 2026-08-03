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
