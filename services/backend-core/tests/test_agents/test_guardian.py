"""Tests for GuardianAgent — pure deterministic DAG validation.

No mocks or external dependencies needed.
"""

from __future__ import annotations

import pytest

from src.agents.guardian.agent import GuardianAgent
from src.agents.models import KnowledgeUnit


@pytest.fixture
def agent() -> GuardianAgent:
    return GuardianAgent()


def mk_ku(
    id: str,
    name: str = "",
    difficulty: int = 1,
    prerequisites: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=id,
        name=name or id,
        description=f"Description for {id}",
        difficulty=difficulty,
        prerequisites=prerequisites or [],
        key_concepts=[],
    )


class TestCycleDetection:
    """DAG cycle detection — DFS-based algorithm."""

    @pytest.mark.asyncio
    async def test_no_cycles_linear_chain(self, agent: GuardianAgent) -> None:
        """A linear chain A→B→C has no cycles."""
        units = [
            mk_ku("kp-c", prerequisites=["kp-b"]),
            mk_ku("kp-b", prerequisites=["kp-a"]),
            mk_ku("kp-a"),
        ]
        result = await agent.run(units)
        assert result.is_valid is True
        assert result.cycle_details == []

    @pytest.mark.asyncio
    async def test_no_cycles_branching(self, agent: GuardianAgent) -> None:
        """Branching DAG: A→B, A→C, B→D, C→D (diamond shape)."""
        units = [
            mk_ku("kp-d", prerequisites=["kp-b", "kp-c"]),
            mk_ku("kp-b", prerequisites=["kp-a"]),
            mk_ku("kp-c", prerequisites=["kp-a"]),
            mk_ku("kp-a"),
        ]
        result = await agent.run(units)
        assert result.is_valid is True
        assert result.cycle_details == []

    @pytest.mark.asyncio
    async def test_simple_cycle_a_to_b_to_a(self, agent: GuardianAgent) -> None:
        """Direct 2-node cycle A→B→A is detected."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-b"]),
            mk_ku("kp-b", prerequisites=["kp-a"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is False
        assert len(result.cycle_details) >= 1

    @pytest.mark.asyncio
    async def test_complex_cycle_multi_node(self, agent: GuardianAgent) -> None:
        """3-node cycle A→B→C→A is detected."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-c"]),
            mk_ku("kp-b", prerequisites=["kp-a"]),
            mk_ku("kp-c", prerequisites=["kp-b"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is False
        assert len(result.cycle_details) >= 1

    @pytest.mark.asyncio
    async def test_self_loop_detected(self, agent: GuardianAgent) -> None:
        """A node that depends on itself creates a trivial cycle."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-a"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is False
        assert len(result.cycle_details) >= 1

    @pytest.mark.asyncio
    async def test_empty_knowledge_units(self, agent: GuardianAgent) -> None:
        """Empty input returns valid (vacuously true)."""
        result = await agent.run([])
        assert result.is_valid is True
        assert result.cycle_details == []

    @pytest.mark.asyncio
    async def test_single_node_no_edges(self, agent: GuardianAgent) -> None:
        """A single isolated node is always valid."""
        units = [mk_ku("kp-a")]
        result = await agent.run(units)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_disconnected_components(self, agent: GuardianAgent) -> None:
        """Multiple independent sub-graphs: valid if each is acyclic."""
        units = [
            mk_ku("kp-a"),
            mk_ku("kp-b", prerequisites=["kp-a"]),
            mk_ku("kp-c"),
            mk_ku("kp-d", prerequisites=["kp-c"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is True
        assert result.cycle_details == []


class TestDifficultyMonotonicity:
    """Prerequisite must not be harder than dependent."""

    @pytest.mark.asyncio
    async def test_monotonic_increasing(self, agent: GuardianAgent) -> None:
        """Difficulty increases along dependency chain."""
        units = [
            mk_ku("kp-a", difficulty=1),
            mk_ku("kp-b", difficulty=2, prerequisites=["kp-a"]),
            mk_ku("kp-c", difficulty=3, prerequisites=["kp-b"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is True
        assert result.monotonicity_violations == []

    @pytest.mark.asyncio
    async def test_monotonicity_violation_detected(self, agent: GuardianAgent) -> None:
        """A harder prerequisite (3→1) is flagged."""
        units = [
            # kp-a (easy, difficulty=1) depends on kp-b (hard, difficulty=3) — violation!
            mk_ku("kp-a", difficulty=1, prerequisites=["kp-b"]),
            mk_ku("kp-b", difficulty=3),
        ]
        result = await agent.run(units)
        assert result.is_valid is True  # monotonicity is a warning, not fatal
        assert len(result.monotonicity_violations) >= 1
        assert "kp-b" in result.monotonicity_violations[0]
        assert "kp-a" in result.monotonicity_violations[0]

    @pytest.mark.asyncio
    async def test_plateau_allowed(self, agent: GuardianAgent) -> None:
        """Same difficulty level is allowed (not a violation)."""
        units = [
            mk_ku("kp-a", difficulty=2),
            mk_ku("kp-b", difficulty=2, prerequisites=["kp-a"]),
        ]
        result = await agent.run(units)
        assert result.monotonicity_violations == []

    @pytest.mark.asyncio
    async def test_violation_only_for_direct_prereqs(self, agent: GuardianAgent) -> None:
        """Only direct prerequisite relationships are checked.

        kp-c (diff=2) → kp-a (diff=1): prereq is easier, OK.
        kp-b (diff=5) → kp-a (diff=1): prereq is easier, OK.
        No violations in either case.
        """
        units = [
            mk_ku("kp-a", difficulty=1),
            mk_ku("kp-b", difficulty=5, prerequisites=["kp-a"]),
            mk_ku("kp-c", difficulty=2, prerequisites=["kp-a"]),
        ]
        result = await agent.run(units)
        assert len(result.monotonicity_violations) == 0


class TestDanglingReferences:
    """Prerequisite must reference an existing node."""

    @pytest.mark.asyncio
    async def test_all_references_valid(self, agent: GuardianAgent) -> None:
        """All prerequisites exist in the unit list."""
        units = [
            mk_ku("kp-a"),
            mk_ku("kp-b", prerequisites=["kp-a"]),
        ]
        result = await agent.run(units)
        assert result.dangling_references == []

    @pytest.mark.asyncio
    async def test_dangling_reference_detected(self, agent: GuardianAgent) -> None:
        """A prerequisite that doesn't exist is flagged."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-nonexistent"]),
        ]
        result = await agent.run(units)
        assert result.is_valid is False
        assert len(result.dangling_references) == 1
        assert "kp-nonexistent" in result.dangling_references[0]

    @pytest.mark.asyncio
    async def test_multiple_dangling_references(self, agent: GuardianAgent) -> None:
        """Multiple missing prerequisites are all flagged."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-missing1", "kp-missing2"]),
        ]
        result = await agent.run(units)
        assert len(result.dangling_references) == 2

    @pytest.mark.asyncio
    async def test_dangling_prereq_outside_scope(self, agent: GuardianAgent) -> None:
        """Prereq not in the current proposal but could exist in KG.

        NOTE: The current implementation only checks if prereq IDs exist
        within the proposed knowledge_units list. A prereq may already
        exist in the Neo4j graph — this is a known limitation.
        """
        units = [
            mk_ku("kp-existing"),
            mk_ku("kp-new", prerequisites=["kp-existing", "kp-outside-scope"]),
        ]
        result = await agent.run(units)
        assert len(result.dangling_references) == 1
        assert "kp-outside-scope" in result.dangling_references[0]


class TestEdgeCases:
    """Boundary and edge cases."""

    @pytest.mark.asyncio
    async def test_many_fifty_nodes_chain(self, agent: GuardianAgent) -> None:
        """A chain of 50 nodes should be validated quickly."""
        units = []
        for i in range(50):
            prereqs = [f"kp-{i}"] if i > 0 else []
            units.append(mk_ku(f"kp-{i + 1}", difficulty=(i % 5) + 1, prerequisites=prereqs))
        result = await agent.run(units)
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_all_checks_run_on_invalid_input(self, agent: GuardianAgent) -> None:
        """All three checks run even if one fails."""
        units = [
            mk_ku("kp-a", prerequisites=["kp-b"]),
            mk_ku("kp-b", prerequisites=["kp-a"]),  # cycle
            mk_ku("kp-c", prerequisites=["kp-nonexistent"]),  # dangling
            mk_ku("kp-d", difficulty=1, prerequisites=["kp-e"]),
            mk_ku("kp-e", difficulty=3, prerequisites=["kp-d"]),  # violation
        ]
        result = await agent.run(units)
        # Cycle detected
        assert len(result.cycle_details) >= 1
        # Dangling detected
        assert len(result.dangling_references) >= 1
        # Monotonicity violation (warning, not fatal)
        assert len(result.monotonicity_violations) >= 1
        # Overall validity
        assert result.is_valid is False
