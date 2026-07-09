"""Guardian agent — DAG cycle detection and structure validation.

Purely deterministic — no LLM dependency.  Validates a proposed set of
:class:`KnowledgeUnit` nodes before they enter the generation phase.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

from src.agents.models import GuardianOutput, KnowledgeUnit

logger = logging.getLogger(__name__)


class GuardianAgent:
    """Validates knowledge-unit proposals for structural correctness."""

    async def run(
        self,
        knowledge_units: list[KnowledgeUnit],
    ) -> GuardianOutput:
        """Validate the proposed knowledge graph structure.

        Checks are performed in order:
        1. DAG cycle detection  (fatal)
        2. Difficulty monotonicity  (warning)
        3. Dangling prerequisite references  (fatal)
        """
        output = GuardianOutput()

        if not knowledge_units:
            output.is_valid = True
            return output

        ids = {ku.id for ku in knowledge_units}
        id_map = {ku.id: ku for ku in knowledge_units}

        # ── 1. DAG cycle detection (DFS) ────────────────────────────────
        adj: dict[str, list[str]] = defaultdict(list)
        for ku in knowledge_units:
            for prereq in ku.prerequisites:
                if prereq in ids:
                    adj[prereq].append(ku.id)

        cycles = self._find_cycles(adj, ids)
        if cycles:
            output.cycle_details = [
                " → ".join(cycle) for cycle in cycles
            ]
            logger.warning("Guardian detected %d cycle(s): %s", len(cycles), output.cycle_details)

        # ── 2. Difficulty monotonicity ──────────────────────────────────
        for ku in knowledge_units:
            for prereq_id in ku.prerequisites:
                prereq = id_map.get(prereq_id)
                if prereq and prereq.difficulty > ku.difficulty:
                    msg = (
                        f"'{prereq_id}' (difficulty={prereq.difficulty}) "
                        f"→ '{ku.id}' (difficulty={ku.difficulty}): "
                        "prerequisite harder than dependent"
                    )
                    output.monotonicity_violations.append(msg)

        # ── 3. Dangling references ──────────────────────────────────────
        for ku in knowledge_units:
            for prereq_id in ku.prerequisites:
                if prereq_id not in ids:
                    msg = f"'{ku.id}' references unknown prerequisite '{prereq_id}'"
                    output.dangling_references.append(msg)

        # ── Verdict ─────────────────────────────────────────────────────
        if output.cycle_details or output.dangling_references:
            output.is_valid = False

        return output

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _find_cycles(
        adj: dict[str, list[str]],
        vertices: set[str],
    ) -> list[list[str]]:
        """Return all elementary cycles in the directed graph.

        Uses a DFS-based approach (Tarjan's algorithm simplified for
        cycle detection in small DAGs — the knowledge plan is rarely
        larger than a few dozen nodes).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {v: WHITE for v in vertices}
        parent: dict[str, str | None] = {v: None for v in vertices}
        cycles: list[list[str]] = []

        def dfs(u: str, stack: list[str]) -> None:
            color[u] = GRAY
            stack.append(u)

            for v in adj.get(u, []):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    # Found a cycle: extract from stack
                    idx = stack.index(v)
                    cycle = stack[idx:] + [v]
                    cycles.append(cycle)
                elif color[v] == WHITE:
                    parent[v] = u
                    dfs(v, stack)

            stack.pop()
            color[u] = BLACK

        for v in vertices:
            if color[v] == WHITE:
                dfs(v, [])

        return cycles
