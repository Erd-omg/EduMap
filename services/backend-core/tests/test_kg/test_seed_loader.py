"""Tests for SeedLoader — path resolution, statement cleaning, and seed verification.

Database-dependent methods (load_seed_data) are tested with mocked connections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.kg.seed_loader import SeedLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def loader(mock_conn: AsyncMock) -> SeedLoader:
    return SeedLoader(conn=mock_conn)


# ---------------------------------------------------------------------------
# _clean_statement — pure static method
# ---------------------------------------------------------------------------

class TestCleanStatement:
    """Unit tests for the static _clean_statement method."""

    def test_strips_comments(self) -> None:
        """_clean_statement is called *after* ;-splitting, so it preserves ;."""
        raw = "// This is a comment\nMATCH (n) RETURN n;"
        assert SeedLoader._clean_statement(raw) == "MATCH (n) RETURN n;"

    def test_removes_inline_comment(self) -> None:
        raw = "MATCH (n) RETURN n // inline"
        assert SeedLoader._clean_statement(raw) == "MATCH (n) RETURN n"

    def test_empty_lines_and_whitespace(self) -> None:
        raw = "\n  \nMATCH (n)\n  RETURN n\n"
        assert SeedLoader._clean_statement(raw) == "MATCH (n) RETURN n"

    def test_only_comments_returns_empty(self) -> None:
        assert SeedLoader._clean_statement("// just a comment") == ""

    def test_blank_string_returns_empty(self) -> None:
        assert SeedLoader._clean_statement("") == ""
        assert SeedLoader._clean_statement("   ") == ""
        assert SeedLoader._clean_statement("\n\n\n") == ""

    def test_multi_line_statement(self) -> None:
        raw = """
        MERGE (kp:KnowledgePoint {
            id: "001",
            name: "加法运算"
        })
        """
        result = SeedLoader._clean_statement(raw)
        assert "MERGE" in result
        assert "id:" in result
        assert "name:" in result
        # No newlines or extra whitespace clumps
        assert "\n" not in result

    def test_comment_between_lines(self) -> None:
        raw = """CREATE (n:Test {id: "1"})
        // explanatory comment
        SET n.name = "test"
        """
        result = SeedLoader._clean_statement(raw)
        assert "//" not in result
        assert "CREATE" in result
        assert "SET" in result


# ---------------------------------------------------------------------------
# _seed_path — path resolution logic
# ---------------------------------------------------------------------------

class TestSeedPath:
    """Test path resolution of the seed Cypher file."""

    def test_seed_path_returns_path_object(self, loader: SeedLoader) -> None:
        path = loader._seed_path()
        assert isinstance(path, Path)
        # The path should end with scripts/db/seed/neo4j-seed.cypher
        assert path.name == "neo4j-seed.cypher"
        assert path.parent.name == "seed"
        assert path.parent.parent.name == "db"

    def test_seed_path_traversal_depth(self, loader: SeedLoader) -> None:
        """_seed_path goes up 3 parent dirs from the module file (src/kg/seed_loader.py).

        The logic:
            Path(__file__).resolve().parent.parent.parent / "scripts" / "db" / "seed" / "neo4j-seed.cypher"

        So given __file__ = /app/src/kg/seed_loader.py:
            parent (1) = /app/src/kg
            parent (2) = /app/src
            parent (3) = /app
            + scripts/db/seed/neo4j-seed.cypher = /app/scripts/db/seed/neo4j-seed.cypher
        """
        path = loader._seed_path()
        parts = path.parts
        # Expect .../<something>/scripts/db/seed/neo4j-seed.cypher
        assert "scripts" in parts
        assert "db" in parts
        assert "seed" in parts
        assert path.name == "neo4j-seed.cypher"

    def test_seed_path_resolves_from_module_location(self, loader: SeedLoader) -> None:
        """Verify the path structure matches the expected layout.

        The production code assumes Docker layout (``/app/scripts/...``).
        In development the file lives at the repo root; the assertion here
        only checks component names, not that the file exists on disk.
        """
        path = loader._seed_path()
        # Expect:  …/scripts/db/seed/neo4j-seed.cypher
        assert path.name == "neo4j-seed.cypher"
        assert path.parent.name == "seed"
        assert path.parent.parent.name == "db"
        assert "scripts" in path.parts


# ---------------------------------------------------------------------------
# verify_seed — mocked DB connection
# ---------------------------------------------------------------------------

class TestVerifySeed:
    """Tests for verify_seed, which queries Neo4j for graph statistics."""

    @pytest.mark.asyncio
    async def test_verify_seed_returns_dict_with_keys(
        self, loader: SeedLoader,
    ) -> None:
        """verify_seed returns a dict with expected top-level keys."""
        loader.conn.execute_read = AsyncMock(side_effect=[
            [{"total": 42}],     # node count
            [{"total": 17}],     # edge count
            [{"total": 3}],      # course count
            [{"name": "idx1", "type": "RANGE", "labelsOrTypes": ["KnowledgePoint"], "properties": ["id"]}],  # indexes
        ])

        result = await loader.verify_seed()
        assert isinstance(result, dict)
        assert "node_count" in result
        assert "edge_count" in result
        assert "course_count" in result
        assert "indexes" in result
        assert result["node_count"] == 42
        assert result["edge_count"] == 17
        assert result["course_count"] == 3
        assert len(result["indexes"]) == 1

    @pytest.mark.asyncio
    async def test_verify_seed_all_counts_zero_when_empty(
        self, loader: SeedLoader,
    ) -> None:
        """An empty graph returns all-zero counts."""
        loader.conn.execute_read = AsyncMock(side_effect=[
            [{"total": 0}],
            [{"total": 0}],
            [{"total": 0}],
            [],
        ])

        result = await loader.verify_seed()
        assert result["node_count"] == 0
        assert result["edge_count"] == 0
        assert result["course_count"] == 0
        assert result["indexes"] == []

    @pytest.mark.asyncio
    async def test_verify_seed_handles_db_errors_gracefully(
        self, loader: SeedLoader,
    ) -> None:
        """If a query fails, the count is set to 0 and execution continues."""
        loader.conn.execute_read = AsyncMock(side_effect=[
            Exception("Connection lost"),   # node count fails
            [{"total": 5}],                # edge count succeeds
            [{"total": 1}],                # course count succeeds
            Exception("Not available"),     # indexes fails
        ])

        result = await loader.verify_seed()
        assert result["node_count"] == 0      # gracefully handled
        assert result["edge_count"] == 5
        assert result["course_count"] == 1
        assert result["indexes"] == []


# ---------------------------------------------------------------------------
# load_seed_data — statement counting logic
# ---------------------------------------------------------------------------

class TestLoadSeedData:
    """Tests for load_seed_data with a mocked connection and filesystem."""

    @pytest.mark.asyncio
    async def test_missing_seed_file_raises(self, loader: SeedLoader) -> None:
        """If the seed file does not exist, FileNotFoundError is raised."""
        with patch.object(loader, "_seed_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/neo4j-seed.cypher")
            with pytest.raises(FileNotFoundError, match="Seed file not found"):
                await loader.load_seed_data()

    @pytest.mark.asyncio
    async def test_empty_file_returns_zero_counts(self, loader: SeedLoader, tmp_path: Path) -> None:
        """An empty seed file results in zero nodes and edges created."""
        seed_file = tmp_path / "neo4j-seed.cypher"
        seed_file.write_text("", encoding="utf-8")

        with patch.object(loader, "_seed_path", return_value=seed_file):
            response = await loader.load_seed_data()

        assert response.nodes_created == 0
        assert response.edges_created == 0

    @pytest.mark.asyncio
    async def test_counts_node_and_edge_statements(self, loader: SeedLoader, tmp_path: Path) -> None:
        """Node CREATE/MERGE statements increment nodes_created; edge statements increment edges_created."""
        seed_file = tmp_path / "neo4j-seed.cypher"
        seed_file.write_text(
            """
            MERGE (kp:KnowledgePoint {id: "1"});
            MERGE (kp:KnowledgePoint {id: "2"});
            MERGE (kp:KnowledgePoint {id: "3"});
            MATCH (a {id: "1"}), (b {id: "2"})
            MERGE (a)-[:PREREQUISITE_OF]->(b);
            MATCH (a {id: "1"}), (b {id: "3"})
            MERGE (a)-[:RELATED_TO]->(b);
            """,
            encoding="utf-8",
        )

        with patch.object(loader, "_seed_path", return_value=seed_file):
            response = await loader.load_seed_data()

        assert response.nodes_created == 3
        assert response.edges_created == 2
        # Total calls to execute_write: 5 (all succeeded)
        assert loader.conn.execute_write.call_count == 5

    @pytest.mark.asyncio
    async def test_create_index_skip_on_error(self, loader: SeedLoader, tmp_path: Path) -> None:
        """CREATE INDEX statements that fail are silently skipped."""
        seed_file = tmp_path / "neo4j-seed.cypher"
        seed_file.write_text(
            "CREATE INDEX kp_id IF NOT EXISTS FOR (kp:KnowledgePoint) ON (kp.id);",
            encoding="utf-8",
        )

        loader.conn.execute_write.side_effect = Exception("Index already exists")

        with patch.object(loader, "_seed_path", return_value=seed_file):
            response = await loader.load_seed_data()

        # Should not crash — index failure is caught
        assert response.nodes_created == 0
        assert response.edges_created == 0
        loader.conn.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_statement_failure_raises(self, loader: SeedLoader, tmp_path: Path) -> None:
        """A failed non-index statement is re-raised."""
        seed_file = tmp_path / "neo4j-seed.cypher"
        seed_file.write_text(
            "MERGE (kp:KnowledgePoint {id: 'bad'});",
            encoding="utf-8",
        )

        loader.conn.execute_write.side_effect = Exception("Syntax error")

        with patch.object(loader, "_seed_path", return_value=seed_file):
            with pytest.raises(Exception, match="Syntax error"):
                await loader.load_seed_data()

    @pytest.mark.asyncio
    async def test_comment_only_statements_skipped(self, loader: SeedLoader, tmp_path: Path) -> None:
        """Statements that become empty after cleaning are skipped."""
        seed_file = tmp_path / "neo4j-seed.cypher"
        seed_file.write_text(
            "// just a comment\n;\n// another comment;\n  \nMERGE (kp:Test {id: '1'});",
            encoding="utf-8",
        )

        with patch.object(loader, "_seed_path", return_value=seed_file):
            response = await loader.load_seed_data()

        # Only the single MERGE statement was executed
        assert loader.conn.execute_write.call_count == 1
        assert response.nodes_created == 1
