"""Demo course data loader — executes the Cypher seed file against Neo4j."""

import logging
import re
from pathlib import Path

from src.kg.connection import Neo4jPool
from src.kg.models import SeedLoadResponse

logger = logging.getLogger(__name__)

# Pattern: statements end with semicolon (optionally preceded by a comment line)
_STATEMENT_SPLIT = re.compile(r";\s*")


class SeedLoader:
    """Read and execute the Neo4j seed Cypher file.

    The seed file lives at ``scripts/db/seed/neo4j-seed.cypher`` relative to
    the project root.
    """

    def __init__(self, conn: Neo4jPool):
        self.conn = conn

    def _seed_path(self) -> Path:
        """Resolve the seed Cypher file path from the project root.

        Works in both development (project root) and Docker (/app).
        Seed file lives at ``scripts/db/seed/neo4j-seed.cypher``.
        """
        # __file__ = /app/src/kg/seed_loader.py → 3 × parent = /app
        return (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "db"
            / "seed"
            / "neo4j-seed.cypher"
        )

    async def load_seed_data(self) -> SeedLoadResponse:
        """Read ``neo4j-seed.cypher`` and execute each statement.

        Comments (``// …``) and blank lines are stripped.  Statements are
        split on ``;`` — note that the file must not contain string literals
        with semicolons for this simple split to work.

        Returns:
            SeedLoadResponse with counts of nodes and edges created.
        """
        seed_file = self._seed_path()
        if not seed_file.exists():
            raise FileNotFoundError(f"Seed file not found: {seed_file}")

        raw = seed_file.read_text(encoding="utf-8")
        statements = _STATEMENT_SPLIT.split(raw)

        nodes_created = 0
        edges_created = 0

        for stmt in statements:
            # Strip comments and blank lines
            cleaned = self._clean_statement(stmt)
            if not cleaned:
                continue

            # Skip CREATE INDEX statements (may fail if index exists)
            if cleaned.upper().startswith("CREATE INDEX"):
                try:
                    await self.conn.execute_write(cleaned)
                except Exception:
                    logger.warning("Index creation skipped (may already exist): %s", cleaned[:80])
                continue

            try:
                result = await self.conn.execute_write(cleaned)
                # Rough counting: MERGE/CREATE statements with SET usually create 1 node
                if any(kw in cleaned.upper() for kw in ("MERGE (", "CREATE (")):
                    if "HAS_TOPIC" in cleaned or "PREREQUISITE_OF" in cleaned or "RELATED_TO" in cleaned:
                        edges_created += 1
                    else:
                        nodes_created += 1
                logger.debug("Executed: %s …", cleaned[:80])
            except Exception:
                logger.exception("Seed statement failed: %s", cleaned[:120])
                raise

        logger.info(
            "Seed load complete — %d nodes, %d edges",
            nodes_created,
            edges_created,
        )
        return SeedLoadResponse(
            nodes_created=nodes_created,
            edges_created=edges_created,
        )

    async def load_embeddings(
        self,
        chroma_host: str = "chromadb",
        chroma_port: int = 8000,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ) -> dict:
        """Generate and upsert ChromaDB embeddings for all KPs in Neo4j.

        Args:
            chroma_host: ChromaDB hostname.
            chroma_port: ChromaDB port.
            embedding_model: Sentence-transformers model name.

        Returns:
            Dict with status and counts.
        """
        try:
            from sentence_transformers import SentenceTransformer
            from src.kg.vector_index import VectorIndex
        except ImportError as exc:
            logger.warning("Cannot load embeddings (import error: %s)", exc)
            return {"status": "skipped", "reason": str(exc)}

        # Load KPs from Neo4j
        kps = await self.conn.execute_read(
            "MATCH (k:KnowledgePoint) RETURN k.id AS id, k.name AS name, "
            "k.description AS description, k.difficulty AS difficulty, "
            "k.category AS category ORDER BY k.id"
        )
        if not kps:
            logger.info("No KPs found — skipping embedding load")
            return {"status": "skipped", "reason": "no KPs", "count": 0}

        logger.info("Loading embeddings for %d KPs...", len(kps))
        try:
            model = SentenceTransformer(embedding_model)
        except Exception as exc:
            logger.warning("Failed to load embedding model (%s) — skipping", exc)
            return {"status": "error", "reason": f"model load failed: {exc}"}

        vi = VectorIndex(host=chroma_host, port=chroma_port)

        texts = [
            f"{kp['name']}: {kp.get('description', '')} (难度: {kp.get('difficulty', '')}, 分类: {kp.get('category', '')})"
            for kp in kps
        ]
        ids = [kp["id"] for kp in kps]

        import time
        t0 = time.time()
        embeddings = model.encode(texts, show_progress_bar=False)
        logger.info("Embeddings generated in %.1fs", time.time() - t0)

        for i, kp_id in enumerate(ids):
            vi.upsert(
                kp_id=kp_id,
                embedding=embeddings[i].tolist(),
                metadata={
                    "name": kps[i]["name"],
                    "description": kps[i].get("description", ""),
                    "difficulty": str(kps[i].get("difficulty", "")),
                    "category": kps[i].get("category", ""),
                },
            )

        size = vi.collection_size()
        logger.info("Embedding load complete — %d upserted, collection size=%d", len(kps), size)
        return {"status": "completed", "count": len(kps), "collection_size": size}

    async def verify_seed(self) -> dict:
        """Return basic statistics about the seeded graph.

        Returns a dict with keys ``node_count``, ``edge_count``,
        ``course_count``, and ``indexes``.
        """
        node_count = 0
        edge_count = 0
        course_count = 0
        indexes = []

        try:
            result = await self.conn.execute_read(
                "MATCH (kp:KnowledgePoint) RETURN count(kp) AS total"
            )
            node_count = result[0]["total"] if result else 0
        except Exception:
            logger.warning("Could not count KnowledgePoints")

        try:
            result = await self.conn.execute_read(
                """
                MATCH ()-[r:PREREQUISITE_OF|RELATED_TO|HAS_TOPIC]->()
                RETURN count(r) AS total
                """
            )
            edge_count = result[0]["total"] if result else 0
        except Exception:
            logger.warning("Could not count edges")

        try:
            result = await self.conn.execute_read(
                "MATCH (c:Course) RETURN count(c) AS total"
            )
            course_count = result[0]["total"] if result else 0
        except Exception:
            logger.warning("Could not count Courses")

        try:
            result = await self.conn.execute_read(
                "SHOW INDEXES YIELD name, type, labelsOrTypes, properties"
            )
            indexes = [
                {
                    "name": row.get("name", ""),
                    "type": row.get("type", ""),
                    "labelsOrTypes": row.get("labelsOrTypes", []),
                    "properties": row.get("properties", []),
                }
                for row in result
            ]
        except Exception:
            logger.warning("Could not list indexes")

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "course_count": course_count,
            "indexes": indexes,
        }

    @staticmethod
    def _clean_statement(stmt: str) -> str:
        """Strip comments and whitespace from a Cypher statement."""
        lines = []
        for line in stmt.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            # Remove inline comments (but be careful with string literals)
            if "//" in line:
                line = line.split("//")[0].strip()
            lines.append(line)
        return " ".join(lines).strip()
