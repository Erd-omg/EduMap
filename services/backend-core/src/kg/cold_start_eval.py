"""Cold-start evaluation module — corpus density, KG coverage, confidence score."""

import logging

from src.kg.connection import Neo4jPool

logger = logging.getLogger(__name__)


class ColdStartEvaluator:
    """Evaluate corpus readiness for a given topic area.

    Confidence formula::

        confidence = w1 * corpus_density + w2 * kg_coverage + w3 * cross_validate_count

    Where:
        - corpus_density: ratio of documents found in ChromaDB vs a target minimum (5)
        - kg_coverage:   ratio of nodes+edges vs a target minimum (8 nodes, 10 edges)
        - cross_validate_count: how many independent sources verified the topic (capped at 3)

    Default weights: w1=0.4, w2=0.4, w3=0.2
    """

    def __init__(
        self,
        conn: Neo4jPool,
        w1: float = 0.4,
        w2: float = 0.4,
        w3: float = 0.2,
    ):
        self.conn = conn
        self.weights = {"corpus_density": w1, "kg_coverage": w2, "cross_validate": w3}

    async def evaluate(self) -> dict:
        """Run all evaluations and return a combined confidence report."""
        corpus = await self._corpus_density()
        kg = await self._kg_coverage()
        cross_val = self._cross_validate_count()

        confidence = (
            self.weights["corpus_density"] * corpus["score"]
            + self.weights["kg_coverage"] * kg["score"]
            + self.weights["cross_validate"] * cross_val["score"]
        )

        return {
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
            "factors": {
                "corpus_density": {
                    "weight": self.weights["corpus_density"],
                    "score": corpus["score"],
                    "detail": corpus["detail"],
                },
                "kg_coverage": {
                    "weight": self.weights["kg_coverage"],
                    "score": kg["score"],
                    "detail": kg["detail"],
                },
                "cross_validate": {
                    "weight": self.weights["cross_validate"],
                    "score": cross_val["score"],
                    "detail": cross_val["detail"],
                },
            },
            "verdict": self._verdict(confidence),
        }

    async def _corpus_density(self) -> dict:
        """Evaluate ChromaDB document density.

        For the prototype we count the number of distinct KnowledgePoint
        labels stored in Neo4j as a proxy for "documents in corpus."
        In production this would query ChromaDB collection stats.
        """
        target = 5  # minimum documents expected
        try:
            count = await self.conn.execute_read(
                "MATCH (kp:KnowledgePoint) RETURN count(kp) AS cnt"
            )
            doc_count = count[0]["cnt"] if count else 0
            score = min(1.0, doc_count / target)
            return {
                "score": score,
                "detail": {
                    "topic_count": doc_count,
                    "target_minimum": target,
                    "note": "Using KnowledgePoint count as proxy for corpus density",
                },
            }
        except Exception as exc:
            logger.warning("Corpus density evaluation failed: %s", exc)
            return {"score": 0.0, "detail": {"error": str(exc)}}

    async def _kg_coverage(self) -> dict:
        """Evaluate knowledge graph coverage — node + edge counts vs targets."""
        node_target = 8
        edge_target = 10
        try:
            nodes = await self.conn.execute_read(
                "MATCH (kp:KnowledgePoint) RETURN count(kp) AS cnt"
            )
            edges = await self.conn.execute_read(
                "MATCH ()-[r]->() RETURN count(r) AS cnt"
            )
            node_count = nodes[0]["cnt"] if nodes else 0
            edge_count = edges[0]["cnt"] if edges else 0

            node_score = min(1.0, node_count / node_target) if node_target else 0
            edge_score = min(1.0, edge_count / edge_target) if edge_target else 0
            score = 0.6 * node_score + 0.4 * edge_score

            return {
                "score": round(score, 4),
                "detail": {
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "node_target": node_target,
                    "edge_target": edge_target,
                },
            }
        except Exception as exc:
            logger.warning("KG coverage evaluation failed: %s", exc)
            return {"score": 0.0, "detail": {"error": str(exc)}}

    @staticmethod
    def _cross_validate_count() -> dict:
        """Return cross-validation score (static prototype).

        In production this would count how many independent source documents
        or LLM passes corroborate the same topic.  For the prototype we
        return a conservative default.
        """
        return {
            "score": 0.5,
            "detail": {
                "validated_sources": 0,
                "note": "Prototype default — no real cross-validation implemented yet",
            },
        }

    @staticmethod
    def _verdict(confidence: float) -> str:
        """Map confidence score to a human-readable verdict."""
        if confidence >= 0.7:
            return "ready — sufficient corpus for quality generation"
        if confidence >= 0.4:
            return "limited — generation possible but quality may vary; consider uploading more material"
        return "insufficient — upload at least one reference document before generating"
