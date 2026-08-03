"""Edge CRUD and path query repository."""

import logging

from src.kg.connection import Neo4jPool
from src.kg.models import KnowledgeEdge, KnowledgePoint

logger = logging.getLogger(__name__)


def _row_to_kp(row: dict) -> KnowledgePoint:
    """Convert a Cypher result node to a KnowledgePoint."""
    node = next(
        (row[k] for k in ("n", "kp", "node", "source", "target") if k in row),
        None,
    )
    if node is None:
        node = row
    props = dict(node) if isinstance(node, dict) else {}
    return KnowledgePoint(
        id=props.get("id", ""),
        name=props.get("name", ""),
        description=props.get("description", ""),
        difficulty=props.get("difficulty", 1),
        category=props.get("category", ""),
        prerequisites=props.get("prerequisites", []),
        merged_from_ids=props.get("merged_from_ids", []),
        canonical=props.get("canonical", True),
    )


# Validated relationship types — only these are allowed in Cypher queries.
VALID_RELATION_TYPES = frozenset({"PREREQUISITE_OF", "RELATED_TO", "HAS_TOPIC"})


class EdgeRepository:
    """CRUD + path queries for KnowledgeEdge relationships."""

    def __init__(self, conn: Neo4jPool):
        self.conn = conn

    @staticmethod
    def _validate_relation_type(relation_type: str) -> str:
        """Validate and return a relation type, raising ValueError if invalid."""
        if relation_type not in VALID_RELATION_TYPES:
            raise ValueError(
                f"Invalid relation type '{relation_type}'. "
                f"Allowed: {', '.join(sorted(VALID_RELATION_TYPES))}"
            )
        return relation_type

    async def create_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> bool:
        """MERGE a directed edge between two knowledge points.

        Returns True if the edge was created/already exists.
        """
        relation_type = self._validate_relation_type(relation_type)
        query = """
        MATCH (source:KnowledgePoint {id: $source_id})
        MATCH (target:KnowledgePoint {id: $target_id})
        CALL apoc.create.relationship(source, $relation_type, {}, target)
        YIELD rel
        RETURN count(rel) AS created
        """
        result = await self.conn.execute_write(
            query, {"source_id": source_id, "target_id": target_id, "relation_type": relation_type}
        )
        return result[0]["created"] > 0 if result else False

    async def delete_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> bool:
        """Delete an edge. Returns True if deleted."""
        relation_type = self._validate_relation_type(relation_type)
        query = """
        MATCH (source:KnowledgePoint {id: $source_id})
            -[r:PREREQUISITE_OF|RELATED_TO|HAS_TOPIC]->
            (target:KnowledgePoint {id: $target_id})
        WHERE type(r) = $relation_type
        DELETE r
        RETURN count(r) AS deleted
        """
        result = await self.conn.execute_write(
            query, {"source_id": source_id, "target_id": target_id, "relation_type": relation_type}
        )
        return result[0]["deleted"] > 0 if result else False

    async def get_shortest_path(
        self, source_id: str, target_id: str
    ) -> list[KnowledgePoint]:
        """Find the shortest path between two knowledge points."""
        query = """
        MATCH path = shortestPath(
            (source:KnowledgePoint {id: $source_id})
            -[:PREREQUISITE_OF|RELATED_TO*]-
            (target:KnowledgePoint {id: $target_id})
        )
        UNWIND nodes(path) AS n
        RETURN DISTINCT n
        """
        result = await self.conn.execute_read(
            query, {"source_id": source_id, "target_id": target_id}
        )
        return [_row_to_kp(row) for row in result]

    async def get_all_edges_for_course(
        self, course_id: str
    ) -> list[KnowledgeEdge]:
        """Get all edges within a course subgraph."""
        query = """
        MATCH (c:Course {id: $course_id})-[:HAS_TOPIC]->(kp:KnowledgePoint)
        MATCH (kp)-[r:PREREQUISITE_OF|RELATED_TO]->(other:KnowledgePoint)
        WHERE (other)<-[:HAS_TOPIC]-(c)
        RETURN DISTINCT {
            source: startNode(r).id,
            target: endNode(r).id,
            relation_type: type(r)
        } AS edge
        """
        result = await self.conn.execute_read(query, {"course_id": course_id})
        edges: list[KnowledgeEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for row in result:
            edge = row.get("edge", {})
            if not isinstance(edge, dict):
                continue
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rtype = edge.get("relation_type", "")
            if src and tgt and rtype:
                key = (src, tgt, rtype)
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        KnowledgeEdge(
                            source=src,
                            target=tgt,
                            relation_type=rtype,  # type: ignore[arg-type]
                        )
                    )
        return edges
