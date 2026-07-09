"""KnowledgePoint CRUD and traversal repository."""

import logging
from typing import Any

from src.kg.connection import Neo4jPool
from src.kg.models import (
    KnowledgeEdge,
    KnowledgeGraphResponse,
    KnowledgePoint,
    KnowledgePointCreate,
    KnowledgePointBase,
)

logger = logging.getLogger(__name__)


def _row_to_kp(row: dict) -> KnowledgePoint:
    """Convert a Cypher result row to a KnowledgePoint."""
    node = row.get("kp") or row.get("n") or row.get("ancestor") or row.get("related") or row.get("startNode") or row
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


def _prereq_ids_to_kp(kp: KnowledgePoint, prereq_ids: list[str]) -> KnowledgePoint:
    """Attach prerequisite IDs to a KnowledgePoint."""
    kp.prerequisites = prereq_ids
    return kp


class KnowledgePointRepository:
    """CRUD + traversal for KnowledgePoint nodes."""

    def __init__(self, conn: Neo4jPool):
        self.conn = conn

    async def create(self, kp: KnowledgePointCreate) -> KnowledgePoint:
        """Create or MERGE a knowledge point by its id."""
        query = """
        MERGE (kp:KnowledgePoint {id: $id})
        SET kp.name = $name,
            kp.description = $description,
            kp.difficulty = $difficulty,
            kp.category = $category,
            kp.canonical = true,
            kp.merged_from_ids = []
        RETURN kp
        """
        result = await self.conn.execute_write(query, kp.model_dump())
        if not result:
            raise RuntimeError(f"Failed to create knowledge point: {kp.id}")
        return _row_to_kp(result[0])

    async def get(self, kp_id: str) -> KnowledgePoint | None:
        """Get a knowledge point by id, including its prerequisite edges."""
        query = """
        MATCH (kp:KnowledgePoint {id: $id})
        OPTIONAL MATCH (kp)<-[:PREREQUISITE_OF]-(prereq:KnowledgePoint)
        RETURN kp, collect(DISTINCT prereq.id) AS prereq_ids
        """
        result = await self.conn.execute_read(query, {"id": kp_id})
        if not result:
            return None
        row = result[0]
        kp = _row_to_kp(row)
        return _prereq_ids_to_kp(kp, row.get("prereq_ids", []))

    async def update(self, kp_id: str, data: dict[str, Any]) -> KnowledgePoint | None:
        """Update a knowledge point's properties by id."""
        if not data:
            return await self.get(kp_id)

        set_clauses = ", ".join(f"kp.{k} = ${k}" for k in data)
        query = f"""
        MATCH (kp:KnowledgePoint {{id: $id}})
        SET {set_clauses}
        RETURN kp
        """
        params = {"id": kp_id, **data}
        result = await self.conn.execute_write(query, params)
        if not result:
            return None
        return _row_to_kp(result[0])

    async def delete(self, kp_id: str) -> bool:
        """DETACH DELETE a knowledge point by id. Returns True if deleted."""
        query = """
        MATCH (kp:KnowledgePoint {id: $id})
        DETACH DELETE kp
        RETURN count(kp) AS deleted
        """
        result = await self.conn.execute_write(query, {"id": kp_id})
        return result[0]["deleted"] > 0 if result else False

    async def list_all(self) -> list[KnowledgePoint]:
        """Return all knowledge points."""
        query = """
        MATCH (kp:KnowledgePoint)
        RETURN kp
        ORDER BY kp.id
        """
        result = await self.conn.execute_read(query)
        return [_row_to_kp(row) for row in result]

    async def get_by_course(self, course_id: str) -> list[KnowledgePoint]:
        """Get all knowledge points belonging to a course."""
        query = """
        MATCH (c:Course {id: $id})-[:HAS_TOPIC]->(kp:KnowledgePoint)
        RETURN kp
        ORDER BY kp.id
        """
        result = await self.conn.execute_read(query, {"id": course_id})
        return [_row_to_kp(row) for row in result]

    async def get_prerequisites(
        self, kp_id: str, depth: int = 3
    ) -> list[KnowledgePoint]:
        """Recursively fetch prerequisite ancestors up to a given depth."""
        query = """
        MATCH path = (kp:KnowledgePoint {id: $kp_id})
            <-[:PREREQUISITE_OF*0..$depth]-(ancestor:KnowledgePoint)
        RETURN ancestor, length(path) AS depth
        ORDER BY depth
        """
        result = await self.conn.execute_read(
            query, {"kp_id": kp_id, "depth": depth}
        )
        seen: set[str] = set()
        points: list[KnowledgePoint] = []
        for row in result:
            kp = _row_to_kp(row)
            if kp.id not in seen and kp.id != kp_id:
                seen.add(kp.id)
                points.append(kp)
        return points

    async def get_prerequisite_chain(self, kp_id: str) -> list[KnowledgePoint]:
        """Get the full chain from root prerequisites up to the given node.

        Returns an ordered list starting from the most fundamental prerequisite
        to the node itself.
        """
        query = """
        MATCH path = (root:KnowledgePoint)
            -[:PREREQUISITE_OF*0..]->(kp:KnowledgePoint {id: $kp_id})
        WHERE NOT EXISTS {
            MATCH (root)<-[:PREREQUISITE_OF]-(:KnowledgePoint)
        }
        WITH root, kp, length(path) AS depth
        RETURN root, kp.id AS target_id
        ORDER BY depth
        """
        result = await self.conn.execute_read(query, {"kp_id": kp_id})
        if not result:
            return []

        # Re-fetch each node along the chain for full details
        chain_ids = []
        seen: set[str] = set()
        for row in result:
            root_props = dict(row.get("root", {})) if isinstance(row.get("root"), dict) else {}
            rid = root_props.get("id", "")
            if rid and rid not in seen and rid != kp_id:
                seen.add(rid)
                chain_ids.append(rid)

        if not chain_ids:
            return []

        points: list[KnowledgePoint] = []
        for cid in chain_ids:
            kp = await self.get(cid)
            if kp:
                points.append(kp)
        return points

    async def get_related(self, kp_id: str) -> list[KnowledgePoint]:
        """Get knowledge points related via RELATED_TO edges (bidirectional)."""
        query = """
        MATCH (kp:KnowledgePoint {id: $kp_id})-[:RELATED_TO]-(related:KnowledgePoint)
        RETURN DISTINCT related
        """
        result = await self.conn.execute_read(query, {"kp_id": kp_id})
        return [_row_to_kp(row) for row in result]

    async def search_by_name(self, query_str: str) -> list[KnowledgePoint]:
        """Case-insensitive search by name (CONTAINS)."""
        query = """
        MATCH (kp:KnowledgePoint)
        WHERE toLower(kp.name) CONTAINS toLower($query)
        RETURN kp
        ORDER BY kp.id
        """
        result = await self.conn.execute_read(query, {"query": query_str})
        return [_row_to_kp(row) for row in result]

    async def get_course_graph(self, course_id: str) -> KnowledgeGraphResponse:
        """Return the entire subgraph for a course including nodes and edges."""
        query = """
        MATCH (c:Course {id: $course_id})-[:HAS_TOPIC]->(kp:KnowledgePoint)
        OPTIONAL MATCH (kp)-[r:PREREQUISITE_OF|RELATED_TO]->(other:KnowledgePoint)
        WHERE (other)<-[:HAS_TOPIC]-(c)
        WITH collect(DISTINCT kp) AS nodes,
             collect(DISTINCT {
                 source: startNode(r).id,
                 target: endNode(r).id,
                 relation_type: type(r)
             }) AS edges
        RETURN nodes, edges
        """
        result = await self.conn.execute_read(query, {"course_id": course_id})
        if not result:
            return KnowledgeGraphResponse(nodes=[], edges=[])

        row = result[0]
        nodes_raw: list[dict] = row.get("nodes", []) or []
        edges_raw: list[dict] = row.get("edges", []) or []

        nodes = []
        seen_ids: set[str] = set()
        for n in nodes_raw:
            props = dict(n) if isinstance(n, dict) else {}
            pid = props.get("id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                nodes.append(KnowledgePoint(
                    id=pid,
                    name=props.get("name", ""),
                    description=props.get("description", ""),
                    difficulty=props.get("difficulty", 1),
                    category=props.get("category", ""),
                    merged_from_ids=props.get("merged_from_ids", []),
                    canonical=props.get("canonical", True),
                ))

        edges = []
        seen_edges: set[tuple[str, str, str]] = set()
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            src = e.get("source") or ""
            tgt = e.get("target") or ""
            rtype = e.get("relation_type") or ""
            if src and tgt and rtype:
                key = (src, tgt, rtype)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append(KnowledgeEdge(
                        source=src,
                        target=tgt,
                        relation_type=rtype,  # type: ignore[arg-type]
                    ))

        return KnowledgeGraphResponse(nodes=nodes, edges=edges)

    async def count_all(self) -> int:
        """Count all KnowledgePoint nodes."""
        query = """
        MATCH (kp:KnowledgePoint)
        RETURN count(kp) AS total
        """
        result = await self.conn.execute_read(query)
        return result[0]["total"] if result else 0
