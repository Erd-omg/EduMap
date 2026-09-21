"""FastAPI router for Knowledge Graph operations."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.kg.connection import Neo4jPool
from src.kg.cold_start_eval import ColdStartEvaluator
from src.kg.models import (
    DedupResult,
    KnowledgeEdge,
    KnowledgeGraphResponse,
    KnowledgePoint,
    KnowledgePointCreate,
    MergeRequest,
    SeedLoadResponse,
)
from src.kg.repositories.edge_repo import EdgeRepository, VALID_RELATION_TYPES
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
from src.kg.seed_loader import SeedLoader
from src.kg.semantic_dedup import SemanticDedupService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/kg", tags=["knowledge-graph"])


# ── Dependency injection ──────────────────────────────────────────────


async def get_kp_repo(request: Request) -> KnowledgePointRepository:
    """Provide a KnowledgePointRepository scoped to the request."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    return KnowledgePointRepository(pool)


async def get_edge_repo(request: Request) -> EdgeRepository:
    """Provide an EdgeRepository scoped to the request."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    return EdgeRepository(pool)


async def get_dedup_service(request: Request) -> SemanticDedupService:
    """Provide a SemanticDedupService scoped to the request."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    kp_repo = KnowledgePointRepository(pool)
    edge_repo = EdgeRepository(pool)
    return SemanticDedupService(kp_repo, edge_repo)


async def get_seed_loader(request: Request) -> SeedLoader:
    """Provide a SeedLoader scoped to the request."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    return SeedLoader(pool)


async def get_cold_start_evaluator(request: Request) -> ColdStartEvaluator:
    """Provide a ColdStartEvaluator scoped to the request."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    return ColdStartEvaluator(pool)


# ── Knowledge Point CRUD ──────────────────────────────────────────────


@router.get("/nodes/{node_id}", response_model=KnowledgePoint | None)
async def get_node(
    node_id: str,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Get a single knowledge point by its ID."""
    kp = await repo.get(node_id)
    if not kp:
        raise HTTPException(status_code=404, detail=f"KnowledgePoint not found: {node_id}")
    return kp


@router.post("/nodes", response_model=KnowledgePoint, status_code=201)
async def create_node(
    body: KnowledgePointCreate,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Create a new knowledge point."""
    try:
        return await repo.create(body)
    except Exception as exc:
        logger.exception("Failed to create KnowledgePoint")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/nodes/{node_id}", response_model=KnowledgePoint)
async def update_node(
    node_id: str,
    body: dict[str, Any],
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Partially update a knowledge point's properties."""
    kp = await repo.update(node_id, body)
    if not kp:
        raise HTTPException(status_code=404, detail=f"KnowledgePoint not found: {node_id}")
    return kp


@router.delete("/nodes/{node_id}", status_code=204)
async def delete_node(
    node_id: str,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Delete a knowledge point (DETACH DELETE)."""
    deleted = await repo.delete(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"KnowledgePoint not found: {node_id}")


# ── Course Graph ──────────────────────────────────────────────────────


@router.get("/courses")
async def list_courses(
    request: Request,
):
    """List all courses from Neo4j."""
    pool: Neo4jPool = request.app.state.neo4j_pool
    query = (
        "MATCH (c:Course) RETURN c.id AS id, c.name AS name, "
        "c.description AS description, c.difficulty AS difficulty ORDER BY c.id"
    )
    result = await pool.execute_read(query)
    courses = []
    for row in result:
        courses.append({
            "id": row.get("id", ""),
            "name": row.get("name", ""),
            "description": row.get("description", ""),
            "difficulty": row.get("difficulty", 1),
        })
    return {"courses": courses, "total": len(courses)}


@router.get(
    "/courses/{course_id}/graph",
    response_model=KnowledgeGraphResponse,
)
async def get_course_graph(
    course_id: str,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Get the entire subgraph for a course (nodes + edges)."""
    return await repo.get_course_graph(course_id)


# ── Traversal ─────────────────────────────────────────────────────────


@router.get(
    "/traverse/prerequisites/{node_id}",
    response_model=list[KnowledgePoint],
)
async def get_prerequisites(
    node_id: str,
    depth: int = Query(3, ge=1, le=10),
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Get prerequisites for a knowledge point up to a given depth."""
    return await repo.get_prerequisites(node_id, depth)


@router.get(
    "/traverse/prerequisites/{node_id}/path",
    response_model=list[KnowledgePoint],
)
async def get_prerequisite_path(
    node_id: str,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Get the full prerequisite chain from root to the given node."""
    return await repo.get_prerequisite_chain(node_id)


@router.get(
    "/traverse/related/{node_id}",
    response_model=list[KnowledgePoint],
)
async def get_related(
    node_id: str,
    repo: KnowledgePointRepository = Depends(get_kp_repo),
):
    """Get knowledge points related to the given node."""
    return await repo.get_related(node_id)


# ── Path ──────────────────────────────────────────────────────────────


@router.get("/path", response_model=list[KnowledgePoint])
async def shortest_path(
    source: str = Query(...),
    target: str = Query(...),
    edge_repo: EdgeRepository = Depends(get_edge_repo),
):
    """Find the shortest path between two knowledge points."""
    path = await edge_repo.get_shortest_path(source, target)
    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"No path found between {source} and {target}",
        )
    return path


# ── Edge CRUD ─────────────────────────────────────────────────────────


@router.post("/edges", status_code=201)
async def create_edge(
    body: KnowledgeEdge,
    edge_repo: EdgeRepository = Depends(get_edge_repo),
):
    """Create a relationship between two knowledge points."""
    if body.relation_type not in VALID_RELATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid relation type '{body.relation_type}'. Allowed: {', '.join(sorted(VALID_RELATION_TYPES))}",
        )
    created = await edge_repo.create_edge(
        body.source, body.target, body.relation_type
    )
    if not created:
        raise HTTPException(
            status_code=400,
            detail="Edge could not be created (check source/target exist)",
        )
    return {"status": "created"}


@router.delete("/edges", status_code=204)
async def delete_edge(
    source: str = Query(...),
    target: str = Query(...),
    relation_type: str = Query(...),
    edge_repo: EdgeRepository = Depends(get_edge_repo),
):
    """Delete a relationship between two knowledge points."""
    if relation_type not in VALID_RELATION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid relation type '{relation_type}'. Allowed: {', '.join(sorted(VALID_RELATION_TYPES))}",
        )
    deleted = await edge_repo.delete_edge(source, target, relation_type)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Edge not found",
        )


# ── Deduplication ─────────────────────────────────────────────────────


@router.post("/dedup/trigger", response_model=list[DedupResult])
async def trigger_dedup(
    body: dict[str, str] | None = None,
    svc: SemanticDedupService = Depends(get_dedup_service),
):
    """Trigger semantic deduplication scan.

    Optionally pass ``{"node_id": "kp-xxx"}`` to scan only against that node.
    """
    node_id = (body or {}).get("node_id")
    return await svc.find_duplicates(node_id)


@router.post("/dedup/merge", response_model=KnowledgePoint)
async def merge_nodes(
    body: MergeRequest,
    svc: SemanticDedupService = Depends(get_dedup_service),
):
    """Merge source knowledge point into target knowledge point."""
    try:
        return await svc.merge(body.source_id, body.target_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Merge failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Seed Data ─────────────────────────────────────────────────────────


@router.post("/seed/load", response_model=SeedLoadResponse)
async def load_seed(
    loader: SeedLoader = Depends(get_seed_loader),
):
    """Load the demo course seed data into Neo4j."""
    try:
        return await loader.load_seed_data()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Seed load failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/seed/verify")
async def verify_seed(
    loader: SeedLoader = Depends(get_seed_loader),
):
    """Return statistics about the seeded graph data."""
    return await loader.verify_seed()


# ── Cold-Start Evaluation ─────────────────────────────────────────────


@router.get("/eval/cold-start")
async def cold_start_evaluation(
    evaluator: ColdStartEvaluator = Depends(get_cold_start_evaluator),
):
    """Evaluate corpus readiness and return confidence score.

    Useful for determining whether the system has enough data to
    generate quality learning resources for a given topic area.
    """
    return await evaluator.evaluate()
