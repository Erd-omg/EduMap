"""Pydantic models for the Knowledge Graph data layer."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class KnowledgePointBase(BaseModel):
    """Base fields for a knowledge point node."""

    id: str
    name: str
    description: str
    difficulty: Annotated[int, Field(ge=1, le=5)] = Field(ge=1, le=5)
    category: str = ""
    prerequisites: list[str] = []


class KnowledgePointCreate(KnowledgePointBase):
    """Payload for creating a new knowledge point."""


class KnowledgePoint(KnowledgePointBase):
    """Full knowledge point node as stored in Neo4j."""

    merged_from_ids: list[str] = []
    canonical: bool = True
    embedding: list[float] | None = None


class KnowledgeEdge(BaseModel):
    """A directed/undirected edge between two knowledge points."""

    source: str
    target: str
    relation_type: Literal["PREREQUISITE_OF", "RELATED_TO", "HAS_TOPIC"]


class Course(BaseModel):
    """A course node."""

    id: str
    name: str
    description: str
    difficulty: int = 1


class KnowledgeGraphResponse(BaseModel):
    """Complete subgraph for a course."""

    nodes: list[KnowledgePoint]
    edges: list[KnowledgeEdge]


class DedupResult(BaseModel):
    """Result of a semantic deduplication comparison."""

    source_id: str
    target_id: str
    similarity: float
    merged: bool


class MergeRequest(BaseModel):
    """Request payload to merge two knowledge points."""

    source_id: str
    target_id: str


class SeedLoadResponse(BaseModel):
    """Response after loading seed data."""

    nodes_created: int
    edges_created: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
