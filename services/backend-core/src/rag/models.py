"""Pydantic models for RAG retrieval and Mentor Q&A."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

SourceType = Literal["chroma", "neo4j"]


class RAGResult(BaseModel):
    """A single result from hybrid search."""

    content: str
    source_type: SourceType
    source_id: str
    source_name: str
    score: float = 0.0
    metadata: dict = {}


class RAGContext(BaseModel):
    """Assembled context for LLM prompt."""

    context_str: str = ""
    sources: list[RAGResult] = []


class MentorQuery(BaseModel):
    """Request to the Mentor agent."""

    user_id: str = "anonymous"
    query: str
    conversation_history: list[dict] | None = None


class MentorSource(BaseModel):
    """A source citation for the frontend."""

    id: str
    name: str
    type: SourceType = "neo4j"
    score: float = 0.0
    summary: str = ""
    # Optional link back to the original uploaded resource (chroma chunk
    # sources carry resource_id in their metadata; neo4j/KP sources don't).
    resource_id: str | None = None
    # Character span of this chunk within the original document, so the
    # frontend can highlight the cited passage rather than just naming the
    # chunk. Both are None when the span is unknown (older indexed chunks,
    # or a chunk the locator could not place) — ``None`` is deliberate and
    # must not be rendered as 0, which would point at the document's start.
    char_start: int | None = None
    char_end: int | None = None
    # 1-based page of the cited passage, when the source document has pages.
    page_number: int | None = None


class MentorResponse(BaseModel):
    """Full response from Mentor agent."""

    answer: str = ""
    sources: list[MentorSource] = []
    confidence: float = 0.5
    usage: dict[str, int] | None = None


class EvalMetricValue(BaseModel):
    """A single evaluation metric value."""
    value: float
    description: str = ""


class EvalReport(BaseModel):
    """Full evaluation report for a RAG benchmark run."""
    status: str = "completed"
    n_queries: int = 0
    metrics: dict[str, Any] = {}
    generated_at: str = ""
    course_id: str | None = None


class RerankPreviewRequest(BaseModel):
    """Request to preview reranking results."""
    query: str
    top_k: int = 5
    candidate_k: int = 15


class RerankPreviewResponse(BaseModel):
    """Reranking preview showing before/after scores."""
    query: str
    original_results: list[RAGResult] = []
    reranked_results: list[RAGResult] = []
    improvements: dict[str, float] = {}
