"""Pydantic models for RAG retrieval and Mentor Q&A."""

from __future__ import annotations

from typing import Literal

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


class MentorResponse(BaseModel):
    """Full response from Mentor agent."""

    answer: str = ""
    sources: list[MentorSource] = []
    confidence: float = 0.5
