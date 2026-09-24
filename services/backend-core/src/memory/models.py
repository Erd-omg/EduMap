"""Pydantic models for the three-tier memory system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events stored in episodic memory."""
    MENTOR_QUERY = "mentor_query"
    QUIZ_ANSWER = "quiz_answer"
    GENERATION = "generation"
    REVIEW = "review"
    PROFILE_UPDATE = "profile_update"


class MemoryType(str, Enum):
    """Types of semantic memory entries."""
    USER_PREFERENCE = "user_preference"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    SKILL_SUMMARY = "skill_summary"
    LEARNING_STYLE = "learning_style"


# ── Episodic Memory ──────────────────────────────────────────

class EpisodicEntry(BaseModel):
    """A single episodic memory record — one interaction with the system."""
    id: str | None = None
    user_id: str
    session_id: str
    event_type: EventType | str
    input: str | None = None
    output: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance_score: float = 0.5
    # Embedding of the interaction text. Used by ``recall_relevant`` for the
    # relevance term; ``None`` for rows written before the column existed, and
    # for callers that do not supply a vector.
    embedding: list[float] | None = None
    created_at: datetime | None = None


# ── Semantic Memory ──────────────────────────────────────────

class SemanticEntry(BaseModel):
    """A single semantic memory entry — distilled knowledge about user/domain."""
    id: str | None = None
    user_id: str
    memory_type: MemoryType | str
    key: str
    value: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ── Short-term Memory (Session-scoped) ───────────────────────

class ConversationTurn(BaseModel):
    """A single turn in a conversation session."""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionMemory(BaseModel):
    """Per-session state held in short-term memory (Redis)."""
    session_id: str
    user_id: str
    conversation: list[ConversationTurn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Sensory Memory (Per-request) ─────────────────────────────

class SensoryInput(BaseModel):
    """Raw input context for the current request — ephemeral."""
    user_id: str
    session_id: str | None = None
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """Combined context loaded from all three memory tiers for a pipeline run."""
    sensory: SensoryInput | None = None
    conversation_history: list[EpisodicEntry] = Field(default_factory=list)
    user_profile: dict[str, Any] = Field(default_factory=dict)
    session_memory: SessionMemory | None = None
    semantic_recall: list[SemanticEntry] = Field(default_factory=list)
