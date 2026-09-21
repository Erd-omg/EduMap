"""Pydantic models for personalized learning paths."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ── Status types ──────────────────────────────────────────────────────

KpStatus = Literal["ready", "in_progress", "completed", "locked"]
ContentType = Literal["explanation", "exercise", "visualization", "code", "reading"]


# ── Path nodes ──────────────────────────────────────────────────────────


class PathNode(BaseModel):
    """A single knowledge point in the personalized learning path."""

    kp_id: str
    name: str
    description: str = ""
    difficulty: int = 1
    status: KpStatus = "locked"
    prerequisites: list[str] = []
    prerequisites_met: bool = False
    recommended_content_types: list[ContentType] = ["explanation"]


class PersonalizedPath(BaseModel):
    """Full personalized learning path for a course."""

    course_id: str
    user_id: str
    nodes: list[PathNode] = []
    total_count: int = 0
    mastered_count: int = 0
    completed_count: int = 0
    progress_percent: float = 0.0
    created_at: str = ""


# ── Progress ─────────────────────────────────────────────────────────────


class ProgressRecord(BaseModel):
    """Record of user progress on a single knowledge point."""

    user_id: str
    course_id: str
    kp_id: str
    status: KpStatus = "in_progress"
    quiz_score: float | None = None
    completed_at: str | None = None


# ── Recommendations ──────────────────────────────────────────────────────


class PathRecommendation(BaseModel):
    """Next-step recommendation for the user."""

    next_kp_id: str
    next_kp_name: str = ""
    reason: str = ""
    recommended_content_type: ContentType = "explanation"
    estimated_session_min: int = 20


class ContentTypeSuggestion(BaseModel):
    """Content type prioritization based on learner profile."""

    kp_id: str = ""
    content_types: list[ContentType] = ["explanation"]
    dominant_style: str = ""
    rationale: str = ""
