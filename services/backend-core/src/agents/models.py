"""Shared Pydantic models for agent I/O contracts in Phase 3.

Every agent reads its input from ``EduMapState`` and writes structured output
back via these models, which are serialised to dicts for LangGraph state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# ── Type aliases matching shared-types/resource.ts ──────────────────────

ResourceType = Literal["explanation", "mindmap", "exercise", "reading", "visualization", "code"]
QuestionType = Literal["choice", "true_false", "fill_blank"]


# ── Planner ──────────────────────────────────────────────────────────────


class KnowledgeUnit(BaseModel):
    """A single extracted knowledge point from the planner."""

    id: str
    name: str
    description: str
    difficulty: int  # 1-5
    prerequisites: list[str] = []
    key_concepts: list[str] = []


class GenerationPlan(BaseModel):
    """Structured plan output by Planner, validated by Guardian."""

    primary_kp_id: str
    content_types: list[ResourceType] = ["explanation", "exercise"]
    knowledge_units: list[KnowledgeUnit] = []


class PlannerOutput(BaseModel):
    """Output from the Planner agent."""

    plan: GenerationPlan
    summary: str = ""


# ── Guardian ─────────────────────────────────────────────────────────────


class GuardianOutput(BaseModel):
    """Output from the Guardian agent — validation results."""

    is_valid: bool = True
    cycle_details: list[str] = []
    monotonicity_violations: list[str] = []
    dangling_references: list[str] = []


# ── Designer ──────────────────────────────────────────────────────────────


class GeneratedResource(BaseModel):
    """A piece of generated content (explanation, exercise, etc.)."""

    type: ResourceType
    title: str
    content: str  # markdown body
    kp_id: str
    difficulty: int = 1
    metadata: dict = {}  # personalisation meta, generation params


class DesignerOutput(BaseModel):
    """Output from the Designer agent."""

    resources: list[GeneratedResource] = []


# ── Coder ─────────────────────────────────────────────────────────────────


class CoderOutput(BaseModel):
    """Output from the Coder agent."""

    code: str = ""
    language: str = "python"
    ast_valid: bool = False
    execution_result: str = ""
    execution_success: bool = False
    fix_iterations: int = 0


# ── Content Auditor ──────────────────────────────────────────────────────


class AuditEntry(BaseModel):
    """Single audit verdict from Content Auditor."""

    resource_index: int
    passed: bool
    similarity_score: float = 0.0
    reason: str | None = None


class AuditorOutput(BaseModel):
    """Output from the Content Auditor agent."""

    entries: list[AuditEntry] = []
    all_passed: bool = True


# ── Assessment ────────────────────────────────────────────────────────────


class QuizQuestion(BaseModel):
    """Assessment micro-quiz question."""

    id: str
    type: QuestionType = "choice"
    content: str
    options: list[str] | None = None
    correct_answer: str
    knowledge_point_id: str


class AssessmentOutput(BaseModel):
    """Output from Assessment agent."""

    quiz: list[QuizQuestion] = []
    mastery_delta: dict[str, float] = {}  # kp_id -> delta
    confidence: float = 0.5
