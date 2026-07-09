"""Assessment agent — micro-quiz generation and profile feedback.

Generates 1-3 quiz questions for a knowledge point and computes
a ``mastery_delta`` estimate based on the user's current profile.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from src.agents.models import AssessmentOutput, KnowledgeUnit, QuizQuestion
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class AssessmentAgent:
    """Generates micro-quizzes for a knowledge point."""

    def __init__(self, llm_adapter: BaseLLMAdapter) -> None:
        self._llm = llm_adapter

    async def run(
        self,
        knowledge_unit: KnowledgeUnit,
        user_profile: dict | None = None,
    ) -> AssessmentOutput:
        """Generate quiz questions and estimate mastery impact."""
        prompt = PromptRegistry.get(
            "assessment/v1-micro-quiz",
            kp_name=knowledge_unit.name,
            kp_description=knowledge_unit.description,
            difficulty=knowledge_unit.difficulty,
        )
        response = await self._llm.generate(prompt)
        data = self._parse_quiz_json(response.content)

        questions: list[QuizQuestion] = []
        for q_data in data.get("questions", []):
            questions.append(
                QuizQuestion(
                    id=q_data.get("id", f"q-{len(questions)+1}"),
                    type=q_data.get("type", "choice"),  # type: ignore[arg-type]
                    content=q_data.get("content", ""),
                    options=q_data.get("options"),
                    correct_answer=q_data.get("correct_answer", ""),
                    knowledge_point_id=knowledge_unit.id,
                )
            )

        mastery_delta: dict[str, float] = {}
        # Start with LLM's estimate (may be empty)
        llm_delta = data.get("estimated_mastery_delta", {})
        if isinstance(llm_delta, dict):
            for kp_id, delta in llm_delta.items():
                mastery_delta[kp_id] = max(0.0, min(1.0, float(delta)))

        # If no LLM estimate, use a sensible default
        if not mastery_delta:
            mastery_delta[knowledge_unit.id] = 0.1

        # Confidence based on profile depth
        confidence = 0.5
        if user_profile and user_profile.get("confidence_scores"):
            scores = user_profile["confidence_scores"]
            if isinstance(scores, dict) and scores:
                confidence = sum(float(v) for v in scores.values()) / len(scores)

        return AssessmentOutput(
            quiz=questions,
            mastery_delta=mastery_delta,
            confidence=round(confidence, 2),
        )

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse_quiz_json(text: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM output with fallback."""
        # Try ```json ... ``` block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        raw = match.group(1).strip() if match else text
        # Try outermost { ... }
        brace_start = raw.find("{")
        if brace_start != -1:
            raw = raw[brace_start:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Assessment JSON parse failed: %s", exc)
            return {"questions": [], "estimated_mastery_delta": {}}
