"""Assessment agent — micro-quiz generation and profile feedback.

Generates 1-3 quiz questions for a knowledge point and computes
a ``mastery_delta`` estimate based on the user's current profile.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from src.agents.assessment.quiz_bank import get_questions
from src.agents.models import AssessmentOutput, KnowledgeUnit, QuizQuestion
from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class AssessmentAgent(BaseAgent):
    """Generates micro-quizzes for a knowledge point."""

    def __init__(self, llm_adapter: BaseLLMAdapter, **kwargs) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="assessment",
            config=AgentConfig(max_retries=2, temperature=0.5),
            **kwargs,
        )

    async def run(self, input: AgentInput) -> AssessmentOutput:
        """Harness-compatible run — wraps legacy logic."""
        knowledge_unit = KnowledgeUnit(**input.extra.get("knowledge_unit", {}))
        user_profile = input.extra.get("user_profile")
        return await self.run_legacy(knowledge_unit, user_profile)

    async def run_legacy(
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
        # Try LLM generation first; fallback to quiz bank on any error
        questions: list[QuizQuestion] = []
        mastery_delta: dict[str, float] = {}
        try:
            response = await self._llm.generate(prompt)
            data = self._parse_quiz_json(response.content)

            # Try LLM-generated questions
            llm_questions = data.get("questions", [])
            if llm_questions:
                for q_data in llm_questions:
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

                # Start with LLM's estimate
                llm_delta = data.get("estimated_mastery_delta", {})
                if isinstance(llm_delta, dict):
                    for kp_id, delta in llm_delta.items():
                        mastery_delta[kp_id] = max(0.0, min(1.0, float(delta)))
        except Exception as exc:
            logger.warning(
                "LLM quiz generation failed for kp=%s: %s — falling back to quiz bank",
                knowledge_unit.id, exc,
            )

        if not questions:
            # Fallback to pre-built quiz bank
            questions = get_questions(knowledge_unit.id)
            logger.info(
                "Using quiz bank for kp=%s (%d questions)",
                knowledge_unit.id, len(questions),
            )

        # If still no questions and no mastery_delta, use sensible default
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
