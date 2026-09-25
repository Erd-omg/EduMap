"""Assessment agent — micro-quiz generation and profile feedback.

Generates 1-3 quiz questions for a knowledge point and computes
quiz questions and, on request, grades submitted answers.
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
                            # Carry the unit's difficulty onto each question so
                            # grading can weight answers by item hardness.  The
                            # LLM is not asked for a difficulty: an unvalidated
                            # self-report would reintroduce the very problem
                            # this replaces.
                            difficulty=knowledge_unit.difficulty,
                        )
                    )

                # NOTE: the LLM's ``estimated_mastery_delta`` is deliberately
                # NOT used.  It was an opinion formed at question-*writing*
                # time, before anyone answered anything, and it fell back to a
                # hard-coded 0.1.  Mastery is now measured from the learner's
                # graded responses — see ``grade_attempt`` below.
                #
                # The field is still read so that a deployment whose prompt has
                # not been updated does not error, but it never reaches the
                # output.
                if data.get("estimated_mastery_delta"):
                    logger.debug(
                        "Ignoring LLM-supplied estimated_mastery_delta for kp=%s "
                        "(mastery is measured from graded responses now)",
                        knowledge_unit.id,
                    )
        except Exception as exc:
            logger.warning(
                "LLM quiz generation failed for kp=%s: %s — falling back to quiz bank",
                knowledge_unit.id, exc,
            )

        if not questions:
            # Fallback to pre-built quiz bank
            questions = get_questions(knowledge_unit.id)
            # The bank is a static table keyed by kp_id and carries no
            # difficulty, so stamp the unit's difficulty on before grading —
            # otherwise every banked question would score as an average item
            # and the IRT estimate would lose the ability to weight them.
            questions = [
                q.model_copy(update={"difficulty": knowledge_unit.difficulty})
                if q.difficulty is None
                else q
                for q in questions
            ]
            logger.info(
                "Using quiz bank for kp=%s (%d questions)",
                knowledge_unit.id, len(questions),
            )

        # NOTE: no mastery value is produced here, by design.
        #
        # This method only *writes* a quiz; nothing has been answered yet, so
        # there is no evidence about mastery to report. The old code emitted the
        # LLM's guess (defaulting to a hard-coded 0.1) into
        # ``AssessmentOutput.mastery_delta`` — a field **no consumer ever read**,
        # so the number was both an assertion rather than a measurement *and*
        # dead weight. Mastery is measured from graded responses by
        # :meth:`grade_attempt`, which returns it to the caller directly.

        # Confidence based on profile depth
        confidence = 0.5
        if user_profile and user_profile.get("confidence_scores"):
            scores = user_profile["confidence_scores"]
            if isinstance(scores, dict) and scores:
                confidence = sum(float(v) for v in scores.values()) / len(scores)

        return AssessmentOutput(
            quiz=questions,
            confidence=round(confidence, 2),
        )

    async def grade_attempt(
        self,
        questions: list[QuizQuestion],
        answers: dict[str, Any] | list[Any],
        *,
        knowledge_point_id: str | None = None,
        prior_mastery: float | None = None,
    ) -> dict[str, Any]:
        """Grade a submitted attempt and estimate mastery from the responses.

        This is the counterpart to :meth:`run_legacy` that closes the loop: the
        agent writes the questions, the learner answers, and *here* the mastery
        number is computed from what actually happened rather than asserted
        beforehand.

        Kept deliberately thin — all the statistics live in
        :mod:`src.agents.assessment.grading`, which is a pure function of
        (questions, answers) and is tested without a model in the loop.

        Args:
            questions: the questions that were presented.
            answers: ``{question_id: answer}`` or positional answers.
            knowledge_point_id: which KP to key the delta under.  Defaults to
                the first question's ``knowledge_point_id``.
            prior_mastery: the learner's mastery before this attempt, if known.
                Without it the delta is measured against the neutral 0.5.

        Returns:
            ``{"graded": GradedQuiz, "mastery_delta": {kp_id: float}}``.
            The delta is returned here rather than stored on
            ``AssessmentOutput`` because the two are produced at different
            times: the output describes a quiz that has not been answered, while
            a delta can only exist after grading.
        """
        from src.agents.assessment.grading import (
            grade_quiz,
            mastery_delta_from_grading,
        )

        graded = grade_quiz(questions, answers)

        kp_id = knowledge_point_id
        if kp_id is None and questions:
            kp_id = questions[0].knowledge_point_id
        if kp_id is None:
            kp_id = "unknown"

        delta = mastery_delta_from_grading(graded, prior_mastery=prior_mastery)

        logger.info(
            "Graded attempt for kp=%s: %d/%d correct, mastery=%.3f, delta=%+.3f",
            kp_id, graded.n_correct, graded.n_questions, graded.mastery, delta,
        )

        return {
            "graded": graded,
            "mastery_delta": {kp_id: round(delta, 4)},
            "score": graded.score,
        }

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
