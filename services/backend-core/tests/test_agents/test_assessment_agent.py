"""Tests for AssessmentAgent — quiz writing and answer grading.

Why this file is new
--------------------
``tests/test_agents/`` contained no AssessmentAgent tests at all before this.
That is why the old behaviour was never caught: ``run_legacy`` fabricated a
``mastery_delta`` from the LLM's opinion (falling back to a hard-coded 0.1) at
question-*writing* time, and nothing noticed, because nothing asserted anything
about it.

Two contracts are pinned here:

1. **Writing a quiz carries no mastery at all.** At that point nobody has
   answered, so there is nothing to report. The field that used to hold the
   LLM's guess (no consumer ever read it) has been removed.
2. **Grading measures mastery from responses.** The number must move with the
   answers, and a harder question answered correctly must count for more.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.assessment.agent import AssessmentAgent
from src.agents.assessment.quiz_bank import get_questions
from src.agents.models import KnowledgeUnit, QuizQuestion
from src.prompts import PromptRegistry


@pytest.fixture(autouse=True)
def _load_prompts() -> None:
    """The agent renders its prompt from the registry, so it must be loaded.

    ``PromptRegistry`` populates a class-level cache that nothing loads
    automatically in a fresh process — without this, ``run_legacy`` raises
    ``KeyError: Prompt 'assessment/v1-micro-quiz' not found``.
    """
    PromptRegistry.load()


def _agent(llm_response: str | None = None) -> AssessmentAgent:
    """Agent with a mocked LLM adapter.

    A response containing valid quiz JSON exercises the LLM path; ``None``
    makes ``generate`` raise so the quiz-bank fallback is exercised instead.
    """
    llm = MagicMock()
    if llm_response is None:
        llm.generate = AsyncMock(side_effect=RuntimeError("no llm"))
    else:
        result = MagicMock()
        result.content = llm_response
        llm.generate = AsyncMock(return_value=result)
    return AssessmentAgent(llm_adapter=llm)


_QUIZ_JSON = json.dumps({
    "questions": [
        {
            "id": "q1",
            "type": "choice",
            "content": "1+1=?",
            "options": ["1", "2", "3"],
            "correct_answer": "2",
        },
        {
            "id": "q2",
            "type": "choice",
            "content": "2+2=?",
            "options": ["3", "4", "5"],
            "correct_answer": "4",
        },
    ],
    "estimated_mastery_delta": {"kp1": 0.9},
})


class TestQuizGeneration:
    @pytest.mark.asyncio
    async def test_writing_a_quiz_carries_no_mastery_field(self) -> None:
        """Quiz writing must not assert a mastery value at all.

        The old ``AssessmentOutput.mastery_delta`` held the LLM's guess
        (defaulting to 0.1) and no consumer ever read it. The field is gone;
        mastery now comes from ``grade_attempt``. Asserting absence is stronger
        than asserting an empty dict — an empty dict would let the field
        quietly return.
        """
        agent = _agent(_QUIZ_JSON)
        unit = KnowledgeUnit(id="kp1", name="加法", description="d", difficulty=2)

        out = await agent.run_legacy(unit)

        assert len(out.quiz) == 2
        assert not hasattr(out, "mastery_delta"), (
            "quiz writing must not carry a mastery field; mastery is measured "
            "from graded responses, not asserted before anyone answers"
        )

    @pytest.mark.asyncio
    async def test_llm_supplied_mastery_delta_does_not_reach_the_output(self) -> None:
        """The prompt still returns the field; it must be discarded."""
        agent = _agent(_QUIZ_JSON)
        unit = KnowledgeUnit(id="kp1", name="加法", description="d", difficulty=2)

        out = await agent.run_legacy(unit)

        assert "mastery_delta" not in out.model_dump()

    @pytest.mark.asyncio
    async def test_questions_carry_the_unit_difficulty(self) -> None:
        """Grading needs item difficulty; the LLM is not trusted to supply it."""
        agent = _agent(_QUIZ_JSON)
        unit = KnowledgeUnit(id="kp1", name="加法", description="d", difficulty=4)

        out = await agent.run_legacy(unit)

        assert all(q.difficulty == 4 for q in out.quiz)

    @pytest.mark.asyncio
    async def test_falls_back_to_quiz_bank_without_reporting_mastery(self) -> None:
        agent = _agent(llm_response=None)  # force the bank path
        unit = KnowledgeUnit(id="kp-intro", name="导论", description="d", difficulty=3)

        out = await agent.run_legacy(unit)

        assert not hasattr(out, "mastery_delta")
        assert out.quiz, "kp-intro exists in the bank, so questions are expected"

    @pytest.mark.asyncio
    async def test_bank_questions_get_the_unit_difficulty(self) -> None:
        """The bank table has no difficulty column, so it must be stamped on.

        Without this, banked questions would all score as average items and the
        IRT estimate would be blind to how hard they actually are.
        """
        agent = _agent(llm_response=None)
        unit = KnowledgeUnit(id="kp-intro", name="导论", description="d", difficulty=5)

        out = await agent.run_legacy(unit)

        assert out.quiz
        assert all(q.difficulty == 5 for q in out.quiz)

    @pytest.mark.asyncio
    async def test_bank_stamping_does_not_clobber_an_existing_difficulty(self) -> None:
        """A bank entry that already declares difficulty keeps its own value."""
        agent = _agent(llm_response=None)
        unit = KnowledgeUnit(id="kp-intro", name="导论", description="d", difficulty=2)

        original = get_questions("kp-intro")
        assert original, "bank must have entries for this kp"
        # Stamp one entry so the "do not overwrite" branch is exercised.
        patched = [original[0].model_copy(update={"difficulty": 4}), *original[1:]]

        import src.agents.assessment.agent as agent_module

        original_getter = agent_module.get_questions
        agent_module.get_questions = lambda _kp: list(patched)
        try:
            out = await agent.run_legacy(unit)
        finally:
            agent_module.get_questions = original_getter

        assert out.quiz[0].difficulty == 4, "explicit difficulty must survive"
        assert all(q.difficulty == 2 for q in out.quiz[1:])


class TestGradeAttempt:
    def _questions(self, difficulty: int | None = 3) -> list[QuizQuestion]:
        return [
            QuizQuestion(
                id="q1", content="1+1=?", correct_answer="2",
                knowledge_point_id="kp1", difficulty=difficulty,
            ),
            QuizQuestion(
                id="q2", content="2+2=?", correct_answer="4",
                knowledge_point_id="kp1", difficulty=difficulty,
            ),
        ]

    @pytest.mark.asyncio
    async def test_all_correct_yields_positive_delta(self) -> None:
        agent = _agent(_QUIZ_JSON)

        result = await agent.grade_attempt(
            self._questions(), {"q1": "2", "q2": "4"},
            knowledge_point_id="kp1", prior_mastery=0.4,
        )

        assert result["score"] == pytest.approx(1.0)
        assert result["mastery_delta"]["kp1"] > 0

    @pytest.mark.asyncio
    async def test_all_wrong_yields_negative_delta(self) -> None:
        agent = _agent(_QUIZ_JSON)

        result = await agent.grade_attempt(
            self._questions(), {"q1": "9", "q2": "9"},
            knowledge_point_id="kp1", prior_mastery=0.8,
        )

        assert result["score"] == pytest.approx(0.0)
        assert result["mastery_delta"]["kp1"] < 0

    @pytest.mark.asyncio
    async def test_delta_is_bounded(self) -> None:
        agent = _agent(_QUIZ_JSON)
        result = await agent.grade_attempt(
            self._questions(), {"q1": "2", "q2": "4"},
            knowledge_point_id="kp1", prior_mastery=0.0,
        )
        assert -1.0 <= result["mastery_delta"]["kp1"] <= 1.0

    @pytest.mark.asyncio
    async def test_harder_questions_count_for_more(self) -> None:
        """Same answers, harder items => stronger evidence of mastery."""
        agent = _agent(_QUIZ_JSON)

        easy = await agent.grade_attempt(
            self._questions(difficulty=1), {"q1": "2", "q2": "4"},
            knowledge_point_id="kp1",
        )
        hard = await agent.grade_attempt(
            self._questions(difficulty=5), {"q1": "2", "q2": "4"},
            knowledge_point_id="kp1",
        )

        assert hard["graded"].mastery > easy["graded"].mastery

    @pytest.mark.asyncio
    async def test_kp_inferred_from_first_question(self) -> None:
        """Callers that do not pass a KP still get a keyed delta."""
        agent = _agent(_QUIZ_JSON)
        result = await agent.grade_attempt(self._questions(), {"q1": "2", "q2": "4"})
        assert set(result["mastery_delta"]) == {"kp1"}

    @pytest.mark.asyncio
    async def test_positional_answers_accepted(self) -> None:
        agent = _agent(_QUIZ_JSON)
        result = await agent.grade_attempt(self._questions(), ["2", "4"])
        assert result["score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_graded_detail_is_returned(self) -> None:
        agent = _agent(_QUIZ_JSON)
        result = await agent.grade_attempt(
            self._questions(), {"q1": "2", "q2": "9"}
        )
        graded = result["graded"]
        assert graded.n_correct == 1
        assert graded.n_questions == 2
        assert len(graded.per_question) == 2
