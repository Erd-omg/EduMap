"""Quiz grading and mastery estimation from observed answers.

This closes the loop the Assessment agent used to leave open.

Before
------
``AssessmentAgent`` generated questions and, *in the same LLM call*, asked the
model to invent an ``estimated_mastery_delta`` — before any learner had
answered anything.  Nothing ever compared a learner's answer to the key, so
there was no observed performance to estimate mastery from; the missing value
fell back to a hard-coded ``0.1``.

After
-----
Grading is deterministic (compare against ``correct_answer``), and mastery is
estimated from the graded responses with the 1PL/Rasch model in
:mod:`src.agents.assessment.irt`.  The LLM's role shrinks to *writing*
questions; it no longer opines on how well the learner did.

Splitting these apart also means the helper is a pure function of
(questions, answers) — testable without a model, and impossible to
accidentally tune by prompt changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.agents.assessment.irt import (
    ItemResponse,
    MasteryEstimate,
    estimate_ability,
)

logger = logging.getLogger(__name__)

# A difficulty prior in logits, used when a question carries no explicit
# difficulty.  0.0 = "assume an average item", which is the honest default for
# a freshly written question bank that has never been calibrated.
DEFAULT_ITEM_DIFFICULTY = 0.0

# 1..5 (the scale used by KnowledgeUnit.difficulty) mapped onto the IRT logit
# scale.  The linear map is deliberate: with no calibration data there is no
# basis for anything fancier, and a monotone map is enough for the estimate to
# be better than assuming every item is identical.
_DIFFICULTY_MIN = 1
_DIFFICULTY_MAX = 5
_DIFFICULTY_LOGIT_SPAN = 2.0


def difficulty_to_logit(difficulty: float | None) -> float:
    """Map a 1–5 author-assigned difficulty onto the IRT logit scale.

    ``None`` or an out-of-range value yields the neutral default rather than
    an extrapolated extreme — an unlabelled item is treated as average, not
    as impossibly hard.
    """
    if difficulty is None:
        return DEFAULT_ITEM_DIFFICULTY
    try:
        value = float(difficulty)
    except (TypeError, ValueError):
        return DEFAULT_ITEM_DIFFICULTY
    if not (_DIFFICULTY_MIN <= value <= _DIFFICULTY_MAX):
        return DEFAULT_ITEM_DIFFICULTY
    # 1 -> -span/2, 3 -> 0, 5 -> +span/2
    midpoint = (_DIFFICULTY_MIN + _DIFFICULTY_MAX) / 2
    return (value - midpoint) / (midpoint - _DIFFICULTY_MIN) * (_DIFFICULTY_LOGIT_SPAN / 2)


def _normalise(text: object) -> str:
    """Normalise an answer for comparison.

    Case- and whitespace-insensitive.  Multi-choice answers arrive as letters
    ("B") from the UI but may be stored as full option text, so both sides are
    stripped and lowercased; a blank answer never matches.
    """
    if text is None:
        return ""
    return str(text).strip().lower()


def grade_answer(question, answer: object) -> bool:
    """Whether *answer* matches the question's key.

    A blank answer is always wrong, even when the key is also blank — an
    unanswered question must not be scored as correct.
    """
    submitted = _normalise(answer)
    if not submitted:
        return False
    expected = _normalise(getattr(question, "correct_answer", ""))
    return submitted == expected


@dataclass
class GradedQuiz:
    """Outcome of grading one quiz attempt."""

    n_questions: int = 0
    n_correct: int = 0
    per_question: list[dict] = field(default_factory=list)
    mastery: float = 0.5
    ability: float = 0.0
    standard_error: float = float("inf")
    # Kept so callers can log/persist the raw responses for later refitting.
    responses: list[ItemResponse] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction correct — the number the rest of the system consumes."""
        return self.n_correct / self.n_questions if self.n_questions else 0.0

    def to_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "n_correct": self.n_correct,
            "score": round(self.score, 4),
            "mastery": round(self.mastery, 4),
            "ability": round(self.ability, 4),
            "standard_error": round(self.standard_error, 4),
        }


def grade_quiz(
    questions: Sequence,
    answers: Mapping[str, object] | Iterable[object],
    *,
    difficulties: Mapping[str, float] | None = None,
) -> GradedQuiz:
    """Grade a quiz and estimate mastery from the observed responses.

    Args:
        questions: objects with ``id`` and ``correct_answer`` (``QuizQuestion``).
        answers: either ``{question_id: answer}`` or an iterable of answers
            positionally aligned with *questions*.
        difficulties: optional ``{question_id: difficulty_1_to_5}`` override.
            Falls back to each question's own ``difficulty`` attribute, then to
            the neutral default.

    Returns:
        A :class:`GradedQuiz`.  An empty *questions* sequence yields a
        default-mastery result rather than raising — a quiz that failed to
        generate must not look like a perfect or a zero score.
    """
    question_list = list(questions)

    if isinstance(answers, Mapping):
        lookup = {str(k): v for k, v in answers.items()}
        get_answer = lambda q, i: lookup.get(str(getattr(q, "id", i)))  # noqa: E731
    else:
        answer_list = list(answers)
        get_answer = lambda q, i: answer_list[i] if i < len(answer_list) else None  # noqa: E731

    responses: list[ItemResponse] = []
    per_question: list[dict] = []
    n_correct = 0

    for idx, question in enumerate(question_list):
        submitted = get_answer(question, idx)
        correct = grade_answer(question, submitted)
        n_correct += int(correct)

        qid = str(getattr(question, "id", idx))
        raw_difficulty = None
        if difficulties and qid in difficulties:
            raw_difficulty = difficulties[qid]
        else:
            raw_difficulty = getattr(question, "difficulty", None)

        responses.append(
            ItemResponse(
                item_id=qid,
                correct=correct,
                difficulty=difficulty_to_logit(raw_difficulty),
            )
        )
        per_question.append(
            {"id": qid, "correct": correct, "difficulty": round(responses[-1].difficulty, 3)}
        )

    estimate: MasteryEstimate = estimate_ability(responses)

    return GradedQuiz(
        n_questions=len(question_list),
        n_correct=n_correct,
        per_question=per_question,
        mastery=estimate.mastery,
        ability=estimate.ability,
        standard_error=estimate.standard_error,
        responses=responses,
    )


def mastery_delta_from_grading(
    graded: GradedQuiz,
    prior_mastery: float | None = None,
) -> float:
    """Signed change in mastery implied by a graded attempt.

    The Assessment agent's contract is a *delta*, so grading is expressed in
    those terms.  Without a prior the delta is measured against the neutral
    0.5, which reads as "this attempt moved the estimate to here from
    unknown"; with a prior it is the honest before/after difference.

    Bounded to [-1, 1] to match the existing ``mastery_delta`` consumer.
    """
    baseline = 0.5 if prior_mastery is None else prior_mastery
    delta = graded.mastery - baseline
    return max(-1.0, min(1.0, delta))
