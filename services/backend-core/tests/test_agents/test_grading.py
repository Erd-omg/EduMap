"""Tests for quiz grading and mastery-from-observed-answers.

The behaviour these pin down replaces an LLM's guess with a computed result,
so the assertions check the *arithmetic and the IRT link*, not just that a
number came back. A test that only asserted ``0 <= mastery <= 1`` would pass
for a constant function.
"""

from __future__ import annotations

import pytest

from src.agents.assessment.grading import (
    DEFAULT_ITEM_DIFFICULTY,
    difficulty_to_logit,
    grade_answer,
    grade_quiz,
    mastery_delta_from_grading,
)


class _Q:
    """Minimal stand-in for QuizQuestion (only the fields grading reads)."""

    def __init__(self, id: str, correct: str, difficulty=None):
        self.id = id
        self.correct_answer = correct
        self.difficulty = difficulty


class TestDifficultyToLogit:
    def test_midpoint_is_neutral(self) -> None:
        assert difficulty_to_logit(3) == pytest.approx(0.0)

    def test_monotone_increasing(self) -> None:
        values = [difficulty_to_logit(d) for d in (1, 2, 3, 4, 5)]
        assert values == sorted(values)

    def test_symmetric_around_neutral(self) -> None:
        assert difficulty_to_logit(1) == pytest.approx(-difficulty_to_logit(5))

    def test_none_is_neutral(self) -> None:
        assert difficulty_to_logit(None) == pytest.approx(DEFAULT_ITEM_DIFFICULTY)

    def test_out_of_range_is_neutral_not_extreme(self) -> None:
        """An unlabelled/typo'd difficulty must not produce a wild item."""
        assert difficulty_to_logit(0) == pytest.approx(0.0)
        assert difficulty_to_logit(99) == pytest.approx(0.0)

    def test_non_numeric_is_neutral(self) -> None:
        assert difficulty_to_logit("hard") == pytest.approx(0.0)


class TestGradeAnswer:
    def test_exact_match(self) -> None:
        assert grade_answer(_Q("q1", "B"), "B")

    def test_case_and_whitespace_insensitive(self) -> None:
        assert grade_answer(_Q("q1", "b"), "  B  ")

    def test_wrong_answer(self) -> None:
        assert not grade_answer(_Q("q1", "B"), "C")

    def test_blank_answer_is_wrong_even_against_blank_key(self) -> None:
        """An unanswered question must never score as correct."""
        assert not grade_answer(_Q("q1", ""), "")
        assert not grade_answer(_Q("q1", "B"), None)
        assert not grade_answer(_Q("q1", "B"), "   ")

    def test_numeric_answers(self) -> None:
        assert grade_answer(_Q("q1", "42"), 42)


class TestGradeQuiz:
    def test_all_correct(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(4)]
        g = grade_quiz(qs, {"q0": "A", "q1": "A", "q2": "A", "q3": "A"})
        assert g.n_correct == 4
        assert g.score == pytest.approx(1.0)
        assert g.n_questions == 4

    def test_all_wrong(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(4)]
        g = grade_quiz(qs, {"q0": "B", "q1": "B", "q2": "B", "q3": "B"})
        assert g.n_correct == 0
        assert g.score == pytest.approx(0.0)

    def test_positional_answers(self) -> None:
        qs = [_Q("q0", "A"), _Q("q1", "B")]
        g = grade_quiz(qs, ["A", "B"])
        assert g.n_correct == 2

    def test_missing_answer_counts_as_wrong(self) -> None:
        """A submission covering only part of the quiz must not inflate score."""
        qs = [_Q("q0", "A"), _Q("q1", "B")]
        g = grade_quiz(qs, {"q0": "A"})
        assert g.n_correct == 1
        assert g.score == pytest.approx(0.5)

    def test_empty_quiz_does_not_divide_by_zero(self) -> None:
        g = grade_quiz([], {})
        assert g.n_questions == 0
        assert g.score == pytest.approx(0.0)
        assert g.mastery == pytest.approx(0.5)

    def test_perfect_score_raises_mastery_above_neutral(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "A" for i in range(6)})
        assert g.mastery > 0.5

    def test_zero_score_lowers_mastery_below_neutral(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "Z" for i in range(6)})
        assert g.mastery < 0.5

    def test_mastery_is_monotone_in_score(self) -> None:
        """More correct answers must mean higher estimated mastery."""
        qs = [_Q(f"q{i}", "A") for i in range(8)]
        masteries = []
        for n_correct in range(9):
            answers = {
                f"q{i}": ("A" if i < n_correct else "Z") for i in range(8)
            }
            masteries.append(grade_quiz(qs, answers).mastery)
        assert masteries == sorted(masteries)

    def test_hard_questions_yield_higher_mastery_than_easy_ones(self) -> None:
        """Same all-correct pattern, harder items => stronger evidence."""
        hard = [_Q(f"h{i}", "A", difficulty=5) for i in range(6)]
        easy = [_Q(f"e{i}", "A", difficulty=1) for i in range(6)]
        hard_m = grade_quiz(hard, {f"h{i}": "A" for i in range(6)}).mastery
        easy_m = grade_quiz(easy, {f"e{i}": "A" for i in range(6)}).mastery
        assert hard_m > easy_m

    def test_explicit_difficulty_override_wins(self) -> None:
        qs = [_Q("q0", "A", difficulty=1)]
        g = grade_quiz(qs, {"q0": "A"}, difficulties={"q0": 5})
        assert g.per_question[0]["difficulty"] == pytest.approx(
            difficulty_to_logit(5)
        )

    def test_responses_are_retained_for_refitting(self) -> None:
        qs = [_Q("q0", "A"), _Q("q1", "B")]
        g = grade_quiz(qs, {"q0": "A", "q1": "Z"})
        assert [r.correct for r in g.responses] == [True, False]

    def test_to_dict_shape(self) -> None:
        qs = [_Q("q0", "A")]
        d = grade_quiz(qs, {"q0": "A"}).to_dict()
        assert set(d) == {
            "n_questions", "n_correct", "score",
            "mastery", "ability", "standard_error",
        }


class TestMasteryDelta:
    def test_improvement_is_positive(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "A" for i in range(6)})
        assert mastery_delta_from_grading(g, prior_mastery=0.3) > 0

    def test_regression_is_negative(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "Z" for i in range(6)})
        assert mastery_delta_from_grading(g, prior_mastery=0.8) < 0

    def test_no_prior_measures_against_neutral(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "A" for i in range(6)})
        assert mastery_delta_from_grading(g) == pytest.approx(g.mastery - 0.5)

    def test_delta_is_bounded(self) -> None:
        qs = [_Q(f"q{i}", "A") for i in range(6)]
        g = grade_quiz(qs, {f"q{i}": "A" for i in range(6)})
        assert -1.0 <= mastery_delta_from_grading(g, prior_mastery=0.0) <= 1.0
