"""Tests for the IRT (1PL/Rasch) mastery estimator.

Motivation: EduMap's mastery number used to be whatever the LLM asserted at
quiz-generation time (with a hard-coded 0.1 fallback). These tests pin the
replacement to a model that is *identifiable from the maths* rather than
plausible-sounding — so the expected values below are derived independently,
not read off the implementation.

Key properties under test:
  * Monotonicity — more ability must mean higher P(correct).
  * Recovery — from data generated at a known ability, the estimator must
    find it back within a tolerance.
  * Bounded on degenerate input — all-correct / all-wrong must not diverge to
    infinity (the plain MLE does; the weak prior is what prevents it).
  * SE shrinks with more responses — the estimate reports its own uncertainty.
"""

from __future__ import annotations

import math
import random

import pytest

from src.agents.assessment.irt import (
    ItemResponse,
    ability_to_mastery,
    estimate_ability,
    estimate_item_difficulty,
    probability_correct,
)


class TestProbabilityCorrect:
    def test_equal_ability_and_difficulty_is_one_half(self) -> None:
        assert probability_correct(0.0, 0.0) == pytest.approx(0.5)

    def test_higher_ability_increases_probability(self) -> None:
        """Independent check: 1 logit above an item => σ(1) ≈ 0.7311."""
        assert probability_correct(1.0, 0.0) == pytest.approx(0.7310585786, abs=1e-9)

    def test_one_logit_below_is_the_complement(self) -> None:
        assert probability_correct(-1.0, 0.0) == pytest.approx(
            1 - 0.7310585786, abs=1e-9
        )

    def test_easier_item_raises_probability(self) -> None:
        assert probability_correct(0.0, -2.0) > probability_correct(0.0, 0.0)

    def test_monotone_in_ability(self) -> None:
        probs = [probability_correct(a / 10, 0.0) for a in range(-30, 31)]
        assert probs == sorted(probs)

    def test_extreme_inputs_do_not_overflow(self) -> None:
        """σ(-1000) must be 0-ish, not an OverflowError."""
        assert probability_correct(-1000.0, 0.0) == pytest.approx(0.0, abs=1e-9)
        assert probability_correct(1000.0, 0.0) == pytest.approx(1.0, abs=1e-9)


class TestAbilityToMastery:
    def test_zero_ability_is_half(self) -> None:
        assert ability_to_mastery(0.0) == pytest.approx(0.5)

    def test_monotone_and_bounded(self) -> None:
        values = [ability_to_mastery(a / 5) for a in range(-20, 21)]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)


class TestEstimateAbility:
    def _responses(self, ability: float, n: int, seed: int = 1,
                   difficulty: float = 0.0) -> list[ItemResponse]:
        rng = random.Random(seed)
        p = probability_correct(ability, difficulty)
        return [
            ItemResponse(item_id=f"i{k}", correct=rng.random() < p, difficulty=difficulty)
            for k in range(n)
        ]

    def test_recovers_known_ability(self) -> None:
        """Data generated at θ=1.5 must be fitted back to ≈1.5."""
        responses = self._responses(1.5, n=400, seed=11)
        est = estimate_ability(responses)
        assert est.ability == pytest.approx(1.5, abs=0.35)

    def test_recovers_low_ability(self) -> None:
        responses = self._responses(-1.2, n=400, seed=12)
        est = estimate_ability(responses)
        assert est.ability == pytest.approx(-1.2, abs=0.35)

    def test_all_correct_is_pulled_toward_the_prior(self) -> None:
        """A perfect run must not yield a maximal ability estimate.

        The estimator is MAP, not MLE: the prior is what stops an all-correct
        pattern from running to the top of the search range. Asserting only
        "finite and < 4.0" would pass even with the prior removed, because the
        search interval itself is bounded — that would be a weak assertion
        (the bound would come from the optimiser, not the model). What must
        hold is that the estimate lands *strictly inside* the range and below
        what an unbounded MLE would report.
        """
        responses = [ItemResponse(item_id=f"i{k}", correct=True) for k in range(10)]
        est = estimate_ability(responses)
        assert math.isfinite(est.ability)
        assert est.ability < 3.5, "prior should pull a perfect run away from the ceiling"
        assert est.mastery < 0.98

    def test_stronger_prior_pulls_harder(self) -> None:
        """With more prior weight the same data yields a less extreme estimate."""
        responses = [ItemResponse(item_id=f"i{k}", correct=True) for k in range(10)]
        weak = estimate_ability(responses, prior=0.0, prior_weight=0.1)
        strong = estimate_ability(responses, prior=0.0, prior_weight=50.0)
        assert strong.ability < weak.ability

    def test_all_wrong_is_pulled_toward_the_prior(self) -> None:
        responses = [ItemResponse(item_id=f"i{k}", correct=False) for k in range(10)]
        est = estimate_ability(responses)
        assert math.isfinite(est.ability)
        assert est.ability > -3.5
        assert est.mastery > 0.02

    def test_no_responses_returns_prior(self) -> None:
        est = estimate_ability([])
        assert est.n_responses == 0
        assert est.ability == pytest.approx(0.0)
        assert est.mastery == pytest.approx(0.5)
        assert est.standard_error == float("inf")

    def test_harder_items_lower_the_estimate(self) -> None:
        """Same responses, harder items => the learner looks more able."""
        # all correct against hard items implies higher ability than against easy
        hard = [ItemResponse(item_id=f"h{k}", correct=True, difficulty=1.0) for k in range(8)]
        easy = [ItemResponse(item_id=f"e{k}", correct=True, difficulty=-1.0) for k in range(8)]
        assert estimate_ability(hard).ability > estimate_ability(easy).ability

    def test_standard_error_shrinks_with_more_responses(self) -> None:
        few = estimate_ability(self._responses(0.0, n=5, seed=3))
        many = estimate_ability(self._responses(0.0, n=200, seed=3))
        assert many.standard_error < few.standard_error

    def test_residuals_are_reported_per_item(self) -> None:
        responses = [
            ItemResponse(item_id="a", correct=True),
            ItemResponse(item_id="b", correct=False),
        ]
        est = estimate_ability(responses)
        assert set(est.per_item_residual) == {"a", "b"}

    def test_deterministic(self) -> None:
        responses = self._responses(0.7, n=50, seed=5)
        assert estimate_ability(responses).ability == estimate_ability(responses).ability


class TestEstimateItemDifficulty:
    def test_easy_item_is_estimated_low(self) -> None:
        """Everyone at θ=0 passing => the item is easier than average."""
        outcomes = [(0.0, True) for _ in range(20)]
        assert estimate_item_difficulty(outcomes) < 0.0

    def test_hard_item_is_estimated_high(self) -> None:
        rng = random.Random(4)
        outcomes = [(0.0, rng.random() < 0.1) for _ in range(200)]
        assert estimate_item_difficulty(outcomes) > 0.5

    def test_no_data_returns_prior(self) -> None:
        assert estimate_item_difficulty([], prior=1.25) == pytest.approx(1.25)

    def test_recovers_known_difficulty(self) -> None:
        """Responses sampled at θ=1.0 against b=2.0 => P≈σ(-1)≈0.269."""
        rng = random.Random(9)
        p = probability_correct(1.0, 2.0)
        outcomes = [(1.0, rng.random() < p) for _ in range(800)]
        assert estimate_item_difficulty(outcomes) == pytest.approx(2.0, abs=0.35)
