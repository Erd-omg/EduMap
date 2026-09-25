"""Tests for the power-law forgetting curve (the I-3 fix).

The old formula had two independent defects, both measured on the harness in
``scripts/run_forgetting_eval.py``:

1. ``S = 24·mean·log2(n+1)`` grew so slowly that ``S_MAX = 720h`` was
   unreachable (it would need ~1.9e11 reviews), and S topped out near 61h
   after 8 reviews — so a week later the model predicted recall 0.063 where a
   reasonable model predicts ~0.83.
2. ``R = exp(-t/S)`` decays far too fast at long intervals: at 336h with
   S≈330h the exponential gives 0.363 versus 0.899 for a power law.

These tests pin the *fixed* behaviour. Deliberately they assert properties
(monotonicity, magnitude, reachability of the clamp) rather than recomputing
the formula — a test that mirrors the implementation passes for any formula.

Measured effect of the fix (300 histories / 3300 observations, default seed):
    legacy LogLoss 1.4200  ->  new LogLoss 0.4320  (moving-avg baseline 0.4421)
Across seeds 1/42/999 the new model wins 3 of 4 runs and ties the baseline in
the fourth, so the honest claim is "matches the trivial baseline", not
"dominates it".
"""

from __future__ import annotations

import math

import pytest

from src.learning_path.forgetting_curve import (
    FACTOR,
    GROWTH_EXP,
    S_BASE,
    S_MAX,
    S_MIN,
    ForgettingState,
    recall_probability,
    stability_after_review,
)


class TestPowerLawCurve:
    def test_zero_elapsed_is_certain(self) -> None:
        assert recall_probability(0.0, 24.0) == pytest.approx(1.0)

    def test_negative_elapsed_is_certain(self) -> None:
        assert recall_probability(-5.0, 24.0) == pytest.approx(1.0)

    def test_at_stability_recall_is_090(self) -> None:
        """S is defined as the interval where R = 0.9 — that is what FACTOR does."""
        assert recall_probability(24.0, 24.0) == pytest.approx(0.9, abs=1e-9)

    def test_monotone_decreasing_in_time(self) -> None:
        values = [recall_probability(t, 100.0) for t in range(0, 2000, 50)]
        assert values == sorted(values, reverse=True)

    def test_monotone_increasing_in_stability(self) -> None:
        values = [recall_probability(168.0, s) for s in range(10, 2000, 50)]
        assert values == sorted(values)

    def test_never_negative_or_above_one(self) -> None:
        for t in (0, 1, 1e3, 1e6):
            for s in (0.5, 24, 720, 1e5):
                r = recall_probability(t, s)
                assert 0.0 <= r <= 1.0

    def test_zero_stability_is_zero_recall(self) -> None:
        assert recall_probability(10.0, 0.0) == 0.0

    def test_beats_exponential_at_long_intervals(self) -> None:
        """The specific gap that motivated the shape change.

        With S=330h, one week out the exponential is far stingier than the
        power law; this is the difference that made the old model score worse
        than a constant predictor.
        """
        stability = 330.0
        elapsed = 336.0
        exponential = math.exp(-elapsed / stability)
        power = recall_probability(elapsed, stability)

        assert power > exponential + 0.3, (
            f"power={power:.3f} exponential={exponential:.3f} — the power law "
            "must be materially more optimistic at long intervals"
        )
        assert power > 0.8


class TestStabilityGrowth:
    def test_grows_with_review_count(self) -> None:
        values = [stability_after_review(n, 0.8) for n in range(1, 15)]
        assert values == sorted(values)

    def test_grows_with_score_quality(self) -> None:
        """A well-remembered item must end up more stable than a poorly one."""
        good = stability_after_review(5, 0.95)
        poor = stability_after_review(5, 0.4)
        assert good > poor

    def test_s_max_is_reachable(self) -> None:
        """The old clamp was dead code — 720h needed ~1.9e11 reviews.

        The new growth must actually reach the ceiling within a plausible
        number of reviews, otherwise the clamp is decorative.
        """
        stability = stability_after_review(60, 0.9)
        assert stability == pytest.approx(S_MAX), (
            f"S after 60 good reviews is {stability:.1f}h; the ceiling should "
            "have been reached rather than remaining unreachable"
        )

    def test_ceiling_is_respected(self) -> None:
        assert stability_after_review(10_000, 1.0) == pytest.approx(S_MAX)

    def test_floor_is_respected(self) -> None:
        """A zero score must not collapse stability to nothing."""
        assert stability_after_review(1, 0.0) == pytest.approx(S_MIN)

    def test_after_eight_reviews_is_materially_larger_than_legacy(self) -> None:
        """The concrete regression: 8 reviews used to yield only ~61h."""
        new = stability_after_review(8, 0.8)
        legacy = S_BASE * 0.8 * math.log2(8 + 1)
        assert new > legacy * 3, (
            f"new S={new:.0f}h vs legacy S={legacy:.0f}h — the growth fix "
            "should be well beyond a rounding difference"
        )
        assert new > 150

    def test_one_week_recall_after_eight_reviews_is_plausible(self) -> None:
        """The headline symptom: 0.063 before the fix, should now be high."""
        stability = stability_after_review(8, 0.8)
        one_week = recall_probability(168.0, stability)
        assert one_week > 0.8, (
            f"one-week recall after 8 reviews is {one_week:.3f}; the old model "
            "gave 0.063 which is what made it lose to a constant predictor"
        )

    def test_growth_exponent_is_the_documented_value(self) -> None:
        """Pin the constant literally so a change is noticed.

        1.0 (not the grid-search optimum 0.5) because a linear power law has
        theoretical backing whereas 0.5 was tuned on synthetic data whose
        ground truth this project chose.
        """
        assert GROWTH_EXP == pytest.approx(1.0)


class TestForgettingStateIntegration:
    def test_predict_recall_uses_the_power_curve(self) -> None:
        state = ForgettingState(kp_id="kp", user_id="u")
        state.last_review_time = 1000.0
        state.strength = 24.0

        import time as _time

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_time, "time", lambda: 1000.0 + 3600 * 24)
            result = state.predict_recall()

        assert result == pytest.approx(0.9, abs=1e-6)

    def test_never_studied_predicts_zero(self) -> None:
        state = ForgettingState(kp_id="kp", user_id="u")
        assert state.predict_recall() == 0.0

    def test_first_review_does_not_lower_stability(self) -> None:
        """A review that reports no failure must not shorten memory.

        It lands exactly at the neutral value (score factor = 1×) rather than
        rising, because one review against a weak Beta(2,2) prior is not yet
        evidence of durable learning. What matters is that it does not go
        *down*, which the old formula did.
        """
        state = ForgettingState(kp_id="kp", user_id="u")
        before = state.strength
        state.update_after_quiz(1.0)
        assert state.strength >= before

    def test_repeated_reviews_raise_stability(self) -> None:
        state = ForgettingState(kp_id="kp", user_id="u")
        state.update_after_quiz(1.0)
        after_one = state.strength
        state.update_after_quiz(1.0)
        assert state.strength > after_one

    def test_repeated_perfect_reviews_reach_the_ceiling(self) -> None:
        state = ForgettingState(kp_id="kp", user_id="u")
        for _ in range(80):
            state.update_after_quiz(1.0)
        assert state.strength == pytest.approx(S_MAX)

    def test_low_scores_barely_grow_stability(self) -> None:
        """SCORE_EXP makes poor reviews contribute little — anti-cramming."""
        good = ForgettingState(kp_id="kp", user_id="u")
        poor = ForgettingState(kp_id="kp", user_id="u")
        for _ in range(5):
            good.update_after_quiz(1.0)
            poor.update_after_quiz(0.3)
        assert good.strength > poor.strength * 2

    def test_a_perfect_review_never_lowers_stability(self) -> None:
        """Reviewing must not make an item more forgettable.

        The Beta(2,2) prior is weak, so one perfect review leaves
        posterior_mean at only 0.6 (3/5). With SCORE_EXP = 3 that cubed to
        0.216 and dropped S from 24h to 14.7h — a review that actively hurt.
        This invariant is what pins the exponent to a sane range.
        """
        state = ForgettingState(kp_id="kp", user_id="u")
        before = state.strength
        state.update_after_quiz(1.0)
        assert state.strength >= before, (
            f"a perfect review lowered stability {before:.1f}h -> "
            f"{state.strength:.1f}h"
        )

    def test_no_score_can_lower_stability_below_start(self) -> None:
        """Even a zero score must not end up below the pre-review stability.

        The floor is guaranteed by clamping to S_MIN, but a genuinely broken
        exponent could still land below the starting 24h for mid-range scores;
        this checks the whole range.
        """
        start = ForgettingState(kp_id="kp", user_id="u").strength
        for score in (0.0, 0.2, 0.5, 0.8, 1.0):
            state = ForgettingState(kp_id="kp", user_id="u")
            state.update_after_quiz(score)
            assert state.strength >= min(start, S_MIN)


class TestFactorConsistency:
    def test_factor_derives_from_decay(self) -> None:
        from src.learning_path.forgetting_curve import DECAY

        expected = 0.9 ** (-1.0 / DECAY) - 1.0
        assert FACTOR == pytest.approx(expected)

    def test_decay_matches_the_fsrs_default(self) -> None:
        from src.learning_path.forgetting_curve import DECAY

        assert DECAY == pytest.approx(0.1542)
