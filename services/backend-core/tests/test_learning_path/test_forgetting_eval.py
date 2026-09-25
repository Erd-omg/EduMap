"""Tests for the forgetting-curve evaluation harness.

The harness exists to produce a *trustworthy* number about which recall model
is best.  If the metrics were insensitive, that number would be meaningless —
so the central tests here are ones that fail when a metric is broken:

* AUC must separate a good model from an inverted one.
* A model that always predicts the base rate must score exactly 0.5 AUC.
* LogLoss must punish confident mistakes harder than hedged ones.
* RMSE(bins) must be ~0 for a perfectly calibrated model and large for a
  mis-calibrated one.

A test asserting only "0 <= log_loss" would pass for a stub returning a
constant.
"""

from __future__ import annotations

import math
import random

import pytest

from src.learning_path.forgetting_eval import (
    ModelReport,
    ReviewObservations,
    auc,
    evaluate_model,
    log_loss,
    predict_all,
    rmse,
)


def _obs(predicted: float, recalled: bool) -> ReviewObservations:
    return ReviewObservations(
        elapsed_hours=1.0, predicted=predicted, recalled=recalled, prior_count=1
    )


class TestLogLoss:
    def test_perfect_confident_predictions_are_near_zero(self) -> None:
        obs = [_obs(0.99, True), _obs(0.01, False)]
        assert log_loss(obs) < 0.02

    def test_worst_confident_predictions_are_large(self) -> None:
        obs = [_obs(0.99, False), _obs(0.01, True)]
        assert log_loss(obs) > 4.0

    def test_hedged_prediction_costs_less_than_confident_error(self) -> None:
        hedged = log_loss([_obs(0.5, True)])
        confident = log_loss([_obs(0.99, False)])
        assert hedged < confident

    def test_exactly_hand_computable(self) -> None:
        """Two observations: -ln(0.8) and -ln(1-0.3), averaged."""
        obs = [_obs(0.8, True), _obs(0.3, False)]
        expected = (-math.log(0.8) + -math.log(0.7)) / 2
        assert log_loss(obs) == pytest.approx(expected, abs=1e-9)

    def test_empty_is_zero(self) -> None:
        assert log_loss([]) == 0.0

    def test_extreme_probabilities_do_not_produce_inf(self) -> None:
        """p=0 with a positive outcome must be clamped, not infinite."""
        obs = [_obs(0.0, True), _obs(1.0, False)]
        assert math.isfinite(log_loss(obs))


class TestAUC:
    def test_perfect_separation_is_one(self) -> None:
        obs = [_obs(0.9, True), _obs(0.8, True), _obs(0.2, False), _obs(0.1, False)]
        assert auc(obs) == pytest.approx(1.0)

    def test_inverted_separation_is_zero(self) -> None:
        obs = [_obs(0.1, True), _obs(0.2, True), _obs(0.8, False), _obs(0.9, False)]
        assert auc(obs) == pytest.approx(0.0)

    def test_constant_prediction_is_chance(self) -> None:
        """A model that predicts the same value everywhere cannot discriminate."""
        obs = [_obs(0.5, True), _obs(0.5, False), _obs(0.5, True), _obs(0.5, False)]
        assert auc(obs) == pytest.approx(0.5)

    def test_ties_count_as_half(self) -> None:
        obs = [_obs(0.5, True), _obs(0.5, False)]
        assert auc(obs) == pytest.approx(0.5)

    def test_single_class_returns_chance(self) -> None:
        assert auc([_obs(0.9, True), _obs(0.8, True)]) == pytest.approx(0.5)
        assert auc([_obs(0.1, False)]) == pytest.approx(0.5)

    def test_empty_is_chance(self) -> None:
        assert auc([]) == pytest.approx(0.5)

    def test_matches_brute_force_pairwise_definition(self) -> None:
        """Cross-check the O(n log n) rank implementation against the
        O(n²) definition: fraction of (pos, neg) pairs correctly ordered."""
        rng = random.Random(42)
        obs = [
            _obs(round(rng.random(), 3), rng.random() < 0.5)
            for _ in range(60)
        ]
        pos = [o.predicted for o in obs if o.recalled]
        neg = [o.predicted for o in obs if not o.recalled]

        wins = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    wins += 1.0
                elif p == n:
                    wins += 0.5
        brute = wins / (len(pos) * len(neg))
        assert auc(obs) == pytest.approx(brute, abs=1e-9)


class TestRMSE:
    def test_perfectly_calibrated_bins_are_near_zero(self) -> None:
        """Prediction matches observed rate exactly in each bucket."""
        obs = [_obs(0.9, True)] * 9 + [_obs(0.9, False)] * 1  # 90% observed
        assert rmse(obs) < 0.02

    def test_miscalibrated_bins_are_large(self) -> None:
        obs = [_obs(0.9, False)] * 10  # predicted 0.9, observed 0.0
        assert rmse(obs) > 0.8

    def test_empty_is_zero(self) -> None:
        assert rmse([]) == 0.0


class TestPredictAll:
    def test_first_review_is_skipped(self) -> None:
        """There is no prior interval for the initial exposure."""
        histories = [[(0.0, 0.9), (10.0, 0.8), (20.0, 0.7)]]
        obs = predict_all(lambda e, n, m: 0.5, histories)
        assert len(obs) == 2
        assert [o.elapsed_hours for o in obs] == [10.0, 20.0]

    def test_recalled_uses_the_060_threshold(self) -> None:
        histories = [[(0.0, 0.0), (1.0, 0.59), (2.0, 0.6)]]
        obs = predict_all(lambda e, n, m: 0.5, histories)
        assert [o.recalled for o in obs] == [False, True]

    def test_prior_count_and_mean_are_passed_in_order(self) -> None:
        seen: list[tuple[float, int, float]] = []

        def model(elapsed: float, n: int, mean_prior: float) -> float:
            seen.append((elapsed, n, mean_prior))
            return 0.5

        histories = [[(0.0, 0.8), (5.0, 0.6), (7.0, 1.0)]]
        predict_all(model, histories)

        assert seen[0][1] == 1  # one prior review
        assert seen[0][2] == pytest.approx(0.8)
        assert seen[1][1] == 2
        assert seen[1][2] == pytest.approx(0.7)  # mean of 0.8, 0.6

    def test_history_with_only_first_review_yields_nothing(self) -> None:
        assert predict_all(lambda e, n, m: 0.5, [[(0.0, 0.9)]]) == []

    def test_empty_dataset(self) -> None:
        assert predict_all(lambda e, n, m: 0.5, []) == []


class TestEvaluateModel:
    def test_oracle_beats_inverted(self) -> None:
        """The end-to-end property the whole harness exists to provide."""
        rng = random.Random(3)
        # Build a dataset where recall depends monotonically on elapsed time.
        histories = []
        for _ in range(40):
            hist = [(0.0, 0.9)]
            for k in range(1, 8):
                elapsed = k * 30.0
                score = 0.9 if rng.random() < max(0.1, 1 - k / 8) else 0.2
                hist.append((elapsed, score))
            histories.append(hist)

        def good(e: float, n: int, m: float) -> float:
            return max(0.05, 1.0 - e / 250.0)

        def bad(e: float, n: int, m: float) -> float:
            return min(0.95, e / 250.0)

        good_r = evaluate_model("good", good, histories)
        bad_r = evaluate_model("bad", bad, histories)

        assert good_r.auc > bad_r.auc
        assert good_r.log_loss < bad_r.log_loss

    def test_report_has_expected_shape(self) -> None:
        histories = [[(0.0, 0.9), (10.0, 0.8)]]
        r = evaluate_model("m", lambda e, n, m: 0.7, histories)
        assert isinstance(r, ModelReport)
        assert r.name == "m"
        assert r.n == 1
        d = r.to_dict()
        assert set(d) == {"name", "n", "log_loss", "rmse", "auc"}

    def test_calibration_curve_reports_counts(self) -> None:
        histories = [[(0.0, 0.9)] + [(5.0, 0.8)] * 20]
        r = evaluate_model("m", lambda e, n, m: 0.75, histories)
        assert r.calibration
        assert sum(count for _, _, count in r.calibration) == 20
