"""Forgetting-curve model evaluation — LogLoss / RMSE / AUC.

Why this module exists
----------------------
EduMap's forgetting curve was self-designed (``R = exp(-t/S)`` with a scalar
strength) and had never been measured against any baseline or against the
published state of the art.  Without a number it was impossible to say whether
it was good, bad, or merely redundant.

The public benchmark to compare against is ``srs-benchmark`` (10,000 Anki
users, ~350M reviews), whose headline result is uncomfortable: a **0-parameter
moving average** of recent outcomes scores LogLoss 0.3369 — *better* than
21-parameter FSRS-6 at 0.3460.  So the first honest question is not "should we
use FSRS?" but "do we even beat the trivial baseline?".

This module answers that question by scoring any recall-prediction model on
the same three metrics the public benchmark uses:

    LogLoss  — calibration.  Lower is better.
    RMSE     — mean squared error of predicted vs observed recall, binned.
               Lower is better.
    AUC      — discrimination: can the model rank recalled above forgotten?
               Higher is better.  A model can be well-calibrated but useless
               at ranking, so AUC is not redundant with LogLoss.

Design: a **model is just a callable** ``(elapsed_hours, n_prior_reviews, mean_prior_score) -> p``.
That keeps the evaluation independent of any particular implementation — the
same harness scores the hand-rolled curve, a baseline, and FSRS without
knowing what any of them are.

Reference:
    srs-benchmark  https://github.com/open-spaced-repetition/srs-benchmark
    FSRS algorithm https://expertium.github.io/Algorithm.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

# A model maps (hours since last review, prior review count, mean prior score)
# to P(recall) in [0, 1].  The three inputs are the minimum any of the
# candidates needs; models may ignore any of them.
RecallModel = Callable[[float, int, float], float]


@dataclass(frozen=True)
class ReviewObservations:
    """One (prediction, outcome) pair extracted from a review history.

    Attributes:
        elapsed_hours: time since the previous review of this item.
        predicted: model's P(recall) at that moment.
        recalled: the observed outcome (True if the review was passed).
        prior_count: how many reviews preceded this one.
    """

    elapsed_hours: float
    predicted: float
    recalled: bool
    prior_count: int


@dataclass
class ModelReport:
    """Aggregate metrics for one model."""

    name: str
    n: int = 0
    log_loss: float = 0.0
    rmse: float = 0.0
    auc: float = 0.0
    # per-bucket calibration detail: (predicted_mean, observed_rate, count)
    calibration: list[tuple[float, float, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "log_loss": round(self.log_loss, 4),
            "rmse": round(self.rmse, 4),
            "auc": round(self.auc, 4),
        }


# ── Metric primitives ────────────────────────────────────────────────────

def _clamp(p: float) -> float:
    """Keep probabilities strictly inside (0, 1).

    log(0) is -inf and log(1) is 0; a confidently-wrong prediction would
    otherwise produce an infinite loss and poison the mean.
    """
    return min(max(p, 1e-6), 1 - 1e-6)


def log_loss(observations: Sequence[ReviewObservations]) -> float:
    """Binary cross-entropy of predicted recall vs observed outcome."""
    if not observations:
        return 0.0
    total = 0.0
    for obs in observations:
        p = _clamp(obs.predicted)
        total += -math.log(p) if obs.recalled else -math.log(1 - p)
    return total / len(observations)


def rmse(observations: Sequence[ReviewObservations], *, bins: int = 20) -> float:
    """RMSE between predicted and observed recall, computed on *bins*.

    Per-observation RMSE is meaningless for a binary outcome (the residual is
    either p or 1-p), so predictions are grouped by predicted probability and
    the error is measured between each bin's mean prediction and its observed
    recall rate — the same "RMSE(bins)" the srs-benchmark reports.
    """
    if not observations:
        return 0.0

    width = 1.0 / bins
    buckets: dict[int, list[ReviewObservations]] = {}
    for obs in observations:
        idx = min(int(_clamp(obs.predicted) / width), bins - 1)
        buckets.setdefault(idx, []).append(obs)

    if not buckets:
        return 0.0

    total = 0.0
    for group in buckets.values():
        mean_pred = sum(o.predicted for o in group) / len(group)
        observed = sum(1.0 for o in group if o.recalled) / len(group)
        total += (mean_pred - observed) ** 2
    return math.sqrt(total / len(buckets))


def auc(observations: Sequence[ReviewObservations]) -> float:
    """Area under the ROC curve, via the rank-sum (Mann-Whitney U) identity.

    Ties in predicted probability contribute 0.5, which is the standard
    convention.  Returns 0.5 when one class is absent — the model has had no
    opportunity to discriminate, and 0.5 is the uninformative value.
    """
    positives = [o.predicted for o in observations if o.recalled]
    negatives = [o.predicted for o in observations if not o.recalled]
    if not positives or not negatives:
        return 0.5

    # Sort all predictions once, then count how many negative ranks each
    # positive outranks.  O(n log n).
    order = sorted(range(len(observations)), key=lambda i: observations[i].predicted)
    ranks = [0.0] * len(observations)
    i = 0
    while i < len(order):
        j = i
        # group ties so they share the average rank
        while j + 1 < len(order) and (
            observations[order[j + 1]].predicted == observations[order[i]].predicted
        ):
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    pos_rank_sum = sum(
        ranks[idx] for idx, o in enumerate(observations) if o.recalled
    )
    n_pos, n_neg = len(positives), len(negatives)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ── Evaluation driver ───────────────────────────────────────────────────

def predict_all(
    model: RecallModel,
    histories: Sequence[Sequence[tuple[float, float]]],
) -> list[ReviewObservations]:
    """Turn raw review histories into (prediction, outcome) observations.

    Args:
        model: the recall model under test.
        histories: one sequence per user/item, each a list of
            ``(elapsed_hours_since_previous_review, score)`` ordered by time.
            The **first** entry of each history is the initial exposure and is
            skipped — there is no prior review to measure an interval from,
            and every candidate model would predict its prior for it.

    Returns:
        Flattened observations across all histories.
    """
    observations: list[ReviewObservations] = []
    for history in histories:
        prior_scores: list[float] = []
        for idx, (elapsed, score) in enumerate(history):
            if idx > 0:
                mean_prior = (
                    sum(prior_scores) / len(prior_scores) if prior_scores else 0.5
                )
                predicted = _clamp(model(elapsed, len(prior_scores), mean_prior))
                observations.append(
                    ReviewObservations(
                        elapsed_hours=elapsed,
                        predicted=predicted,
                        # A review counts as "recalled" when the learner passed
                        # it.  0.6 is the threshold used elsewhere in the
                        # codebase for "mastered enough to count".
                        recalled=score >= 0.6,
                        prior_count=len(prior_scores),
                    )
                )
            prior_scores.append(score)
    return observations


def evaluate_model(
    name: str,
    model: RecallModel,
    histories: Sequence[Sequence[tuple[float, float]]],
    *,
    bins: int = 20,
) -> ModelReport:
    """Score one model on a set of histories."""
    observations = predict_all(model, histories)
    report = ModelReport(
        name=name,
        n=len(observations),
        log_loss=log_loss(observations),
        rmse=rmse(observations, bins=bins),
        auc=auc(observations),
    )
    report.calibration = _calibration_curve(observations, bins=bins)
    return report


def _calibration_curve(
    observations: Sequence[ReviewObservations], *, bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, count) per bucket — for plotting/debug."""
    if not observations:
        return []
    width = 1.0 / bins
    buckets: dict[int, list[ReviewObservations]] = {}
    for obs in observations:
        idx = min(int(_clamp(obs.predicted) / width), bins - 1)
        buckets.setdefault(idx, []).append(obs)
    out: list[tuple[float, float, int]] = []
    for idx in sorted(buckets):
        group = buckets[idx]
        mean_pred = sum(o.predicted for o in group) / len(group)
        observed = sum(1.0 for o in group if o.recalled) / len(group)
        out.append((round(mean_pred, 4), round(observed, 4), len(group)))
    return out
