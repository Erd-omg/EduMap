"""Synthetic review-history generator for forgetting-curve evaluation.

Why synthetic data
------------------
The production database holds essentially no review history (``learning_progress``
had 1 row; the review pipeline had never been used).  Without a series of
(time, score) observations there is nothing to fit or to validate a
forgetting-curve model against.

Generating data obviously cannot *validate* a model against reality.  What it
can do is:

1. **Prove the evaluation harness discriminates.**  If a knowingly-bad model
   scores as well as a knowingly-good one, the harness is broken and any
   conclusion drawn from it — including on real data later — is worthless.
   ``test_forgetting_eval.py`` asserts exactly this.
2. **Establish the ballpark** before real data exists, so that the first real
   measurement has something to be compared against.

The generator simulates a learner as a *ground-truth* memory process and then
samples review outcomes from it.  Crucially the ground truth uses a **power-law
forgetting curve** (the shape FSRS fits to real data), not EduMap's exponential
one — otherwise the benchmark would be rigged in EduMap's favour.

Reference: FSRS uses a power forgetting curve ``R = (1 + factor·t/S)^(-decay)``
after v4 (v3 was exponential).  See https://expertium.github.io/Algorithm.html
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticLearner:
    """Ground-truth memory parameters for one simulated learner/item pair."""

    stability_hours: float = 24.0
    # FSRS-style power curve decay; higher = faster forgetting.
    decay: float = 0.5
    # Probability of a lucky correct answer without true recall (guessing).
    guess_rate: float = 0.05
    # Probability of a slip: knowing it but answering wrong.
    slip_rate: float = 0.05

    def true_recall(self, elapsed_hours: float, prior_reviews: int) -> float:
        """P(recall) under the ground truth.

        Stability grows with successful repetition (a power law in the number
        of prior reviews), which is what produces the spacing effect the
        schedulers are trying to exploit.
        """
        if elapsed_hours <= 0:
            return 1.0
        stability = self.stability_hours * (1.0 + prior_reviews) ** 0.6
        base = (1.0 + (elapsed_hours / (9.0 * stability))) ** (-self.decay * 2.0)
        return min(1.0, max(0.0, base))

    def sample_outcome(self, elapsed_hours: float, prior_reviews: int, rng: random.Random) -> float:
        """Sample an observed score in [0, 1] for one review.

        Models both failure modes: a genuine miss, and a guess that passes
        without recall.  Scores are continuous because real quiz scores are.
        """
        p = self.true_recall(elapsed_hours, prior_reviews)
        truly_recalled = rng.random() < p
        if truly_recalled:
            # correct, unless a slip occurs
            if rng.random() < self.slip_rate:
                return round(rng.uniform(0.2, 0.55), 2)
            return round(rng.uniform(0.65, 1.0), 2)
        # not recalled — may still guess correctly
        if rng.random() < self.guess_rate:
            return round(rng.uniform(0.6, 0.8), 2)
        return round(rng.uniform(0.0, 0.5), 2)


def generate_history(
    learner: SyntheticLearner,
    *,
    n_reviews: int = 12,
    rng: random.Random,
    min_gap_hours: float = 4.0,
    max_gap_hours: float = 24.0 * 14,
) -> list[tuple[float, float]]:
    """One item's review history as ``[(gap_hours, score), ...]``.

    Gaps are drawn log-uniformly so that both short (same-day cramming) and
    long (two-week lapse) intervals appear — the model has to be scored across
    the whole range to be meaningfully evaluated.
    """
    import math

    import random as _random

    history: list[tuple[float, float]] = []
    for i in range(n_reviews):
        if i == 0:
            gap = 0.0
        else:
            log_min = math.log(min_gap_hours)
            log_max = math.log(max_gap_hours)
            gap = math.exp(rng.uniform(log_min, log_max))
        score = learner.sample_outcome(gap, i, rng)
        history.append((round(gap, 3), score))
    return history


def generate_dataset(
    *,
    n_items: int = 300,
    n_reviews: int = 12,
    seed: int = 20260923,
) -> list[list[tuple[float, float]]]:
    """Generate ``n_items`` review histories with per-item learner variation.

    Stability is drawn from a wide log-normal so the dataset contains both
    fast and slow forgetters — a model that ignores the item entirely cannot
    do well on it, which is the point.
    """
    import math

    rng = random.Random(seed)
    dataset: list[list[tuple[float, float]]] = []
    for _ in range(n_items):
        stability = math.exp(rng.gauss(math.log(24.0), 0.8))
        learner = SyntheticLearner(
            stability_hours=min(max(stability, 2.0), 2000.0),
            decay=rng.uniform(0.3, 0.7),
        )
        dataset.append(generate_history(learner, n_reviews=n_reviews, rng=rng))
    return dataset
