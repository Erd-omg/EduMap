"""Item Response Theory (1PL / Rasch) for mastery estimation.

Why this module exists
----------------------
EduMap's mastery estimate came from the LLM: the Assessment agent asked the
model to emit ``estimated_mastery_delta`` *at quiz-generation time* — before
anyone had answered anything — and fell back to a hard-coded ``0.1`` when the
model omitted it.  That is an assertion, not a measurement.

This module computes mastery from **observed responses** instead, using the
simplest defensible psychometric model: 1PL / Rasch.

    P(correct | θ, b) = σ(θ − b)        σ = logistic function

where θ is the learner's ability and b is the item's difficulty, both on the
same logit scale.

Why 1PL rather than 2PL/3PL
---------------------------
1PL has one parameter per item (difficulty) and one per learner (ability),
with **equal discrimination assumed across items**.  Trade-offs:

* It is identifiable from small samples.  2PL (per-item discrimination) and
  3PL (adding a guessing parameter) need substantially more responses per item
  before their extra parameters are estimable — with a small item bank they
  overfit and produce *worse* ability estimates than 1PL.
* It needs no calibration corpus.  EduMap's question bank is small and
  freshly written, so there is no history to fit discrimination from.

This matches the finding in the knowledge-tracing literature that fancier
models (DKT) only beat simple ones under specific evaluation setups — see the
Xiong et al. (EDM 2016) reproduction, which cut DKT's reported AUC advantage
substantially.  Starting simple is the defensible choice; 2PL can be added
once there is enough response data to identify it.

References:
    Rasch (1960), Probabilistic Models for Some Intelligence and Attainment Tests
    Corbett & Anderson (1994), BKT — doi:10.1007/BF01099821
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Ability is expressed in logits.  ±4 logits spans roughly 2%–98% success
# probability against an average-difficulty item, which covers any realistic
# learner without letting an extreme estimate dominate.
MIN_ABILITY = -4.0
MAX_ABILITY = 4.0


@dataclass
class ItemResponse:
    """One observed response: a learner answering one item."""

    item_id: str
    correct: bool
    difficulty: float = 0.0
    # Optional per-item discrimination, used only by the 2PL-style update.
    # 1PL ignores it.
    discrimination: float = 1.0


@dataclass
class MasteryEstimate:
    """Result of estimating ability from responses."""

    ability: float                    # θ, in logits
    n_responses: int
    standard_error: float             # 1/sqrt(Fisher information)
    # Ability mapped onto [0, 1] via the logistic CDF — the number the rest of
    # the system consumes as "mastery".
    mastery: float
    per_item_residual: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ability": round(self.ability, 4),
            "mastery": round(self.mastery, 4),
            "n_responses": self.n_responses,
            "standard_error": round(self.standard_error, 4),
        }


def probability_correct(ability: float, difficulty: float, discrimination: float = 1.0) -> float:
    """P(correct) under the logistic IRT model."""
    return _sigmoid(discrimination * (ability - difficulty))


def ability_to_mastery(ability: float) -> float:
    """Map ability (logits) onto a [0, 1] mastery score.

    Uses the logistic CDF so that ability 0 (an average learner facing an
    average item) maps to 0.5 rather than to some arbitrary midpoint, and the
    mapping is monotone and bounded.
    """
    return _sigmoid(ability)


def _log_likelihood(responses: list[ItemResponse], ability: float) -> float:
    total = 0.0
    for r in responses:
        p = probability_correct(ability, r.difficulty, r.discrimination)
        p = min(max(p, 1e-9), 1 - 1e-9)
        total += math.log(p) if r.correct else math.log(1 - p)
    return total


def estimate_ability(
    responses: list[ItemResponse],
    *,
    prior: float = 0.0,
    prior_weight: float = 1.0,
    iterations: int = 60,
) -> MasteryEstimate:
    """Maximum-likelihood ability estimate with a weak Gaussian prior (MAP).

    The prior matters: with all-correct or all-incorrect responses the plain
    MLE diverges to ±infinity.  A weak prior centred at 0 keeps the estimate
    finite and expresses the honest state of knowledge ("we don't know yet").

    Args:
        responses: observed responses, any order.
        prior: prior mean ability in logits.
        prior_weight: prior precision, in pseudo-observations.  1.0 expresses
            "as informative as one extra response" — deliberately weak.
        iterations: bisection steps; 60 over a 8-logit range resolves far
            beyond display precision.

    Returns:
        A :class:`MasteryEstimate`.  With no responses, returns the prior.
    """
    if not responses:
        return MasteryEstimate(
            ability=prior,
            n_responses=0,
            standard_error=float("inf"),
            mastery=ability_to_mastery(prior),
        )

    def objective(theta: float) -> float:
        """MAP objective: log-likelihood minus prior penalty."""
        return _log_likelihood(responses, theta) - (
            prior_weight * (theta - prior) ** 2 / 2.0
        )

    # The logistic log-likelihood is concave in theta, so a golden-section /
    # ternary search on a bounded interval finds the global maximum without
    # needing derivatives (and without Newton's divergence issues on
    # perfect-response patterns).
    lo, hi = MIN_ABILITY, MAX_ABILITY
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            lo = m1
        else:
            hi = m2
    ability = (lo + hi) / 2.0

    # Fisher information for 1PL: Σ p(1-p).  SE = 1/sqrt(info).
    info = sum(
        (lambda p: p * (1 - p))(
            probability_correct(ability, r.difficulty, r.discrimination)
        )
        for r in responses
    )
    se = 1.0 / math.sqrt(info) if info > 1e-9 else float("inf")

    residuals = {
        r.item_id: round(
            (1.0 if r.correct else 0.0)
            - probability_correct(ability, r.difficulty, r.discrimination),
            4,
        )
        for r in responses
    }

    return MasteryEstimate(
        ability=ability,
        n_responses=len(responses),
        standard_error=se,
        mastery=ability_to_mastery(ability),
        per_item_residual=residuals,
    )


def estimate_item_difficulty(
    outcomes: list[tuple[float, bool]],
    *,
    prior: float = 0.0,
    prior_weight: float = 1.0,
) -> float:
    """Estimate one item's difficulty from (respondent_ability, correct) pairs.

    Inverts the Rasch link: a learner of ability θ passes an item of
    difficulty b with probability σ(θ − b).  Given the observed pass rate for
    a group of learners whose abilities are known, the difficulty that best
    explains it is the one making expected and observed pass rates agree.

    Args:
        outcomes: ``[(respondent_ability, was_correct), ...]`` for this item.
        prior: difficulty prior mean (used when there is no data).
        prior_weight: prior precision in pseudo-responses.

    Returns:
        The MAP difficulty estimate.  Returns ``prior`` with no outcomes.

    Note:
        This is one half of a coordinate-ascent loop: estimate abilities from
        difficulties, then difficulties from abilities, and repeat.  A single
        call against stale abilities is an improvement, not a solution.
    """
    if not outcomes:
        return prior

    def objective(b: float) -> float:
        total = 0.0
        for ability, correct in outcomes:
            p = _sigmoid(ability - b)
            p = min(max(p, 1e-9), 1 - 1e-9)
            total += math.log(p) if correct else math.log(1 - p)
        return total - prior_weight * (b - prior) ** 2 / 2.0

    lo, hi = MIN_ABILITY, MAX_ABILITY
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) < objective(m2):
            lo = m1
        else:
            hi = m2
    return (lo + hi) / 2.0


def fit_abilities_and_difficulties(
    responses: list[ItemResponse],
    *,
    rounds: int = 12,
) -> tuple[dict[str, MasteryEstimate], dict[str, float]]:
    """Jointly fit learner abilities and item difficulties (Rasch, MAP).

    Alternates the two coordinate steps until the difficulties stabilise.
    Abilities are estimated per *item_id group* only when the caller supplies
    a single learner's responses; for a multi-learner matrix the caller should
    group by learner and call :func:`estimate_ability` per group, then feed
    the results back here.  This helper covers the common single-learner case
    used by the Assessment agent.

    Returns:
        ``({item_id: MasteryEstimate}, {item_id: difficulty})``.
    """
    difficulties = {r.item_id: r.difficulty for r in responses}

    for _ in range(rounds):
        # Step 1: given difficulties, estimate the learner's ability.
        rescored = [
            ItemResponse(
                item_id=r.item_id,
                correct=r.correct,
                difficulty=difficulties.get(r.item_id, r.difficulty),
                discrimination=r.discrimination,
            )
            for r in responses
        ]
        ability_est = estimate_ability(rescored)

        # Step 2: given that ability, refine each difficulty.
        by_item: dict[str, list[tuple[float, bool]]] = {}
        for r in responses:
            by_item.setdefault(r.item_id, []).append((ability_est.ability, r.correct))

        updated = {
            item_id: estimate_item_difficulty(pairs, prior=difficulties.get(item_id, 0.0))
            for item_id, pairs in by_item.items()
        }
        if all(
            abs(updated[k] - difficulties.get(k, updated[k])) < 1e-6 for k in updated
        ):
            difficulties = updated
            break
        difficulties = updated

    # Report per-item estimates against the converged difficulties.
    final = [
        ItemResponse(
            item_id=r.item_id,
            correct=r.correct,
            difficulty=difficulties.get(r.item_id, r.difficulty),
            discrimination=r.discrimination,
        )
        for r in responses
    ]
    ability_est = estimate_ability(final)
    per_item = {item_id: ability_est for item_id in difficulties}
    return per_item, difficulties
