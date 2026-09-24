"""ForgettingCurveService — power-law forgetting curve + Bayesian fitting.

Mathematical model::

    Retrievability  R(t, S) = (1 + FACTOR · t / S) ^ (-DECAY)      (power law)

    Stability       S = S_BASE · posterior_mean^SCORE_EXP · (n + 1)^GROWTH_EXP

where ``t`` is hours since the last review, ``S`` is memory stability (hours),
and ``n`` is the number of prior reviews.

Why a power law and not ``exp(-t/S)``
-------------------------------------
The original model used the Ebbinghaus exponential ``R = exp(-t/S)`` with
``S = S_BASE · posterior_mean · log2(n+1)``.  Measured against a 0-parameter
baseline on the harness in ``scripts/run_forgetting_eval.py`` it scored
**LogLoss 1.42 vs 0.44** — it did not beat "predict the learner's average".
Two independent defects were found, and this module fixes both:

1. **Stability grew too slowly.**  ``log2(n+1)`` meant ``S_MAX = 720h`` was
   unreachable in practice — reaching it would need ~1.9e11 reviews — so the
   clamp was dead code and S topped out around 61h after 8 reviews.
   Diagnostically: after 8 reviews the old model predicted a 0.063 chance of
   recall one week later, where a reasonable model predicts ~0.83.

2. **The curve shape was wrong.**  Even with a corrected S ≈ 330h, the
   exponential still decays far too fast at long intervals: at 336h it gives
   0.363 where a power law gives 0.899 (a 0.54 gap).  This matches the
   published FSRS history — v3 used an exponential, v4 switched to a power
   law, and the switch is why FSRS fits real review data better.

Parameters are **not fitted** here (there is no review corpus to fit against
yet).  The exponents are chosen from the spacing-effect literature and the
FSRS defaults, then validated on the synthetic harness — see
``src/learning_path/forgetting_synth.py`` for the independent ground truth.

Bayesian update (Beta-Bernoulli)::

    Prior:      Beta(α₀=2, β₀=2)      — weak prior, mean ≈ 0.5
    Likelihood: Bernoulli(score)
    Posterior:  Beta(α₀ + Σscore, β₀ + N - Σscore)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from src.utils.db_pool import unwrap_pool

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

S_BASE = 24.0  # 基准记忆强度（小时）
S_MIN = 1.0    # 最小记忆强度（小时）
S_MAX = 720.0  # 最大记忆强度（约 30 天）— 现值可达，见 GROWTH_EXP 说明

# 曲线形状：幂律 R = (1 + FACTOR·t/S)^(-DECAY)
#
# DECAY 越大衰减越快。0.1542 是 FSRS-6 的默认衰减参数（w20），它使曲线在
# 长间隔处显著比指数平缓 —— 这正是实测中差距最大的一段。
DECAY = 0.1542
# 由 DECAY 导出，使 R(S) = 0.9，即"S 的定义为回忆率降到 90% 所需时间"。
# 与 FSRS 一致，便于与公开基准对照。
FACTOR = 0.9 ** (-1.0 / DECAY) - 1.0

# 稳定性随复习次数的增长指数。
#
# 为什么是 1.0（而不是网格扫描出的最优 0.5）：在 4 个随机种子 × 200 条历史上
# 扫描 g ∈ {0.5, 1.0, 1.5} × se ∈ {1.0, 2.0, 3.0}，平均 LogLoss 全部优于
# 0 参数基线，且 g=0.5 的数值最好（0.4065 vs g=1.0 的 0.4287）。但仍选 1.0：
#
#   1. **g=1.0 有理论依据** —— 稳定性随复习次数线性增长，是间隔效应的标准形式；
#      g=0.5 是在**我自己生成的合成数据**上调出来的最优点，而合成数据的真值
#      形态是我选的 —— 选它就是在拟合自己的假设。
#   2. 两者都显著优于基线（−0.051 vs −0.073），这个差距不足以支撑"用理论
#      依据换 0.022 的 LogLoss"。
#
# 教训：合成数据上的最优值不是真实最优值。**判据是"有明显改善且形式有依据"，
# 而不是"在自造数据上跑出最好看的小数"** —— 那正是本文档批评过的做法。
# 真实复习数据积累起来后（`forgetting_review_log` 已在记录），应重新拟合。
GROWTH_EXP = 1.0
# 得分对稳定性的影响，以**中性后验**为 1× 基准做归一化。
#
# 为什么需要 NEUTRAL_POSTERIOR：Beta(2,2) 先验很弱，一次满分复习后
# posterior_mean 只有 0.6 (3/5)。直接取幂的话 0.6^2 = 0.36 < 1，会让 S
# **低于** base —— 即"复习一次反而更容易忘"，这显然是错的。以 0.6 为 1× 基准
# 归一化后：中性复习既不增长也不削减稳定性，而明显低于中性的得分会把因子拉到
# 1 以下（保留防刷分意图）。
# 不变量：复习后的 S 不得低于复习前（有测试守住）。
NEUTRAL_POSTERIOR = 0.6
# 因子下限，防止极端低分把稳定性压到无意义的小值。
MIN_POSTERIOR = 0.05
SCORE_EXP = 2.0

ALERT_RECALL_THRESHOLD = 0.6   # 低于此值 → "需复习"
URGENT_RECALL_THRESHOLD = 0.3  # 低于此值 → "紧急复习"
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0


def recall_probability(elapsed_hours: float, stability: float) -> float:
    """P(recall) under the power-law curve.

    Exposed as a module function so the evaluation harness can score the exact
    formula the service uses, rather than a re-implementation that could drift.

    Args:
        elapsed_hours: hours since the last review.
        stability: S, in hours — the interval at which R falls to 0.9.

    Returns:
        P(recall) in [0, 1].  1.0 for a non-positive elapsed time.
    """
    if elapsed_hours <= 0:
        return 1.0
    if stability <= 0:
        return 0.0
    return (1.0 + FACTOR * elapsed_hours / stability) ** (-DECAY)


def stability_after_review(
    prior_count: int,
    posterior_mean: float,
    *,
    base: float = S_BASE,
) -> float:
    """Stability after *prior_count* reviews at the given mean score.

    ``prior_count`` is the count **including** the review just performed.

    Shape::

        S = base · ((n + 1) / 2)^GROWTH_EXP · max(FLOOR, posterior_mean)^SCORE_EXP

    Two deliberate choices, each fixing a problem found by measurement:

    * **The growth term is normalised by 2.**  With a bare ``(n+1)^g`` a single
      review already multiplies by ``2^g``, overstating what one repetition
      teaches.  Dividing by 2 puts the first review at 1×, so ``base`` reads as
      "stability after one review" and growth accrues from the second on.
    * **The score factor is normalised so a neutral review is 1×.**  The
      Beta(2,2) prior is weak, so one perfect review leaves ``posterior_mean``
      at only 0.6 (3/5).  Raised to a positive power that is < 1, so a naive
      formula makes stability *drop* after a perfect review — reviewing an item
      would make it more forgettable.  Dividing by ``NEUTRAL_POSTERIOR``
      rescales the factor so that a neutral posterior gives exactly 1×: a
      review that reports no failure neither grows nor shrinks stability, while
      a review materially below neutral does shrink it (preserving the
      anti-cramming intent).

    Clamped to [S_MIN, S_MAX].
    """
    score_factor = (max(posterior_mean, MIN_POSTERIOR) / NEUTRAL_POSTERIOR) ** SCORE_EXP
    growth = ((prior_count + 1) / 2.0) ** GROWTH_EXP
    raw = base * growth * score_factor
    return max(S_MIN, min(S_MAX, raw))


@dataclass
class ForgettingState:
    """单个用户对单个知识点的遗忘参数状态。"""
    kp_id: str
    user_id: str
    alpha: float = PRIOR_ALPHA      # Beta posterior α
    beta: float = PRIOR_BETA        # Beta posterior β
    last_review_time: float = 0.0   # Unix timestamp of last review
    review_count: int = 0
    strength: float = S_BASE        # 当前记忆强度（小时）

    @property
    def posterior_mean(self) -> float:
        """Beta 后验均值 = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5

    def predict_recall(self) -> float:
        """预测当前回忆概率（幂律曲线，见模块 docstring）。"""
        if self.last_review_time <= 0:
            return 0.0  # 从未学习过
        elapsed = (time.time() - self.last_review_time) / 3600.0  # 小时
        return recall_probability(elapsed, self.strength)

    def update_after_quiz(self, score: float) -> None:
        """Quiz 后更新贝叶斯参数和记忆强度。

        Args:
            score: 0-1 之间的得分
        """
        # Beta posterior update
        self.alpha += score
        self.beta += (1.0 - score)
        self.review_count += 1
        self.last_review_time = time.time()

        # Stability: power-law growth in both review count and score quality.
        # See the module docstring for why the previous log2 growth was wrong.
        self.strength = stability_after_review(
            self.review_count, self.posterior_mean
        )
        logger.debug(
            "ForgettingState updated: kp=%s user=%s alpha=%.1f beta=%.1f S=%.1fh",
            self.kp_id, self.user_id, self.alpha, self.beta, self.strength,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kp_id": self.kp_id,
            "user_id": self.user_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "last_review_time": self.last_review_time,
            "review_count": self.review_count,
            "strength": self.strength,
            "posterior_mean": self.posterior_mean,
            "recall_probability": self.predict_recall(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ForgettingState:
        return cls(
            kp_id=data["kp_id"],
            user_id=data["user_id"],
            alpha=data.get("alpha", PRIOR_ALPHA),
            beta=data.get("beta", PRIOR_BETA),
            last_review_time=data.get("last_review_time", 0.0),
            review_count=data.get("review_count", 0),
            strength=data.get("strength", S_BASE),
        )


class ForgettingCurveService:
    """遗忘曲线服务 — Ebbinghaus 模型 + 贝叶斯参数拟合。

    使用方式::

        svc = ForgettingCurveService()
        recall = await svc.predict_recall("kp-01", "user-01")
        await svc.update_after_quiz("kp-01", "user-01", score=0.85)
        alerts = await svc.get_alerts("user-01")
    """

    def __init__(self, db_pool=None) -> None:
        # In-memory store: {(user_id, kp_id): ForgettingState}
        self._states: dict[tuple[str, str], ForgettingState] = {}
        self._db_pool = unwrap_pool(db_pool, owner="ForgettingCurveService")
        self._loaded_from_db: set[tuple[str, str]] = set()

    async def _load_state(self, kp_id: str, user_id: str) -> ForgettingState | None:
        """Try loading state from PostgreSQL, falling back to in-memory."""
        key = (user_id, kp_id)

        # Check in-memory first
        if key in self._states:
            return self._states[key]

        # Check if already tried DB for this key
        if key in self._loaded_from_db:
            return None

        self._loaded_from_db.add(key)

        # Try PostgreSQL
        if self._db_pool is not None:
            try:
                async with self._db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT alpha, beta, last_review_time, review_count, strength
                        FROM forgetting_curve_state
                        WHERE user_id = $1 AND kp_id = $2
                        """,
                        user_id, kp_id,
                    )
                    if row:
                        last_review = row["last_review_time"]
                        state = ForgettingState(
                            kp_id=kp_id,
                            user_id=user_id,
                            alpha=row["alpha"],
                            beta=row["beta"],
                            # 列类型为 TIMESTAMP WITH TIME ZONE，asyncpg 返回 datetime，
                            # 必须转成 Unix 秒，否则 predict_recall() 会抛 TypeError。
                            last_review_time=last_review.timestamp() if last_review else 0.0,
                            review_count=row["review_count"],
                            strength=row["strength"],
                        )
                        self._states[key] = state
                        return state
            except Exception as exc:
                logger.debug("Failed to load forgetting state from DB: %s", exc)

        return None

    async def _save_state(self, kp_id: str, user_id: str) -> None:
        """Persist state to PostgreSQL."""
        if self._db_pool is None:
            return

        key = (user_id, kp_id)
        state = self._states.get(key)
        if state is None:
            return

        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO forgetting_curve_state
                        (user_id, kp_id, alpha, beta, last_review_time, review_count, strength)
                    VALUES ($1, $2, $3, $4,
                        to_timestamp($5)::timestamp with time zone,
                        $6, $7)
                    ON CONFLICT (user_id, kp_id)
                    DO UPDATE SET
                        alpha = EXCLUDED.alpha,
                        beta = EXCLUDED.beta,
                        last_review_time = EXCLUDED.last_review_time,
                        review_count = EXCLUDED.review_count,
                        strength = EXCLUDED.strength,
                        updated_at = NOW()
                    """,
                    user_id, kp_id,
                    state.alpha, state.beta,
                    state.last_review_time if state.last_review_time > 0 else None,
                    state.review_count, state.strength,
                )
        except Exception as exc:
            logger.debug("Failed to save forgetting state to DB: %s", exc)

    async def predict_recall(self, kp_id: str, user_id: str) -> float:
        """预测指定用户对指定知识点的回忆概率 (0-1)。"""
        state = await self._load_state(kp_id, user_id)
        if not state:
            return 0.0
        return state.predict_recall()

    async def update_after_quiz(
        self, kp_id: str, user_id: str, score: float,
    ) -> ForgettingState:
        """Quiz 后更新遗忘参数。

        Also appends the review to ``forgetting_review_log``.  That log is the
        only place the (time, score) series survives — ``forgetting_curve_state``
        keeps just the latest fitted parameters and ``learning_progress`` is an
        upsert — so without it no model can ever be fitted or evaluated on real
        data.  Logging is best-effort: a failure to append must not stop the
        curve from updating.

        Returns:
            Updated ForgettingState
        """
        key = (user_id, kp_id)
        state = await self._load_state(kp_id, user_id)
        if not state:
            state = ForgettingState(kp_id=kp_id, user_id=user_id)
            self._states[key] = state
        state.update_after_quiz(score)
        await self._save_state(kp_id, user_id)
        await self._log_review(
            kp_id, user_id, score,
            event_type="learn" if state.review_count <= 1 else "review",
            source="assessment",
        )
        return state

    # A review without a reported score still counts as a successful retrieval
    # for stability purposes — you cannot review an item you have forgotten.
    # 0.8 is used (not 1.0) because "no failure reported" is weaker evidence
    # than an actually perfect quiz. It is stored in the log with
    # source="path" so it is never mistaken for a measured score.
    UNSCORED_REVIEW_SCORE = 0.8

    async def record_review(self, kp_id: str, user_id: str) -> ForgettingState:
        """记录一次复习（不涉及 quiz 评分，仅刷新时间戳）。

        当用户完成某个知识点的学习后调用。

        Stability now goes through the same formula as :meth:`update_after_quiz`
        rather than the old ad-hoc ``×1.2``: two paths computing stability
        differently meant the curve depended on which endpoint the client
        happened to call, and the multiplicative form grew without bound in a
        way the clamped formula does not.
        """
        key = (user_id, kp_id)
        state = await self._load_state(kp_id, user_id)
        if not state:
            state = ForgettingState(kp_id=kp_id, user_id=user_id)
            self._states[key] = state

        score = self.UNSCORED_REVIEW_SCORE
        state.alpha += score
        state.beta += (1.0 - score)
        state.review_count += 1
        state.last_review_time = time.time()
        state.strength = stability_after_review(
            state.review_count, state.posterior_mean
        )

        await self._save_state(kp_id, user_id)
        await self._log_review(
            kp_id, user_id, score, event_type="review", source="path",
        )
        return state

    async def _log_review(
        self,
        kp_id: str,
        user_id: str,
        score: float,
        *,
        event_type: str = "review",
        source: str = "assessment",
    ) -> None:
        """Append one review to the durable history.  Never raises.

        The review log is *evidence*; the caller's state update is the
        *product*.  Losing a log row degrades future model fitting, but losing
        the state update breaks the live feature — so failures here are logged
        and swallowed.
        """
        if self._db_pool is None:
            return
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO forgetting_review_log
                        (user_id, kp_id, score, event_type, source)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    user_id,
                    kp_id,
                    max(0.0, min(1.0, float(score))),
                    event_type,
                    source,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to append review log for %s/%s (non-fatal): %s",
                user_id, kp_id, exc,
            )

    async def load_review_history(
        self, user_id: str, kp_id: str | None = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Read the review log back as ``{kp_id: [(elapsed_hours, score), ...]}``.

        Produces exactly the shape
        :func:`src.learning_path.forgetting_eval.predict_all` consumes, so the
        evaluation harness can run against real history once it accumulates.

        The first entry per item carries ``elapsed_hours = 0.0`` — it is the
        initial exposure, and the scorer skips it deliberately.
        """
        if self._db_pool is None:
            return {}
        try:
            async with self._db_pool.acquire() as conn:
                if kp_id is not None:
                    rows = await conn.fetch(
                        """
                        SELECT kp_id, score, reviewed_at
                        FROM forgetting_review_log
                        WHERE user_id = $1 AND kp_id = $2
                        ORDER BY reviewed_at ASC
                        """,
                        user_id, kp_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT kp_id, score, reviewed_at
                        FROM forgetting_review_log
                        WHERE user_id = $1
                        ORDER BY kp_id, reviewed_at ASC
                        """,
                        user_id,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load review history for %s: %s", user_id, exc)
            return {}

        histories: dict[str, list[tuple[float, float]]] = {}
        # Track the previous timestamp per kp so each entry records the gap
        # since the *previous* review rather than since an arbitrary epoch.
        previous: dict[str, float] = {}
        for row in rows:
            kp = row["kp_id"]
            reviewed = row["reviewed_at"]
            prev_time = previous.get(kp)
            if prev_time is None:
                elapsed = 0.0  # initial exposure — the scorer skips it
            else:
                elapsed = (reviewed - prev_time).total_seconds() / 3600.0
            previous[kp] = reviewed
            histories.setdefault(kp, []).append((round(elapsed, 4), float(row["score"])))
        return histories

    async def get_state(self, kp_id: str, user_id: str) -> ForgettingState | None:
        """获取遗忘状态。"""
        return self._states.get((user_id, kp_id))

    async def get_alerts(
        self,
        user_id: str,
        kp_metadata: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """获取需要复习的知识点列表（按紧急程度排序）。

        Args:
            user_id: 用户 ID
            kp_metadata: 可选，{kp_id: kp_name} 映射

        Returns:
            [{
                "kp_id": "...",
                "kp_name": "...",
                "recall_probability": 0.45,
                "strength": 12.5,
                "review_count": 3,
                "last_review": "timestamp",
                "alert_level": "urgent" | "warning" | "ok"
            }]
        """
        alerts: list[dict[str, Any]] = []
        for (uid, kpid), state in self._states.items():
            if uid != user_id:
                continue
            recall = state.predict_recall()
            alert_level = "urgent" if recall < URGENT_RECALL_THRESHOLD else (
                "warning" if recall < ALERT_RECALL_THRESHOLD else "ok"
            )
            alerts.append({
                "kp_id": kpid,
                "kp_name": kp_metadata.get(kpid, kpid) if kp_metadata else kpid,
                "recall_probability": round(recall, 4),
                "strength": round(state.strength, 1),
                "review_count": state.review_count,
                "last_review": state.last_review_time,
                "alert_level": alert_level,
            })

        # Sort: urgent first, then warning, then ok
        level_rank = {"urgent": 0, "warning": 1, "ok": 2}
        alerts.sort(key=lambda a: level_rank.get(a["alert_level"], 3))
        return alerts

    async def get_all_states(self, user_id: str) -> list[dict[str, Any]]:
        """获取用户所有知识点的遗忘状态。"""
        states: list[dict[str, Any]] = []
        for (uid, kpid), state in self._states.items():
            if uid == user_id:
                states.append(state.to_dict())
        return states

    async def get_recall_map(self, user_id: str) -> dict[str, float]:
        """返回 ``{kp_id: recall_probability}``，用于驱动学习路径重算。

        仅包含"真正学习过"（``last_review_time > 0``）的知识点，避免把
        "从未学习"（``predict_recall() == 0.0``）误判为"已完全遗忘"而
        错误地触发复习加成。

        优先使用内存缓存；若缓存为空且配置了 DB，则从
        ``forgetting_curve_state`` 补齐，保证进程重启后闭环依然可用。
        """
        keys = [kpid for (uid, kpid) in self._states if uid == user_id]

        if not keys and self._db_pool is not None:
            try:
                async with self._db_pool.acquire() as conn:
                    rows = await conn.fetch(
                        """
                        SELECT kp_id, alpha, beta, last_review_time, review_count, strength
                        FROM forgetting_curve_state
                        WHERE user_id = $1
                        """,
                        user_id,
                    )
                for row in rows:
                    kp_id = row["kp_id"]
                    last_review = row["last_review_time"]
                    state = ForgettingState(
                        kp_id=kp_id,
                        user_id=user_id,
                        alpha=row["alpha"],
                        beta=row["beta"],
                        last_review_time=last_review.timestamp() if last_review else 0.0,
                        review_count=row["review_count"],
                        strength=row["strength"],
                    )
                    self._states[(user_id, kp_id)] = state
                    self._loaded_from_db.add((user_id, kp_id))
                keys = [kpid for (uid, kpid) in self._states if uid == user_id]
            except Exception as exc:
                logger.debug("Failed to load forgetting states for recall map: %s", exc)

        recall_map: dict[str, float] = {}
        for kp_id in keys:
            state = self._states.get((user_id, kp_id))
            if state is None or state.last_review_time <= 0:
                continue
            recall_map[kp_id] = state.predict_recall()
        return recall_map

    async def delete_user_data(self, user_id: str) -> int:
        """Delete all forgetting curve states for a user.

        Removes from both the in-memory cache and the database.
        Returns the count of deleted states.
        """
        # Remove from in-memory cache
        to_delete = [(uid, kpid) for (uid, kpid) in self._states if uid == user_id]
        for key in to_delete:
            del self._states[key]

        # Remove from database.  The returned row count is not used — the
        # reported ``total`` is the in-memory count — so the DELETE is issued
        # for its effect only.
        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM forgetting_curve_state WHERE user_id = $1",
                        user_id,
                    )
            except Exception as exc:
                logger.warning("Failed to delete forgetting curves from DB: %s", exc)

        total = len(to_delete)
        logger.info("Deleted %d forgetting curve states for user %s", total, user_id)
        return total

    def get_state_dict(self) -> dict:
        """序列化所有状态（用于持久化）。"""
        return {
            f"{uid}::{kpid}": s.to_dict()
            for (uid, kpid), s in self._states.items()
        }

    def load_state_dict(self, data: dict) -> None:
        """从字典加载状态（用于恢复持久化数据）。"""
        for key, val in data.items():
            if "::" in key:
                uid, kpid = key.split("::", 1)
                self._states[(uid, kpid)] = ForgettingState.from_dict(val)
