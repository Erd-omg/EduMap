"""ForgettingCurveService — Ebbinghaus遗忘曲线 + 贝叶斯参数拟合。

数学模型::

    回忆概率 R = exp(-t / S)

    其中 t 是距上次复习的小时数, S 是记忆强度参数（小时）。

贝叶斯更新 (Beta-Bernoulli)::

    Prior:  Beta(α₀=2, β₀=2)  — 弱先验, 均值 ≈ 0.5
    Likelihood:  Bernoulli(score)
    Posterior:   Beta(α₀ + Σscore, β₀ + N - Σscore)

    记忆强度 = S_base × P(recall | posterior)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

S_BASE = 24.0  # 基准记忆强度（小时）— 对应约 1 天后回忆率 ~37%
S_MIN = 1.0    # 最小记忆强度（小时）
S_MAX = 720.0  # 最大记忆强度（约 30 天）
ALERT_RECALL_THRESHOLD = 0.6   # 低于此值 → "需复习"
URGENT_RECALL_THRESHOLD = 0.3  # 低于此值 → "紧急复习"
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0


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
        """预测当前回忆概率 R = exp(-t / S)。"""
        if self.last_review_time <= 0:
            return 0.0  # 从未学习过
        elapsed = (time.time() - self.last_review_time) / 3600.0  # 小时
        if elapsed <= 0:
            return 1.0
        return math.exp(-elapsed / self.strength)

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

        # Memory strength: S = S_BASE × posterior_mean
        # For repeated reviews, strength increases sub-linearly
        self.strength = max(
            S_MIN,
            min(
                S_MAX,
                S_BASE * self.posterior_mean * math.log2(self.review_count + 1),
            ),
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
        self._db_pool = db_pool  # Optional asyncpg pool for persistence
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
        return state

    async def record_review(self, kp_id: str, user_id: str) -> ForgettingState:
        """记录一次复习（不涉及 quiz 评分，仅刷新时间戳）。

        当用户完成某个知识点的学习后调用。
        """
        key = (user_id, kp_id)
        state = await self._load_state(kp_id, user_id)
        if not state:
            state = ForgettingState(kp_id=kp_id, user_id=user_id)
            self._states[key] = state
        state.last_review_time = time.time()
        state.review_count += 1
        # 轻微增强记忆强度
        state.strength = max(
            S_MIN,
            min(S_MAX, state.strength * 1.2),
        )
        await self._save_state(kp_id, user_id)
        return state

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
