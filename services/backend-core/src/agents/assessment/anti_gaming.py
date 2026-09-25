"""Anti-gaming mechanism — prevents score gaming through multi-factor weighting.

PRD 4.4 四项机制::

    1. 多因子加权 — 综合掌握度、时间间隔、难度匹配度
    2. 贝叶斯缓慢更新 — posterior = (prior*N + new) / (N+1)
    3. 间隔突击复习检测 — 短时间高频完成 → 降权
    4. 元认知透明度 — 每次评估返回详细的得分因子分解
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.utils.db_pool import unwrap_pool

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

CRAM_WINDOW_HOURS = 2.0    # 检测"突击"的时间窗口（小时）
CRAM_THRESHOLD_COUNT = 5    # 在时间窗口内完成超过此数 → 判定突击
WEIGHT_MASTERY = 0.5        # 掌握度权重
WEIGHT_TIME_GAP = 0.3       # 时间间隔权重
WEIGHT_DIFFICULTY = 0.2     # 难度匹配度权重
MAX_SCORE = 1.0
MIN_SCORE = 0.0


@dataclass
class ActivityRecord:
    """单次学习/测评活动记录。"""
    timestamp: float
    score: float  # 0-1
    kp_id: str
    difficulty: int  # 1-5


@dataclass
class AntiGamingState:
    """用户对单个知识点的反游戏化状态。"""
    kp_id: str
    user_id: str
    # Bayesian posterior
    prior_count: float = 1.0      # 先验等效样本量
    prior_mean: float = 0.5       # 先验均值
    # Recent activity for cramming detection
    recent_activities: list[ActivityRecord] = field(default_factory=list)
    # 最新加权得分
    weighted_score: float = 0.0

    @property
    def posterior_mean(self) -> float:
        """贝叶斯后验均值 = (prior_count * prior_mean + sum_scores) / (prior_count + N)"""
        total_weighted = self.prior_count * self.prior_mean
        total_weight = self.prior_count
        for act in self.recent_activities[-20:]:  # 最多用最近 20 条
            total_weighted += act.score
            total_weight += 1.0
        return total_weighted / total_weight if total_weight > 0 else 0.5

    @property
    def posterior_confidence(self) -> float:
        """后验置信度 = N / (N + prior_count) — 样本越多越置信"""
        n = len(self.recent_activities)
        return n / (n + self.prior_count) if (n + self.prior_count) > 0 else 0.0


class AntiGamingService:
    """反游戏化服务。

    使用方式::

        svc = AntiGamingService()
        adjusted = await svc.calculate_weighted_score("kp-01", "user-01", 0.95, difficulty=2)
        # adjusted 可能是 0.7（因时间间隔太短而降权）
    """

    def __init__(self, db_pool=None) -> None:
        # {(user_id, kp_id): AntiGamingState}
        self._states: dict[tuple[str, str], AntiGamingState] = {}
        self._db_pool = unwrap_pool(db_pool, owner="AntiGamingService")
        self._loaded_from_db: set[tuple[str, str]] = set()

    async def _load_state(self, kp_id: str, user_id: str) -> AntiGamingState | None:
        """Try loading anti-gaming state from PostgreSQL."""
        key = (user_id, kp_id)

        if key in self._states:
            return self._states[key]
        if key in self._loaded_from_db:
            return None

        self._loaded_from_db.add(key)

        if self._db_pool is not None:
            try:
                async with self._db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        """
                        SELECT alpha, beta, last_event_time, pattern_flags
                        FROM anti_gaming_state
                        WHERE user_id = $1 AND kp_id = $2
                        """,
                        user_id, kp_id,
                    )
                    if row:
                        state = AntiGamingState(
                            kp_id=kp_id,
                            user_id=user_id,
                            prior_count=row["alpha"],
                            prior_mean=row["beta"],
                        )
                        # Note: recent_activities are kept in-memory for performance
                        # pattern_flags stored in JSONB for future use
                        self._states[key] = state
                        return state
            except Exception as exc:
                logger.debug("Failed to load anti-gaming state from DB: %s", exc)

        return None

    async def _save_state(self, kp_id: str, user_id: str) -> None:
        """Persist anti-gaming state to PostgreSQL."""
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
                    INSERT INTO anti_gaming_state
                        (user_id, kp_id, alpha, beta, last_event_time, pattern_flags)
                    VALUES ($1, $2, $3, $4, NOW(), '{}')
                    ON CONFLICT (user_id, kp_id)
                    DO UPDATE SET
                        alpha = EXCLUDED.alpha,
                        beta = EXCLUDED.beta,
                        last_event_time = NOW(),
                        updated_at = NOW()
                    """,
                    user_id, kp_id,
                    state.prior_count, state.prior_mean,
                )
        except Exception as exc:
            logger.debug("Failed to save anti-gaming state to DB: %s", exc)

    async def calculate_weighted_score(
        self,
        kp_id: str,
        user_id: str,
        raw_score: float,
        difficulty: int = 3,
    ) -> dict[str, Any]:
        """计算加权得分，返回详细因子分解。

        Returns::

            {
                "weighted_score": 0.72,
                "factors": {
                    "mastery_factor": 0.95,
                    "time_gap_factor": 0.60,
                    "difficulty_factor": 0.80,
                },
                "weights": {"mastery": 0.5, "time_gap": 0.3, "difficulty": 0.2},
                "is_cramming": False,
                "details": "...",
            }
        """
        key = (user_id, kp_id)
        state = await self._load_state(kp_id, user_id)
        if not state:
            state = AntiGamingState(kp_id=kp_id, user_id=user_id)
            self._states[key] = state

        # 1. 检测突击复习
        is_cramming = self._detect_cramming(state)
        cramming_penalty = 0.7 if is_cramming else 1.0

        # 2. 多因子加权
        mastery_factor = self._compute_mastery_factor(state, raw_score)
        time_gap_factor = self._compute_time_gap_factor(state)
        difficulty_factor = self._compute_difficulty_factor(difficulty, raw_score)

        # 3. 综合加权
        composite = (
            WEIGHT_MASTERY * mastery_factor
            + WEIGHT_TIME_GAP * time_gap_factor
            + WEIGHT_DIFFICULTY * difficulty_factor
        ) * cramming_penalty

        weighted_score = max(MIN_SCORE, min(MAX_SCORE, composite))

        # 4. 记录活动（用于下次检测）
        state.recent_activities.append(
            ActivityRecord(
                timestamp=time.time(),
                score=weighted_score,
                kp_id=kp_id,
                difficulty=difficulty,
            )
        )
        state.weighted_score = weighted_score

        # Persist to DB
        await self._save_state(kp_id, user_id)

        logger.debug(
            "AntiGaming: kp=%s user=%s raw=%.2f weighted=%.2f cramming=%s",
            kp_id, user_id, raw_score, weighted_score, is_cramming,
        )

        return {
            "weighted_score": round(weighted_score, 4),
            "raw_score": raw_score,
            "factors": {
                "mastery_factor": round(mastery_factor, 4),
                "time_gap_factor": round(time_gap_factor, 4),
                "difficulty_factor": round(difficulty_factor, 4),
                "cramming_penalty": round(cramming_penalty, 4),
            },
            "weights": {
                "mastery": WEIGHT_MASTERY,
                "time_gap": WEIGHT_TIME_GAP,
                "difficulty": WEIGHT_DIFFICULTY,
            },
            "is_cramming": is_cramming,
            "details": self._generate_details(
                mastery_factor, time_gap_factor, difficulty_factor,
                cramming_penalty, is_cramming, state,
            ),
        }

    async def get_state(self, kp_id: str, user_id: str) -> dict[str, Any] | None:
        """获取反游戏化状态详情。"""
        state = await self._load_state(kp_id, user_id)
        if not state:
            return None
        return {
            "kp_id": state.kp_id,
            "kp_name": state.kp_id,
            "activities_24h": sum(
                1 for a in state.recent_activities
                if a.timestamp > time.time() - 86400
            ),
            "posterior_mean": round(state.posterior_mean, 4),
            "posterior_confidence": round(state.posterior_confidence, 4),
            "weighted_score": round(state.weighted_score, 4),
            "cramming_detected": self._detect_cramming(state),
        }

    # ── Internal methods ──────────────────────────────────────────────

    @staticmethod
    def _detect_cramming(state: AntiGamingState) -> bool:
        """检测间隔突击复习。

        在 CRAM_WINDOW_HOURS 小时内完成超过 CRAM_THRESHOLD_COUNT 次 → 判定突击。
        """
        now = time.time()
        cutoff = now - CRAM_WINDOW_HOURS * 3600
        count = sum(1 for a in state.recent_activities if a.timestamp > cutoff)
        return count >= CRAM_THRESHOLD_COUNT

    @staticmethod
    def _compute_mastery_factor(state: AntiGamingState, new_score: float) -> float:
        """掌握度因子 — 贝叶斯缓慢更新。

        已有高掌握度且新得分与历史一致 → 高分。
        已有高掌握度但新得分异常高 → 抑制（防止刷分）。
        """
        posterior = state.posterior_mean
        # 如果后验很低，新得分应该被鼓励（有提升空间）
        if posterior < 0.3:
            return min(1.0, new_score * 1.2)
        # 后验高且新得分也高 → 正常
        if new_score >= posterior:
            diff = new_score - posterior
            return posterior + diff * 0.5  # 抑制过度上涨
        # 后验高但新得分低 → 接受（可能真的忘了）
        return new_score

    @staticmethod
    def _compute_time_gap_factor(state: AntiGamingState) -> float:
        """时间间隔因子。

        两次学习间隔太短 → 降权。间隔越长（在合理范围内）→ 加分。
        最优间隔：24-72 小时。
        """
        if not state.recent_activities:
            return 1.0  # 首次学习 → 满分

        now = time.time()
        last_activity = state.recent_activities[-1]
        hours_since_last = (now - last_activity.timestamp) / 3600.0

        if hours_since_last < 0.5:
            return 0.3  # 30分钟内重复 → 严重降权
        elif hours_since_last < 2:
            return 0.6  # 2小时内 → 中度降权
        elif hours_since_last < 8:
            return 0.8  # 8小时内 → 轻度降权
        elif hours_since_last < 72:
            return 1.0  # 1-3天 → 最优间隔
        elif hours_since_last < 168:
            return 0.9  # 1周内 → 少量降权（可能遗忘）
        else:
            return 0.7  # 超过1周 → 更多降权

    @staticmethod
    def _compute_difficulty_factor(difficulty: int, score: float) -> float:
        """难度匹配度因子。

        在简单知识点上得高分 → 正常评分。
        在困难知识点上得高分 → 加分奖励。
        """
        if difficulty <= 2:
            return min(1.0, score * 0.9)  # 简单 → 轻微抑制
        elif difficulty <= 3:
            return score  # 中等 → 不变
        elif difficulty <= 4:
            return min(1.0, score * 1.1)  # 困难 → 奖励
        else:
            return min(1.0, score * 1.2)  # 非常困难 → 更多奖励

    @staticmethod
    def _generate_details(
        mastery: float,
        time_gap: float,
        difficulty: float,
        cramming: float,
        is_cramming: bool,
        state: AntiGamingState,
    ) -> str:
        """生成人类可读的评分说明。"""
        parts = []
        if mastery < 0.5:
            parts.append("掌握度较低，得分有提升空间")
        elif mastery > 0.8:
            parts.append(f"掌握度较高（历史均值 {state.posterior_mean:.1%}），新得分适度计入")

        if time_gap < 0.5:
            parts.append("两次学习间隔过短，时间因子降权")
        elif time_gap > 0.9:
            parts.append("学习间隔合理，时间因子正常")

        if difficulty > 3:
            parts.append(f"知识点难度 {difficulty}/5，有难度奖励")

        if is_cramming:
            cram_count = sum(
                1 for a in state.recent_activities
                if a.timestamp > time.time() - CRAM_WINDOW_HOURS * 3600
            )
            parts.append(f"检测到突击复习（{CRAM_WINDOW_HOURS}h 内 {cram_count} 次），总得分降权")

        return "；".join(parts) if parts else "评分正常"
