"""Forgetting curve check tool — checks learner's memory state for a knowledge point."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.tools.base import BaseTool, ToolParameter, ToolResult, ToolSpec

if TYPE_CHECKING:
    from src.learning_path.forgetting_curve import ForgettingCurveService

logger = logging.getLogger(__name__)


class ForgettingCheckTool(BaseTool):
    """Tool for checking the forgetting curve state of a learner on a KP."""

    def __init__(
        self,
        forgetting_service: ForgettingCurveService | None = None,
    ) -> None:
        self._forgetting = forgetting_service

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="forgetting_check",
            description="检查学习者在某个知识点的遗忘曲线状态。了解学生对特定知识点的记忆程度，判断是否需要复习。",
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    description="用户 ID",
                    required=True,
                ),
                ToolParameter(
                    name="kp_id",
                    type="string",
                    description="知识点 ID",
                    required=True,
                ),
            ],
        )

    async def execute(self, user_id: str, kp_id: str, **kwargs) -> ToolResult:
        if self._forgetting is None:
            return ToolResult(success=False, error="Forgetting curve service not available")

        try:
            # predict_recall uses (kp_id, user_id) parameter order
            recall_prob = await self._forgetting.predict_recall(kp_id=kp_id, user_id=user_id)

            # get_state returns ForgettingState or None, uses (kp_id, user_id)
            state = await self._forgetting.get_state(kp_id=kp_id, user_id=user_id)

            # Determine alert level from recall probability
            from src.learning_path.forgetting_curve import ALERT_RECALL_THRESHOLD, URGENT_RECALL_THRESHOLD
            if recall_prob < URGENT_RECALL_THRESHOLD:
                alert = "🔴 紧急复习！回忆概率极低"
            elif recall_prob < ALERT_RECALL_THRESHOLD:
                alert = "🟡 建议复习，记忆在衰退"
            else:
                alert = "🟢 记忆状态良好"

            strength = state.strength if state else 0
            review_count = state.review_count if state else 0

            output = (
                f"知识点 {kp_id} 的记忆状态：\n"
                f"  回忆概率: {recall_prob:.2%}\n"
                f"  记忆强度: {strength:.1f} 小时\n"
                f"  复习次数: {review_count}\n"
                f"  状态: {alert}\n"
            )
            return ToolResult(
                success=True,
                output=output,
                data={"recall_probability": recall_prob, "strength": strength, "review_count": review_count},
            )
        except Exception as exc:
            logger.exception("ForgettingCheckTool failed")
            return ToolResult(success=False, error=str(exc))
