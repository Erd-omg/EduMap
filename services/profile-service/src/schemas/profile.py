from pydantic import BaseModel
from typing import Any


class ProfileUpdateRequest(BaseModel):
    """Request body for PUT /api/v1/profiles/{user_id}.

    ``profile_data`` 是核心画像（包含 LLM 分析出的 6 维 UserProfile + confidence_scores
    + 显示设置等所有自定义键）。为兼容既有客户端，``confidence_scores`` 与
    ``notifications_enabled`` 也可作为顶层可选字段单独传入——router 会把它们合入
    ``profile_data`` 的对应子字段，使这两条数据永远只有一份真理来源。
    """

    profile_data: dict[str, Any]
    confidence_scores: dict[str, float] | None = None
    notifications_enabled: bool | None = None


class AnalyzeRequest(BaseModel):
    user_message: str
    conversation_history: list[dict] | None = None


class AnalyzeResponse(BaseModel):
    user_id: str
    profile: dict[str, Any]
    confidence_scores: dict[str, float]
    analysis_text: str = ""


class ProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, Any]
    version: int
    updated_at: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
