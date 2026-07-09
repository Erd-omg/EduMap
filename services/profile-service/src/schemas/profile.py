from pydantic import BaseModel
from typing import Any


class ProfileUpdateRequest(BaseModel):
    profile_data: dict[str, Any]


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
