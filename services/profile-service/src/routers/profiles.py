from fastapi import APIRouter, HTTPException, Request

from src.schemas.profile import ProfileUpdateRequest

router = APIRouter(prefix="/api/v1/profiles", tags=["profiles"])


@router.get("/{user_id}")
async def get_profile(user_id: str, request: Request) -> dict:
    repo = request.app.state.profile_repo
    profile = await repo.get_profile(user_id)
    if profile is None:
        return {"user_id": user_id, "profile": {}, "version": 0, "updated_at": None}
    return profile


@router.put("/{user_id}")
async def update_profile(
    user_id: str, body: ProfileUpdateRequest, request: Request
) -> dict:
    """新建/更新用户画像。

    为了消除"双 Profile"问题——前端 SSE 推送的 6 维画像 confidence_scores
    与 settings 页面写入的显式偏好 notifications_enabled 各自拥有独立写入路径
    会互相覆盖——把顶层可选字段合入同一个 ``profile_data`` 的子字段：

    - ``confidence_scores`` → ``profile_data["confidence_scores"]``
    - ``notifications_enabled`` → ``profile_data["notifications_enabled"]``

    合入时只覆盖存在的字段，保留其他既有键值，避免双向写入时把对方的字段抹掉。
    """
    repo = request.app.state.profile_repo
    # 先读已存画像做顶层 key 级合并（"最新写入"语义）：局部 PUT 只覆盖请求
    # 中出现的顶层字段，其余字段保留，避免把其他端/来源写入的字段抹掉
    # （profile-service 是唯一真理源）。嵌套 dict 仍整体替换。
    stored = await repo.get_profile(user_id)
    stored_data = (stored or {}).get("profile") or {}
    if not isinstance(stored_data, dict):
        stored_data = {}
    merged = {**stored_data, **dict(body.profile_data)}
    if body.confidence_scores is not None:
        merged["confidence_scores"] = dict(body.confidence_scores)
    if body.notifications_enabled is not None:
        merged["notifications_enabled"] = bool(body.notifications_enabled)
    return await repo.upsert_profile(user_id, merged)


@router.delete("/{user_id}", status_code=204)
async def delete_profile(user_id: str, request: Request) -> None:
    repo = request.app.state.profile_repo
    deleted = await repo.delete_profile(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")


@router.get("/")
async def list_profiles(
    request: Request, skip: int = 0, limit: int = 100
) -> list[dict]:
    repo = request.app.state.profile_repo
    return await repo.list_profiles(skip, limit)
