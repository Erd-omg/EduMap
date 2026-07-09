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
    repo = request.app.state.profile_repo
    return await repo.upsert_profile(user_id, body.profile_data)


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
