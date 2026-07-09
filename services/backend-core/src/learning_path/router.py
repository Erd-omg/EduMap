"""Learning Path API router — personalized learning paths + adaptive routing.

Endpoints::

    GET    /api/v1/learning-path/{course_id}?user_id=xxx
    GET    /api/v1/learning-path/{course_id}/next?user_id=xxx&last_kp_id=xxx
    POST   /api/v1/learning-path/progress
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.kg.repositories.edge_repo import EdgeRepository
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
from src.learning_path.models import (
    ContentTypeSuggestion,
    PathNode,
    PathRecommendation,
    PersonalizedPath,
)
from src.learning_path.path_service import PathService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/learning-path", tags=["learning-path"])


# ── Dependencies ─────────────────────────────────────────────────────


async def get_path_service(request: Request) -> PathService:
    """Provide a PathService from app state."""
    svc: PathService = request.app.state.path_service
    return svc


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/{course_id}", response_model=PersonalizedPath)
async def get_personalized_path(
    course_id: str,
    user_id: str = Query("anonymous"),
    svc: PathService = Depends(get_path_service),
):
    """Get a full personalized learning path for a course."""
    # If user has a profile, fetch it from profile-service
    profile = None
    if user_id and user_id != "anonymous":
        try:
            profile = await _fetch_profile(user_id)
        except Exception:
            logger.warning("Could not fetch profile for user %s, using defaults", user_id)

    return await svc.get_personalized_path(course_id, user_id, profile)


@router.get("/{course_id}/next")
async def get_next_recommendation(
    course_id: str,
    user_id: str = Query("anonymous"),
    last_kp_id: str | None = Query(None),
    svc: PathService = Depends(get_path_service),
):
    """Get the next recommended knowledge point."""
    profile = None
    if user_id and user_id != "anonymous":
        try:
            profile = await _fetch_profile(user_id)
        except Exception:
            pass

    rec = await svc.get_next_recommendation(course_id, user_id, last_kp_id, profile)
    if not rec:
        raise HTTPException(status_code=404, detail="No next recommendation available — all KPs completed?")

    return rec


@router.post("/progress")
async def record_progress(
    body: dict,
    request: Request,
):
    """Record user progress and return the updated path.

    Body::

        {"user_id": "...", "course_id": "...", "kp_id": "...",
         "status": "completed", "quiz_score": 0.85}
    """
    svc: PathService = request.app.state.path_service
    updated_path = await svc.record_progress(
        user_id=body.get("user_id", "anonymous"),
        course_id=body.get("course_id", ""),
        kp_id=body.get("kp_id", ""),
        status=body.get("status", "completed"),
        quiz_score=body.get("quiz_score"),
    )

    # Also return next recommendation
    profile = None
    try:
        profile = await _fetch_profile(body.get("user_id", "anonymous"))
    except Exception:
        pass
    next_rec = await svc.get_next_recommendation(
        body.get("course_id", ""),
        body.get("user_id", "anonymous"),
        body.get("kp_id"),
        profile,
    )

    return {
        "updated_path": updated_path.model_dump(),
        "next_recommendation": next_rec.model_dump() if next_rec else None,
    }


@router.get("/{course_id}/content-style")
async def get_content_style_suggestion(
    course_id: str,
    user_id: str = Query("anonymous"),
):
    """Get content type recommendations based on learning style."""
    profile = None
    if user_id and user_id != "anonymous":
        try:
            profile = await _fetch_profile(user_id)
        except Exception:
            pass
    suggestion = PathService.get_content_type_suggestion(profile)
    return suggestion.model_dump()


# ── Internal helpers ──────────────────────────────────────────────────


async def _fetch_profile(user_id: str) -> dict | None:
    """Fetch user profile from the profile-service via httpx."""
    import httpx
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"http://profile-service:8001/api/v1/profiles/{user_id}")
        if resp.status_code == 200:
            data = resp.json()
            return data.get("profile")
    return None
