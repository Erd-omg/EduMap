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
from src.learning_path.forgetting_curve import ForgettingCurveService
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


async def get_forgetting_service(request: Request) -> ForgettingCurveService:
    """Provide a ForgettingCurveService from app state."""
    svc: ForgettingCurveService = request.app.state.forgetting_service
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


# ── Quiz generation endpoint ─────────────────────────────────────────


@router.post("/quiz/generate")
async def generate_quiz(
    body: dict,
    request: Request,
):
    """Generate quiz questions for a knowledge point on demand.

    Body::

        {"kp_id": "...", "kp_name": "...", "kp_description": "...", "difficulty": 3}
    """
    from src.agents.assessment.agent import AssessmentAgent
    from src.agents.models import KnowledgeUnit

    llm = getattr(request.app.state, "llm_adapter", None)
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not available")

    agent = AssessmentAgent(llm_adapter=llm)
    ku = KnowledgeUnit(
        id=body.get("kp_id", ""),
        name=body.get("kp_name", ""),
        description=body.get("kp_description", ""),
        difficulty=body.get("difficulty", 3),
    )
    result = await agent.run_legacy(knowledge_unit=ku)

    return {
        "kp_id": body.get("kp_id"),
        "questions": [q.model_dump() for q in result.quiz],
        "confidence": result.confidence,
    }


# ── Forgetting Curve endpoints ────────────────────────────────────────


@router.get("/forgetting/{kp_id}")
async def get_forgetting_state(
    kp_id: str,
    user_id: str = Query("anonymous"),
    fc: ForgettingCurveService = Depends(get_forgetting_service),
):
    """Get forgetting curve state for a single KP."""
    state = await fc.get_state(kp_id, user_id)
    if not state:
        return {
            "kp_id": kp_id,
            "user_id": user_id,
            "recall_probability": 0.0,
            "status": "unknown",
        }
    return state.to_dict()


@router.post("/forgetting/review")
async def record_forgetting_review(
    body: dict,
    request: Request,
):
    """Record a review (study session) for a KP — refreshes the forgetting curve.

    Body::

        {"user_id": "...", "kp_id": "..."}
    """
    fc: ForgettingCurveService = request.app.state.forgetting_service
    state = await fc.record_review(
        kp_id=body.get("kp_id", ""),
        user_id=body.get("user_id", "anonymous"),
    )
    return state.to_dict()


@router.get("/forgetting/alerts/{user_id}")
async def get_forgetting_alerts(
    user_id: str,
    fc: ForgettingCurveService = Depends(get_forgetting_service),
):
    """Get forgetting curve alerts (KPs needing review)."""
    alerts = await fc.get_alerts(user_id)
    return {"alerts": alerts, "total": len(alerts)}


@router.get("/dashboard/{user_id}")
async def get_dashboard(
    user_id: str,
    request: Request,
):
    """Aggregated dashboard data for the learning dashboard page.

    Returns forgetting curve alerts, mastery distribution, and activity data.
    """
    fc: ForgettingCurveService = getattr(request.app.state, "forgetting_service", None)
    path_svc: PathService = getattr(request.app.state, "path_service", None)

    if not fc:
        raise HTTPException(status_code=503, detail="Forgetting service not available")

    # Forgetting curve alerts
    alerts = await fc.get_alerts(user_id)

    # All forgetting states
    all_states = await fc.get_all_states(user_id)

    # Compute mastery distribution
    mastery_dist = {
        "urgent": sum(1 for a in alerts if a["alert_level"] == "urgent"),
        "warning": sum(1 for a in alerts if a["alert_level"] == "warning"),
        "ok": sum(1 for a in alerts if a["alert_level"] == "ok"),
        "unknown": 0,
    }

    # Average recall
    avg_recall = (
        sum(s["recall_probability"] for s in all_states) / len(all_states)
        if all_states else 0.0
    )

    return {
        "alerts": alerts,
        "alert_count": len(alerts),
        "mastery_distribution": mastery_dist,
        "total_kps_tracked": len(all_states),
        "average_recall": round(avg_recall, 4),
        "all_states": all_states,
    }


# ── Anti-Gaming endpoints ────────────────────────────────────────────


@router.post("/anti-gaming/score")
async def calculate_anti_gaming_score(
    body: dict,
    request: Request,
):
    """Calculate weighted score with anti-gaming factors.

    Body::

        {"kp_id": "...", "user_id": "anonymous", "raw_score": 0.95, "difficulty": 3}
    """
    svc = getattr(request.app.state, "anti_gaming_service", None)
    if not svc:
        raise HTTPException(status_code=503, detail="Anti-gaming service not available")

    result = await svc.calculate_weighted_score(
        kp_id=body.get("kp_id", ""),
        user_id=body.get("user_id", "anonymous"),
        raw_score=body.get("raw_score", 0.5),
        difficulty=body.get("difficulty", 3),
    )
    return result


@router.get("/anti-gaming/state/{kp_id}")
async def get_anti_gaming_state(
    kp_id: str,
    user_id: str = Query("anonymous"),
    request: Request = None,
):
    """Get anti-gaming state for a single KP."""
    svc = getattr(request.app.state, "anti_gaming_service", None)
    if not svc:
        raise HTTPException(status_code=503, detail="Anti-gaming service not available")

    state = await svc.get_state(kp_id, user_id)
    if not state:
        return {"kp_id": kp_id, "user_id": user_id, "activities_24h": 0}
    return state


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
