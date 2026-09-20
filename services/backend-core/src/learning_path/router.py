"""Learning Path API router — personalized learning paths + adaptive routing.

Endpoints::

    GET    /api/v1/learning-path/{course_id}?user_id=xxx
    GET    /api/v1/learning-path/{course_id}/next?user_id=xxx&last_kp_id=xxx
    POST   /api/v1/learning-path/progress
    GET    /api/v1/learning-path/recommendations/{user_id}?limit=20
    GET    /api/v1/learning-path/notifications/{user_id}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.kg.repositories.edge_repo import EdgeRepository
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
from src.learning_path.forgetting_curve import (
    ALERT_RECALL_THRESHOLD,
    URGENT_RECALL_THRESHOLD,
    ForgettingCurveService,
)
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
    request: Request,
    user_id: str = Query("anonymous"),
    last_kp_id: str | None = Query(None),
    svc: PathService = Depends(get_path_service),
):
    """Get the next recommended knowledge point.

    会拉取用户的遗忘状态并注入 ``forgetting_map``，使已经遗忘的知识点
    （即使已标记 completed）重新进入复习候选。
    """
    profile = None
    if user_id and user_id != "anonymous":
        try:
            profile = await _fetch_profile(user_id)
        except Exception:
            pass

    forgetting_map = await _get_forgetting_map(request, user_id)
    rec = await svc.get_next_recommendation(
        course_id, user_id, last_kp_id, profile, forgetting_map=forgetting_map,
    )
    if not rec:
        raise HTTPException(status_code=404, detail="No next recommendation available — all KPs completed?")

    # 推荐结果落库（失败不影响响应）
    await svc.save_recommendation(user_id, course_id, rec, source="next")

    return rec


@router.post("/progress")
async def record_progress(
    body: dict,
    request: Request,
):
    """Record user progress and return the updated path.

    Body::

        {"user_id": "...", "course_id": "...", "kp_id": "...",
         "status": "completed", "quiz_score": 0.85, "difficulty": 3}

    若提供 ``quiz_score``，本端点会自动串联 **反游戏化加权** 与 **遗忘曲线更新**，
    并使用加权后的得分落库、驱动下一次推荐，使
    "学习 → 评估 → 遗忘预测 → 再推荐" 无需前端额外调用即可闭环。
    """
    svc: PathService = request.app.state.path_service
    user_id = body.get("user_id", "anonymous")
    course_id = body.get("course_id", "")
    kp_id = body.get("kp_id", "")
    raw_score = body.get("quiz_score")
    effective_score = raw_score
    anti_gaming_result: dict | None = None

    if raw_score is not None:
        # ── 1. 反游戏化多因子加权（含突击刷分检测）──────────────────
        anti_svc = getattr(request.app.state, "anti_gaming_service", None)
        if anti_svc is not None:
            try:
                anti_gaming_result = await anti_svc.calculate_weighted_score(
                    kp_id=kp_id,
                    user_id=user_id,
                    raw_score=float(raw_score),
                    difficulty=int(body.get("difficulty", 3)),
                )
                effective_score = anti_gaming_result["weighted_score"]
            except Exception as exc:
                logger.warning("Anti-gaming scoring failed for %s/%s: %s", user_id, kp_id, exc)

        # ── 2. 遗忘曲线更新（使用加权得分，刷分不会虚增记忆强度）────
        fc = getattr(request.app.state, "forgetting_service", None)
        if fc is not None:
            try:
                await fc.update_after_quiz(kp_id, user_id, float(effective_score))
            except Exception as exc:
                logger.warning("Forgetting update failed for %s/%s: %s", user_id, kp_id, exc)

    # ── 3. 落库（存加权得分，避免刷分污染掌握度）────────────────────
    updated_path = await svc.record_progress(
        user_id=user_id,
        course_id=course_id,
        kp_id=kp_id,
        status=body.get("status", "completed"),
        quiz_score=effective_score,
    )

    # ── 4. 拉取遗忘状态 → 驱动再推荐 ─────────────────────────────────
    profile = None
    try:
        profile = await _fetch_profile(user_id)
    except Exception:
        pass

    forgetting_map = await _get_forgetting_map(request, user_id)
    next_rec = await svc.get_next_recommendation(
        course_id,
        user_id,
        kp_id,
        profile,
        forgetting_map=forgetting_map,
    )
    if next_rec is not None:
        # 闭环后产生的新推荐落库（失败不影响响应）
        await svc.save_recommendation(user_id, course_id, next_rec, source="after_progress")

    response: dict = {
        "updated_path": updated_path.model_dump(),
        "next_recommendation": next_rec.model_dump() if next_rec else None,
    }

    if anti_gaming_result is not None:
        response["anti_gaming"] = {
            "raw_score": anti_gaming_result["raw_score"],
            "weighted_score": anti_gaming_result["weighted_score"],
            "factors": anti_gaming_result["factors"],
            "is_cramming": anti_gaming_result["is_cramming"],
            "details": anti_gaming_result["details"],
        }

    forgetting_info = await _get_forgetting_info(request, kp_id, user_id)
    if forgetting_info is not None:
        response["forgetting"] = forgetting_info

    return response


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


# ── Recommendation history & notifications ────────────────────────────


@router.get("/recommendations/{user_id}")
async def get_recommendation_history(
    user_id: str,
    request: Request,
    limit: int = Query(20, ge=1, le=100),
):
    """查询用户最近落库的推荐历史（最新在前）。"""
    svc: PathService = request.app.state.path_service
    history = await svc.get_recommendation_history(user_id, limit)
    return {"user_id": user_id, "recommendations": history, "total": len(history)}


@router.get("/notifications/{user_id}")
async def get_notifications(
    user_id: str,
    request: Request,
):
    """聚合通知：遗忘复习预警 + 最近推荐。

    遵循用户在 profile 中的 ``notifications_enabled`` 开关（默认开启）；
    关闭时返回空列表且 ``enabled: false``，前端不再展示。
    """
    # ── 1. 读取通知开关（读不到 profile 时默认开启）─────────────────
    enabled = True
    if user_id and user_id != "anonymous":
        try:
            profile = await _fetch_profile(user_id)
            if profile and isinstance(profile.get("notifications_enabled"), bool):
                enabled = profile["notifications_enabled"]
        except Exception as exc:
            logger.warning(
                "Failed to read notifications_enabled for %s: %s", user_id, exc
            )

    if not enabled:
        return {"user_id": user_id, "enabled": False, "notifications": [], "total": 0}

    notifications: list[dict] = []

    # ── 2. 遗忘复习预警（urgent / warning）─────────────────────────
    fc = getattr(request.app.state, "forgetting_service", None)
    if fc is not None:
        try:
            alerts = await fc.get_alerts(user_id)
            for a in alerts:
                if a.get("alert_level") not in ("urgent", "warning"):
                    continue
                recall = a.get("recall_probability", 0.0)
                notifications.append({
                    "type": "review_reminder",
                    "severity": a["alert_level"],
                    "kp_id": a["kp_id"],
                    "kp_name": a.get("kp_name", a["kp_id"]),
                    "title": f"「{a.get('kp_name', a['kp_id'])}」即将遗忘",
                    "message": (
                        f"当前回忆概率 {recall:.0%}，"
                        f"建议尽快复习（已复习 {a.get('review_count', 0)} 次）"
                    ),
                })
        except Exception as exc:
            logger.warning("Failed to build review alerts for %s: %s", user_id, exc)

    # ── 3. 最近一次推荐（来自落库的历史）───────────────────────────
    path_svc: PathService | None = getattr(request.app.state, "path_service", None)
    if path_svc is not None:
        history = await path_svc.get_recommendation_history(user_id, limit=1)
        if history:
            latest = history[0]
            notifications.append({
                "type": "next_step",
                "severity": "info",
                "kp_id": latest["kp_id"],
                "kp_name": latest.get("kp_name") or latest["kp_id"],
                "title": f"推荐下一步：{latest.get('kp_name') or latest['kp_id']}",
                "message": latest.get("reason") or "根据你的学习进度与遗忘状态推荐",
                "created_at": latest.get("created_at"),
            })

    return {
        "user_id": user_id,
        "enabled": True,
        "notifications": notifications,
        "total": len(notifications),
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
    """Fetch user profile from the profile-service via httpx.

    地址取自 ``settings.profile_service_base``：host 模式默认
    ``http://localhost:8001``，容器模式由 docker-compose 覆盖为
    ``http://profile-service:8001``。失败仅记 warning 并返回 None——
    通知开关读不到时默认开启，不影响主流程。
    """
    import httpx

    from src.config import settings

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.profile_service_base}/api/v1/profiles/{user_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("profile")
    except Exception as exc:
        logger.warning(
            "Failed to fetch profile for %s from %s: %s",
            user_id,
            settings.profile_service_base,
            exc,
        )
    return None


def _alert_level(recall: float) -> str:
    """按遗忘阈值把回忆概率分级（与 forgetting_curve 保持一致）。"""
    if recall < URGENT_RECALL_THRESHOLD:
        return "urgent"
    if recall < ALERT_RECALL_THRESHOLD:
        return "warning"
    return "ok"


async def _get_forgetting_map(request: Request, user_id: str) -> dict[str, float]:
    """构建 ``{kp_id: recall_probability}``，供路径重算的 W_FORGET 使用。"""
    fc = getattr(request.app.state, "forgetting_service", None)
    if fc is None or not user_id or user_id == "anonymous":
        return {}
    try:
        return await fc.get_recall_map(user_id)
    except Exception as exc:
        logger.warning("Failed to build forgetting map for %s: %s", user_id, exc)
        return {}


async def _get_forgetting_info(
    request: Request, kp_id: str, user_id: str,
) -> dict | None:
    """返回单个知识点的遗忘状态摘要，用于元认知透明度展示。"""
    fc = getattr(request.app.state, "forgetting_service", None)
    if fc is None or not kp_id or not user_id or user_id == "anonymous":
        return None
    try:
        state = await fc.get_state(kp_id, user_id)
    except Exception as exc:
        logger.debug("Failed to read forgetting state for %s/%s: %s", user_id, kp_id, exc)
        return None
    if state is None:
        return None
    # 从未学习过（last_review_time == 0）时 predict_recall() 恒为 0.0，
    # 直接上报会误导为"已完全遗忘"，因此视为无遗忘数据。
    if state.last_review_time <= 0:
        return None
    recall = state.predict_recall()
    return {
        "kp_id": kp_id,
        "recall_probability": round(recall, 4),
        "alert_level": _alert_level(recall),
        "strength": round(state.strength, 1),
        "review_count": state.review_count,
    }
