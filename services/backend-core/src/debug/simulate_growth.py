"""Growth simulation endpoint — simulates learning progress over time.

Debug-only endpoint for demo and testing purposes.  Uses existing
PathService and ForgettingCurveService to generate realistic learning
trajectories.
"""

from __future__ import annotations

import logging
import random

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/debug", tags=["debug"])


@router.post("/simulate-growth")
async def simulate_growth(
    request: Request,
    user_id: str = Query("anonymous"),
    course_id: str = Query(..., description="Course ID to simulate"),
    days: int = Query(7, ge=1, le=30, description="Number of days to simulate"),
):
    """Simulate learning progress over N days.

    For each simulated day:
    1. Pick 1-3 random knowledge points from the course.
    2. Record progress (completed) for each one.
    3. Record a quiz review with a random score.
    4. Snapshot the forgetting curve state + learning path.

    Returns a timeline of daily snapshots and a final summary.
    """
    path_service = getattr(request.app.state, "path_service", None)
    forgetting_service = getattr(request.app.state, "forgetting_service", None)

    if not path_service:
        raise HTTPException(status_code=503, detail="PathService not available")
    if not forgetting_service:
        raise HTTPException(status_code=503, detail="ForgettingCurveService not available")

    # Get the initial path to discover KPs in the course
    try:
        initial_path = await path_service.get_personalized_path(
            course_id=course_id, user_id=user_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load course graph: {exc}",
        )

    all_kps = [n.kp_id for n in initial_path.nodes]
    if not all_kps:
        raise HTTPException(status_code=404, detail="No knowledge points found for this course")

    timeline: list[dict] = []
    total_quizzes = 0

    # Seed RNG for reproducible results
    rng = random.Random(f"{user_id}-{course_id}-{days}")

    for day in range(1, days + 1):
        # Simulate 1-3 KPs studied per day
        kps_today = rng.sample(
            all_kps,
            k=rng.randint(1, min(3, len(all_kps))),
        )
        day_quizzes = 0

        for kp_id in kps_today:
            # Simulate quiz score (typically improves over time)
            base_score = 0.6 + (day / days) * 0.3  # 0.6 → 0.9 over the period
            score = min(1.0, base_score + rng.uniform(-0.15, 0.15))
            score = max(0.3, score)

            # Record progress via PathService
            try:
                await path_service.record_progress(
                    user_id=user_id,
                    course_id=course_id,
                    kp_id=kp_id,
                    status="completed",
                    quiz_score=score,
                )
            except Exception as exc:
                logger.warning("record_progress failed for kp=%s: %s", kp_id, exc)

            # Update forgetting curve
            try:
                await forgetting_service.update_after_quiz(
                    kp_id=kp_id, user_id=user_id, score=score,
                )
            except Exception as exc:
                logger.warning("update_after_quiz failed for kp=%s: %s", kp_id, exc)

            day_quizzes += 1
            total_quizzes += 1

        # Snapshot forgetting curve state for all KPs
        forgetting_snapshot = {}
        for kp_id in all_kps:
            try:
                recall = await forgetting_service.predict_recall(
                    kp_id=kp_id, user_id=user_id,
                )
                state = await forgetting_service.get_state(
                    kp_id=kp_id, user_id=user_id,
                )
                forgetting_snapshot[kp_id] = {
                    "recall_probability": round(recall, 4),
                    "review_count": state.review_count if state else 0,
                    "strength_hours": round(state.strength, 1) if state else 0,
                }
            except Exception as exc:
                forgetting_snapshot[kp_id] = {"error": str(exc)}

        # Snapshot learning path
        try:
            current_path = await path_service.get_personalized_path(
                course_id=course_id, user_id=user_id,
            )
            path_snapshot = {
                "total": current_path.total_count,
                "completed": current_path.completed_count,
                "mastered": current_path.mastered_count,
                "progress_pct": round(current_path.progress_percent, 1),
            }
        except Exception as exc:
            path_snapshot = {"error": str(exc)}

        timeline.append({
            "day": day,
            "kps_studied": kps_today,
            "quizzes_taken": day_quizzes,
            "path": path_snapshot,
            "forgetting": forgetting_snapshot,
        })

        # Simulate time passing (reduce wall-clock wait)
        # No actual sleep — we just advance the logical day

    # Final summary
    try:
        alerts = await forgetting_service.get_alerts(user_id=user_id)
    except Exception:
        alerts = []

    try:
        final_path = await path_service.get_personalized_path(
            course_id=course_id, user_id=user_id,
        )
    except Exception:
        final_path = None

    return {
        "simulation": {
            "user_id": user_id,
            "course_id": course_id,
            "days_simulated": days,
            "total_quizzes": total_quizzes,
        },
        "final_path": {
            "total": final_path.total_count if final_path else 0,
            "completed": final_path.completed_count if final_path else 0,
            "mastered": final_path.mastered_count if final_path else 0,
            "progress_pct": round(final_path.progress_percent, 1) if final_path else 0,
        } if final_path else None,
        "alerts_count": len(alerts),
        "timeline": timeline,
        "note": "Simulation uses approximate models — results are for demo purposes only.",
    }
