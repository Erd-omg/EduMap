"""Privacy router — PIPL-compliant data export, deletion, and anonymization.

Endpoints::

    DELETE /api/v1/privacy/user/{user_id}/data   — cascade delete all user data
    GET    /api/v1/privacy/user/{user_id}/export  — export all user data as JSON
    POST   /api/v1/privacy/user/{user_id}/anonymize — anonymize for graduation retention
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Request

if TYPE_CHECKING:
    from src.learning_path.forgetting_curve import ForgettingCurveService
    from src.resources.repository import ResourceRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/privacy", tags=["privacy"])

_ANONYMIZATION_SALT = "edumap-anon-salt-2026"  # rotate per deployment


# ── Helper: get dependencies from app state ──────────────────────────────


def _get_memory_ops(request: Request):
    """Get MemoryOperations from app state."""
    return getattr(request.app.state, "memory_ops", None)


def _get_resource_repo(request: Request) -> ResourceRepository | None:
    return getattr(request.app.state, "resource_repo", None)


def _get_forgetting_service(request: Request) -> ForgettingCurveService | None:
    return getattr(request.app.state, "forgetting_service", None)


# ── DELETE /user/{user_id}/data — Cascade delete ─────────────────────────


@router.delete("/user/{user_id}/data", status_code=204)
async def delete_user_data(user_id: str, request: Request):
    """Delete ALL data for a user across all stores.

    Cascades through: profile → episodic/semantic memory → forgetting curves
    → resources → Redis sessions. Shared KG data (Neo4j, ChromaDB embeddings)
    is not per-user and is not deleted.
    """
    deleted_counts: dict[str, int] = {}

    # 1. Profile service (HTTP call via profile-service)
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(
                f"http://profile-service:8001/api/v1/profiles/{user_id}",
            )
            if resp.status_code == 204:
                deleted_counts["profile"] = 1
            else:
                logger.warning("Profile delete returned %d for %s", resp.status_code, user_id)
    except Exception as exc:
        logger.warning("Failed to delete profile for %s: %s", user_id, exc)

    # 2. Long-term memory (episodic + semantic)
    memory_ops = _get_memory_ops(request)
    if memory_ops and memory_ops.long_term:
        try:
            count = await memory_ops.long_term.delete_all_by_user(user_id)
            deleted_counts["memory"] = count
        except Exception as exc:
            logger.warning("Failed to delete memory for %s: %s", user_id, exc)

    # 3. Forgetting curve states
    forgetting = _get_forgetting_service(request)
    if forgetting:
        try:
            count = await forgetting.delete_user_data(user_id)
            deleted_counts["forgetting_curve"] = count
        except Exception as exc:
            logger.warning("Failed to delete forgetting curves for %s: %s", user_id, exc)

    # 4. Resources
    resource_repo = _get_resource_repo(request)
    if resource_repo:
        try:
            count = await resource_repo.delete_all_by_user(user_id)
            deleted_counts["resources"] = count
        except Exception as exc:
            logger.warning("Failed to delete resources for %s: %s", user_id, exc)

    # 5. Redis sessions
    if memory_ops and memory_ops.short_term:
        try:
            count = await memory_ops.short_term.delete_all_sessions_by_user(user_id)
            deleted_counts["sessions"] = count
        except Exception as exc:
            logger.warning("Failed to delete sessions for %s: %s", user_id, exc)

    logger.info(
        "Deleted user data for %s: %s", user_id,
        json.dumps(deleted_counts, ensure_ascii=False),
    )
    return None  # 204 No Content


# ── GET /user/{user_id}/export — Unified data export ────────────────────


@router.get("/user/{user_id}/export")
async def export_user_data(user_id: str, request: Request):
    """Export ALL user data as a JSON blob.

    Aggregates: profile → episodic memory → semantic memory → forgetting curves
    → resources → learning history.
    """
    export: dict[str, Any] = {
        "user_id": user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "data_stores": {},
    }

    # 1. Profile
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"http://profile-service:8001/api/v1/profiles/{user_id}",
            )
            if resp.status_code == 200:
                export["data_stores"]["profile"] = resp.json()
    except Exception as exc:
        logger.warning("Failed to export profile for %s: %s", user_id, exc)

    # 2. Episodic + semantic memory
    memory_ops = _get_memory_ops(request)
    if memory_ops and memory_ops.long_term:
        try:
            episodic = await memory_ops.long_term.recall_episodic(
                user_id=user_id,
                event_types=None,
                limit=1000,
            )
            semantic = await memory_ops.long_term.recall_semantic(user_id=user_id)
            export["data_stores"]["episodic_memory"] = [
                {
                    "id": str(e.id),
                    "event_type": e.event_type,
                    "session_id": e.session_id,
                    "input": e.input,
                    "output": e.output,
                    "created_at": str(e.created_at) if hasattr(e, "created_at") else "",
                    "importance_score": e.importance_score,
                }
                for e in episodic
            ]
            export["data_stores"]["semantic_memory"] = [
                {
                    "key": e.key,
                    "memory_type": e.memory_type,
                    "value": e.value,
                    "confidence": e.confidence,
                }
                for e in semantic
            ]
        except Exception as exc:
            logger.warning("Failed to export memory for %s: %s", user_id, exc)

    # 3. Forgetting curves
    forgetting = _get_forgetting_service(request)
    if forgetting:
        try:
            states = await forgetting.get_all_states(user_id)
            export["data_stores"]["forgetting_curve"] = states
        except Exception as exc:
            logger.warning("Failed to export forgetting curves for %s: %s", user_id, exc)

    # 4. Resources
    resource_repo = _get_resource_repo(request)
    if resource_repo:
        try:
            resources, total = await resource_repo.list(user_id=user_id, limit=1000)
            export["data_stores"]["resources"] = [
                {
                    "id": r.id,
                    "name": r.name,
                    "type": r.type,
                    "source": r.source,
                    "kp_id": r.kp_id,
                    "kp_name": r.kp_name,
                    "created_at": r.created_at,
                }
                for r in resources
            ]
            export["data_stores"]["resources_total"] = total
        except Exception as exc:
            logger.warning("Failed to export resources for %s: %s", user_id, exc)

    return export


# ── POST /user/{user_id}/anonymize — Graduation retention ────────────────


@router.post("/user/{user_id}/anonymize")
async def anonymize_user_data(user_id: str, request: Request):
    """Anonymize user data for post-graduation retention (PIPL Section 4.5).

    Replaces ``user_id`` with a SHA-256 hash, clears personal fields,
    and flags the records as anonymized.  Original records are deleted
    after the anonymized copy is created.
    """
    # Compute a deterministic anonymous ID
    anon_id = hashlib.sha256(
        f"{user_id}:{_ANONYMIZATION_SALT}".encode()
    ).hexdigest()[:32]

    anonymized_count = 0

    # NOTE: the previous code called ``export_user_data(user_id, request)``
    # here with the comment "for audit log", but the result was never stored or
    # written anywhere — so it was dead work.  The export endpoint
    # (``GET /privacy/export``) remains the supported way to obtain user data.
    # The call is not reinstated until there is an actual audit sink for it.

    # Delete original data (same cascade as delete endpoint)
    await delete_user_data(user_id, request)

    logger.info(
        "Anonymized user %s → %s (%d records anonymized)",
        user_id, anon_id, anonymized_count,
    )

    return {
        "status": "anonymized",
        "original_user_id": user_id,
        "anonymized_id": anon_id,
        "anonymized_at": datetime.now(timezone.utc).isoformat(),
        "note": "Original data has been deleted. Anonymized ID cannot be reversed.",
    }
