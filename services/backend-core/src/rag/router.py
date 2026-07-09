"""RAG / Mentor API router — streaming RAG-powered Q&A.

Endpoints::

    GET  /api/v1/mentor/stream/{user_id}?query=xxx   — SSE streaming answer
    GET  /api/v1/mentor/search?query=xxx&top_k=5      — search only (preview)
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query, Request
from sse_starlette.sse import EventSourceResponse

from src.rag.models import MentorSource
from src.rag.rag_service import RAGRetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mentor", tags=["mentor"])


# ── Dependencies ─────────────────────────────────────────────────────


async def get_rag_service(request: Request) -> RAGRetrievalService:
    """Provide the RAGRetrievalService from app state."""
    svc: RAGRetrievalService = request.app.state.rag_service
    return svc


async def get_mentor_agent(request: Request):
    """Provide the MentorAgent from app state."""
    return request.app.state.mentor_agent


# ── Endpoints ─────────────────────────────────────────────────────────


@router.get("/stream/{user_id}")
async def stream_mentor(
    user_id: str,
    query: str = Query(...),
    request: Request = None,
):
    """SSE streaming Mentor Q&A.

    Events::

        event: source
        data: {"sources": [{"id":"kp-xx", "name":"...", "type":"neo4j", "score":0.9}]}

        event: token
        data: {"content": "根据资料[1]..."}

        event: complete
        data: {"sources": [...], "confidence": 0.85}
    """
    agent = getattr(request.app.state, "mentor_agent", None)
    if not agent:
        return EventSourceResponse(
            async_gen_error("Mentor agent not configured")
        )

    async def event_generator():
        async for event in agent.answer_stream(query=query, user_id=user_id):
            yield {
                "event": event["type"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/search")
async def search_knowledge(
    query: str = Query(...),
    top_k: int = Query(5, ge=1, le=20),
    svc: RAGRetrievalService = Depends(get_rag_service),
):
    """Search knowledge base without generating an answer.

    Useful for frontend preview / autocomplete.
    """
    results = await svc.search(query, top_k=top_k)
    return {
        "query": query,
        "results": [
            {
                "id": r.source_id,
                "name": r.source_name,
                "type": r.source_type,
                "score": r.score,
                "summary": r.content[:200],
            }
            for r in results
        ],
        "total": len(results),
    }


# ── Helpers ──────────────────────────────────────────────────────────


async def async_gen_error(msg: str):
    yield {"event": "error", "data": json.dumps({"content": msg})}
