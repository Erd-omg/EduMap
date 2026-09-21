import json
import logging

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.schemas.profile import AnalyzeRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.post("/analyze")
async def analyze_conversation(
    body: AnalyzeRequest, request: Request
) -> dict:
    """Non-streaming analysis endpoint."""
    analyzer = request.app.state.profile_analyzer
    result = await analyzer.analyze(body.user_message, body.conversation_history)
    return result


@router.get("/stream/{user_id}")
async def stream_analysis(
    user_id: str, message: str, request: Request
) -> EventSourceResponse:
    """SSE streaming analysis endpoint. Frontend connects with EventSource."""
    analyzer = request.app.state.profile_analyzer

    async def event_generator():
        try:
            async for event in analyzer.analyze_stream(message):
                if event["type"] == "token":
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": event["content"]}),
                    }
                elif event["type"] == "profile_update":
                    # Persist profile
                    try:
                        repo = request.app.state.profile_repo
                        await repo.upsert_profile(user_id, event["profile"])
                    except Exception as persist_err:
                        logger.warning("Failed to persist profile: %s", persist_err)
                    yield {
                        "event": "profile_update",
                        "data": json.dumps({
                            "profile": event["profile"],
                            "confidence_scores": event["confidence_scores"],
                        }),
                    }
                elif event["type"] == "complete":
                    yield {
                        "event": "complete",
                        "data": json.dumps({"status": "done"}),
                    }
                elif event["type"] == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps({"content": event["content"]}),
                    }
        except Exception:
            logger.exception("SSE event_generator failed")
            yield {
                "event": "error",
                "data": json.dumps({"content": "分析服务出错，请重试"}),  # don't leak internal error details
            }

    return EventSourceResponse(event_generator())
