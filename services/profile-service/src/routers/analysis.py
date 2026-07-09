import json

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from src.schemas.profile import AnalyzeRequest

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
        async for event in analyzer.analyze_stream(message):
            if event["type"] == "token":
                yield {
                    "event": "token",
                    "data": json.dumps({"content": event["content"]}),
                }
            elif event["type"] == "profile_update":
                # Persist profile
                repo = request.app.state.profile_repo
                await repo.upsert_profile(user_id, event["profile"])
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

    return EventSourceResponse(event_generator())
