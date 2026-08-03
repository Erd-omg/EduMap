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

from src.rag.models import MentorSource, EvalReport, RerankPreviewRequest, RerankPreviewResponse
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
        # Load conversation history from memory system
        conversation_history = None
        try:
            memory_ops = getattr(request.app.state, "memory_ops", None)
            if memory_ops:
                from src.memory.models import EventType
                episodic = await memory_ops.long_term.recall_episodic(
                    user_id=user_id,
                    event_types=[EventType.MENTOR_QUERY],
                    limit=10,
                )
                if episodic:
                    conversation_history = [
                        {"input": e.input, "output": e.output}
                        for e in episodic
                    ]
        except Exception as exc:
            logger.debug("Failed to load conversation history: %s", exc)

        try:
            async for event in agent.answer_stream(
                query=query,
                user_id=user_id,
                conversation_history=conversation_history,
            ):
                yield {
                    "event": event["type"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }

            # Record this interaction to episodic memory
            try:
                if memory_ops:
                    await memory_ops.record_interaction(
                        user_id=user_id,
                        event_type="mentor_query",
                        input_text=query,
                        output_text="(streamed)",
                        importance=0.5,
                    )
            except Exception as exc:
                logger.debug("Failed to record mentor interaction: %s", exc)

        except Exception as exc:
            logger.exception("Mentor stream error")
            yield {
                "event": "error",
                "data": json.dumps({"content": f"生成回答时出错: {exc}"}, ensure_ascii=False),
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


@router.get("/evaluate", response_model=EvalReport)
async def evaluate_rag(
    request: Request,
    n_queries: int = Query(10, ge=1, le=50),
    course_id: str | None = Query(None),
    use_sample_queries: bool = Query(False, alias="sample"),
):
    """Run RAG quality evaluation benchmark.

    Measures retrieval precision, recall, MRR, NDCG, and hit rate.
    By default generates test queries from the KG (KP names as queries).
    Pass ``sample=true`` to use the hand-curated student query dataset
    from ``datasets/sample_queries.json`` for more realistic evaluation.
    """
    from src.rag.evaluation.benchmark import RAGEvalBenchmark
    from src.rag.evaluation.evaluator import RAGEvaluator

    rag = getattr(request.app.state, "rag_service", None)
    kp_repo = getattr(request.app.state, "neo4j_pool", None)
    if not rag or not kp_repo:
        return EvalReport(status="error", n_queries=0, metrics={"error": "RAG service or KG not available"})

    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    kp_repo_inst = KnowledgePointRepository(kp_repo) if not use_sample_queries else None

    evaluator = RAGEvaluator(k_values=[1, 3, 5])
    llm = getattr(request.app.state, "llm_adapter", None) if use_sample_queries else None

    benchmark = RAGEvalBenchmark(
        rag_service=rag,
        kp_repo=kp_repo_inst,  # type: ignore[arg-type]
        evaluator=evaluator,
    )

    result = await benchmark.run_benchmark(
        n_queries=n_queries,
        course_id=course_id,
        use_sample_queries=use_sample_queries,
        llm=llm,
    )
    return EvalReport(**result)


@router.post("/rerank-preview", response_model=RerankPreviewResponse)
async def rerank_preview(
    body: RerankPreviewRequest,
    request: Request,
):
    """Preview reranking results — shows before/after scores.

    Useful for tuning reranker parameters and comparing strategies.
    """
    rag = getattr(request.app.state, "rag_service", None)
    if not rag:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="RAG service not available")

    reranker = getattr(request.app.state, "reranker", None)
    if not reranker or not reranker.is_loaded:
        return RerankPreviewResponse(
            query=body.query,
            original_results=[],
            reranked_results=[],
            improvements={"error": "Reranker not loaded. Enable with RERANKER_ENABLED=true"},
        )

    # Get candidates
    raw_results = await rag.search(body.query, top_k=body.candidate_k)
    # Remove the reranker from rag_service temporarily to get original scores
    original_results = raw_results[:body.top_k]

    # Apply reranking
    reranked = await reranker.rerank(body.query, raw_results, top_k=body.top_k)

    # Calculate score deltas
    orig_by_id = {r.source_id: r.score for r in original_results}
    improvements: dict[str, float] = {}
    for r in reranked:
        orig_score = orig_by_id.get(r.source_id, 0)
        delta = round(r.score - orig_score, 4)
        improvements[r.source_name or r.source_id] = delta

    return RerankPreviewResponse(
        query=body.query,
        original_results=original_results,
        reranked_results=reranked,
        improvements=improvements,
    )


# ── Helpers ──────────────────────────────────────────────────────────


async def async_gen_error(msg: str):
    yield {"event": "error", "data": json.dumps({"content": msg})}
