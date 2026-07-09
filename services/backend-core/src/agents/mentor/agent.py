"""Mentor agent — RAG-constrained Q&A for learning assistance.

Uses :class:`RAGRetrievalService` for hybrid search and the LLM adapter
to generate context-grounded answers with source citations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from src.rag.models import MentorResponse, MentorSource
from src.rag.rag_service import RAGRetrievalService
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class MentorAgent:
    """RAG-constrained learning Q&A agent."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        rag_service: RAGRetrievalService,
    ) -> None:
        self._llm = llm_adapter
        self._rag = rag_service

    async def answer(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_history: list[dict] | None = None,
    ) -> MentorResponse:
        """Generate a context-grounded answer with sources."""
        # 1. Retrieve
        results = await self._rag.search(query)
        context = await self._rag.assemble_context(results)

        # 2. Build prompt
        prompt = PromptRegistry.get(
            "mentor/v1-mentor",
            context=context.context_str,
            query=query,
        )

        # 3. Generate
        response = await self._llm.generate(prompt)

        return MentorResponse(
            answer=response.content,
            sources=[
                MentorSource(
                    id=s.source_id,
                    name=s.source_name,
                    type=s.source_type,
                    score=s.score,
                    summary=s.content[:200],
                )
                for s in context.sources
            ],
            confidence=min(0.9, 0.5 + 0.1 * len(context.sources)),
        )

    async def answer_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming answer — yields source, token, and complete events."""
        # 1. Retrieve
        results = await self._rag.search(query)
        context = await self._rag.assemble_context(results)

        # 2. Yield sources first (so frontend shows them immediately)
        sources = [
            MentorSource(
                id=s.source_id,
                name=s.source_name,
                type=s.source_type,
                score=s.score,
                summary=s.content[:200],
            )
            for s in context.sources
        ]
        yield {
            "type": "source",
            "data": {"sources": [s.model_dump() for s in sources]},
        }

        # 3. Build prompt and stream
        prompt = PromptRegistry.get(
            "mentor/v1-mentor",
            context=context.context_str,
            query=query,
        )

        full_answer = ""
        async for chunk in self._llm.generate_stream(prompt):
            full_answer += chunk
            yield {"type": "token", "data": {"content": chunk}}

        # 4. Complete
        yield {
            "type": "complete",
            "data": {
                "sources": [s.model_dump() for s in sources],
                "confidence": min(0.9, 0.5 + 0.1 * len(sources)),
            },
        }
