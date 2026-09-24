"""Mentor agent — RAG-constrained Q&A for learning assistance.

Uses :class:`RAGRetrievalService` for hybrid search and the LLM adapter
to generate context-grounded answers with source citations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.rag.models import MentorResponse, MentorSource
from src.rag.rag_service import RAGRetrievalService
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


def _format_conversation_history(history: list[dict] | None) -> str:
    """Format conversation history into a condensed string for the prompt."""
    if not history:
        return "（暂无对话历史）"

    parts: list[str] = []
    for entry in history[-5:]:
        user_msg = entry.get("input", entry.get("query", ""))
        assistant_msg = entry.get("output", entry.get("answer", ""))
        if user_msg:
            parts.append(f"用户: {user_msg[:500]}")
        if assistant_msg:
            parts.append(f"助手: {assistant_msg[:500]}")

    formatted = "\n".join(parts)
    return formatted if formatted else "（暂无对话历史）"


def _as_offset(value: object) -> int | None:
    """Coerce a metadata value to a character offset, or None if unusable.

    Accepts ints **and int-valued floats**: character counts travel through
    JSON and numpy, so ``1200`` can arrive as ``1200.0``. An earlier version
    used ``isinstance(value, int)``, which silently dropped those — leaving
    ``char_start=None`` alongside a valid ``char_end``. Half a span is worse
    than none: the frontend's "is the span known?" check then fails and no
    highlight is shown at all, even though the offset was perfectly available.

    Non-integral floats are rejected rather than truncated: a fractional offset
    would indicate the value is not a character index at all.
    """
    if isinstance(value, bool):
        # bool is an int subclass; True/False are not offsets.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf
            return None
        if not value.is_integer():
            return None
        return int(value)
    return None


def _source_span(metadata: dict | None) -> dict:
    """Extract ``char_start``/``char_end`` from chunk metadata, if present.

    Returns a dict suitable for ``MentorSource(**span)``. Missing keys are
    omitted rather than defaulted to 0 — ``None`` means "span unknown" and the
    frontend must not highlight from it.

    Both endpoints must be present for either to be passed: a one-sided span is
    unrenderable, and emitting it would make the caller's "span known?" test
    behave differently from what the values imply.
    """
    if not metadata:
        return {}
    start = _as_offset(metadata.get("char_start"))
    end = _as_offset(metadata.get("char_end"))
    if start is None or end is None:
        return {}
    if end <= start:
        # An inverted or empty span cannot be highlighted; treat as absent.
        return {}
    return {"char_start": start, "char_end": end}


class MentorAgent(BaseAgent):
    """RAG-constrained learning Q&A agent.

    ``ToolInjectionMixin`` / ``MemoryAwareMixin`` are inherited via ``BaseAgent``;
    listing them again here would be redundant (and creates an MRO conflict).
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        rag_service: RAGRetrievalService,
        **kwargs,
    ) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="mentor",
            # Tools disabled: RAG context is injected directly; the prompt-injected
            # !tool:() syntax would otherwise leak into the streamed answer unparsed.
            config=AgentConfig(max_retries=1, temperature=0.3, enable_memory=True, enable_tools=False),
            **kwargs,
        )
        self._rag = rag_service

    async def run(self, input: AgentInput) -> MentorResponse:
        """Harness-compatible run — wraps legacy answer logic."""
        query = input.task_input
        conversation_history = input.extra.get("conversation_history")
        return await self._answer_impl(query, conversation_history)

    async def _answer_impl(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
    ) -> MentorResponse:
        """Core answer implementation used by both harness and streaming."""
        results = await self._rag.search(query)
        context = await self._rag.assemble_context(results)

        conv_str = _format_conversation_history(conversation_history)
        prompt = PromptRegistry.get(
            "mentor/v1-mentor",
            context=context.context_str,
            query=query,
            conversation_history=conv_str,
        )

        # Inject tool prompt block if tools are enabled
        tool_block = self._build_tool_prompt_block()
        if tool_block:
            prompt += f"\n\n{tool_block}"

        response = await self._get_llm().generate(prompt)

        return MentorResponse(
            answer=response.content,
            sources=[
                MentorSource(
                    id=s.source_id,
                    name=s.source_name,
                    type=s.source_type,
                    score=s.score,
                    summary=s.content[:200],
                    resource_id=s.metadata.get("resource_id"),
                    **_source_span(s.metadata),
                )
                for s in context.sources
            ],
            confidence=round(min(
                0.95,
                0.3 + 0.15 * len(context.sources)
                + 0.15 * (
                    sum(s.score for s in context.sources) / max(len(context.sources), 1)
                    if context.sources else 0
                ),
            ), 4),
            usage=response.usage,
        )

    async def answer(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_history: list[dict] | None = None,
    ) -> MentorResponse:
        """Legacy non-streaming answer — delegates to harness."""
        report = await self.execute(AgentInput(
            task_input=query,
            user_id=user_id,
            extra={"conversation_history": conversation_history},
        ))
        return report.output

    async def answer_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming answer — yields source, token, and complete events.

        Streaming bypasses the harness ``execute()`` lifecycle since it
        is fundamentally different from request-response.  Memory operations
        are handled via mixin methods directly.
        """
        results = await self._rag.search(query)
        context = await self._rag.assemble_context(results)

        sources = [
            MentorSource(
                id=s.source_id,
                name=s.source_name,
                type=s.source_type,
                score=s.score,
                summary=s.content[:200],
                # Chroma chunk metadata carries resource_id — lets the
                # frontend open the original uploaded material.
                resource_id=s.metadata.get("resource_id"),
                # ...and the character span, so the frontend can highlight the
                # cited passage instead of only naming the chunk.
                **_source_span(s.metadata),
            )
            for s in context.sources
        ]
        yield {
            "type": "source",
            "data": {"sources": [s.model_dump() for s in sources]},
        }

        conv_str = _format_conversation_history(conversation_history)
        prompt = PromptRegistry.get(
            "mentor/v1-mentor",
            context=context.context_str,
            query=query,
            conversation_history=conv_str,
        )

        # Inject tool prompt block
        tool_block = self._build_tool_prompt_block()
        if tool_block:
            prompt += f"\n\n{tool_block}"

        full_answer = ""
        async for chunk in self._get_llm().generate_stream(prompt):
            full_answer += chunk
            yield {"type": "token", "data": {"content": chunk}}

        has_sources = len(sources) > 0
        if not has_sources:
            disclaimer = "\n\n---\n*（注：当前没有上传学习资料，以上回答基于 AI 知识生成，请注意甄别）*"
            full_answer += disclaimer
            yield {"type": "token", "data": {"content": disclaimer}}

        yield {
            "type": "complete",
            "data": {
                "sources": [s.model_dump() for s in sources],
                "confidence": round(min(
                    0.95,
                    0.3 + 0.15 * len(sources)
                    + 0.15 * (
                        sum(s.score for s in sources) / max(len(sources), 1)
                        if sources else 0
                    ),
                ), 4),
                "no_sources": not has_sources,
                "usage": self._llm._last_usage if hasattr(self._llm, '_last_usage') else {},
            },
        }

        # Record interaction to memory after streaming completes
        await self._save_interaction(
            user_id=user_id,
            session_id=None,
            input_text=query,
            output_text=full_answer,
        )
