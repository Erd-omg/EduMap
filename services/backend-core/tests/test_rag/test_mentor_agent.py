"""Tests for MentorAgent — RAG-constrained Q&A with mock LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.mentor.agent import MentorAgent
from src.harness.types import AgentConfig, AgentInput
from src.rag.models import RAGContext, RAGResult, MentorResponse
from src.rag.rag_service import RAGRetrievalService


_DEFAULT_RESULTS = [
    RAGResult(
        content="数组插入复杂度为O(n)。",
        source_type="neo4j",
        source_id="kp-array",
        source_name="数组",
        score=0.85,
    ),
    RAGResult(
        content="链表插入复杂度为O(1)。",
        source_type="neo4j",
        source_id="kp-linkedlist",
        source_name="链表",
        score=0.75,
    ),
]


@pytest.fixture
def mock_rag() -> MagicMock:
    """Create a mock RAGRetrievalService that returns predictable results."""
    rag = MagicMock(spec=RAGRetrievalService)

    rag.search = AsyncMock(return_value=_DEFAULT_RESULTS)

    async def assemble_side_effect(
        results: list[RAGResult], max_chars: int = 2000
    ) -> RAGContext:
        parts = []
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r.source_name} (来源: {r.source_type})"
            parts.append(f"{header}\n{r.content[:500]}")
        return RAGContext(
            context_str="\n---\n".join(parts),
            sources=results,
        )

    rag.assemble_context = AsyncMock(side_effect=assemble_side_effect)
    return rag


@pytest.fixture
def mock_rag_no_results() -> MagicMock:
    """Mock RAG service that returns no results."""
    rag = MagicMock(spec=RAGRetrievalService)
    rag.search = AsyncMock(return_value=[])

    async def assemble_side_effect(
        results: list[RAGResult], max_chars: int = 2000
    ) -> RAGContext:
        return RAGContext(context_str="（未找到相关参考资料）", sources=[])

    rag.assemble_context = AsyncMock(side_effect=assemble_side_effect)
    return rag


@pytest.fixture
def mentor_agent(mock_rag: MagicMock, mock_llm) -> MentorAgent:
    """Create a MentorAgent with mock LLM and mock RAG."""
    agent = MentorAgent(
        llm_adapter=mock_llm,
        rag_service=mock_rag,
    )
    return agent


@pytest.fixture
def mentor_agent_no_sources(mock_rag_no_results: MagicMock, mock_llm) -> MentorAgent:
    """MentorAgent that finds no sources."""
    agent = MentorAgent(
        llm_adapter=mock_llm,
        rag_service=mock_rag_no_results,
    )
    return agent


class TestMentorAgent:
    """Core MentorAgent behavior."""

    async def test_answer_returns_valid_response(self, mentor_agent: MentorAgent) -> None:
        """Basic Q&A returns a MentorResponse with answer and sources."""
        response = await mentor_agent._answer_impl("数组和链表有什么区别？")

        assert isinstance(response, MentorResponse)
        assert response.answer != ""
        assert len(response.sources) == 2

    async def test_answer_calls_rag_search(self, mentor_agent: MentorAgent, mock_rag) -> None:
        """MentorAgent calls RAG search with the user query."""
        await mentor_agent._answer_impl("什么是数组？")
        mock_rag.search.assert_awaited_once()
        call_args = mock_rag.search.call_args
        assert call_args is not None
        assert "数组" in call_args[0][0]

    async def test_confidence_with_sources(self, mentor_agent: MentorAgent) -> None:
        """Confidence is > 0.5 when sources are found."""
        response = await mentor_agent._answer_impl("数组和链表")
        assert response.confidence > 0.5

    async def test_confidence_without_sources(
        self, mentor_agent_no_sources: MentorAgent
    ) -> None:
        """Confidence is lower when no sources are found."""
        response = await mentor_agent_no_sources._answer_impl("未知话题")
        assert response.confidence < 0.5

    async def test_sources_contain_metadata(self, mentor_agent: MentorAgent) -> None:
        """Sources include id, name, type, score."""
        response = await mentor_agent._answer_impl("数组")
        assert len(response.sources) > 0
        source = response.sources[0]
        assert source.id
        assert source.name
        assert source.type in ("neo4j", "chroma")
        assert 0.0 <= source.score <= 1.0

    async def test_empty_query_returns_response(self, mentor_agent: MentorAgent) -> None:
        """An empty query still returns a response (not a crash)."""
        response = await mentor_agent._answer_impl("")
        assert isinstance(response, MentorResponse)


class TestMentorStreaming:
    """MentorAgent streaming behavior."""

    async def test_answer_stream_yields_events(self, mentor_agent: MentorAgent) -> None:
        """Streaming yields source, token, and complete events."""
        events = []
        async for event in mentor_agent.answer_stream("数组"):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "source" in event_types
        assert "token" in event_types
        assert "complete" in event_types

    async def test_stream_no_sources_appends_disclaimer(
        self, mentor_agent_no_sources: MentorAgent
    ) -> None:
        """When no sources found, streaming appends a disclaimer."""
        events = []
        async for event in mentor_agent_no_sources.answer_stream("未知话题"):
            events.append(event)

        complete_event = [e for e in events if e["type"] == "complete"][0]
        assert complete_event["data"].get("no_sources") is True


class TestMentorHarness:
    """MentorAgent through harness execute()."""

    async def test_harness_execute_returns_answer(
        self, mentor_agent: MentorAgent
    ) -> None:
        """Harness execute() route returns a MentorResponse."""
        report = await mentor_agent.execute(
            AgentInput(
                task_input="数组和链表的区别",
                user_id="test-user",
            )
        )
        response = report.output
        assert isinstance(response, MentorResponse)
        assert response.answer != ""
