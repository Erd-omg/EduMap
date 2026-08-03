"""Tests for BaseAgent — lifecycle and execution pipeline.

Uses MockLLMAdapter for LLM-dependent paths and mocks for memory/tools.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.harness.errors import AgentError, ConfigurationError
from src.harness.types import AgentConfig, AgentInput, ExecutionReport


class SimpleTestAgent:
    """Minimal agent for testing BaseAgent lifecycle.

    We test BaseAgent's lifecycle by creating a minimal concrete subclass
    rather than importing a real agent (which might have complex dependencies).
    """

    # We'll test directly by instantiating BaseAgent with a minimal subclass.
    pass


@pytest.fixture
def mock_llm():
    from tests.mocks.llm_adapter import MockLLMAdapter

    return MockLLMAdapter(response="测试回复")


# ── Import and use BaseAgent via a concrete subclass ─────────────────


def _make_agent_cls():
    """Dynamically create a minimal BaseAgent subclass for testing."""
    from src.harness.base import BaseAgent as _BaseAgent

    class TestAgent(_BaseAgent):
        async def run(self, input: AgentInput) -> str:
            return f"processed: {input.task_input}"

    return TestAgent


class TestLifecycle:
    """BaseAgent lifecycle: before_run → run → after_run."""

    @pytest.mark.asyncio
    async def test_execute_calls_lifecycle_in_order(self) -> None:
        """Execute calls before_run → run → after_run in sequence."""
        TestAgent = _make_agent_cls()

        call_order: list[str] = []

        class TrackingAgent(TestAgent):
            async def before_run(self, input: AgentInput) -> AgentInput:
                call_order.append("before_run")
                return input

            async def run(self, input: AgentInput) -> str:
                call_order.append("run")
                return "result"

            async def after_run(self, output: str) -> str:
                call_order.append("after_run")
                return output

        agent = TrackingAgent()
        report = await agent.execute(AgentInput(task_input="test"))
        assert call_order == ["before_run", "run", "after_run"]
        assert report.success is True
        assert report.output == "result"

    @pytest.mark.asyncio
    async def test_execute_returns_execution_report(self) -> None:
        """Execute returns an ExecutionReport with metadata."""
        TestAgent = _make_agent_cls()

        agent = TestAgent(agent_name="test_agent")
        report = await agent.execute(AgentInput(task_input="hello"))

        assert isinstance(report, ExecutionReport)
        assert report.agent_name == "test_agent"
        assert report.success is True
        assert report.duration_ms > 0
        assert report.retries == 0

    @pytest.mark.asyncio
    async def test_run_receives_task_input(self) -> None:
        """The run method receives the full AgentInput."""
        TestAgent = _make_agent_cls()

        class InspectAgent(TestAgent):
            async def run(self, input: AgentInput) -> dict:
                return {"received": input.task_input, "user": input.user_id}

        agent = InspectAgent()
        report = await agent.execute(
            AgentInput(task_input="do_something", user_id="user123")
        )
        assert report.success is True
        assert report.output["received"] == "do_something"
        assert report.output["user"] == "user123"


class TestErrorHandling:
    """Error propagation in BaseAgent lifecycle."""

    @pytest.mark.asyncio
    async def test_run_error_reported_in_report(self) -> None:
        """Exception in run() is captured in ExecutionReport."""
        TestAgent = _make_agent_cls()

        class FailingAgent(TestAgent):
            async def run(self, input: AgentInput) -> str:
                raise ValueError("something went wrong")

        agent = FailingAgent()
        report = await agent.execute(AgentInput(task_input="test"))

        assert report.success is False
        assert report.error is not None
        assert "something went wrong" in report.error
        assert report.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_before_run_error_non_fatal(self) -> None:
        """Error in before_run propagates (non-retryable) and sets success=False.

        RuntimeError is not a retryable exception in RetryHandler, so it
        propagates to execute()'s outer exception handler.
        """
        TestAgent = _make_agent_cls()

        class BrokenBeforeAgent(TestAgent):
            async def before_run(self, input: AgentInput) -> AgentInput:
                raise RuntimeError("memory unavailable")

        agent = BrokenBeforeAgent()
        report = await agent.execute(AgentInput(task_input="test"))

        # RuntimeError is not retryable, so it's caught at the top level
        assert report.success is False
        assert "memory unavailable" in report.error

    @pytest.mark.asyncio
    async def test_agent_error_propagated(self) -> None:
        """AgentError subclasses are captured with type name."""
        TestAgent = _make_agent_cls()

        class FailingAgent(TestAgent):
            async def run(self, input: AgentInput) -> str:
                raise ConfigurationError("bad config", agent_name="test")

        agent = FailingAgent()
        report = await agent.execute(AgentInput(task_input="test"))

        assert report.success is False
        assert report.error_type == "ConfigurationError"
        assert "bad config" in report.error


class TestLLMIntegration:
    """LLM adapter integration via _get_llm()."""

    @pytest.mark.asyncio
    async def test_get_llm_returns_adapter(self) -> None:
        """_get_llm() returns the configured LLM adapter."""
        from tests.mocks.llm_adapter import MockLLMAdapter

        TestAgent = _make_agent_cls()

        agent = TestAgent(llm_adapter=MockLLMAdapter(response="测试回复"))
        llm = agent._get_llm()
        assert llm is not None
        result = await llm.generate("test")
        assert result.content == "测试回复"

    @pytest.mark.asyncio
    async def test_get_llm_raises_when_not_configured(self) -> None:
        """_get_llm() raises ConfigurationError when no adapter set."""
        TestAgent = _make_agent_cls()

        agent = TestAgent()
        with pytest.raises(ConfigurationError, match="LLM adapter not configured"):
            agent._get_llm()


class TestMemoryLifecycle:
    """Memory context loading in before_run."""

    @pytest.mark.asyncio
    async def test_memory_context_loaded_when_enabled(self) -> None:
        """When memory is enabled, before_run loads memory context."""
        from src.memory.models import EventType

        TestAgent = _make_agent_cls()

        mock_memory = AsyncMock()
        mock_episodic = [MagicMock(input="q1", output="a1")]
        mock_memory.long_term.recall_episodic = AsyncMock(return_value=mock_episodic)

        agent = TestAgent(
            memory_ops=mock_memory,
            config=AgentConfig(enable_memory=True),
        )
        report = await agent.execute(
            AgentInput(task_input="test", user_id="user1")
        )

        assert report.success is True
        assert report.memory_context_loaded is True
        mock_memory.long_term.recall_episodic.assert_awaited_once_with(
            user_id="user1",
            event_types=[EventType.MENTOR_QUERY],
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_memory_not_loaded_when_disabled(self) -> None:
        """When memory is disabled, before_run skips memory loading."""
        TestAgent = _make_agent_cls()

        mock_memory = AsyncMock()

        agent = TestAgent(
            memory_ops=mock_memory,
            config=AgentConfig(enable_memory=False),
        )
        await agent.execute(AgentInput(task_input="test"))

        mock_memory.long_term.recall_episodic.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_memory_graceful_failure(self) -> None:
        """Memory load failure is non-fatal, execution continues."""
        TestAgent = _make_agent_cls()

        mock_memory = AsyncMock()
        mock_memory.long_term.recall_episodic = AsyncMock(
            side_effect=Exception("Redis unavailable")
        )

        agent = TestAgent(
            memory_ops=mock_memory,
            config=AgentConfig(enable_memory=True),
        )
        report = await agent.execute(AgentInput(task_input="test"))

        assert report.success is True
        assert report.memory_context_loaded is False


class TestToolPromptBlock:
    """Tool prompt block construction."""

    def test_tools_disabled_returns_empty(self) -> None:
        """When tools are disabled, build_prompt_block returns ''."""
        TestAgent = _make_agent_cls()

        agent = TestAgent(
            config=AgentConfig(enable_tools=False),
        )
        block = agent._build_tool_prompt_block()
        assert block == ""

    def test_no_registry_returns_empty(self) -> None:
        """When no tool_registry set, returns '' even if tools enabled."""
        TestAgent = _make_agent_cls()

        agent = TestAgent(
            config=AgentConfig(enable_tools=True),
        )
        block = agent._build_tool_prompt_block()
        assert block == ""

    def test_tools_enabled_returns_block(self) -> None:
        """When tools enabled and registered, returns prompt block."""
        TestAgent = _make_agent_cls()

        mock_registry = MagicMock()
        mock_registry.build_prompt_block = MagicMock(return_value="!tool:search(query=...)")

        agent = TestAgent(
            tool_registry=mock_registry,
            config=AgentConfig(enable_tools=True),
        )
        block = agent._build_tool_prompt_block()
        assert block == "!tool:search(query=...)"


class TestAgentName:
    """Agent name / identity."""

    @pytest.mark.asyncio
    async def test_custom_agent_name(self) -> None:
        """Custom agent name is used in report."""
        TestAgent = _make_agent_cls()

        agent = TestAgent(agent_name="my_agent")
        report = await agent.execute(AgentInput(task_input="test"))
        assert report.agent_name == "my_agent"

    @pytest.mark.asyncio
    async def test_default_agent_name_is_class_name(self) -> None:
        """Default agent name is the class name."""
        TestAgent = _make_agent_cls()

        agent = TestAgent()
        assert agent.agent_name == "TestAgent"


class TestAgentConfig:
    """AgentConfig defaults and overrides."""

    def test_default_config(self) -> None:
        """Default AgentConfig has expected values."""
        config = AgentConfig()
        assert config.max_retries == 3
        assert config.before_retries == 2
        assert config.retry_base_delay == 1.0
        assert config.temperature == 0.3
        assert config.max_tokens == 2048
        assert config.enable_tools is True
        assert config.enable_memory is False

    def test_custom_config(self) -> None:
        """Custom config overrides defaults."""
        config = AgentConfig(
            max_retries=5,
            enable_memory=True,
            temperature=0.7,
        )
        assert config.max_retries == 5
        assert config.enable_memory is True
        assert config.temperature == 0.7
