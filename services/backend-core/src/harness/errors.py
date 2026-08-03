"""Agent error hierarchy."""

from typing import Any


class AgentError(Exception):
    """Base error for all agent harness errors."""

    def __init__(
        self,
        message: str,
        *,
        agent_name: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.details = details or {}
        super().__init__(message)


class LLMGenerationError(AgentError):
    """LLM call failed after retries or returned unusable output."""
    pass


class StructuredOutputError(AgentError):
    """LLM output could not be parsed into the expected schema."""

    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        **kwargs: Any,
    ) -> None:
        self.raw_output = raw_output
        super().__init__(message, **kwargs)


class ToolExecutionError(AgentError):
    """A tool call failed during execution."""
    pass


class MemoryError(AgentError):
    """Memory operation failed (non-fatal for most agents)."""
    pass


class ConfigurationError(AgentError):
    """Agent or harness misconfiguration."""
    pass


class CircuitBreakerOpenError(AgentError):
    """Circuit breaker is open — API calls suspended until recovery window passes."""
    pass
