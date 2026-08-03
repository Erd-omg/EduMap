"""Agent Harness — standardized agent execution layer.

Provides BaseAgent ABC, structured output, retry, observability,
tool injection, and memory awareness for all EduMap agents.
"""

from src.harness.base import BaseAgent
from src.harness.errors import (
    AgentError,
    ConfigurationError,
    LLMGenerationError,
    MemoryError,
    StructuredOutputError,
    ToolExecutionError,
)
from src.harness.observability import Timer, log_execution_report
from src.harness.retry import RetryHandler
from src.harness.structured import OutputSchema
from src.harness.types import AgentConfig, AgentInput, ExecutionReport

__all__ = [
    "BaseAgent",
    "AgentInput",
    "AgentConfig",
    "ExecutionReport",
    "OutputSchema",
    "RetryHandler",
    "Timer",
    "log_execution_report",
    "AgentError",
    "LLMGenerationError",
    "StructuredOutputError",
    "ToolExecutionError",
    "MemoryError",
    "ConfigurationError",
]
