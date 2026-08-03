"""Shared type definitions for the Agent Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentInput:
    """Standard input envelope for all agents.

    Every agent receives this from the harness. Subclassing agents
    extract what they need from ``task_input`` and ``extra``.
    """

    task_input: str
    user_id: str = ""
    session_id: str = ""
    memory_context: dict | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionReport:
    """Standard execution report returned by ``BaseAgent.execute()``."""

    agent_name: str
    success: bool = False
    output: Any = None
    error: str | None = None
    error_type: str | None = None
    duration_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    retries: int = 0
    memory_context_loaded: bool = False
    output_schema: str | None = None


@dataclass
class AgentConfig:
    """Per-agent configuration overrides."""

    max_retries: int = 3
    before_retries: int = 2
    retry_base_delay: float = 1.0
    temperature: float = 0.3
    max_tokens: int = 2048
    enable_tools: bool = True
    enable_memory: bool = False
