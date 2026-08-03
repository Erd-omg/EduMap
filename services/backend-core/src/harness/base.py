"""BaseAgent — unified base class for all EduMap agents.

Lifecycle (orchestrated by ``execute()``)::

    1. before_run(AgentInput) -> AgentInput
    2. run(AgentInput) -> AgentOutput (any type)
    3. after_run(AgentOutput) -> AgentOutput

Each step is wrapped with:
- Retry logic (via RetryHandler)
- Timing (via Timer)
- Logging (structured JSON reports)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.harness.errors import AgentError, ConfigurationError
from src.harness.observability import Timer, log_execution_report
from src.harness.retry import RetryHandler
from src.harness.types import AgentConfig, AgentInput, ExecutionReport

if TYPE_CHECKING:
    from src.memory.operations import MemoryOperations
    from src.tools.registry import ToolRegistry
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Unified base class for all EduMap agents.

    Subclasses must implement ``run()``. They may override
    ``before_run()`` and ``after_run()`` for lifecycle hooks.
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_ops: MemoryOperations | None = None,
        agent_name: str = "",
        config: AgentConfig | None = None,
    ) -> None:
        self._llm = llm_adapter
        self._tool_registry = tool_registry
        self._memory_ops = memory_ops
        self._agent_name = agent_name or self.__class__.__name__
        self._config = config or AgentConfig()
        self._logger = logging.getLogger(f"{__name__}.{self._agent_name}")

    # ── Lifecycle hooks (subclass may override) ──────────────────

    async def before_run(self, input: AgentInput) -> AgentInput:
        """Pre-execution hook.

        Default: loads memory context if ``config.enable_memory`` is True.
        """
        if self._config.enable_memory and self._memory_ops:
            try:
                from src.memory.models import EventType

                episodic = await self._memory_ops.long_term.recall_episodic(
                    user_id=input.user_id,
                    event_types=[EventType.MENTOR_QUERY],
                    limit=10,
                )
                if episodic:
                    input.memory_context = {
                        "conversation_history": [
                            {"input": e.input, "output": e.output}
                            for e in episodic
                        ]
                    }
                self._logger.debug("Memory context loaded for %s", self._agent_name)
            except Exception as exc:
                self._logger.warning(
                    "Memory context load failed for %s (non-fatal): %s",
                    self._agent_name, exc,
                )
        return input

    @abstractmethod
    async def run(self, input: AgentInput) -> Any:
        """Core agent logic.

        Subclasses implement this with their own return type.
        The return value is stored in ``ExecutionReport.output``.
        """
        ...

    async def after_run(self, output: Any) -> Any:
        """Post-execution hook. Default: no-op."""
        return output

    # ── Main entry point ─────────────────────────────────────────

    async def execute(self, input: AgentInput) -> ExecutionReport:
        """Standard execution pipeline.

        1. Run ``before_run()`` (with retry)
        2. Run ``run()`` (with retry for transient LLM errors)
        3. Run ``after_run()``
        4. Build and return ``ExecutionReport``
        """
        report = ExecutionReport(agent_name=self._agent_name)
        timer = Timer()
        timer.start()

        try:
            # Step 1: before_run (memory loading, enrichment)
            input = await RetryHandler.with_retry(
                self.before_run,
                input,
                max_retries=self._config.before_retries,
                base_delay=self._config.retry_base_delay,
                on_retry=lambda a, e: self._logger.warning(
                    "before_run retry %d for %s: %s", a + 1, self._agent_name, e,
                ),
            )
            report.memory_context_loaded = input.memory_context is not None

            # Step 2: run (core logic) with circuit breaker awareness
            retry_count = [0]

            # If the LLM adapter has a circuit breaker, bridge it into retry
            cb_check = None
            cb_record = None
            if hasattr(self._llm, '_is_circuit_open') and hasattr(self._llm, '_record_failure'):
                cb_check = lambda: self._llm._is_circuit_open()  # type: ignore[union-attr]
                cb_record = lambda: self._llm._record_failure()   # type: ignore[union-attr]

            output = await RetryHandler.with_circuit_breaker(
                self.run,
                input,
                circuit_check=cb_check,
                circuit_record_failure=cb_record,
                max_retries=self._config.max_retries,
                base_delay=self._config.retry_base_delay,
                on_retry=lambda a, e: (
                    retry_count.__setitem__(0, a + 1),
                    self._logger.warning(
                        "run retry %d for %s: %s", a + 1, self._agent_name, e,
                    ),
                ),
            )
            report.retries = retry_count[0]

            # Step 3: after_run (record interaction, audit)
            output = await self.after_run(output)
            report.success = True
            report.output = output

            # Transfer LLM token usage from adapter into report
            if hasattr(self._llm, '_last_usage') and self._llm._last_usage:
                report.token_usage = dict(self._llm._last_usage)

        except AgentError as exc:
            report.success = False
            report.error = str(exc)
            report.error_type = type(exc).__name__
            self._logger.error(
                "AgentError in %s: [%s] %s",
                self._agent_name, report.error_type, report.error,
            )
        except Exception as exc:
            report.success = False
            report.error = str(exc)
            report.error_type = type(exc).__name__
            self._logger.exception(
                "Unexpected error in %s.execute()", self._agent_name,
            )

        report.duration_ms = timer.stop()
        log_execution_report(report)
        return report

    # ── Convenience helpers for subclasses ────────────────────────

    def _get_llm(self) -> BaseLLMAdapter:
        """Get the LLM adapter or raise ConfigurationError."""
        if self._llm is None:
            raise ConfigurationError(
                f"{self._agent_name}: LLM adapter not configured",
                agent_name=self._agent_name,
            )
        return self._llm

    def _build_tool_prompt_block(self) -> str:
        """Build the tool system-prompt block if tools are enabled."""
        if not self._config.enable_tools or not self._tool_registry:
            return ""
        return self._tool_registry.build_prompt_block()

    @property
    def tool_registry(self) -> ToolRegistry | None:
        return self._tool_registry

    @property
    def agent_name(self) -> str:
        return self._agent_name
