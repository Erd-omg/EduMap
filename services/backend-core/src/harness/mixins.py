"""Agent mixins for tool injection and memory awareness.

Mixed into ``BaseAgent`` subclasses that opt in.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.memory.operations import MemoryOperations
    from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolInjectionMixin:
    """Mixin that adds tool call parsing and execution to an agent.

    The agent's ``run()`` must call ``self._handle_tool_calls()``
    after each LLM generation to process tool invocations.
    """

    def _handle_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse and execute tool calls embedded in LLM output.

        Looks for ``!tool:name(key=value, ...)`` patterns.
        Returns a list of tool-result dicts for enrichment.
        """
        registry: ToolRegistry | None = getattr(self, "_tool_registry", None)
        if not registry:
            return []

        pattern = r"!tool:(\w+)\(([^)]*)\)"
        results: list[dict[str, Any]] = []

        for match in re.finditer(pattern, text):
            tool_name = match.group(1)
            args_str = match.group(2)
            args = self._parse_tool_args(args_str)
            tool_result = registry.execute(tool_name, **(args or {}))
            results.append({
                "tool": tool_name,
                "args": args,
                "result": tool_result,
            })

        if results:
            logger.info(
                "Executed %d tool call(s): %s",
                len(results), [r["tool"] for r in results],
            )

        return results

    @staticmethod
    def _parse_tool_args(args_str: str) -> dict[str, Any]:
        """Parse ``key1=value1, key2=value2`` format into a dict."""
        if not args_str.strip():
            return {}
        args: dict[str, Any] = {}
        for part in args_str.split(","):
            part = part.strip()
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            args[key] = value
        return args


class MemoryAwareMixin:
    """Mixin that adds memory-aware before_run/after_run to an agent."""

    async def _load_memory_context(
        self,
        user_id: str,
        session_id: str | None = None,
        query: str = "",
    ) -> dict[str, Any]:
        """Load memory context from all three tiers."""
        memory_ops: MemoryOperations | None = getattr(self, "_memory_ops", None)
        if not memory_ops:
            return {}

        try:
            ctx = await memory_ops.build_context(
                user_id=user_id,
                session_id=session_id,
                query=query,
            )
            return ctx.model_dump() if hasattr(ctx, "model_dump") else {}
        except Exception as exc:
            logger.warning("Failed to load memory context: %s", exc)
            return {}

    async def _save_interaction(
        self,
        user_id: str,
        session_id: str | None,
        input_text: str,
        output_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction to both short-term and long-term memory."""
        memory_ops: MemoryOperations | None = getattr(self, "_memory_ops", None)
        if not memory_ops:
            return

        try:
            await memory_ops.record_interaction(
                user_id=user_id,
                session_id=session_id,
                event_type="mentor_query",
                input_text=input_text,
                output_text=output_text,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning("Failed to record interaction: %s", exc)
