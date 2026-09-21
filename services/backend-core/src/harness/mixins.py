"""Agent mixins for tool injection and memory awareness.

Mixed into ``BaseAgent`` subclasses that opt in.
"""

from __future__ import annotations

import logging
import re
import time
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

    async def _handle_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse and execute tool calls embedded in LLM output.

        Looks for ``!tool:name(key=value, ...)`` patterns.  Returns a list of
        **JSON-serialisable** tool-result dicts, ready to be stored on
        ``ExecutionReport.tool_calls`` and streamed to the client.

        Note the ``await`` on ``registry.execute`` — it is a coroutine.  Missing
        it silently returned an un-awaited coroutine, so no tool ever ran and the
        "result" was unserialisable (which is why callers used to drop it).
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

            started = time.perf_counter()
            tool_result = await registry.execute(tool_name, **(args or {}))
            duration_ms = round((time.perf_counter() - started) * 1000, 2)

            results.append({
                "tool": tool_name,
                "args": args,
                "success": bool(getattr(tool_result, "success", False)),
                "output": str(getattr(tool_result, "output", ""))[:500],
                "error": getattr(tool_result, "error", None),
                "duration_ms": duration_ms,
            })

        if results:
            logger.info(
                "Executed %d tool call(s): %s",
                len(results),
                [
                    {"tool": r["tool"], "success": r["success"],
                     "duration_ms": r["duration_ms"]}
                    for r in results
                ],
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
