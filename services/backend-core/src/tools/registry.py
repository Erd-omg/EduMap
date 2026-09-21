"""ToolRegistry — register, discover, and execute agent tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TOOL_TIMEOUT_SECONDS = 5.0


class ToolRegistry:
    """Registry for agent-callable tools.

    Tools are discovered by name and executed by the registry.
    The registry also provides a formatted prompt block for LLM injection.

    Execution is guarded by a timeout so a slow or hung tool cannot stall the
    whole agent pipeline; a timeout is reported as a failed ``ToolResult``
    rather than propagating, matching how other tool errors are surfaced.
    """

    def __init__(self, timeout_seconds: float = DEFAULT_TOOL_TIMEOUT_SECONDS) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._timeout = timeout_seconds

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its spec name."""
        name = tool.spec.name
        if name in self._tools:
            logger.warning("Overwriting existing tool: %s", name)
        self._tools[name] = tool
        logger.debug("Registered tool: %s", name)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name, bounded by the registry timeout."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        try:
            logger.debug("Executing tool: %s with args=%s", name, kwargs)
            return await asyncio.wait_for(
                tool.execute(**kwargs), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Tool %s timed out after %.1fs", name, self._timeout)
            return ToolResult(
                success=False, error=f"Tool '{name}' timed out after {self._timeout}s"
            )
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return ToolResult(success=False, error=str(exc))

    def build_prompt_block(self) -> str:
        """Build the full tool system prompt block for LLM injection."""
        if not self._tools:
            return ""
        lines = ["## 可用工具", "你可以调用以下工具来帮助回答问题：", ""]
        for tool in self._tools.values():
            lines.append(tool.to_prompt_block())
            lines.append("")
        lines.append(
            "注意：每次只调用一个工具。等待工具返回结果后再继续。\n"
            "如果你认为不需要工具，直接回答即可。"
        )
        return "\n".join(lines)
