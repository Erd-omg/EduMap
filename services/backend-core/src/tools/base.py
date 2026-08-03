"""Tool system — abstract base tool definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """Describes a single parameter for a tool."""
    name: str
    type: str  # "string" | "number" | "boolean" | "object" | "array"
    description: str = ""
    required: bool = False


@dataclass
class ToolSpec:
    """Specification of a tool — name, description, and parameters."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


@dataclass
class ToolResult:
    """Result of executing a tool."""
    success: bool
    output: str = ""
    data: Any = None
    error: str | None = None


class BaseTool(ABC):
    """Abstract base for all agent-callable tools."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Return the tool's specification for LLM prompt injection."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments."""
        ...

    def to_prompt_block(self) -> str:
        """Format tool as an LLM prompt block for function-calling injection."""
        lines = [f"### {self.spec.name}", self.spec.description, "", "参数："]
        for p in self.spec.parameters:
            req = "（必填）" if p.required else "（可选）"
            lines.append(f"  - `{p.name}` ({p.type}) {req}: {p.description}")
        lines.append("")
        lines.append("调用格式：`!tool:{name}(arg1=value1, arg2=value2)`")
        return "\n".join(lines)
