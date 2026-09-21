"""Structured output support: schema injection for LLM prompts and parsing.

Two modes:
1. **Function-calling mode** — inject schema as a tool definition.
2. **Prompt-injection mode (fallback)** — inject schema as JSON instructions
   into the prompt, then parse with Pydantic.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OutputSchema:
    """Wraps a Pydantic model as an agent output schema.

    Provides:
    - JSON Schema representation for function-calling mode
    - Prompt instructions for fallback injection
    - Parser that handles markdown fences + raw JSON
    """

    def __init__(self, model: type[BaseModel], schema_id: str = "") -> None:
        self.model = model
        self.schema_id = schema_id or model.__name__
        self._json_schema = model.model_json_schema()

    # ── Prompt injection helpers ──────────────────────────────────

    def to_function_tool_def(self) -> dict[str, Any]:
        """Return an OpenAI-compatible tool definition dict."""
        return {
            "type": "function",
            "function": {
                "name": self.schema_id,
                "description": f"Structured output matching {self.schema_id}",
                "parameters": self._json_schema,
            },
        }

    def to_prompt_instructions(self) -> str:
        """Return human-readable JSON-schema instructions for prompt injection."""
        schema_str = json.dumps(self._json_schema, ensure_ascii=False, indent=2)
        return (
            f"你必须以 JSON 格式输出，严格匹配以下 Schema：\n"
            f"```json\n{schema_str}\n```\n"
            f"只输出 JSON，不要其他内容。"
        )

    def parse(self, text: str) -> BaseModel:
        """Parse JSON from LLM output and validate against the Pydantic model.

        Handles:
        - `` ```json ... ``` `` markdown fences
        - Raw ``{ ... }`` JSON
        - Partial/trailing text after JSON
        """
        raw = text.strip()

        # Strip markdown fences
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

        # Find first { or [
        brace_start = raw.find("{")
        if brace_start == -1:
            brace_start = raw.find("[")
        if brace_start != -1:
            raw = raw[brace_start:]

        # Parse + validate
        try:
            data = json.loads(raw)
            return self.model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            # A response cut off at the token cap yields JSON that ends
            # mid-object ("Expecting property name enclosed in double
            # quotes").  This is common when the model runs long, so try to
            # salvage the complete leading elements before giving up.
            repaired = self._repair_truncated(raw)
            if repaired is not None:
                try:
                    logger.warning(
                        "OutputSchema.parse recovered a truncated %s response",
                        self.schema_id,
                    )
                    return self.model.model_validate(repaired)
                except ValidationError:
                    pass

            logger.warning(
                "OutputSchema.parse failed for %s: %s\nRaw: %.200s",
                self.schema_id, exc, text,
            )
            raise

    @staticmethod
    def _repair_truncated(raw: str) -> dict | None:
        """Salvage a JSON object that was truncated mid-way.

        Strategy: scan once, tracking string state and bracket depth, and
        remember the position after each *completed* value at every depth.
        Then cut at the deepest point that still yields a balanced structure
        and close the remaining containers.

        Returns the parsed dict, or None when the text is too broken to fix.
        """
        stack: list[str] = []          # open brackets, in order
        in_string = False
        escaped = False
        # All positions (exclusive) immediately after a completed value, in order.
        boundaries: list[int] = []

        for i, ch in enumerate(raw):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                    boundaries.append(i + 1)
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
                boundaries.append(i + 1)

        if not stack:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None

        # Try the most recent completed value first; walk backwards until the
        # remaining text parses after closing the still-open containers.
        for end in sorted(boundaries, reverse=True):
            candidate = raw[:end].rstrip().rstrip(",")
            # Recompute what is still open at this cut point.
            open_stack: list[str] = []
            in_str = False
            esc = False
            for ch in candidate:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    open_stack.append(ch)
                elif ch in "}]" and open_stack:
                    open_stack.pop()
            closers = "".join(
                "}" if b == "{" else "]" for b in reversed(open_stack)
            )
            try:
                return json.loads(candidate + closers)
            except json.JSONDecodeError:
                continue
        return None
