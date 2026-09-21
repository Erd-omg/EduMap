"""Coder agent — code generation, AST analysis, and sandbox execution.

Generates code examples for a knowledge point, validates syntax via AST,
executes in the sandbox service, and iteratively fixes on failure.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from src.agents.models import CoderOutput, KnowledgeUnit
from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_MAX_FIX_ITERATIONS = 2
_SANDBOX_TIMEOUT = 15


class CoderAgent(BaseAgent):
    """Generates, validates, and fixes code examples for a KP."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        sandbox_url: str = "http://sandbox-service:8002",
        **kwargs,
    ) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="coder",
            config=AgentConfig(max_retries=3, temperature=0.3),
            **kwargs,
        )
        self._sandbox_url = sandbox_url.rstrip("/")

    async def run(self, input: AgentInput) -> CoderOutput:
        """Harness-compatible run — wraps legacy logic."""
        knowledge_unit = KnowledgeUnit(**input.extra.get("knowledge_unit", {}))
        language = input.extra.get("language", "python")
        return await self._run_legacy(knowledge_unit, language)

    async def _run_legacy(
        self,
        knowledge_unit: KnowledgeUnit,
        language: str = "python",
    ) -> CoderOutput:
        """Generate code, validate AST, execute in sandbox, fix if needed."""
        output = CoderOutput(language=language)

        # 1. Initial generation
        prompt = PromptRegistry.get(
            "coder/v1-generate",
            kp_name=knowledge_unit.name,
            concepts=", ".join(knowledge_unit.key_concepts),
            language=language,
        )
        response = await self._llm.generate(prompt)
        code = self._extract_code(response.content)
        output.code = code

        # 2. AST validation + sandbox execution with fix loop
        for attempt in range(_MAX_FIX_ITERATIONS + 1):
            output.code = code
            output.fix_iterations = attempt

            # AST validation (Python only)
            if language in ("python", "py", "python3"):
                try:
                    ast.parse(code)
                    output.ast_valid = True
                except SyntaxError as exc:
                    output.ast_valid = False
                    if attempt < _MAX_FIX_ITERATIONS:
                        code = await self._fix_code(code, str(exc), knowledge_unit)
                        continue
                    output.ast_valid = False
                    output.execution_success = False
                    output.execution_result = f"AST validation failed: {exc}"
                    return output

            # Sandbox execution
            try:
                result = await self._execute_in_sandbox(code, language)
                output.execution_success = result.get("status") == "success"
                output.execution_result = (
                    result.get("stdout", "") or result.get("stderr", "") or str(result)
                )
                if output.execution_success:
                    break  # All good
                if attempt < _MAX_FIX_ITERATIONS:
                    # Try to fix based on sandbox error
                    error_msg = result.get("stderr", "") or result.get("stdout", "")
                    code = await self._fix_code(code, error_msg, knowledge_unit)
                else:
                    output.execution_result = (
                        result.get("stderr", "") or result.get("stdout", "")
                    )
            except Exception as exc:
                logger.warning("Coder: sandbox execution failed (attempt %d): %s", attempt, exc)
                output.execution_result = str(exc)
                if attempt < _MAX_FIX_ITERATIONS:
                    code = await self._fix_code(code, str(exc), knowledge_unit)
                else:
                    break

        return output

    # ── Internal ────────────────────────────────────────────────────────

    async def _fix_code(
        self,
        code: str,
        error: str,
        knowledge_unit: KnowledgeUnit,
    ) -> str:
        """Request LLM to fix code based on error."""
        try:
            prompt = PromptRegistry.get(
                "coder/v1-fix",
                code=code,
                error=error,
                kp_name=knowledge_unit.name,
            )
            response = await self._llm.generate(prompt)
            return self._extract_code(response.content)
        except Exception as exc:
            logger.warning("Code fix failed: %s", exc)
            return code

    async def _execute_in_sandbox(
        self,
        code: str,
        language: str,
    ) -> dict[str, Any]:
        """POST code to the sandbox service for execution."""
        async with httpx.AsyncClient(timeout=_SANDBOX_TIMEOUT + 5) as client:
            resp = await client.post(
                f"{self._sandbox_url}/execute",
                json={
                    "code": code,
                    "language": language,
                    "timeout_seconds": _SANDBOX_TIMEOUT,
                    "memory_limit_mb": 128,
                },
            )
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract code from markdown code blocks, or return text as-is."""
        # Try ```python ... ``` or ``` ... ``` blocks
        match = re.search(r"```(?:\w+)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
