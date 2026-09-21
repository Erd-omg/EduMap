"""Planner agent — knowledge extraction from user input.

Uses LLM to parse a free-text learning request into a structured
:class:`GenerationPlan` with :class:`KnowledgeUnit` nodes.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.agents.models import GenerationPlan, KnowledgeUnit, PlannerOutput
from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

# Cap on how much tool output is folded back into the planner prompt.  The
# structured call is capped at 2048 output tokens; injecting a large tool
# result makes the JSON response liable to be truncated mid-object.
_MAX_TOOL_CONTEXT = 600


class _PlannerLLMOutput(BaseModel):
    """Schema matching what the LLM returns for planner/v1-extract."""
    knowledge_units: list[dict] = Field(default_factory=list)
    primary_kp_id: str = ""
    content_types: list[str] = Field(default_factory=lambda: ["explanation", "exercise"])
    summary: str = ""


class PlannerAgent(BaseAgent):
    """Extracts a structured knowledge plan from a user's learning request."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        kp_repo: KnowledgePointRepository,
        **kwargs,
    ) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="planner",
            config=AgentConfig(max_retries=2, temperature=0.3),
            **kwargs,
        )
        self._kp_repo = kp_repo

    async def run(self, input: AgentInput) -> PlannerOutput:
        """Parse user input → KnowledgeUnit[] + GenerationPlan.

        Uses structured output (``generate_structured``) which tries
        function-calling first, then falls back to prompt-injected JSON.

        Before the structured extraction we offer the tool block to the LLM and
        execute any ``!tool:name(...)`` calls it emits, then feed the results
        back in.  Tools let the planner consult the knowledge graph for the
        task at hand instead of guessing knowledge-unit ids.
        """
        llm = self._get_llm()
        prompt = PromptRegistry.get("planner/v1-extract", task_input=input.task_input)

        # ── Tool round (best-effort) ────────────────────────────────────
        # Only runs when tools are enabled and a registry is wired in.  Any
        # failure degrades to "no tool context" rather than failing the plan.
        #
        # The injected context is deliberately small (see ``_MAX_TOOL_CONTEXT``):
        # the planner's structured call is capped at 2048 output tokens, and a
        # larger prompt makes the model's JSON response more likely to be
        # truncated mid-object, which fails the whole plan.
        self._last_tool_calls = []
        tool_block = self._build_tool_prompt_block()
        if tool_block:
            tool_context = await self._gather_tool_context(llm, prompt, tool_block)
            if tool_context:
                prompt = f"{prompt}\n\n### 工具查询结果\n{tool_context}"

        # Use structured output — no more regex parsing
        result = await llm.generate_structured(
            prompt,
            schema=_PlannerLLMOutput,
        )

        # Enrich with existing KG data
        enriched_units: list[KnowledgeUnit] = []
        for ku_data in result.knowledge_units:
            ku = KnowledgeUnit(**ku_data)
            existing = await self._kp_repo.get(ku.id)
            if existing:
                logger.info("Planner found existing KP '%s' — merging metadata", ku.id)
                if len(ku.description) < 10 and existing.description:
                    ku.description = existing.description
            enriched_units.append(ku)

        plan = GenerationPlan(
            primary_kp_id=result.primary_kp_id or (enriched_units[0].id if enriched_units else ""),
            content_types=result.content_types or ["explanation", "exercise"],
            knowledge_units=enriched_units,
        )
        summary = result.summary or input.task_input[:120]

        return PlannerOutput(plan=plan, summary=summary)

    async def _gather_tool_context(self, llm, prompt: str, tool_block: str) -> str:
        """Ask the LLM whether it wants a tool, run it, and summarise results.

        Returns a short text block to append to the planner prompt, or an empty
        string when the model asked for no tools (the common case).  Failures
        are swallowed: tool context is an enhancement, not a dependency.

        Note this uses plain ``generate`` (not ``generate_structured``) because
        the model must be free to emit ``!tool:`` syntax rather than JSON.
        """
        try:
            response = await llm.generate(
                f"{prompt}\n\n{tool_block}",
                system_prompt=(
                    "你可以先调用工具补充信息。若需要，只输出一行 "
                    "`!tool:工具名(参数=值)`；若不需要工具，直接输出 `无需工具`。"
                ),
            )
            calls = await self._handle_tool_calls(response.content)
            self._last_tool_calls = calls
            if not calls:
                return ""
            lines = [
                f"- {c['tool']}({c['args']}) → "
                + (c["output"] if c["success"] else f"失败: {c['error']}")
                for c in calls
            ]
            logger.info("Planner used %d tool call(s)", len(calls))
            context = "\n".join(lines)[:_MAX_TOOL_CONTEXT]
            return context
        except Exception as exc:
            logger.warning("Planner tool round failed (continuing without): %s", exc)
            return ""

    # ── Legacy compatibility ────────────────────────────────────────

    async def run_legacy(self, task_input: str, user_id: str = "") -> PlannerOutput:
        """Legacy signature wrapper. Use ``execute()`` instead."""
        import warnings
        warnings.warn("run_legacy() is deprecated, use execute()", DeprecationWarning, stacklevel=2)
        report = await self.execute(AgentInput(task_input=task_input, user_id=user_id))
        return report.output
