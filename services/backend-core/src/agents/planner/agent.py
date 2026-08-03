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
        """
        llm = self._get_llm()
        prompt = PromptRegistry.get("planner/v1-extract", task_input=input.task_input)

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

    # ── Legacy compatibility ────────────────────────────────────────

    async def run_legacy(self, task_input: str, user_id: str = "") -> PlannerOutput:
        """Legacy signature wrapper. Use ``execute()`` instead."""
        import warnings
        warnings.warn("run_legacy() is deprecated, use execute()", DeprecationWarning, stacklevel=2)
        report = await self.execute(AgentInput(task_input=task_input, user_id=user_id))
        return report.output
