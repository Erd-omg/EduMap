"""Planner agent — knowledge extraction from user input.

Uses LLM to parse a free-text learning request into a structured
:class:`GenerationPlan` with :class:`KnowledgeUnit` nodes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from src.agents.models import GenerationPlan, KnowledgeUnit, PlannerOutput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Extracts a structured knowledge plan from a user's learning request."""

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        kp_repo: KnowledgePointRepository,
    ) -> None:
        self._llm = llm_adapter
        self._kp_repo = kp_repo

    async def run(self, task_input: str, user_id: str = "") -> PlannerOutput:
        """Parse user input → KnowledgeUnit[] + GenerationPlan."""
        prompt = PromptRegistry.get("planner/v1-extract", task_input=task_input)
        response = await self._llm.generate(prompt)

        raw = self._extract_json(response.content)
        data = self._parse_llm_output(raw)

        # Enrich with existing KG data
        enriched_units: list[KnowledgeUnit] = []
        for ku_data in data.get("knowledge_units", []):
            ku = KnowledgeUnit(**ku_data)
            existing = await self._kp_repo.get(ku.id)
            if existing:
                logger.info("Planner found existing KP '%s' — merging metadata", ku.id)
                # Merge existing description if ours is too short
                if len(ku.description) < 10 and existing.description:
                    ku.description = existing.description
            enriched_units.append(ku)

        plan = GenerationPlan(
            primary_kp_id=data.get("primary_kp_id", enriched_units[0].id if enriched_units else ""),
            content_types=data.get("content_types", ["explanation", "exercise"]),
            knowledge_units=enriched_units,
        )
        summary = data.get("summary", task_input[:120])

        return PlannerOutput(plan=plan, summary=summary)

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from markdown code blocks if present."""
        # Try ```json ... ``` block first
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: find outermost { ... }
        brace_start = text.find("{")
        if brace_start != -1:
            return text[brace_start:]
        return text

    @staticmethod
    def _parse_llm_output(raw: str) -> dict:
        """Parse the LLM JSON response with graceful fallback."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Planner LLM output not valid JSON: %s", exc)
            # Minimal fallback
            return {
                "knowledge_units": [],
                "primary_kp_id": "",
                "content_types": ["explanation"],
                "summary": raw[:200],
            }
