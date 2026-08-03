"""Designer agent — multimodal content generation.

Generates explanations, exercises, and visualizations for a given
:class:`KnowledgeUnit` using LLM prompts.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from src.agents.models import DesignerOutput, GeneratedResource, KnowledgeUnit
from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_CONTENT_TYPE_PROMPTS: dict[str, str] = {
    "explanation": "designer/v1-explanation",
    "exercise": "designer/v1-exercise",
    "visualization": "designer/v1-visualization",
}


class DesignerAgent(BaseAgent):
    """Generates learning content for a given knowledge point."""

    def __init__(self, llm_adapter: BaseLLMAdapter, **kwargs) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="designer",
            config=AgentConfig(max_retries=2, temperature=0.5),
            **kwargs,
        )

    async def run(self, input: AgentInput) -> DesignerOutput:
        """Harness-compatible run — wraps legacy logic."""
        knowledge_unit = KnowledgeUnit(**input.extra.get("knowledge_unit", {}))
        content_types = input.extra.get("content_types")
        user_profile = input.extra.get("user_profile")
        return await self._run_legacy(knowledge_unit, content_types, user_profile)

    async def _run_legacy(
        self,
        knowledge_unit: KnowledgeUnit,
        content_types: list[str] | None = None,
        user_profile: dict | None = None,
    ) -> DesignerOutput:
        """Generate one or more content pieces for the given KP."""
        resources: list[GeneratedResource] = []
        types = content_types or ["explanation"]

        for ct in types:
            prompt_id = _CONTENT_TYPE_PROMPTS.get(ct)
            if not prompt_id:
                logger.warning("Designer: unknown content type '%s', skipping", ct)
                continue

            prompt = PromptRegistry.get(
                prompt_id,
                kp_name=knowledge_unit.name,
                kp_description=knowledge_unit.description,
                kp_concepts=", ".join(knowledge_unit.key_concepts),
                difficulty=knowledge_unit.difficulty,
                language="python",
                user_profile_json=json.dumps(user_profile or {}, ensure_ascii=False),
            )
            response = await self._llm.generate(prompt)

            content = self._clean_content(response.content, ct)
            title = self._make_title(knowledge_unit.name, ct)
            resources.append(
                GeneratedResource(
                    type=ct,  # type: ignore[arg-type]
                    title=title,
                    content=content,
                    kp_id=knowledge_unit.id,
                    difficulty=knowledge_unit.difficulty,
                    metadata={
                        "rationale": f"Generated for {knowledge_unit.name} (difficulty {knowledge_unit.difficulty})",
                        "target_gaps": knowledge_unit.key_concepts,
                        "confidence": 0.7,
                    },
                )
            )

        return DesignerOutput(resources=resources)

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _clean_content(content: str, content_type: str) -> str:
        """Strip enclosing JSON or markdown fences from raw LLM output."""
        # Remove JSON code fences
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)

        # For visualization, keep the mermaid block as-is
        if content_type == "visualization":
            if not content.startswith("```mermaid"):
                # If LLM didn't wrap in mermaid, wrap it
                content = f"```mermaid\n{content}\n```"
            return content

        # For exercise, try to extract structured JSON
        if content_type == "exercise":
            try:
                data = json.loads(content)
                if "exercises" in data:
                    return json.dumps(data, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass

        return content

    @staticmethod
    def _make_title(kp_name: str, content_type: str) -> str:
        titles = {
            "explanation": f"{kp_name} — 知识点讲解",
            "exercise": f"{kp_name} — 练习题",
            "visualization": f"{kp_name} — 思维导图",
            "code": f"{kp_name} — 代码示例",
        }
        return titles.get(content_type, f"{kp_name} — {content_type}")
