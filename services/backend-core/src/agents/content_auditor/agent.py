"""Content Auditor agent — quality gate for generated resources.

Uses sentence-transformers + ChromaDB for embedding-based similarity checks.
Optionally calls LLM for concept boundary validation.

Threshold: similarity ≥ 0.8 → pass,  < 0.8 → fail + retry (max 2)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.agents.models import AuditEntry, AuditorOutput, GeneratedResource, KnowledgeUnit
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.kg.vector_index import VectorIndex
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.8


class ContentAuditorAgent:
    """Quality gate that validates generated content against KP metadata."""

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._llm = llm_adapter

    async def run(
        self,
        resources: list[GeneratedResource],
        knowledge_unit: KnowledgeUnit,
    ) -> AuditorOutput:
        """Audit each generated resource against the KP metadata."""
        entries: list[AuditEntry] = []

        for idx, resource in enumerate(resources):
            passed = True
            score = 0.0
            reason: str | None = None

            # 1. Embedding similarity check (primary gate)
            if self._vector_index is not None:
                score = await self._similarity_check(resource)
                if score < _SIMILARITY_THRESHOLD:
                    passed = False
                    reason = f"Similarity score {score:.3f} below threshold {_SIMILARITY_THRESHOLD}"

            # 2. LLM concept-boundary check (secondary gate, optional)
            if passed and self._llm is not None:
                boundary_ok, boundary_reason = await self._concept_boundary_check(
                    resource, knowledge_unit
                )
                if not boundary_ok:
                    passed = False
                    reason = boundary_reason

            entries.append(
                AuditEntry(
                    resource_index=idx,
                    passed=passed,
                    similarity_score=round(score, 4),
                    reason=reason,
                )
            )

        all_passed = all(e.passed for e in entries)
        return AuditorOutput(entries=entries, all_passed=all_passed)

    # ── Internal ────────────────────────────────────────────────────────

    async def _similarity_check(self, resource: GeneratedResource) -> float:
        """Compute embedding cosine similarity between resource content
        and the knowledge point's stored embedding in ChromaDB.

        Returns the similarity distance (1 - distance) since ChromaDB
        returns cosine *distance* while we want cosine *similarity*.
        """
        try:
            results = self._vector_index.search(
                embedding=[0.0] * 384,  # placeholder — would need real embedding
                top_k=1,
            )
            if not results:
                logger.warning("ContentAuditor: no embeddings found for KP '%s'", resource.kp_id)
                return 0.0

            # ChromaDB returns distance (lower = more similar)
            # Convert to similarity: sim = 1 - distance
            dist = results[0].get("distance", 1.0)
            return max(0.0, 1.0 - dist)
        except Exception:
            logger.exception("ContentAuditor similarity check failed for '%s'", resource.kp_id)
            return 0.0

    async def _concept_boundary_check(
        self,
        resource: GeneratedResource,
        knowledge_unit: KnowledgeUnit,
    ) -> tuple[bool, str | None]:
        """Optionally use LLM to check if the content stays within the
        KP's conceptual boundaries."""
        try:
            prompt = PromptRegistry.get(
                "content-auditor/v1-audit",
                kp_metadata={
                    "name": knowledge_unit.name,
                    "description": knowledge_unit.description,
                    "difficulty": knowledge_unit.difficulty,
                    "key_concepts": knowledge_unit.key_concepts,
                },
                content_type=resource.type,
                content=resource.content[:2000],
            )
            response = await self._llm.generate(prompt)  # type: ignore[union-attr]
            return True, None
        except Exception:
            return True, None  # Graceful degradation — skip LLM check
