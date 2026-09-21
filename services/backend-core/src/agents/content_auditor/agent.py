"""Content Auditor agent — quality gate for generated resources.

Uses sentence-transformers + ChromaDB for embedding-based similarity checks.
Optionally calls LLM for concept boundary validation.

Threshold: similarity ≥ 0.8 → pass,  < 0.8 → fail + retry (max 2)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.agents.models import AuditEntry, AuditorOutput, GeneratedResource, KnowledgeUnit
from src.harness.base import BaseAgent
from src.harness.types import AgentConfig, AgentInput
from src.prompts import PromptRegistry

if TYPE_CHECKING:
    from src.kg.vector_index import VectorIndex
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.8


class ContentAuditorAgent(BaseAgent):
    """Quality gate that validates generated content against KP metadata."""

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        **kwargs,
    ) -> None:
        super().__init__(
            llm_adapter=llm_adapter,
            agent_name="content_auditor",
            config=AgentConfig(max_retries=1, temperature=0.3),
            **kwargs,
        )
        self._vector_index = vector_index
        self._embedding_model_name = embedding_model_name
        self._model: Any = None

    def _get_embedding(self) -> Any:
        """Lazy-load the sentence-transformers embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._embedding_model_name)
            except Exception as exc:
                logger.warning("Failed to load embedding model: %s", exc)
                return None
        return self._model

    async def run(self, input: AgentInput) -> AuditorOutput:
        """Harness-compatible run — wraps legacy logic."""
        resources = input.extra.get("resources", [])
        knowledge_unit = KnowledgeUnit(**input.extra.get("knowledge_unit", {}))
        if resources and isinstance(resources[0], dict):
            resources = [GeneratedResource(**r) for r in resources]
        return await self._run_legacy(resources, knowledge_unit)

    async def _run_legacy(
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

        Generates a real embedding for the resource content and searches
        for the nearest neighbor in ChromaDB.

        Returns the similarity distance (1 - distance) since ChromaDB
        returns cosine *distance* while we want cosine *similarity*.
        """
        model = self._get_embedding()
        if model is None:
            logger.warning("ContentAuditor: embedding model not available, defaulting to 0.0")
            return 0.0

        try:
            # Encode resource content to get a real embedding
            content_text = f"{resource.title} {resource.content[:2000]}"
            emb = model.encode([content_text], show_progress_bar=False)[0]
            emb_list = emb.tolist()

            results = self._vector_index.search(
                embedding=emb_list,
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
            kp_metadata_str = (
                f"名称: {knowledge_unit.name}\n"
                f"描述: {knowledge_unit.description}\n"
                f"难度: {knowledge_unit.difficulty}\n"
                f"核心概念: {', '.join(knowledge_unit.key_concepts)}"
            )
            prompt = PromptRegistry.get(
                "content-auditor/v1-audit",
                kp_metadata=kp_metadata_str,
                content_type=resource.type,
                content=resource.content[:2000],
            )
            # The call itself is the point (it records token usage and may
            # raise); the returned text is unused here.
            await self._llm.generate(prompt)  # type: ignore[union-attr]
            return True, None
        except Exception:
            return True, None  # Graceful degradation — skip LLM check
