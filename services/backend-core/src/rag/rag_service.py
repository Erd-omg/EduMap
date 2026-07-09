"""RAGRetrievalService — hybrid search combining ChromaDB vector and Neo4j KG."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from src.rag.models import RAGContext, RAGResult, SourceType

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """Hybrid retrieval service for Mentor RAG.

    Searches ChromaDB (vector similarity) and Neo4j (keyword) in parallel,
    merges, ranks, and assembles a context string for LLM consumption.
    """

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        kp_repo: KnowledgePointRepository | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._kp_repo = kp_repo

    async def search(self, query: str, top_k: int = 5) -> list[RAGResult]:
        """Run hybrid search across vector index and KG."""
        results: list[RAGResult] = []

        # ── 1. ChromaDB vector search ──────────────────────────────────
        if self._vector_index is not None:
            try:
                # For prototype, simulate with KP search results
                chroma_results = await self._chroma_search(query, top_k)
                results.extend(chroma_results)
            except Exception as exc:
                logger.warning("ChromaDB search failed: %s", exc)

        # ── 2. Neo4j keyword search ──────────────────────────────────────
        if self._kp_repo is not None:
            try:
                neo4j_results = await self._neo4j_search(query, top_k)
                results.extend(neo4j_results)
            except Exception as exc:
                logger.warning("Neo4j search failed: %s", exc)

        # ── 3. Merge, deduplicate, rank ──────────────────────────────────
        return self._merge_and_rank(results, query)

    async def assemble_context(
        self, results: list[RAGResult], max_chars: int = 2000
    ) -> RAGContext:
        """Assemble search results into a context string for LLM prompts."""
        parts: list[str] = []
        sources: list[RAGResult] = []

        char_count = 0
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r.source_name} (来源: {r.source_type})"
            entry = f"{header}\n{r.content[:500]}\n"
            if char_count + len(entry) > max_chars:
                break
            parts.append(entry)
            char_count += len(entry)
            sources.append(r)

        return RAGContext(
            context_str="\n---\n".join(parts) if parts else "（未找到相关参考资料）",
            sources=sources,
        )

    # ── Internal ────────────────────────────────────────────────────────

    async def _chroma_search(self, query: str, top_k: int) -> list[RAGResult]:
        """Search ChromaDB vector index."""
        if self._vector_index is None:
            return []

        # ChromaDB needs embeddings to search properly.
        # For prototype, we rely on Neo4j as primary source and
        # return empty from ChromaDB since we may not have embeddings
        # for every KP stored in the vector index yet.
        try:
            results = self._vector_index.search(
                embedding=[0.0] * 384,  # placeholder — would need sentence-transformers
                top_k=top_k,
            )
            rag_results: list[RAGResult] = []
            for r in results:
                rid = r.get("id", "")
                rname = (r.get("metadata") or {}).get("name", rid)
                rag_results.append(
                    RAGResult(
                        content=str(r.get("metadata", {})),
                        source_type="chroma",
                        source_id=rid,
                        source_name=rname,
                        score=1.0 - r.get("distance", 0.0),
                    )
                )
            return rag_results
        except Exception:
            logger.exception("ChromaDB search error")
            return []

    async def _neo4j_search(self, query: str, top_k: int) -> list[RAGResult]:
        """Search Neo4j knowledge graph by name/category."""
        if self._kp_repo is None:
            return []

        kps = await self._kp_repo.search_by_name(query)
        results: list[RAGResult] = []
        for kp in kps[:top_k]:
            results.append(
                RAGResult(
                    content=f"{kp.name}: {kp.description} (难度: {kp.difficulty}, 分类: {kp.category})",
                    source_type="neo4j",
                    source_id=kp.id,
                    source_name=kp.name,
                    score=1.0,
                    metadata={
                        "difficulty": kp.difficulty,
                        "category": kp.category,
                        "prerequisites": kp.prerequisites,
                    },
                )
            )
        return results

    @staticmethod
    def _merge_and_rank(
        results: list[RAGResult], query: str
    ) -> list[RAGResult]:
        """Merge, deduplicate by source_id, and rank by score descending."""
        seen: set[str] = set()
        deduped: list[RAGResult] = []

        for r in sorted(results, key=lambda x: x.score, reverse=True):
            key = f"{r.source_type}:{r.source_id}"
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped
