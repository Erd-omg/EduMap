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


_RESOURCE_COLLECTION = "resource_chunks"


class RAGRetrievalService:
    """Hybrid retrieval service for Mentor RAG.

    Searches ChromaDB (vector similarity) and Neo4j (keyword) in parallel,
    merges, ranks, optionally re-ranks with a cross-encoder, and assembles
    a context string for LLM consumption.
    """

    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        kp_repo: KnowledgePointRepository | None = None,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        preloaded_model: Any | None = None,
    ) -> None:
        self._vector_index = vector_index
        self._kp_repo = kp_repo
        self._model: Any | None = preloaded_model
        self._embedding_model = embedding_model
        self._reranker: Any = None  # Set by main.py when reranker is enabled
        self._expand_query_enabled: bool = False  # Set externally
        self._expand_query_max_terms: int = 5     # Set externally

    async def search(self, query: str, top_k: int = 5) -> list[RAGResult]:
        """Run hybrid search across vector index and KG, with optional reranking."""
        # Fetch more candidates when reranker is available
        candidate_k = top_k * 3 if self._reranker is not None else top_k

        results: list[RAGResult] = []

        # ── 0. KG-based query expansion (for ChromaDB only) ─────────────────
        chroma_query = query
        if self._expand_query_enabled and self._kp_repo is not None:
            try:
                chroma_query = await self._expand_query_with_kg(query)
            except Exception as exc:
                logger.warning("Query expansion failed, using original: %s", exc)

        # ── 1. ChromaDB vector search (with expanded query) ────────────────
        if self._vector_index is not None:
            try:
                chroma_results = await self._chroma_search(chroma_query, candidate_k)
                results.extend(chroma_results)
            except Exception as exc:
                logger.warning("ChromaDB search failed: %s", exc)

        # ── 2. Neo4j keyword search (original query — exact name match) ──────
        if self._kp_repo is not None:
            try:
                neo4j_results = await self._neo4j_search(query, candidate_k)
                results.extend(neo4j_results)
            except Exception as exc:
                logger.warning("Neo4j search failed: %s", exc)

        # ── 3. Merge, deduplicate, rank ──────────────────────────────────
        merged = self._merge_and_rank(results, chroma_query)

        # ── 4. Cross-encoder reranking (if available) ────────────────────
        if self._reranker is not None and len(merged) > top_k:
            try:
                reranked = await self._reranker.rerank(chroma_query, merged, top_k=top_k)
                logger.debug("Reranked %d -> %d results", len(merged), len(reranked))
                return reranked
            except Exception as exc:
                logger.warning("Reranking failed, using original scores: %s", exc)

        return merged[:top_k]

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
        """Search ChromaDB vector index.

        Generates a query embedding using sentence-transformers and searches
        both the KP embeddings collection and the resource chunks collection.
        """
        if self._vector_index is None:
            return []

        # Lazy-load embedding model
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = getattr(self, "_embedding_model", "BAAI/bge-small-zh-v1.5")
                self._model = SentenceTransformer(model_name)
            except Exception as exc:
                logger.warning("Failed to load embedding model: %s", exc)
                return []

        try:
            query_emb = self._model.encode([query], show_progress_bar=False)[0]
            query_emb_list = query_emb.tolist()

            results: list[RAGResult] = []

            # Search KP embeddings collection
            try:
                kp_results = self._vector_index.search(query_emb_list, top_k)
                for r in kp_results:
                    meta = r.get("metadata", {})
                    name = meta.get("name", "")
                    desc = meta.get("description", "")
                    category = meta.get("category", "")
                    difficulty = meta.get("difficulty", "")
                    content = name
                    if desc:
                        content += f": {desc}"
                    if category or difficulty:
                        content += f" (分类: {category}, 难度: {difficulty})"
                    results.append(
                        RAGResult(
                            content=content,
                            source_type="chroma",
                            source_id=r.get("id", ""),
                            source_name=name,
                            score=1.0 - r.get("distance", 0),
                            metadata=meta,
                        )
                    )
            except Exception as exc:
                logger.debug("KP collection search: %s", exc)

            # Search resource chunks collection (separate from KP)
            try:
                client = self._vector_index._client  # type: ignore[attr-defined]
                res_collection = client.get_collection(_RESOURCE_COLLECTION)
                chunk_results = res_collection.query(
                    query_embeddings=[query_emb_list],
                    n_results=top_k,
                    include=["metadatas", "distances", "documents"],
                )
                c_ids = chunk_results.get("ids", [[]])[0]
                c_dists = chunk_results.get("distances", [[]])[0]
                c_metas = chunk_results.get("metadatas", [[]])[0]
                c_docs = chunk_results.get("documents", [[]])[0]
                for i in range(len(c_ids)):
                    meta = c_metas[i] if c_metas else {}
                    doc_text = c_docs[i] if c_docs and i < len(c_docs) else ""
                    content = doc_text or meta.get("text_preview", "")
                    results.append(
                        RAGResult(
                            source_id=c_ids[i],
                            source_name=meta.get("resource_name", "上传资料"),
                            source_type="chroma",
                            content=content,
                            metadata=meta,
                            score=1.0 - (c_dists[i] if c_dists else 0),
                        )
                    )
            except Exception as exc:
                logger.debug("Resource chunks collection search: %s", exc)

            return results

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

    async def _expand_query_with_kg(self, query: str) -> str:
        """Expand a short/ambiguous query by appending related KG concept names.

        Finds matching KnowledgePoints via ``search_by_name()``, then traverses
        their ``RELATED_TO`` and ``PREREQUISITE_OF`` edges (depth=1) to collect
        semantically connected concept names. The expanded string is used for
        ChromaDB embedding search, where extra context improves vector similarity.

        Returns:
            Original query unmodified if no expansion found, otherwise
            ``original_query + " term1 term2 ..."`` (max ``_expand_query_max_terms``
            extra terms, deduplicated).
        """
        if not query.strip() or self._kp_repo is None:
            return query

        # Find matching KPs — first try full query, then token-by-token for Chinese
        kps: list[Any] = []
        try:
            kps = await self._kp_repo.search_by_name(query)
        except Exception:
            pass

        if not kps:
            # Tokenize the query and search by each content-bearing token
            search_terms: list[str] = []
            try:
                import jieba as _jieba
                for word in _jieba.cut(query, cut_all=False):
                    w = word.strip().lower()
                    if len(w) >= 2:
                        search_terms.append(w)
            except ImportError:
                # Fallback: split on common Chinese delimiters
                import re as _re
                for part in _re.split(r'[的 了 是 在 有 和 或 与 及 之 吗 呢 什么 怎么 如何 为 对 从 向 与 并 而 且 到 ]+', query):
                    if len(part.strip()) >= 2:
                        search_terms.append(part.strip())

            for term in search_terms:
                try:
                    term_kps = await self._kp_repo.search_by_name(term)
                    kps.extend(term_kps)
                except Exception:
                    continue

        if not kps:
            return query

        # Score matches: use jieba for Chinese-aware token overlap
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        # Try jieba for Chinese tokenization
        try:
            import jieba as _jieba
            chinese_tokens = {w.strip().lower() for w in _jieba.cut(query, cut_all=False)
                              if len(w.strip()) >= 2}
            query_tokens.update(chinese_tokens)
        except ImportError:
            pass
        scored: list[tuple[Any, int]] = []
        for kp in kps:
            name_lower = kp.name.lower()
            overlap = sum(1 for t in query_tokens if t in name_lower)
            if overlap > 0:
                scored.append((kp, overlap))
            elif name_lower in query_lower or query_lower in name_lower:
                scored.append((kp, 2))

        # Sort by overlap score desc, take top 3 as seed KPs
        scored.sort(key=lambda x: x[1], reverse=True)
        seed_kps = [s[0] for s in scored[:3]]

        if not seed_kps:
            seed_kps = kps[:2]  # fallback

        # Collect expansion terms
        extra_terms: list[str] = []
        seen_names: set[str] = set()

        for kp in seed_kps:
            # Skip only if KP name IS the entire query (nothing to expand)
            if kp.name.lower().strip() == query_lower.strip():
                continue
            if kp.name not in seen_names:
                extra_terms.append(kp.name)
                seen_names.add(kp.name)

            if len(extra_terms) >= self._expand_query_max_terms:
                break

            # Traverse RELATED_TO edges
            try:
                related = await self._kp_repo.get_related(kp.id)
            except Exception:
                related = []
            for rkp in related:
                if rkp.name not in seen_names and len(extra_terms) < self._expand_query_max_terms:
                    extra_terms.append(rkp.name)
                    seen_names.add(rkp.name)

            if len(extra_terms) >= self._expand_query_max_terms:
                break

            # Traverse PREREQUISITE_OF edges (depth=1)
            try:
                prereqs = await self._kp_repo.get_prerequisites(kp.id, depth=1)
            except Exception:
                prereqs = []
            for pkp in prereqs:
                if pkp.name not in seen_names and len(extra_terms) < self._expand_query_max_terms:
                    extra_terms.append(pkp.name)
                    seen_names.add(pkp.name)

            if len(extra_terms) >= self._expand_query_max_terms:
                break

        if not extra_terms:
            return query

        expanded = f"{query} {' '.join(extra_terms)}"
        logger.debug("Query expanded (%d extra terms): '%s' -> '%s'", len(extra_terms), query, expanded)
        return expanded

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
