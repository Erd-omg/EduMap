"""RAGRetrievalService — hybrid search combining ChromaDB vector and Neo4j KG."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.rag.models import RAGContext, RAGResult
from src.utils.ttl_cache import TTLLRUCache

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.vector_index import VectorIndex

logger = logging.getLogger(__name__)


_RESOURCE_COLLECTION = "resource_chunks"


def _naive_tokenize(text: str) -> list[str]:
    """轻量切分：ASCII 词按空白分，中文按字符 + 连续 2-gram（jieba 不可用时的兜底）。"""
    out: list[str] = []
    buf: list[str] = []
    for ch in text.lower():
        if ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            if not ch.isspace():
                out.append(ch)
    if buf:
        out.append("".join(buf))
    # 中文 2-gram（提高短查询的重叠召回）
    grams = []
    for i in range(len(text) - 1):
        pair = text[i : i + 2].lower()
        if any(not c.isascii() for c in pair) and not pair.isspace():
            grams.append(pair)
    return out + grams


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
        # LLM query rewrite (set externally; default off — see main.py wiring).
        self._rewriter: Any = None
        self._rewrite_enabled: bool = False
        # Retrieval-result TTL cache. Bypass with search(..., use_cache=False).
        self._result_cache: TTLLRUCache[str, list[RAGResult]] | None = None
        # Hybrid fusion strategy: "rrf" (default) | "minmax" | "score"
        self.fusion_method: str = self.default_fusion_method
        self.fusion_k: int = 60  # RRF 常数

    # 默认融合策略（子类或 main.py 可覆盖；亦可由 Settings.hybrid_fusion_method 注入）
    default_fusion_method: str = "rrf"

    def enable_result_cache(self, maxsize: int = 256, ttl_seconds: float = 300.0) -> None:
        """Turn on memoisation of search results keyed on the normalised query."""
        self._result_cache = TTLLRUCache(maxsize=maxsize, ttl_seconds=ttl_seconds)

    @staticmethod
    def _cache_key(query: str, top_k: int) -> str:
        return f"{top_k}:{' '.join(query.split()).lower()}"

    async def search(
        self, query: str, top_k: int = 5, use_cache: bool = True
    ) -> list[RAGResult]:
        """Run hybrid search across vector index and KG, with optional reranking.

        Args:
            query: The user's query.
            top_k: Number of results to return.
            use_cache: When False, skip (and do not populate) the result cache.
                Evaluation harnesses pass False so repeated runs stay honest.
        """
        if self._result_cache is not None and use_cache:
            cache_key = self._cache_key(query, top_k)
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                # Return copies: RAGResult is a mutable pydantic model and
                # downstream code (reranker, context assembly) may set fields.
                return [r.model_copy() for r in cached]
        else:
            cache_key = ""

        # Fetch more candidates when reranker is available
        candidate_k = top_k * 3 if self._reranker is not None else top_k

        results: list[RAGResult] = []

        # ── -1. LLM query rewrite (vector leg only) ─────────────────────────
        # Only the vector query is rewritten.  Neo4j keyword search matches on
        # exact names, so it keeps the user's real words — mirroring the KG
        # expansion policy below.  Any failure falls back to the original.
        vector_query = query
        if self._rewrite_enabled and self._rewriter is not None:
            try:
                vector_query = await self._rewriter.rewrite(query)
            except Exception as exc:
                logger.warning("Query rewrite failed, using original: %s", exc)
                vector_query = query

        # ── 0. KG-based query expansion (for ChromaDB only) ─────────────────
        chroma_query = vector_query
        if self._expand_query_enabled and self._kp_repo is not None:
            try:
                chroma_query = await self._expand_query_with_kg(vector_query)
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
        merged = self._merge_and_rank(
            results, chroma_query, method=self.fusion_method, k=self.fusion_k
        )

        # ── 4. Cross-encoder reranking (if available) ────────────────────
        final: list[RAGResult] = merged[:top_k]
        if self._reranker is not None and len(merged) > top_k:
            try:
                final = await self._reranker.rerank(chroma_query, merged, top_k=top_k)
                logger.debug("Reranked %d -> %d results", len(merged), len(final))
            except Exception as exc:
                logger.warning("Reranking failed, using original scores: %s", exc)
                final = merged[:top_k]

        if self._result_cache is not None and use_cache and cache_key:
            self._result_cache.put(cache_key, [r.model_copy() for r in final])

        return final

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

    @staticmethod
    def _neo4j_keyword_score(query: str, name: str) -> float:
        """对 Neo4j 关键字命中按匹配质量打分（量纲 [0, 1]）。

        用于替代原先全部填 1.0 的做法，使融合阶段能够分辨「完全匹配」与「模糊包含」。
        规则（从高到低）：
          1. 完全相等                              → 1.00
          2. 一端是另一端的前缀                     → 0.85
          3. 子串包含                              → 0.70
          4. 词项 Jaccard 重叠 > 0                  → 0.50 + 0.45 · jaccard（封顶 0.95）
          5. 无重叠（兜底）                        → 0.40
        """
        q = query.strip().lower()
        n = name.strip().lower()
        if not q or not n:
            return 0.0
        if q == n:
            return 1.0
        if n.startswith(q) or q.startswith(n):
            return 0.85
        if q in n or n in q:
            return 0.7

        # 中文友好的词项切分：优先 jieba（若已安装且加载），否则按字符与 ASCII 段拆分
        try:
            import jieba  # type: ignore[import-not-found]

            q_tokens = {t for t in jieba.lcut(q) if t.strip()}
            n_tokens = {t for t in jieba.lcut(n) if t.strip()}
        except Exception:
            q_tokens = set(_naive_tokenize(q))
            n_tokens = set(_naive_tokenize(n))

        if not q_tokens or not n_tokens:
            return 0.4
        overlap = len(q_tokens & n_tokens)
        if overlap == 0:
            return 0.4
        union = len(q_tokens | n_tokens) or 1
        return min(0.95, 0.5 + 0.45 * (overlap / union))

    @staticmethod
    def _extract_search_terms(query: str) -> list[str]:
        """从自然语言查询中抽出可用于 CONTAINS 检索的词项（去重保序）。"""
        terms: list[str] = []
        seen: set[str] = set()
        try:
            import jieba  # type: ignore[import-not-found]

            words = [w.strip().lower() for w in jieba.cut(query, cut_all=False)]
        except Exception:
            words = [w for w in _naive_tokenize(query) if len(w) >= 2]
        for w in words:
            if len(w) >= 2 and w not in seen:
                seen.add(w)
                terms.append(w)
        return terms

    async def _neo4j_search(self, query: str, top_k: int) -> list[RAGResult]:
        """Search Neo4j knowledge graph by name/category, scoring by match quality."""
        if self._kp_repo is None:
            return []

        kps = await self._kp_repo.search_by_name(query)
        seen_ids: set[str] = set()
        if not kps:
            # 整句 CONTAINS 匹配不到时，按分词逐个检索（自然语言查询兜底）
            for term in self._extract_search_terms(query):
                try:
                    term_kps = await self._kp_repo.search_by_name(term)
                except Exception:
                    continue
                for kp in term_kps:
                    if kp.id not in seen_ids:
                        seen_ids.add(kp.id)
                        kps.append(kp)
        if not kps:
            # jieba 分出的词也全部未命中时（口语化整句，如"什么是二叉树"分出
            # ['什么','是','二叉树']，而图中节点叫"二叉搜索树"），降级用朴素
            # 2-gram 片段再试一轮："二叉树" → "二叉" 仍可 CONTAINS 命中。
            tried = set(self._extract_search_terms(query))
            for term in _naive_tokenize(query):
                if len(term) < 2 or term in tried:
                    continue
                try:
                    term_kps = await self._kp_repo.search_by_name(term)
                except Exception:
                    continue
                for kp in term_kps:
                    if kp.id not in seen_ids:
                        seen_ids.add(kp.id)
                        kps.append(kp)
        results: list[RAGResult] = []
        for kp in kps[:top_k]:
            results.append(
                RAGResult(
                    content=f"{kp.name}: {kp.description} (难度: {kp.difficulty}, 分类: {kp.category})",
                    source_type="neo4j",
                    source_id=kp.id,
                    source_name=kp.name,
                    score=self._neo4j_keyword_score(query, kp.name),
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
            for term in self._extract_search_terms(query):
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

    @classmethod
    def _merge_and_rank(
        cls,
        results: list[RAGResult],
        query: str,
        *,
        method: str | None = None,
        k: int = 60,
    ) -> list[RAGResult]:
        """Merge, deduplicate, and rank across heterogeneous sources.

        ``method`` 决定融合策略（None 时取 ``cls.default_fusion_method``）：
          - "score"  : 各自源内保留最高分；按 score 降序（旧行为，用于回退/对照）
          - "minmax" : 各自源内 min-max 归一化到 [0,1] 后再统一排序
          - "rrf"    : Reciprocal Rank Fusion，score = Σ 1/(k+rank)，只看顺序

        所有策略均按 **``source_id``**（文档身份）去重，``source_type`` 仅作为
        「来自哪一路召回」的溯源标记保留在结果上。

        为什么不能按 ``(source_type, source_id)`` 去重：同一个知识点会同时被
        ChromaDB（向量）和 Neo4j（关键字）两路召回，两路给它的 ``source_id``
        相同但 ``source_type`` 不同。按二元组去重会把**同一篇文档当成两篇**，
        结果是结果列表里出现重复项，且 recall@k 会算出 > 1 的非法值；
        RRF 期望的「同一文档多源得分累加」也永远不会发生。
        """
        if not results:
            return []
        method = (method or cls.default_fusion_method).lower()
        if method == "score":
            return cls._merge_score(results)
        if method == "minmax":
            return cls._merge_minmax(results)
        # default: RRF
        return cls._merge_rrf(results, k=k)

    @classmethod
    def _merge_score(cls, results: list[RAGResult]) -> list[RAGResult]:
        """按 ``source_id`` 去重保留各源最高分，按 score 降序排序。"""
        best: dict[str, RAGResult] = {}
        for r in results:
            prev = best.get(r.source_id)
            if prev is None or r.score > prev.score:
                best[r.source_id] = r
        return sorted(best.values(), key=lambda x: x.score, reverse=True)

    @classmethod
    def _merge_minmax(cls, results: list[RAGResult]) -> list[RAGResult]:
        """各自源内 min-max 归一化到 [0,1] 后再统一排序（保留原始 RAGResult 元数据）。

        全零源（所有 score 相等）取 0.5 中位分；仅一个元素的源直接保留。
        """
        by_source: dict[str, list[RAGResult]] = {}
        for r in results:
            by_source.setdefault(r.source_type, []).append(r)

        normalised: list[RAGResult] = []
        for group in by_source.values():
            scores = [r.score for r in group]
            lo, hi = min(scores), max(scores)
            for r in group:
                if hi == lo:
                    ns = 0.5
                else:
                    ns = (r.score - lo) / (hi - lo)
                normalised.append(r.model_copy(update={"score": ns}))
        return cls._merge_score(normalised)

    @classmethod
    def _merge_rrf(cls, results: list[RAGResult], *, k: int = 60) -> list[RAGResult]:
        """Reciprocal Rank Fusion：每个 source_type 单独按 score 排序后取名次。

        ``score(d) = Σ_sources 1/(k + rank_source(d))``——同一文档若被两路召回，
        两路的倒数名次分**累加**，因此双路都命中的文档会排在只被单路命中的前面。
        按 ``source_id`` 聚合，最终按 rrf_score 降序输出。
        """
        by_source: dict[str, list[RAGResult]] = {}
        for r in results:
            by_source.setdefault(r.source_type, []).append(r)

        rrf_scores: dict[str, float] = {}
        chosen: dict[str, RAGResult] = {}
        for group in by_source.values():
            for rank, r in enumerate(
                sorted(group, key=lambda x: x.score, reverse=True), start=1
            ):
                rrf_scores[r.source_id] = (
                    rrf_scores.get(r.source_id, 0.0) + 1.0 / (k + rank)
                )
                prev = chosen.get(r.source_id)
                if prev is None or r.score > prev.score:
                    chosen[r.source_id] = r

        out: list[RAGResult] = []
        for key, r in chosen.items():
            out.append(r.model_copy(update={"score": rrf_scores[key]}))
        out.sort(key=lambda x: x.score, reverse=True)
        return out
