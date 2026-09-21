"""LLM query rewriting for retrieval.

Lives outside ``rag/evaluation/`` because this is a production dependency of
``RAGRetrievalService`` — the evaluation module only *measures* it.

The rewriter turns colloquial student questions into retrieval-friendly keyword
phrases (colloquial term → textbook term, "+2-4 related terms").  It is
deliberately failure-tolerant: any error, timeout, or empty result falls back
to the original query, so enabling it can never make retrieval *worse* by
throwing — only by returning a different (possibly less useful) query.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.utils.ttl_cache import TTLLRUCache

logger = logging.getLogger(__name__)


class RewrittenQuery(BaseModel):
    """Structured output contract for the LLM query rewriter."""

    rewritten_query: str = Field(..., description="改写后的检索查询")
    search_terms: list[str] = Field(
        default_factory=list, description="抽取出的核心检索术语"
    )
    reason: str = ""


_REWRITE_SYSTEM_PROMPT = """\
你是教育知识库的检索查询改写器。将学生的口语化提问改写为更适合向量检索的查询：

1. 把口语/简称替换为标准术语（如"哈希冲突怎么处理"→"哈希表 冲突解决方法 链地址法 开放地址法"）
2. 补全指代不清的表述，去掉寒暄与无关修饰
3. 保留问题核心意图，可附加 2-4 个相关术语关键词
4. 改写结果应是关键词组合而非完整句子，长度不超过 30 字

只输出 JSON，不要输出其他内容。"""


class QueryRewriter:
    """LLM-backed query rewriter with failure fallback and TTL caching.

    Args:
        llm: Any object exposing ``generate_structured(prompt, schema, system_prompt=...)``.
        timeout_seconds: Per-call ceiling; on timeout the original query is used.
        cache_size: Number of rewrites to memoise.
        cache_ttl_seconds: Lifetime of a memoised rewrite.
    """

    def __init__(
        self,
        llm: Any,
        timeout_seconds: float = 8.0,
        cache_size: int = 256,
        cache_ttl_seconds: float = 3600.0,
    ) -> None:
        self._llm = llm
        self._timeout = timeout_seconds
        self._cache: TTLLRUCache[str, str] = TTLLRUCache(
            maxsize=cache_size, ttl_seconds=cache_ttl_seconds
        )

    @staticmethod
    def _normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    async def rewrite(self, query: str) -> str:
        """Rewrite a query; falls back to the original on any failure.

        Returns the original ``query`` unchanged when the model is unavailable,
        times out, errors, or returns nothing usable.
        """
        normalized = self._normalize(query)
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        rewritten = query
        try:
            output = await asyncio.wait_for(
                self._llm.generate_structured(
                    f"学生提问：{query}",
                    RewrittenQuery,
                    system_prompt=_REWRITE_SYSTEM_PROMPT,
                ),
                timeout=self._timeout,
            )
            candidate = (output.rewritten_query or "").strip()
            if candidate:
                rewritten = candidate
        except Exception as exc:
            logger.warning("Query rewrite failed (%s) — using original", exc)

        self._cache.put(normalized, rewritten)
        return rewritten
