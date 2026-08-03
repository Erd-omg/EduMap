"""Cross-Encoder reranker for RAG retrieval re-ranking.

Uses a cross-encoder model to re-score query-document pairs, producing
more accurate relevance scores than embedding cosine similarity alone.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.models import RAGResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank retrieved documents using a cross-encoder model.

    Cross-encoders jointly encode the query and document, producing
    a relevance score that is typically more accurate than bi-encoder
    (embedding) similarity — at the cost of higher latency per pair.

    Args:
        model_name: Cross-encoder model name or path.
            Default uses a fast MiniLM model (~80MB, ~100-200ms per batch of 50).
            For higher accuracy (but slower), use ``BAAI/bge-reranker-v2-m3`` (~1.1GB).
        batch_size: Number of query-document pairs to score per batch.
        max_length: Maximum token length for the cross-encoder (truncates longer texts).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None

    def load(self) -> None:
        """Load the cross-encoder model. Call once at startup."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )
            logger.info("Loaded cross-encoder model: %s", self._model_name)
        except Exception as exc:
            logger.warning(
                "Failed to load cross-encoder model '%s': %s. "
                "Reranking will be disabled.",
                self._model_name, exc,
            )
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def rerank(
        self,
        query: str,
        results: list[RAGResult],
        top_k: int = 5,
    ) -> list[RAGResult]:
        """Re-rank results by cross-encoder relevance scores.

        Args:
            query: The user's original query.
            results: Retrieved documents to re-rank.
            top_k: Number of top results to return after re-ranking.

        Returns:
            Re-ranked results with updated ``score`` fields.
        """
        if not results or not query:
            return results[:top_k]

        if self._model is None:
            # Model not loaded — return top_k by original score
            logger.debug("Reranker not loaded, using original scores")
            return results[:top_k]

        # Build query-document pairs
        pairs: list[tuple[str, str]] = []
        for r in results:
            content = r.content[:self._max_length * 3]  # rough char limit
            pairs.append((query, content))

        try:
            # Score all pairs
            scores = self._model.predict(pairs, batch_size=self._batch_size)
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed: %s", exc)
            return results[:top_k]

        # Attach scores and re-sort
        for i, score in enumerate(scores):
            # Cross-encoders often output logits — sigmoid to [0,1] range
            if isinstance(score, (int, float)):
                results[i].score = float(score)
            elif isinstance(score, (list, tuple)) and len(score) > 0:
                # Some models return [neg_score, pos_score] — use positive class
                results[i].score = float(score[1]) if len(score) > 1 else float(score[0])

        # Sort by new score descending
        reranked = sorted(results, key=lambda r: r.score, reverse=True)

        logger.debug(
            "Reranked %d results -> top %d (top score: %.4f)",
            len(results), top_k, reranked[0].score if reranked else 0,
        )
        return reranked[:top_k]
