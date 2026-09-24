"""Cross-Encoder reranker for RAG retrieval re-ranking.

Uses a cross-encoder model to re-score query-document pairs, producing
more accurate relevance scores than embedding cosine similarity alone.
"""

from __future__ import annotations

import logging
import math
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

        # Attach scores and re-sort.
        #
        # NOTE: ``predict`` returns a numpy array, so each element is a
        # ``numpy.float32`` — NOT a Python ``float``. The previous version
        # branched on ``isinstance(score, (int, float))`` / ``isinstance(score,
        # (list, tuple))``, and a numpy scalar satisfies *neither*, so no branch
        # ran and **the reranker's scores were silently discarded**: it computed
        # for every query (doubling latency) while the ranking never changed.
        #
        # The scores themselves are sound (a relevant doc scored 7.83 against
        # 1.81 for an irrelevant one in a direct check) — they were simply
        # thrown away. Converting through ``float()`` accepts numpy scalars and
        # Python numbers alike.
        #
        # **Scale matters too.** Cross-encoder output is an unbounded logit,
        # while every other producer of ``RAGResult.score`` in this codebase
        # yields a [0,1] number — fusion scores (~0.016) and cosine similarity
        # alike. The frontend renders ``score`` as a percentage
        # (``Math.round(score * 100)`` plus a bar width of ``score * 100``), so
        # a raw logit shows up as "783%" and overflows its container. Ranking
        # uses the raw logit (monotone, so order is identical) and the stored
        # value is squashed through a logistic so downstream consumers keep the
        # [0,1] contract they were written against.
        scored: list[tuple[float, int]] = []
        for i, score in enumerate(scores):
            value = self._extract_score(score)
            if value is not None:
                scored.append((value, i))

        if not scored:
            # Nothing usable — keep the incoming order rather than returning
            # an arbitrary one.
            logger.warning("Cross-encoder produced no usable scores; keeping input order")
            return results[:top_k]

        # Rank on the raw logit (order-preserving), then write normalised values.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ranked_results: list[RAGResult] = []
        for value, idx in scored:
            result = results[idx]
            result.score = self._to_unit_score(value)
            ranked_results.append(result)

        logger.debug(
            "Reranked %d results -> top %d (top logit: %.4f)",
            len(results), top_k, scored[0][0] if scored else 0,
        )
        return ranked_results[:top_k]

    @staticmethod
    def _to_unit_score(logit: float) -> float:
        """Squash a cross-encoder logit into [0, 1] via the logistic function.

        Every other score producer in this codebase (RRF/score fusion, cosine
        similarity) already yields [0,1], and the frontend renders scores as
        percentages. Returning a raw logit would display as "783%" and overflow
        the progress bar, so the stored value is normalised while ranking uses
        the raw value (the logistic is monotone, so order is unaffected).

        Note these values are *not* calibrated probabilities — a sigmoid of a
        relevance logit is a monotone rescaling, not a likelihood. They are
        suitable for display and ordering, not for thresholding decisions.
        """
        # Clamp the exponent to avoid OverflowError on extreme logits.
        if logit >= 0:
            z = math.exp(-min(logit, 60.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(logit, -60.0))
        return z / (1.0 + z)

    @staticmethod
    def _extract_score(score: Any) -> float | None:
        """Normalise one model output into a float, or None if unusable.

        Handles the shapes a cross-encoder may emit:
          * a numpy or Python scalar (``float()`` accepts both);
          * a 2-element sequence ``[neg, pos]`` — the positive class is taken;
          * a 1-element sequence.
        Returns None when the value cannot be interpreted, so the caller skips
        it rather than writing a nonsense score.
        """
        try:
            if hasattr(score, "__len__") and not isinstance(score, (str, bytes)):
                seq = list(score)
                if not seq:
                    return None
                score = seq[1] if len(seq) > 1 else seq[0]
            return float(score)
        except (TypeError, ValueError):
            return None
