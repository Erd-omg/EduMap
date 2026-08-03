"""Evaluation framework for RAG quality measurement."""

from __future__ import annotations

import logging
from typing import Any

from src.rag.evaluation.metrics import (
    answer_relevancy,
    faithfulness,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


class EvalResult:
    """Result of a single evaluation run."""

    def __init__(
        self,
        precision: dict[int, float],
        recall: dict[int, float],
        mrr: float,
        ndcg: dict[int, float],
        faithfulness_score: float,
        relevancy_score: float,
        n_queries: int,
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.mrr = mrr
        self.ndcg = ndcg
        self.faithfulness_score = faithfulness_score
        self.relevancy_score = relevancy_score
        self.n_queries = n_queries

    @property
    def overall_score(self) -> float:
        """Combined quality score (0-1) weighting all metrics."""
        avg_precision = sum(self.precision.values()) / max(len(self.precision), 1)
        avg_recall = sum(self.recall.values()) / max(len(self.recall), 1)
        avg_ndcg = sum(self.ndcg.values()) / max(len(self.ndcg), 1)
        return round(
            0.25 * avg_precision
            + 0.25 * avg_recall
            + 0.15 * self.mrr
            + 0.15 * avg_ndcg
            + 0.10 * self.faithfulness_score
            + 0.10 * self.relevancy_score,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": {str(k): v for k, v in self.precision.items()},
            "recall": {str(k): v for k, v in self.recall.items()},
            "mrr": self.mrr,
            "ndcg": {str(k): v for k, v in self.ndcg.items()},
            "faithfulness": self.faithfulness_score,
            "relevancy": self.relevancy_score,
            "n_queries": self.n_queries,
            "overall": self.overall_score,
        }


class RAGEvaluator:
    """Offline RAG evaluation pipeline.

    Evaluates retrieval quality (precision, recall, MRR, NDCG) and
    generation quality (faithfulness, answer relevancy) against a
    dataset of query → ground-truth document pairs.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        self._k_values = k_values or [1, 3, 5, 10]

    async def evaluate_retrieval(
        self,
        queries: list[dict],
        retrieval_fn: Any,
    ) -> EvalResult:
        """Evaluate retrieval quality against ground-truth data.

        Args:
            queries: List of dicts with keys ``query``, ``relevant_ids`` (set of str).
            retrieval_fn: Async callable ``(query, top_k) -> list[RAGResult]``.

        Returns:
            EvalResult with retrieval metrics.
        """
        all_precision: dict[int, list[float]] = {k: [] for k in self._k_values}
        all_recall: dict[int, list[float]] = {k: [] for k in self._k_values}
        all_mrr: list[float] = []
        all_ndcg: dict[int, list[float]] = {k: [] for k in self._k_values}

        max_k = max(self._k_values)

        for q_data in queries:
            query = q_data["query"]
            relevant = q_data["relevant_ids"]
            if isinstance(relevant, list):
                relevant = set(relevant)

            results = await retrieval_fn(query, top_k=max_k)

            for k in self._k_values:
                all_precision[k].append(precision_at_k(results, relevant, k))
                all_recall[k].append(recall_at_k(results, relevant, k))
                all_ndcg[k].append(ndcg_at_k(results, relevant, k))

            all_mrr.append(mean_reciprocal_rank(results, relevant))

        return EvalResult(
            precision={k: round(sum(v) / max(len(v), 1), 4) for k, v in all_precision.items()},
            recall={k: round(sum(v) / max(len(v), 1), 4) for k, v in all_recall.items()},
            mrr=round(sum(all_mrr) / max(len(all_mrr), 1), 4),
            ndcg={k: round(sum(v) / max(len(v), 1), 4) for k, v in all_ndcg.items()},
            faithfulness_score=0.0,
            relevancy_score=0.0,
            n_queries=len(queries),
        )

    async def evaluate_generation(
        self,
        qa_pairs: list[dict],
    ) -> EvalResult:
        """Evaluate generation quality (faithfulness + relevancy).

        Args:
            qa_pairs: List of dicts with keys ``query``, ``answer``, ``context_sentences``.
        """
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []

        for pair in qa_pairs:
            # Faithfulness
            f_result = faithfulness(
                pair["answer"],
                pair.get("context_sentences", []),
            )
            faithfulness_scores.append(f_result["faithfulness"])

            # Relevancy
            relevancy_scores.append(answer_relevancy(pair["answer"], pair["query"]))

        return EvalResult(
            precision={},
            recall={},
            mrr=0.0,
            ndcg={},
            faithfulness_score=round(sum(faithfulness_scores) / max(len(faithfulness_scores), 1), 4),
            relevancy_score=round(sum(relevancy_scores) / max(len(relevancy_scores), 1), 4),
            n_queries=len(qa_pairs),
        )
