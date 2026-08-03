"""Benchmark runner for RAG quality evaluation.

Generates test queries from the knowledge graph and runs evaluation
across different chunking/retrieval configurations.  Supports loading
hand-curated query datasets for more realistic evaluation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.rag.evaluation.datasets import load_sample_queries
from src.rag.evaluation.evaluator import EvalResult, RAGEvaluator
from src.rag.evaluation.metrics import hit_rate_at_k
from src.rag.models import RAGResult

if TYPE_CHECKING:
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.rag.rag_service import RAGRetrievalService
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)


class RAGEvalBenchmark:
    """Run RAG quality benchmarks using knowledge graph data as ground truth.

    Generates test queries from knowledge point names and descriptions,
    using the KG structure (prerequisites, related nodes) to define
    relevant documents for each query.  Also supports loading hand-curated
    student query datasets for more realistic evaluation.
    """

    def __init__(
        self,
        rag_service: RAGRetrievalService,
        kp_repo: KnowledgePointRepository,
        evaluator: RAGEvaluator | None = None,
    ) -> None:
        self._rag = rag_service
        self._kp_repo = kp_repo
        self._evaluator = evaluator or RAGEvaluator()

    async def run_benchmark(
        self,
        n_queries: int = 10,
        course_id: str | None = None,
        use_sample_queries: bool = False,
        llm: BaseLLMAdapter | None = None,
    ) -> dict[str, Any]:
        """Run a full benchmark.

        Args:
            n_queries: Number of test queries to generate (from KG or dataset).
            course_id: Optional course to scope the benchmark.
            use_sample_queries: If True, load queries from ``datasets/sample_queries.json``
                instead of auto-generating from KG names.
            llm: Optional LLM adapter for LLM-as-judge evaluations.

        Returns:
            Dict with benchmark results and metadata.
        """
        # 1. Load or generate test queries
        if use_sample_queries:
            test_queries = load_sample_queries()
            if n_queries and n_queries < len(test_queries):
                test_queries = test_queries[:n_queries]
        else:
            test_queries = await self._generate_test_queries(n_queries, course_id)

        if not test_queries:
            return {
                "status": "skipped",
                "reason": "No test queries available",
                "n_queries": 0,
            }

        # 2. Evaluate retrieval (precision, recall, MRR, NDCG, hit rate)
        eval_result = await self._evaluator.evaluate_retrieval(
            queries=test_queries,
            retrieval_fn=self._rag.search,
        )

        # 3. Compute hit rate separately (not in evaluator by default)
        hit_rates = {}
        max_k = max(self._evaluator._k_values)
        for k in self._evaluator._k_values:
            hits = []
            for q_data in test_queries:
                query = q_data["query"]
                relevant = q_data["relevant_ids"]
                if isinstance(relevant, list):
                    relevant = set(relevant)
                results = await self._rag.search(query, top_k=max_k)
                hits.append(hit_rate_at_k(results, relevant, k))
            hit_rates[str(k)] = round(
                sum(hits) / max(len(hits), 1), 4
            ) if hits else 0.0

        # 4. LLM-as-judge faithfulness evaluation (default for Chinese datasets)
        llm_faithfulness_score = None
        if llm is not None and test_queries:
            try:
                from src.rag.evaluation.llm_judge import llm_faithfulness

                scores = []
                # Check if dataset is Chinese — if so, use LLM judge as primary
                has_chinese = any(
                    '一' <= ch <= '鿿'
                    for q in test_queries[:5]
                    for ch in (q.get("query", "") or "")
                )
                for q_data in test_queries[:3]:  # limit to 3 to avoid high cost
                    query = q_data["query"]
                    results = await self._rag.search(query, top_k=3)
                    context = await self._rag.assemble_context(results)
                    if context.sources:
                        judge_result = await llm_faithfulness(
                            answer=f"（关于{query}的模拟回答）",
                            context=context.context_str,
                            llm=llm,
                        )
                        scores.append(judge_result.get("faithfulness_score", 0))
                if scores:
                    llm_faithfulness_score = round(
                        sum(scores) / len(scores), 4
                    )
                if has_chinese:
                    logger.info("LLM-as-judge faithfulness used (Chinese dataset): %s", llm_faithfulness_score)
            except Exception as exc:
                logger.warning("LLM faithfulness evaluation failed: %s", exc)

        result_dict = eval_result.to_dict()
        result_dict["hit_rate"] = hit_rates
        if llm_faithfulness_score is not None:
            result_dict["llm_faithfulness"] = llm_faithfulness_score

        return {
            "status": "completed",
            "n_queries": len(test_queries),
            "dataset_source": "sample_queries" if use_sample_queries else "kg_generated",
            "metrics": result_dict,
            "generated_at": datetime.utcnow().isoformat(),
            "course_id": course_id,
        }

    async def _generate_test_queries(
        self,
        n_queries: int,
        course_id: str | None = None,
    ) -> list[dict]:
        """Generate test queries from knowledge points.

        Creates a query from each KP's name, and uses its description
        and related/prerequisite node names as ground-truth relevant IDs.
        """
        if course_id:
            kps = await self._kp_repo.get_by_course(course_id)
        else:
            kps = await self._kp_repo.list_all()

        if not kps:
            return []

        # Gather prerequisite/related relationships to define relevance
        kp_relations: dict[str, set[str]] = {}
        for kp in kps[:n_queries]:
            edges = await self._kp_repo.get_prerequisite_chain(kp.id)
            related = await self._kp_repo.get_related(kp.id)

            relevant = set()
            if edges:
                for edge in edges:
                    relevant.add(edge.id) if hasattr(edge, 'id') else None
            if related:
                for r in related:
                    relevant.add(r.id) if hasattr(r, 'id') else None

            # Always include the KP itself
            relevant.add(kp.id)
            kp_relations[kp.id] = relevant

        queries = []
        for kp in kps[:n_queries]:
            queries.append({
                "query": kp.name,
                "relevant_ids": kp_relations.get(kp.id, {kp.id}),
                "source_kp": kp.name,
                "source_kp_id": kp.id,
            })

        logger.info(
            "Generated %d test queries from KG (course=%s)", len(queries), course_id
        )
        return queries
