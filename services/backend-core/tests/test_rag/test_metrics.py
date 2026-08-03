"""Tests for RAG evaluation metrics (pure functions, no deps)."""

from __future__ import annotations

from src.rag.evaluation.metrics import (
    answer_relevancy,
    citation_accuracy,
    context_coverage,
    faithfulness,
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionRecall:
    """Precision@K and Recall@K — basic ranking metrics."""

    def test_perfect_precision_at_1(self) -> None:
        """Precision@1 = 1.0 when the top result is relevant."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=1) == 1.0

    def test_zero_precision_at_1(self) -> None:
        """Precision@1 = 0.0 when the top result is irrelevant."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant = {"b"}
        assert precision_at_k(retrieved, relevant, k=1) == 0.0

    def test_precision_at_k_partial(self) -> None:
        """Precision@3 = 2/3 when 2 of top 3 are relevant."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 2.0 / 3.0

    def test_precision_k_zero(self) -> None:
        """Precision with k=0 returns 0.0."""
        retrieved = [{"source_id": "a"}]
        relevant = {"a"}
        assert precision_at_k(retrieved, relevant, k=0) == 0.0

    def test_recall_at_k_all_relevant(self) -> None:
        """Recall@K = 1.0 when all relevant docs are in top K."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0

    def test_recall_at_k_partial(self) -> None:
        """Recall@2 = 1/3 when only 1 of 3 relevant docs is in top 2."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant = {"a", "c", "d"}
        assert recall_at_k(retrieved, relevant, k=2) == 1.0 / 3.0

    def test_recall_empty_relevant(self) -> None:
        """Recall returns 0.0 when the relevant set is empty."""
        retrieved = [{"source_id": "a"}]
        relevant: set[str] = set()
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_handles_string_items(self) -> None:
        """Metrics work with plain string IDs in the retrieved list."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "c"}
        assert precision_at_k(retrieved, relevant, k=3) == 2.0 / 3.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0


class TestMRR:
    """Mean Reciprocal Rank."""

    def test_mrr_first_rank(self) -> None:
        """MRR = 1.0 when the first item is relevant."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_third_rank(self) -> None:
        """MRR = 1/3 when the first relevant item is at rank 3."""
        retrieved = [
            {"source_id": "x"},
            {"source_id": "y"},
            {"source_id": "z"},
        ]
        relevant = {"z"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0 / 3.0

    def test_mrr_no_relevant(self) -> None:
        """MRR = 0.0 when no relevant items are retrieved."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant: set[str] = {"z"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestNDCG:
    """Normalized Discounted Cumulative Gain."""

    def test_perfect_ndcg(self) -> None:
        """NDCG@K = 1.0 when ranking matches ideal order."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, k=2) == 1.0

    def test_ndcg_with_relevance_scores(self) -> None:
        """NDCG with graded relevance (not just binary)."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}]
        scores = {"a": 3.0, "b": 1.0, "c": 2.0}
        ndcg = ndcg_at_k(retrieved, set(scores.keys()), k=3, relevance_scores=scores)
        assert 0.0 < ndcg <= 1.0

    def test_ndcg_k_zero(self) -> None:
        """NDCG with k=0 returns 0.0."""
        retrieved = [{"source_id": "a"}]
        relevant = {"a"}
        assert ndcg_at_k(retrieved, relevant, k=0) == 0.0


class TestFaithfulness:
    """Faithfulness (token overlap heuristic) with Chinese-aware tokenization.

    Uses ``jieba`` for Chinese word segmentation — the old ``str.split()``
    limitation is resolved by the ``_tokenize()`` helper in ``metrics.py``.
    """

    def test_perfect_faithfulness(self) -> None:
        """Answer fully overlaps with context."""
        answer = "数组插入复杂度O(n)。链表插入复杂度O(1)。"
        context = ["数组插入复杂度O(n)", "链表插入复杂度O(1)"]
        result = faithfulness(answer, context)
        assert result["faithfulness"] == 1.0
        assert result["supported_sentences"] == 2

    def test_no_faithfulness(self) -> None:
        """Answer has no overlap with context."""
        answer = "完全无关的内容。"
        context = ["数组插入复杂度O(n)"]
        result = faithfulness(answer, context)
        assert result["faithfulness"] == 0.0

    def test_partial_faithfulness(self) -> None:
        """Only 1 of 2 sentences is supported."""
        answer = "数组插入复杂度O(n)。今天天气很好。"
        context = ["数组插入复杂度O(n)"]
        result = faithfulness(answer, context)
        assert result["faithfulness"] == 0.5
        assert len(result["unsupported"]) == 1

    def test_empty_answer(self) -> None:
        """Empty answer returns perfect faithfulness (vacuously true)."""
        result = faithfulness("", ["一些上下文"])
        assert result["faithfulness"] == 1.0
        assert result["total_sentences"] == 0


class TestAnswerRelevancy:
    """Answer relevancy (Jaccard overlap with query, Chinese-aware)."""

    def test_high_relevancy(self) -> None:
        """Answer contains most query tokens."""
        score = answer_relevancy("数组插入复杂度", "数组插入复杂度")
        assert score > 0.5

    def test_low_relevancy(self) -> None:
        """Answer shares few tokens with query."""
        score = answer_relevancy("今天天气怎么样", "数组插入复杂度O(n)")
        assert score < 0.5

    def test_empty_query(self) -> None:
        """Empty query returns neutral score 0.5."""
        score = answer_relevancy("一些回答", "")
        assert score == 0.5


class TestHitRate:
    """Hit Rate@K — "did we find anything useful?" metric."""

    def test_hit_at_1(self) -> None:
        """Hit@1 = 1 when the top item is relevant."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        assert hit_rate_at_k(retrieved, {"a"}, k=1) == 1.0

    def test_no_hit(self) -> None:
        """Hit@K = 0 when no relevant items are in top K."""
        retrieved = [{"source_id": "a"}, {"source_id": "b"}]
        assert hit_rate_at_k(retrieved, {"c"}, k=2) == 0.0

    def test_hit_at_larger_k(self) -> None:
        """Hit@3 = 1 when the relevant item is at rank 3 (still in top 3)."""
        retrieved = [{"source_id": "x"}, {"source_id": "y"}, {"source_id": "z"}]
        assert hit_rate_at_k(retrieved, {"z"}, k=3) == 1.0

    def test_hit_k_zero(self) -> None:
        """Hit@0 = 0."""
        retrieved = [{"source_id": "a"}]
        assert hit_rate_at_k(retrieved, {"a"}, k=0) == 0.0

    def test_hit_empty_retrieved(self) -> None:
        """Hit@K = 0 when retrieved list is empty."""
        assert hit_rate_at_k([], {"a"}, k=5) == 0.0


class TestContextCoverage:
    """Context coverage — what fraction of answer is covered by context."""

    def test_full_coverage(self) -> None:
        """All answer tokens appear in context."""
        answer = ["数组插入O(n)"]
        context = ["数组插入O(n)", "链表插入O(1)"]
        assert context_coverage(answer, context) == 1.0

    def test_no_coverage(self) -> None:
        """No answer tokens appear in context."""
        answer = ["完全无关内容"]
        context = ["数组插入O(n)"]
        assert context_coverage(answer, context) == 0.0

    def test_empty_answer(self) -> None:
        """Empty answer returns 1.0 (vacuously correct)."""
        assert context_coverage([], ["some context"]) == 1.0

    def test_empty_context(self) -> None:
        """Empty context returns 0.0."""
        assert context_coverage(["一些内容"], []) == 0.0


class TestCitationAccuracy:
    """Citation accuracy — structural check of [N] references."""

    def test_all_citations_valid(self) -> None:
        """Every [N] reference has a corresponding source."""
        answer = "根据资料[1]和[2]，数组插入是O(n)。"
        sources = {"1": "数组", "2": "链表"}
        assert citation_accuracy(answer, sources) == 1.0

    def test_some_invalid_citations(self) -> None:
        """Some [N] references point to non-existent sources."""
        answer = "资料[1]显示复杂度，资料[999]无法确认。"
        sources = {"1": "数组"}
        assert citation_accuracy(answer, sources) == 0.5

    def test_no_citations(self) -> None:
        """No citations = vacuously correct."""
        answer = "这是一个没有引用的回答。"
        sources = {"1": "数组"}
        assert citation_accuracy(answer, sources) == 1.0

    def test_empty_answer(self) -> None:
        """Empty answer with no citations = 1.0."""
        answer = ""
        sources: dict[str, str] = {}
        assert citation_accuracy(answer, sources) == 1.0


class TestChineseFaithfulness:
    """Chinese-specific faithfulness with jieba tokenization.

    These tests use continuous Chinese text (no spaces between Chinese words)
    to prove the ``jieba`` integration properly handles real Chinese input.
    """

    def test_chinese_supported(self) -> None:
        """Chinese answer that is fully supported by context."""
        answer = "数组插入复杂度是O(n)。链表插入复杂度是O(1)。"
        context = ["数组插入复杂度O(n)", "链表插入复杂度O(1)"]
        result = faithfulness(answer, context)
        # Both sentences overlap significantly with context after tokenization
        assert result["faithfulness"] == 1.0
        assert result["supported_sentences"] == 2

    def test_chinese_unsupported(self) -> None:
        """Chinese answer that is NOT supported by context."""
        answer = "量子计算的基本原理是量子叠加和量子纠缠。"
        context = ["数组插入复杂度O(n)", "链表插入复杂度O(1)"]
        result = faithfulness(answer, context)
        # No overlap between quantum computing and data structures
        assert result["faithfulness"] == 0.0
        assert result["supported_sentences"] == 0

    def test_chinese_mixed_english(self) -> None:
        """Mixed Chinese-English answer with typical CS terminology."""
        answer = "数组插入复杂度是O(n)。哈希表查找是O(1)。"
        context = ["数组插入复杂度O(n)", "哈希表查找O(1)"]
        result = faithfulness(answer, context)
        # Both sentences should be supported (English terms preserved by split)
        assert result["faithfulness"] == 1.0
        assert result["supported_sentences"] == 2
