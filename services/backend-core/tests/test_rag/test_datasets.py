"""Tests for evaluation dataset loader."""

from __future__ import annotations

from src.rag.evaluation.datasets import (
    load_queries_by_difficulty,
    load_sample_queries,
)


class TestSampleQueries:
    """Dataset loader tests."""

    def test_load_all_queries(self) -> None:
        """Loading sample queries returns a non-empty list."""
        queries = load_sample_queries()
        assert len(queries) > 0
        assert len(queries) <= 21  # file has 20, plus boundary

    def test_query_format(self) -> None:
        """Each query has the expected fields."""
        queries = load_sample_queries()
        q = queries[0]
        assert "query" in q
        assert "relevant_ids" in q
        assert isinstance(q["relevant_ids"], set)
        assert len(q["relevant_ids"]) > 0

    def test_filter_by_difficulty(self) -> None:
        """Loading by difficulty returns only matching queries."""
        basic = load_queries_by_difficulty("basic")
        advanced = load_queries_by_difficulty("advanced")
        assert len(basic) > 0
        assert len(advanced) > 0
        assert len(basic) > len(advanced)  # more basic than advanced

    def test_relevant_ids_are_valid(self) -> None:
        """Relevant IDs follow the kp-* naming pattern."""
        queries = load_sample_queries()
        for q in queries:
            for rid in q["relevant_ids"]:
                assert rid.startswith("kp-"), f"Invalid ID: {rid}"
