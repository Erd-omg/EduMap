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


class TestAvailableCourses:
    """``available_courses()`` —— 只认真正声明了课程的语料文件。

    真实入口：在 ``_DATASETS_DIR`` 里放文件。守护的 bug：只看文件名的话，
    `expanded_queries_old.json` 这类归档文件会被当成一门课，`--course old`
    会对一个不存在的课程跑基准并把数字当作该课程的结论。
    """

    def _write(self, datasets_dir, name: str, meta: dict) -> None:
        import json

        (datasets_dir / name).write_text(
            json.dumps({"_meta": meta, "queries": []}), encoding="utf-8"
        )

    def test_an_archival_file_is_not_a_course(self, tmp_path, monkeypatch) -> None:
        from src.rag.evaluation import datasets as ds

        self._write(tmp_path, "expanded_queries.json", {"course_id": "cs201"})
        self._write(tmp_path, "expanded_queries_cs301.json", {"course_id": "cs301"})
        # No _meta.course_id and not the historical bare name -> not a course.
        self._write(tmp_path, "expanded_queries_old.json", {})
        monkeypatch.setattr(ds, "_DATASETS_DIR", tmp_path)

        assert ds.available_courses() == ["cs201", "cs301"]

    def test_a_declared_course_is_found_even_with_an_unusual_filename(
        self, tmp_path, monkeypatch
    ) -> None:
        from src.rag.evaluation import datasets as ds

        self._write(tmp_path, "expanded_queries_archive_v2.json", {"course_id": "cs999"})
        monkeypatch.setattr(ds, "_DATASETS_DIR", tmp_path)

        assert ds.available_courses() == ["cs999"]
