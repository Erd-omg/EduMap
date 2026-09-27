"""Tests for evaluation dataset loader."""

from __future__ import annotations

from src.rag.evaluation.datasets import (
    available_courses,
    load_expanded_queries,
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


class TestCourseParameterisation:
    """Datasets must be selectable by course, not hardcoded to cs201.

    The fusion default in ``src/config.py`` was chosen on a cs201-only
    ablation, and the open question was whether that conclusion generalises.
    Answering it requires loading a *different* course's corpus — which was
    impossible while the loaders hardcoded a single filename.
    """

    def test_default_is_still_cs201(self) -> None:
        """The default argument must not change existing behaviour."""
        assert load_expanded_queries() == load_expanded_queries("cs201")

    def test_cs301_corpus_loads(self) -> None:
        queries = load_expanded_queries("cs301")
        assert len(queries) > 0
        for q in queries:
            assert q["relevant_ids"], f"cs301 row has no labels: {q['query']}"
            for rid in q["relevant_ids"]:
                assert rid.startswith("kp-os-"), rid

    def test_unknown_course_yields_empty_not_cs201(self) -> None:
        """A missing course must not silently fall back to another corpus.

        Falling back would mean a benchmark run "for cs301" actually measured
        cs201 — the kind of mislabelled result that is worse than no result.
        """
        assert load_expanded_queries("no-such-course") == []

    def test_sample_loader_accepts_a_course(self) -> None:
        assert load_sample_queries("no-such-course") == []

    def test_available_courses_lists_both(self) -> None:
        courses = available_courses()
        assert "cs201" in courses
        assert "cs301" in courses


class TestCs301CorpusIntegrity:
    """The cs301 labels must point at KPs that exist in its seeded course.

    A corpus whose ``relevant_kp_ids`` reference nothing retrieves nothing,
    scoring every fusion strategy 0.0 — which reads as a measurement rather
    than as the labelling error it is.  This pins the ids against the seed
    used to create the course (``scripts/seed_cs301.py``).
    """

    def _seeded_ids(self) -> set[str]:
        import re
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2] / "scripts" / "seed_cs301.py").read_text()
        return set(re.findall(r'\("(kp-os-[a-z]+)"', src))

    def test_every_label_exists_in_the_course_seed(self) -> None:
        seeded = self._seeded_ids()
        assert seeded, "failed to parse any KP ids out of the cs301 seed"
        used = {rid for q in load_expanded_queries("cs301") for rid in q["relevant_ids"]}
        assert used <= seeded, f"corpus references unseeded KPs: {sorted(used - seeded)}"

    def test_seed_defines_the_whole_course(self) -> None:
        """Every seeded KP is exercised by at least one query."""
        seeded = self._seeded_ids()
        used = {rid for q in load_expanded_queries("cs301") for rid in q["relevant_ids"]}
        assert seeded <= used, f"seeded KPs with no query: {sorted(seeded - used)}"


class TestAvailableCoursesIsNotFooledByStrayFiles:
    """``available_courses`` must report corpora, not filenames.

    It used to derive ids from the filename alone, so an archival file like
    ``expanded_queries_old.json`` was advertised as a loadable course and
    ``--course old`` would benchmark it — reporting numbers for a course that
    does not exist.
    """

    def _write(self, name: str, body: str):
        from src.rag.evaluation.datasets import _DATASETS_DIR

        path = _DATASETS_DIR / name
        path.write_text(body, encoding="utf-8")
        return path

    def test_undeclared_file_is_not_a_course(self) -> None:
        path = self._write("expanded_queries_old.json", '{"queries": []}')
        try:
            assert "old" not in available_courses()
        finally:
            path.unlink()

    def test_declared_course_is_listed(self) -> None:
        path = self._write(
            "expanded_queries_cs999.json",
            '{"_meta": {"course_id": "cs999"}, "queries": []}',
        )
        try:
            assert "cs999" in available_courses()
        finally:
            path.unlink()

    def test_the_real_corpora_are_still_listed(self) -> None:
        courses = available_courses()
        assert "cs201" in courses and "cs301" in courses


class TestCourseScopesEveryLoader:
    """A course parameter that some callers ignore would silently mix corpora.

    ``course_id`` was added to the loaders with a default, so an un-updated
    call site keeps loading cs201 without any error — and scoring a cs301 run
    against cs201's labels yields 0 for everything while looking like a
    measurement.  This pins the call sites that take a course.
    """

    def test_benchmark_helper_scopes_to_its_course(self) -> None:
        """``RAGBenchmarkRunner`` already took ``course_id`` — it must use it."""
        import inspect

        from src.rag.evaluation.benchmark import RAGEvalBenchmark

        src = inspect.getsource(RAGEvalBenchmark.run_benchmark)
        assert "load_sample_queries(course_id" in src, (
            "run_benchmark must pass its course_id to the loader"
        )

    def test_eval_scripts_accept_a_course_flag(self) -> None:
        """Every benchmark entry point that loads a corpus must expose it."""
        import pathlib
        import re

        scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts"
        for name in (
            "run_fusion_ablation.py",
            "run_rag_benchmark.py",
            "run_reranker_ablation.py",
            "run_strategy_comparison.py",
        ):
            src = (scripts / name).read_text(encoding="utf-8")
            assert '--course' in src, f"{name} loads a corpus but has no --course"
            # ...and actually forwards it, rather than accepting and ignoring.
            assert re.search(r"loader\(args\.course\)|_queries\(args\.course\)", src), (
                f"{name} declares --course but never passes it to the loader"
            )


class TestReviewWorksheetStaysAligned:
    """The committed review worksheet must round-trip through a CSV reader.

    A worksheet whose rows have fewer fields than its header silently shifts
    every column after the gap: the reviewer fills in a cell, and the evidence
    columns to its right move one position left.  That happened — rows 1 and 2
    of the first committed worksheet had ``top_rrf``'s content sitting in the
    ``foreign_hits`` column and ``top_score`` empty, because those were the two
    rows whose ``never_retrieved`` cell was blank and the blank was swallowed.

    The generator now writes ``-`` for empty cells and self-checks the round
    trip (``_assert_columns_align``).  This test pins the *committed artifact*,
    so a hand-edit that reintroduces the shift fails here rather than silently
    misinforming the next reviewer.
    """

    _PATH = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "docs" / "cs301-review-worksheet.csv"
    )

    def test_every_row_has_the_same_field_count_as_the_header(self) -> None:
        import csv

        with open(self._PATH, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

        assert rows, "worksheet is empty"
        header, body = rows[0], rows[1:]
        assert len(header) == 9, f"unexpected header: {header}"
        assert body, "worksheet has a header but no rows"

        width = len(header)
        bad = [i for i, r in enumerate(body, 1) if len(r) != width]
        assert not bad, (
            f"rows {bad} have a field count != {width} — the columns after the "
            f"gap are shifted, so every value to their right is mislabelled."
        )

    def test_fix_is_the_fourth_column_and_never_blank(self) -> None:
        """The reviewer's verdict must stay in one place and be explicit.

        ``fix`` is what the review is *for*; if it drifts or goes blank the
        worksheet stops recording a decision.
        """
        import csv

        with open(self._PATH, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))

        header, body = rows[0], rows[1:]
        assert header[3] == "fix", f"column 3 is {header[3]!r}, expected 'fix'"

        blank = [r[0] for r in body if not r[3].strip()]
        assert not blank, (
            f"rows {blank} have an empty `fix` — every row must carry a verdict "
            f"('ok' or a replacement KP list)."
        )
