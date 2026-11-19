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


class TestCourseParameterisation:
    """Datasets must be selectable by course, not hardcoded to cs201.

    The fusion default in ``src/config.py`` was chosen on a cs201-only
    ablation, and the open question was whether that conclusion generalises.
    Answering it requires loading a *different* course's corpus — which was
    impossible while the loaders hardcoded a single filename.

    Restored after merge ``2ab03b2`` silently dropped it (along with the four
    classes below); see issue #6 / the review worksheet thread.
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
        """The KP ids the seed script will create, read from the seed itself.

        Imports ``_KPS`` rather than regex-scraping the script's source: the
        regex form broke on any reformatting and could silently yield an empty
        set (turning ``seeded <= used`` into a tautology).  Importing pins the
        data, not the text.
        """
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "seed_cs301.py"
        spec = importlib.util.spec_from_file_location("seed_cs301", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {k[0] for k in module._KPS}

    def test_every_label_exists_in_the_course_seed(self) -> None:
        seeded = self._seeded_ids()
        assert seeded, "seed_cs301.py defines no KPs — _KPS is empty"
        used = {rid for q in load_expanded_queries("cs301") for rid in q["relevant_ids"]}
        assert used <= seeded, f"corpus references unseeded KPs: {sorted(used - seeded)}"

    def test_seed_defines_the_whole_course(self) -> None:
        """Every seeded KP is exercised by at least one query."""
        seeded = self._seeded_ids()
        used = {rid for q in load_expanded_queries("cs301") for rid in q["relevant_ids"]}
        assert seeded <= used, f"seeded KPs with no query: {sorted(seeded - used)}"


class TestCourseScopesEveryLoader:
    """A course parameter that some callers ignore would silently mix corpora.

    ``course_id`` was added to the loaders with a default, so an un-updated
    call site keeps loading cs201 without any error — and scoring a cs301 run
    against cs201's labels yields 0 for everything while looking like a
    measurement.  This pins the call sites that take a course.
    """

    def test_benchmark_helper_scopes_to_its_course(self, monkeypatch) -> None:
        """``RAGBenchmarkRunner`` must pass its ``course_id`` to the loader.

        Driven through the real ``run_benchmark`` with the loader patched to
        record what it was called with — a behavioural assertion, not a source
        grep.  The earlier version string-matched ``inspect.getsource(...)``
        for ``"load_sample_queries(course_id"``, which passes for a call that
        names the right argument while passing the wrong value, and fails on a
        harmless local rename.  Both directions were verified by mutation.
        """
        import asyncio

        from src.rag.evaluation import benchmark as bench_mod

        seen: list[str] = []

        def _record(course_id=None):
            seen.append(course_id)
            return []  # empty -> run_benchmark short-circuits to "skipped"

        monkeypatch.setattr(bench_mod, "load_sample_queries", _record, raising=False)

        runner = bench_mod.RAGEvalBenchmark.__new__(bench_mod.RAGEvalBenchmark)
        runner._rag = None
        result = asyncio.run(
            runner.run_benchmark(use_sample_queries=True, course_id="cs301")
        )

        assert result["status"] == "skipped"
        assert seen == ["cs301"], (
            f"run_benchmark loaded queries for {seen!r}, not the course it was "
            f"given — a cs301 run would score against cs201's labels."
        )

    def test_eval_scripts_accept_a_course_flag(self) -> None:
        """Every benchmark entry point that loads a corpus must expose --course.

        Inspected via ``ast`` rather than substring search.  A text search
        passes on a mention in a comment or docstring and says nothing about
        the call's shape; the AST sees an actual ``add_argument("--course")``
        call and an actual call expression receiving ``args.course``.
        """
        import ast
        import pathlib

        scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts"
        # Calls that merely *report* the course are not forwarding it.
        logging_calls = {"info", "error", "warning", "debug", "print"}

        for name in (
            "run_fusion_ablation.py",
            "run_rag_benchmark.py",
            "run_reranker_ablation.py",
            "run_strategy_comparison.py",
        ):
            tree = ast.parse((scripts / name).read_text(encoding="utf-8"))

            flags = {
                n.args[0].value
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"
                and n.args
                and isinstance(n.args[0], ast.Constant)
            }
            assert "--course" in flags, f"{name} loads a corpus but has no --course"

            # ...and actually forwards it, rather than accepting and ignoring.
            forwarded = []
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                fname = (
                    call.func.attr if isinstance(call.func, ast.Attribute)
                    else getattr(call.func, "id", None)
                )
                if fname in logging_calls:
                    continue
                if any(
                    isinstance(a, ast.Attribute)
                    and isinstance(a.value, ast.Name)
                    and a.value.id == "args"
                    and a.attr == "course"
                    for a in call.args
                ):
                    forwarded.append(fname or "<call>")
            assert forwarded, (
                f"{name} declares --course but never passes args.course to a "
                f"loader (only logging/reporting calls counted)"
            )


class TestReviewWorksheetStaysAligned:
    """The committed review worksheet must round-trip through a CSV reader.

    A worksheet whose rows have fewer fields than its header silently shifts
    every column after the gap: the reviewer fills in a cell, and the evidence
    columns to its right move one position left.  That happened — rows 1 and 2
    of the first committed worksheet had ``top_rrf``'s content sitting in the
    ``foreign_hits`` column and ``top_score`` empty, because those were the two
    rows whose ``never_retrieved`` cell was blank and the blank was swallowed.

    The generator writes ``-`` for empty cells and self-checks the round trip
    (``_assert_columns_align``).  This test pins the *committed artifact*, so a
    hand-edit that reintroduces the shift fails here rather than silently
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
        """The reviewer's verdict must stay in one place and be explicit."""
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

    def test_the_worksheet_reflects_the_reviewed_labels(self) -> None:
        """The worksheet and the corpus must agree on the review's outcome.

        The worksheet is the evidence behind ``_meta.review_status ==
        "reviewed"``.  If the corpus were re-labelled without regenerating it,
        the committed worksheet would describe a different review than the one
        the metrics rest on.
        """
        import csv

        with open(self._PATH, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        by_query = {r["query"]: r for r in rows}
        corpus = load_expanded_queries("cs301")
        assert len(corpus) == len(rows)

        for q in corpus:
            row = by_query[q["query"]]
            verdict = row["fix"].strip()
            applied = verdict if verdict != "ok" else row["claimed"]
            assert applied.split() == sorted(q["relevant_ids"]), (
                f"worksheet and corpus disagree for {q['query']!r}: "
                f"worksheet says {applied!r}, corpus has "
                f"{sorted(q['relevant_ids'])!r}"
            )
