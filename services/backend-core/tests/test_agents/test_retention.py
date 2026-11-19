"""Tests for checkpoint retention (I-6 follow-up).

Retention is destructive, so the tests focus on the *guard rails* rather than
on happy-path numbers:

* A thread whose age cannot be determined must be **kept**, not deleted —
  deleting state we cannot date is a destructive guess.
* Failures on one thread must not abort the pass, or a single corrupt thread
  would block cleanup forever.
* Retention delegates to ``adelete_thread``, the saver's *supported* API.
  ``aprune`` looks available (it is inherited) but raises
  ``NotImplementedError`` on ``AsyncPostgresSaver`` — mocking hid that, and it
  only surfaced against a real database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.retention import (
    is_thread_stale,
    run_retention,
)


def _pool_returning(rows: list[tuple]) -> MagicMock:
    """A psycopg-pool stand-in whose cursor yields *rows*.

    Retention reads thread timestamps straight from the ``checkpoints`` table,
    so the fake is shaped like a DB cursor rather than like a saver snapshot.
    """
    cursor = AsyncMock()
    cursor.execute = AsyncMock()
    cursor.fetchall = AsyncMock(return_value=rows)

    conn = MagicMock()
    conn.cursor = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=cursor),
        __aexit__=AsyncMock(return_value=None),
    ))
    pool = MagicMock()
    pool.connection = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=conn),
        __aexit__=AsyncMock(return_value=None),
    ))
    return pool


def _saver_with_activity(activities: dict[str, str | None]) -> AsyncMock:
    """Saver whose thread_last_activity query returns *activities*.

    Values are the raw text produced by ``(checkpoint -> 'ts')::text`` — i.e.
    a JSON string **with quotes** — so the parsing path is exercised honestly.
    """
    rows = [
        (tid, None if ts is None else f'"{ts}"')
        for tid, ts in activities.items()
    ]
    saver = AsyncMock()
    saver.conn = _pool_returning(rows)
    saver.adelete_thread = AsyncMock()
    return saver


class TestStaleness:
    @pytest.mark.asyncio
    async def test_old_thread_is_stale(self) -> None:
        saver = _saver_with_activity({"t1": "2026-09-01T00:00:00+00:00"})
        now = datetime(2026, 9, 24, tzinfo=timezone.utc)

        assert await is_thread_stale(saver, "t1", ttl=timedelta(days=7), now=now) is True

    @pytest.mark.asyncio
    async def test_recent_thread_is_kept(self) -> None:
        saver = _saver_with_activity({"t1": "2026-09-23T00:00:00+00:00"})
        now = datetime(2026, 9, 24, tzinfo=timezone.utc)

        assert await is_thread_stale(saver, "t1", ttl=timedelta(days=7), now=now) is False

    @pytest.mark.asyncio
    async def test_undatable_thread_is_kept(self) -> None:
        """No readable timestamp → keep. Deleting undated state is a guess."""
        saver = _saver_with_activity({"t1": None})

        assert await is_thread_stale(
            saver, "t1", ttl=timedelta(days=1), now=datetime(2030, 1, 1, tzinfo=timezone.utc)
        ) is False

    @pytest.mark.asyncio
    async def test_malformed_timestamp_is_kept(self) -> None:
        saver = _saver_with_activity({"t1": "not-a-date"})

        assert await is_thread_stale(
            saver, "t1", ttl=timedelta(days=1), now=datetime(2030, 1, 1, tzinfo=timezone.utc)
        ) is False

    @pytest.mark.asyncio
    async def test_unknown_thread_is_kept(self) -> None:
        """A thread with no row cannot be dated → keep."""
        saver = _saver_with_activity({})

        assert await is_thread_stale(
            saver, "ghost", ttl=timedelta(days=1), now=datetime(2030, 1, 1, tzinfo=timezone.utc)
        ) is False

    @pytest.mark.asyncio
    async def test_naive_timestamp_treated_as_utc(self) -> None:
        saver = _saver_with_activity({"t1": "2026-09-01T00:00:00"})
        now = datetime(2026, 9, 24, tzinfo=timezone.utc)

        assert await is_thread_stale(saver, "t1", ttl=timedelta(days=7), now=now) is True

    @pytest.mark.asyncio
    async def test_prefetched_activity_avoids_a_query(self) -> None:
        """Passing last_activity must not trigger a second lookup."""
        saver = _saver_with_activity({"t1": "2026-09-01T00:00:00+00:00"})
        now = datetime(2026, 9, 24, tzinfo=timezone.utc)

        stale = await is_thread_stale(
            saver, "t1", ttl=timedelta(days=7), now=now,
            last_activity=datetime(2026, 9, 20, tzinfo=timezone.utc),
        )
        assert stale is False


class TestAsUtc:
    """Timestamp values may arrive as datetime (psycopg) or as a string."""

    def test_datetime_passes_through(self) -> None:
        from src.agents.orchestrator.retention import _as_utc

        got = _as_utc(datetime(2026, 9, 24, tzinfo=timezone.utc))
        assert got is not None and got.year == 2026

    def test_quoted_json_string_parses(self) -> None:
        """The ``ts`` column is JSON text, so it arrives wrapped in quotes.

        Asserting only ``is not None`` was too weak to notice the quotes not
        being stripped: the surrounding quotes are part of the value, and a
        parse that ignored the mismatch would still be a sign the JSONB was
        read as something other than a JSON scalar.  Pinning the exact instant
        makes the stripping load-bearing.
        """
        from src.agents.orchestrator.retention import _as_utc

        got = _as_utc('"2026-09-24T07:18:21.777984+00:00"')
        assert got == datetime(
            2026, 9, 24, 7, 18, 21, 777984, tzinfo=timezone.utc,
        )

    def test_unquoted_string_parses(self) -> None:
        from src.agents.orchestrator.retention import _as_utc

        assert _as_utc("2026-09-24T07:18:21+00:00") is not None

    def test_none_and_garbage_yield_none(self) -> None:
        from src.agents.orchestrator.retention import _as_utc

        assert _as_utc(None) is None
        assert _as_utc("") is None
        assert _as_utc('"not-a-date"') is None
        assert _as_utc(12345) is None

    def test_naive_gets_utc(self) -> None:
        from src.agents.orchestrator.retention import _as_utc

        got = _as_utc("2026-09-24T07:00:00")
        assert got is not None and got.tzinfo is not None


class TestTimestampOrdering:
    """Timestamps must be compared chronologically, not as text.

    The query casts to ``timestamptz``. This is a robustness guard rather than
    a bug fix: with LangGraph's fixed 6-digit fractional precision, text order
    happens to agree with time order today — see the accompanying note in
    ``retention.py``. What the cast removes is the *dependency* on that
    formatting, which a differently-formatted timestamp would break.
    """

    @pytest.mark.asyncio
    async def test_query_casts_to_timestamptz_not_text(self) -> None:
        from src.agents.orchestrator.retention import thread_last_activity

        cursor = AsyncMock()
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=cursor),
            __aexit__=AsyncMock(return_value=None),
        ))
        pool = MagicMock()
        pool.connection = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        saver = AsyncMock()
        saver.conn = pool

        await thread_last_activity(saver)

        sql = cursor.execute.call_args[0][0]
        assert "timestamptz" in sql, "timestamps must be cast for comparison"
        assert "::text" not in sql, "no text cast may remain in the ordering query"

    @pytest.mark.asyncio
    async def test_datetime_values_are_accepted(self) -> None:
        """psycopg returns datetime for timestamptz — the cast changes the type."""
        from src.agents.orchestrator.retention import thread_last_activity

        moment = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
        cursor = AsyncMock()
        cursor.execute = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[("t1", moment)])
        conn = MagicMock()
        conn.cursor = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=cursor),
            __aexit__=AsyncMock(return_value=None),
        ))
        pool = MagicMock()
        pool.connection = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=conn),
            __aexit__=AsyncMock(return_value=None),
        ))
        saver = AsyncMock()
        saver.conn = pool

        got = await thread_last_activity(saver)
        assert got["t1"] == moment


def _saver_with_threads(
    activities: dict[str, str | None],
) -> AsyncMock:
    """Saver that answers *both* queries ``run_retention`` makes.

    ``list_thread_ids`` and ``thread_last_activity`` each open their own
    cursor, so a single-row-list fake cannot serve both.  This dispatches on
    the SQL text: the ``DISTINCT thread_id`` query gets the ids, the
    ``MAX`` query gets ``(thread_id, quoted-ts)`` rows.
    """
    ids_rows = [(tid,) for tid in activities]
    ts_rows = [
        (tid, None if ts is None else f'"{ts}"')
        for tid, ts in activities.items()
    ]

    class _DispatchingPool:
        def connection(self):
            cursor = AsyncMock()
            # Explicit, so a fetch-before-execute fails loudly instead of
            # reading an AsyncMock auto-attribute and silently choosing a
            # dataset.
            cursor._last_sql = None

            async def _execute(sql, params=None):
                cursor._last_sql = sql

            async def _fetchall():
                if cursor._last_sql is None:
                    raise AssertionError(
                        "fetchall() before execute() — the fake cannot tell "
                        "which query to answer"
                    )
                return ids_rows if "DISTINCT thread_id" in cursor._last_sql else ts_rows

            cursor.execute = AsyncMock(side_effect=_execute)
            cursor.fetchall = AsyncMock(side_effect=_fetchall)
            conn = MagicMock()
            conn.cursor = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=cursor),
                __aexit__=AsyncMock(return_value=None),
            ))
            return AsyncMock(
                __aenter__=AsyncMock(return_value=conn),
                __aexit__=AsyncMock(return_value=None),
            )

    saver = AsyncMock()
    saver.conn = _DispatchingPool()
    saver.adelete_thread = AsyncMock()
    return saver


_OLD = "2020-01-01T00:00:00+00:00"
_NEW = "2999-01-01T00:00:00+00:00"


class TestRunRetention:
    """``run_retention`` — the module's entry point, previously untested.

    Every other test in this file targets a *helper* (``is_thread_stale``,
    ``_as_utc``).  That is how two mutation survivors went unnoticed: the
    deletion loop and the stale-set filter both live in ``run_retention``,
    which no test called.  A retention job whose delete loop is dead is
    indistinguishable from a working one while the table grows unbounded —
    exactly the failure this module exists to prevent.
    """

    @pytest.mark.asyncio
    async def test_stale_threads_are_actually_deleted(self) -> None:
        """The loop must call ``adelete_thread`` for each stale thread.

        Survivor was ``if await delete_thread(...)`` -> ``if False:``.
        """
        saver = _saver_with_threads({"old-1": _OLD, "old-2": _OLD})
        report = await run_retention(saver)

        assert report.threads_deleted == 2, report.to_dict()
        assert report.threads_examined == 2, report.to_dict()
        deleted = sorted(c.args[0] for c in saver.adelete_thread.await_args_list)
        assert deleted == ["old-1", "old-2"], (
            f"stale threads must be deleted, got {deleted}"
        )

    @pytest.mark.asyncio
    async def test_fresh_threads_are_not_deleted(self) -> None:
        saver = _saver_with_threads({"new-1": _NEW})
        report = await run_retention(saver)

        assert report.threads_deleted == 0, report.to_dict()
        assert report.threads_kept == 1, report.to_dict()
        saver.adelete_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_only_the_stale_ones_are_deleted(self) -> None:
        saver = _saver_with_threads(
            {"old-1": _OLD, "new-1": _NEW, "undatable": None}
        )
        report = await run_retention(saver)

        deleted = [c.args[0] for c in saver.adelete_thread.await_args_list]
        assert deleted == ["old-1"], (
            f"only the old thread may be deleted; undatable is kept. Got {deleted}"
        )
        assert report.to_dict() == {
            "threads_examined": 3,
            "threads_deleted": 1,
            "threads_kept": 2,
            "errors": 0,
        }

    @pytest.mark.asyncio
    async def test_a_failing_delete_is_counted_not_raised(self) -> None:
        """One corrupt thread must not abort the pass (module docstring)."""
        saver = _saver_with_threads({"old-1": _OLD, "old-2": _OLD})
        saver.adelete_thread = AsyncMock(side_effect=RuntimeError("boom"))

        report = await run_retention(saver)

        assert report.errors == 2, report.to_dict()
        assert report.threads_deleted == 0, report.to_dict()

    @pytest.mark.asyncio
    async def test_stale_set_filters_discovery_but_does_not_replace_it(self) -> None:
        """Passing stale ids must still enumerate threads from the saver.

        Survivor was ``stale = [tid for tid in thread_ids if tid in precomputed]``
        -> ``stale = list(precomputed)``.  The difference is what happens to an
        id in the precomputed set that the saver does **not** know about: the
        filter ignores it, the replacement tries to delete it.  Deleting a
        thread that was never discovered means acting on an id from somewhere
        other than the store being cleaned.
        """
        saver = _saver_with_threads({"old-1": _OLD, "new-1": _NEW})

        report = await run_retention(
            saver, stale_thread_ids=["old-1", "ghost-not-in-store"],
        )

        deleted = [c.args[0] for c in saver.adelete_thread.await_args_list]
        assert deleted == ["old-1"], (
            f"stale_thread_ids must filter discovered threads, not replace "
            f"them — a non-existent id must never be passed to the saver. "
            f"Got {deleted}"
        )
        assert report.threads_examined == 2, (
            f"discovery must still run; got {report.to_dict()}"
        )

    @pytest.mark.asyncio
    async def test_empty_stale_set_means_nothing_stale_not_do_nothing(self) -> None:
        """``[]`` is a filter that excludes everything — discovery still runs.

        Documented explicitly in the docstring: an earlier version treated the
        argument as the thread list, so ``[]`` became a silent no-op.
        """
        saver = _saver_with_threads({"old-1": _OLD})

        report = await run_retention(saver, stale_thread_ids=[])

        assert report.threads_examined == 1, (
            f"threads must still be enumerated when the stale set is empty; "
            f"got {report.to_dict()}"
        )
        assert report.threads_deleted == 0
        saver.adelete_thread.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_listing_failure_is_reported_not_raised(self) -> None:
        """Retention is housekeeping: it must not break the service."""
        saver = AsyncMock()
        saver.conn = None  # list_thread_ids returns [] when there is no pool

        report = await run_retention(None)

        assert report.to_dict() == {
            "threads_examined": 0,
            "threads_deleted": 0,
            "threads_kept": 0,
            "errors": 0,
        }
