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
    DEFAULT_THREAD_TTL_DAYS,
    delete_thread,
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
        from src.agents.orchestrator.retention import _as_utc

        assert _as_utc('"2026-09-24T07:18:21.777984+00:00"') is not None

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
