"""Tests for the orchestrator checkpointer (I-6).

What is under test is the *lifecycle and failure policy*, not LangGraph itself:

* A missing backend must degrade to "no checkpointer", never to a service that
  refuses to start — the orchestrator still works with ephemeral state.
* ``stop()`` must be safe when ``start()`` failed partway or was never called,
  otherwise a startup failure turns into a shutdown crash.
* The DSN normalisation must strip asyncpg-only syntax, or psycopg refuses the
  connection.
* ``thread_id`` length is guarded because the Postgres saver keys on it and
  LangGraph's own docs flag the 255-char limit.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator.checkpointing import (
    CheckpointerHolder,
    _normalise_dsn,
    thread_config,
)


class TestDsnNormalisation:
    def test_strips_asyncpg_scheme(self) -> None:
        assert (
            _normalise_dsn("postgresql+asyncpg://u:p@host:5432/db")
            == "postgresql://u:p@host:5432/db"
        )

    def test_strips_psycopg_scheme(self) -> None:
        assert (
            _normalise_dsn("postgresql+psycopg://u:p@host/db")
            == "postgresql://u:p@host/db"
        )

    def test_plain_dsn_untouched(self) -> None:
        dsn = "postgresql://u:p@host:5432/db"
        assert _normalise_dsn(dsn) == dsn

    def test_only_the_scheme_is_replaced(self) -> None:
        """A password containing the scheme text must not be mangled."""
        dsn = "postgresql+asyncpg://u:postgresql+asyncpg@host/db"
        assert _normalise_dsn(dsn) == "postgresql://u:postgresql+asyncpg@host/db"


class TestThreadConfig:
    def test_shape(self) -> None:
        cfg = thread_config("sess-1")
        assert cfg["configurable"]["thread_id"] == "sess-1"
        assert cfg["recursion_limit"] >= 25

    def test_uuid_fits(self) -> None:
        import uuid

        thread_config(str(uuid.uuid4()))  # must not raise

    def test_over_long_id_is_rejected(self) -> None:
        """The Postgres saver keys on thread_id; >255 chars breaks it."""
        with pytest.raises(ValueError) as exc:
            thread_config("x" * 256)
        assert "255" in str(exc.value)

    def test_exactly_255_is_accepted(self) -> None:
        thread_config("x" * 255)


class TestHolderLifecycle:
    @pytest.mark.asyncio
    async def test_start_reports_false_when_backend_missing(self) -> None:
        """A missing package must degrade, not raise."""
        holder = CheckpointerHolder("postgresql://x/y")
        with patch.dict(
            sys.modules,
            {
                "langgraph.checkpoint.postgres.aio": None,
                "psycopg_pool": None,
            },
        ):
            ok = await holder.start()
        assert ok is False
        assert holder.available is False
        assert holder.saver is None

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_never_started(self) -> None:
        """A startup failure must not turn into a shutdown crash."""
        holder = CheckpointerHolder("postgresql://x/y")
        await holder.stop()  # must not raise
        await holder.stop()

    @pytest.mark.asyncio
    async def test_start_failure_cleans_up_pool(self) -> None:
        """A half-open pool must be closed, or the process leaks connections."""
        holder = CheckpointerHolder("postgresql://x/y")
        pool = AsyncMock()
        pool.open = AsyncMock(side_effect=RuntimeError("cannot connect"))

        fake_pool_cls = MagicMock(return_value=pool)
        fake_saver = MagicMock()
        fake_saver.setup = AsyncMock()

        with patch("psycopg_pool.AsyncConnectionPool", fake_pool_cls), patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            MagicMock(return_value=fake_saver),
        ):
            ok = await holder.start()

        assert ok is False
        pool.close.assert_awaited(), "the partially-created pool must be closed"
        assert holder.available is False

    @pytest.mark.asyncio
    async def test_successful_start_creates_tables_and_returns_saver(self) -> None:
        holder = CheckpointerHolder("postgresql://x/y")
        pool = AsyncMock()
        pool.open = AsyncMock()
        pool.close = AsyncMock()
        saver = MagicMock()
        saver.setup = AsyncMock()

        with patch("psycopg_pool.AsyncConnectionPool", MagicMock(return_value=pool)), patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            MagicMock(return_value=saver),
        ):
            ok = await holder.start()

        assert ok is True
        assert holder.available is True
        assert holder.saver is saver
        saver.setup.assert_awaited(), "setup() must run to create the tables"

    @pytest.mark.asyncio
    async def test_stop_clears_the_saver(self) -> None:
        holder = CheckpointerHolder("postgresql://x/y")
        pool = AsyncMock()
        pool.open = AsyncMock()
        pool.close = AsyncMock()
        saver = MagicMock()
        saver.setup = AsyncMock()

        with patch("psycopg_pool.AsyncConnectionPool", MagicMock(return_value=pool)), patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            MagicMock(return_value=saver),
        ):
            await holder.start()
            await holder.stop()

        assert holder.available is False
        pool.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_stop_swallows_close_errors(self) -> None:
        """Shutdown must not fail because the pool was already gone."""
        holder = CheckpointerHolder("postgresql://x/y")
        pool = AsyncMock()
        pool.open = AsyncMock()
        pool.close = AsyncMock(side_effect=RuntimeError("already closed"))
        saver = MagicMock()
        saver.setup = AsyncMock()

        with patch("psycopg_pool.AsyncConnectionPool", MagicMock(return_value=pool)), patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            MagicMock(return_value=saver),
        ):
            await holder.start()
            await holder.stop()  # must not raise

        assert holder.available is False


class TestPoolKwargs:
    @pytest.mark.asyncio
    async def test_pool_is_opened_with_required_kwargs(self) -> None:
        """AsyncPostgresSaver requires autocommit + prepare_threshold=0.

        The pool is created with ``open=False`` and opened explicitly so the
        connection error surfaces as a return value rather than an
        unhandled task exception.
        """
        holder = CheckpointerHolder("postgresql://x/y")
        pool = AsyncMock()
        pool.open = AsyncMock()
        pool.close = AsyncMock()
        saver = MagicMock()
        saver.setup = AsyncMock()
        pool_cls = MagicMock(return_value=pool)

        with patch("psycopg_pool.AsyncConnectionPool", pool_cls), patch(
            "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
            MagicMock(return_value=saver),
        ):
            await holder.start()

        kwargs = pool_cls.call_args.kwargs
        assert kwargs["kwargs"]["autocommit"] is True
        assert kwargs["kwargs"]["prepare_threshold"] == 0
        assert kwargs["open"] is False
        pool.open.assert_awaited()
