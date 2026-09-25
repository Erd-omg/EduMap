"""Durable graph state for the orchestrator (I-6).

Motivation
----------
The orchestrator's progress lived only in Redis, written by hand in
``router.py``. That code re-implements LangGraph's reducer semantics by
consulting ``_STATE_REDUCERS`` node-update by node-update — a workaround whose
own comments record the bug it caused: a hand-rolled last-write-wins merge
silently dropped one parallel branch's resources.

A checkpointer removes the need for that reconstruction. LangGraph persists
each super-step itself, applying the graph's declared reducers, so:

* **Live progress** can be read back from the checkpoint rather than rebuilt.
* **Crash recovery** becomes possible — a run can resume from the last
  completed super-step instead of restarting.
* **Time travel** (inspecting a past state) comes for free.

Design
------
This module owns *only* pool lifecycle and compilation. Deciding what to do
with the persisted state stays in the router, which is why the router keeps
writing its Redis snapshot during the transition (see ``dual-write`` in
``create_graph``'s docstring).

Failure policy: if the checkpoint store cannot be created, the graph is
compiled **without** one and a warning is logged. A working orchestrator with
ephemeral state beats a service that refuses to start; the Redis path still
provides progress visibility in that case.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tables are created on first use. Kept as a module constant so tests can
# assert the setup step actually ran.
_SETUP_REQUIRED = True


class CheckpointerHolder:
    """Owns the async Postgres checkpointer and its connection pool.

    A ``AsyncPostgresSaver`` is backed by a psycopg connection pool that must
    outlive the graph, so it is held here rather than created per-request.

    Usage::

        holder = CheckpointerHolder(dsn)
        await holder.start()          # creates pool + tables
        graph = builder.compile(checkpointer=holder.saver)
        ...
        await holder.stop()
    """

    def __init__(self, dsn: str, *, min_size: int = 1, max_size: int = 5) -> None:
        self._dsn = _normalise_dsn(dsn)
        self._min_size = min_size
        self._max_size = max_size
        self._pool: Any = None
        self._saver: Any = None

    @property
    def saver(self) -> Any:
        """The checkpointer to hand to ``compile()`` — None when unavailable."""
        return self._saver

    @property
    def available(self) -> bool:
        return self._saver is not None

    async def start(self) -> bool:
        """Create the pool and run table setup.

        Returns True when the checkpointer is usable. Never raises: callers get
        ``False`` and compile without a checkpointer rather than failing to
        start the service.
        """
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:
            logger.warning(
                "Checkpointer backend unavailable (%s) — graph state will be "
                "ephemeral. Install langgraph-checkpoint-postgres and "
                "psycopg[binary].",
                exc,
            )
            return False

        try:
            # `autocommit=True` and `prepare_threshold=0` are required by
            # AsyncPostgresSaver; the pool asserts on them at startup.
            self._pool = AsyncConnectionPool(
                conninfo=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            await self._pool.open(wait=True, timeout=10)
            self._saver = AsyncPostgresSaver(self._pool)
            if _SETUP_REQUIRED:
                await self._saver.setup()
            logger.info("Orchestrator checkpointer ready (Postgres)")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to initialise the checkpointer (%s) — continuing with "
                "ephemeral graph state. Redis progress reporting is unaffected.",
                exc,
            )
            await self.stop()
            return False

    async def stop(self) -> None:
        """Close the pool. Safe to call when nothing was opened."""
        pool, self._pool, self._saver = self._pool, None, None
        if pool is None:
            return
        try:
            await pool.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Error closing checkpointer pool: %s", exc)


def _normalise_dsn(dsn: str) -> str:
    """Convert a SQLAlchemy-style DSN to what psycopg expects.

    ``settings.database_url`` is ``postgresql+asyncpg://...`` (asyncpg-only
    syntax). psycopg rejects the ``+asyncpg`` scheme, so it is stripped.
    """
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    if dsn.startswith("postgresql+psycopg://"):
        return dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return dsn


def thread_config(session_id: str) -> dict:
    """Build the ``config`` passed to graph invocation.

    ``thread_id`` is the checkpointer's primary key. ``recursion_limit`` is
    raised above the default because the generation graph legitimately revisits
    nodes through the retry loop.

    Note: LangGraph's own troubleshooting docs warn that ``thread_id`` must
    stay under 255 characters for the Postgres saver. Session ids are UUIDs, so
    this holds — but the guard is here so a future change cannot silently break
    persistence.
    """
    if len(session_id) > 255:
        raise ValueError(
            f"thread_id must be < 255 chars (got {len(session_id)}); the "
            "Postgres checkpoint saver uses it as a key column"
        )
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
