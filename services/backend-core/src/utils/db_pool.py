"""Helpers for accepting an asyncpg pool in a tolerant-but-loud way."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def unwrap_pool(db_pool: Any, *, owner: str = "service") -> Any:
    """Normalise a ``db_pool`` argument to something exposing ``acquire()``.

    Two shapes reach our constructors in practice:

    * a raw ``asyncpg.Pool`` — what ``src/main.py`` passes today, via
      ``memory_db_pool.pool``;
    * a ``MemoryDBPool`` wrapper — which has **no** ``acquire()``, only ``.pool``.

    Passing the wrapper is a real trap: every call site wraps acquisition in
    ``except Exception`` (correctly — a DB blip must not break the live
    feature), so persistence silently degrades to a no-op while the in-memory
    path keeps working.  Nothing surfaces until someone notices that restarting
    the process loses state that should have been durable.

    Rather than depend on every caller unwrapping correctly, accept both and
    unwrap here; report anything else **loudly at construction time**, where it
    can actually be fixed, instead of swallowing it per-query.

    Args:
        db_pool: the candidate pool, or None.
        owner: caller name, for the error message.

    Returns:
        An object exposing ``acquire()``, or None when *db_pool* is None.

    Raises:
        TypeError: when the argument is neither a usable pool nor a None.
    """
    if db_pool is None:
        return None
    if hasattr(db_pool, "acquire"):
        return db_pool

    inner = getattr(db_pool, "pool", None)
    if inner is not None and hasattr(inner, "acquire"):
        logger.debug(
            "%s: unwrapped a pool wrapper (%s) to its .pool",
            owner,
            type(db_pool).__name__,
        )
        return inner

    raise TypeError(
        f"{owner}: db_pool must expose acquire() (an asyncpg.Pool) or wrap one "
        f"in .pool; got {type(db_pool).__name__} with neither. Persistence "
        "would silently no-op if this were swallowed."
    )
