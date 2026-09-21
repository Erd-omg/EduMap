"""A small thread-safe-by-construction LRU cache with per-entry TTL.

Extracted from ``src/analysis/intent.py`` where it backed the intent cache, so
that the RAG retrieval cache (``src/rag/rag_service.py``) and the tool-result
cache (``src/tools/registry.py``) can share one implementation instead of each
hand-rolling an ``OrderedDict``.

Scope notes:

- Intended for single-process use.  The asyncio event loop is single-threaded,
  and every current caller runs on it, so no lock is taken; entries are plain
  values and each method is atomic with respect to the loop.
- Expiry is lazy: an expired entry is dropped when it is next read, and eviction
  order is refreshed on both ``get`` and ``put`` (so a hot entry survives).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLLRUCache(Generic[K, V]):
    """LRU cache with a per-entry time-to-live.

    Args:
        maxsize: Maximum number of live entries before LRU eviction kicks in.
        ttl_seconds: Lifetime of an entry, measured from the moment it is put.
        clock: Monotonic time source; injectable for tests.
    """

    def __init__(
        self,
        maxsize: int = 512,
        ttl_seconds: float = 600.0,
        clock=time.monotonic,
    ) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: OrderedDict[K, tuple[V, float]] = OrderedDict()

    def get(self, key: K) -> V | None:
        """Return the live value for ``key``, or None if missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if self._clock() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        """Insert or refresh ``key``, evicting the least-recently-used if over capacity."""
        self._store[key] = (value, self._clock() + self._ttl)
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def get_or_set(self, key: K, factory) -> V:
        """Return the cached value, computing and storing it via ``factory()`` on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.put(key, value)
        return value

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        """Number of entries held, including ones that have not yet expired."""
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None  # type: ignore[arg-type]

    def stats(self) -> dict[str, Any]:
        """Small snapshot for logging/tests."""
        return {"size": len(self._store), "maxsize": self._maxsize, "ttl": self._ttl}
