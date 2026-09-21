"""Tests for the shared TTL+LRU cache (src/utils/ttl_cache.py)."""

from __future__ import annotations

import pytest

from src.utils.ttl_cache import TTLLRUCache


class _FakeClock:
    """Controllable monotonic clock so TTL tests never sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTTLLRUCache:
    def test_get_returns_none_for_missing_key(self) -> None:
        cache: TTLLRUCache[str, int] = TTLLRUCache()
        assert cache.get("absent") is None

    def test_put_then_get(self) -> None:
        cache: TTLLRUCache[str, int] = TTLLRUCache()
        cache.put("a", 1)
        assert cache.get("a") == 1
        assert len(cache) == 1

    def test_entry_expires_after_ttl(self) -> None:
        clock = _FakeClock()
        cache: TTLLRUCache[str, int] = TTLLRUCache(ttl_seconds=10.0, clock=clock)
        cache.put("a", 1)

        clock.advance(9.9)
        assert cache.get("a") == 1, "must still be live just before the TTL"

        clock.advance(0.2)  # now past 10s
        assert cache.get("a") is None, "must expire once the TTL elapses"

    def test_expired_entry_is_physically_dropped(self) -> None:
        clock = _FakeClock()
        cache: TTLLRUCache[str, int] = TTLLRUCache(ttl_seconds=5.0, clock=clock)
        cache.put("a", 1)
        assert len(cache) == 1

        clock.advance(6.0)
        cache.get("a")  # triggers lazy eviction
        assert len(cache) == 0

    def test_get_refreshes_recency(self) -> None:
        """A read moves the key to the MRU end, so it survives eviction."""
        cache: TTLLRUCache[str, int] = TTLLRUCache(maxsize=2, ttl_seconds=600.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")          # "a" is now most-recently-used
        cache.put("c", 3)       # evicts the LRU entry, which should be "b"

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_put_refreshes_existing_key(self) -> None:
        clock = _FakeClock()
        cache: TTLLRUCache[str, int] = TTLLRUCache(ttl_seconds=10.0, clock=clock)
        cache.put("a", 1)
        clock.advance(8.0)
        cache.put("a", 2)  # resets the TTL window
        clock.advance(8.0)  # 16s after the first put, 8s after the second
        assert cache.get("a") == 2
        assert len(cache) == 1, "re-putting a key must not create a second entry"

    def test_lru_eviction_beyond_maxsize(self) -> None:
        cache: TTLLRUCache[str, int] = TTLLRUCache(maxsize=3, ttl_seconds=600.0)
        for i in range(10):
            cache.put(f"k{i}", i)
        assert len(cache) == 3
        # Only the three most recent survive.
        assert cache.get("k9") == 9
        assert cache.get("k0") is None

    def test_clear_empties_cache(self) -> None:
        cache: TTLLRUCache[str, int] = TTLLRUCache()
        cache.put("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_contains_ignores_expired(self) -> None:
        clock = _FakeClock()
        cache: TTLLRUCache[str, int] = TTLLRUCache(ttl_seconds=5.0, clock=clock)
        cache.put("a", 1)
        assert "a" in cache
        clock.advance(6.0)
        assert "a" not in cache

    def test_get_or_set_computes_only_on_miss(self) -> None:
        cache: TTLLRUCache[str, int] = TTLLRUCache()
        calls = {"n": 0}

        def factory() -> int:
            calls["n"] += 1
            return 42

        assert cache.get_or_set("k", factory) == 42
        assert cache.get_or_set("k", factory) == 42
        assert calls["n"] == 1, "factory must run once; the second call is a hit"

    def test_invalid_construction_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TTLLRUCache(maxsize=0)
        with pytest.raises(ValueError):
            TTLLRUCache(ttl_seconds=0)
