"""Test helpers shared across suites (not a package — import via path).

``new_async_redis()`` returns an in-memory ``redis.asyncio`` client
(``fakeredis``) with ``decode_responses=True`` — the same shape ``main.py``
builds against a real Redis.  It implements ``GETDEL``, so tests exercise the
store's *atomic* single-use path rather than a mock that happens to behave.
"""

from __future__ import annotations


def new_async_redis():
    import fakeredis.aioredis

    return fakeredis.aioredis.FakeRedis(decode_responses=True)
