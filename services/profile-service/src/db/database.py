import asyncpg
from typing import Any


class DatabasePool:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.dsn, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        if not self._pool:
            raise RuntimeError("Database pool not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        if not self._pool:
            raise RuntimeError("Database pool not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        if not self._pool:
            raise RuntimeError("Database pool not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)
