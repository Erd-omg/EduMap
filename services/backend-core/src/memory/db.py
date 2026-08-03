"""Memory database connection pool — asyncpg wrapper.

Reuses the connection pattern from ``profile-service/src/db/database.py``.
"""

from __future__ import annotations

import logging

import asyncpg

logger = logging.getLogger(__name__)


class MemoryDBPool:
    """Async PostgreSQL connection pool for memory tables."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10) -> None:
        # Asyncpg only accepts "postgresql://" or "postgres://" scheme,
        # not "postgresql+asyncpg://" (SQLAlchemy-style DSN)
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self.pool: asyncpg.Pool | None = None

    async def create(self) -> None:
        """Initialize the connection pool."""
        self.pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
        )
        logger.info(
            "MemoryDBPool created (min=%d, max=%s)",
            self._min_size, self._max_size,
        )

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("MemoryDBPool closed")
