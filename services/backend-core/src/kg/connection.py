"""Neo4j async connection pool — compatible with neo4j 6.x driver.

Uses ``session.run()`` directly instead of ``execute_read/execute_write``
to remain compatible with the neo4j 6.x async API changes.
"""

from __future__ import annotations

import logging

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jPool:
    """Async connection pool wrapping a Neo4j driver instance."""

    def __init__(self, uri: str, user: str, password: str, max_connections: int = 50):
        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connections,
        )
        logger.info("Neo4jPool created (uri=%s, pool_size=%d)", uri, max_connections)

    async def close(self) -> None:
        await self._driver.close()
        logger.info("Neo4jPool closed")

    async def execute_read(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        """Execute a read-only Cypher query and return the result."""
        return await self._run_query(query, params or {})

    async def execute_write(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        """Execute a write Cypher query and return the result."""
        return await self._run_query(query, params or {})

    async def _run_query(self, query: str, params: dict) -> list[dict]:
        """Run a Cypher query and return the result as a list of dicts.

        Note: The key ``query`` is reserved for the Cypher string and must
        not appear in ``params`` (would cause ``session.run(query, query=...)``).
        """
        if "query" in params:
            logger.warning(
                "Param key 'query' is reserved — stripping from params. "
                "Rename the variable in the Cypher query (e.g. $query → $search_term)."
            )
            params = {k: v for k, v in params.items() if k != "query"}
        try:
            async with self._driver.session() as session:
                result = await session.run(query, **params)
                return await result.data()
        except Exception:
            logger.exception("Neo4j query failed: %s", query[:120])
            raise
