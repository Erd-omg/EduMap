"""Neo4j async connection pool."""

import logging

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jPool:
    """Async connection pool wrapping a Neo4j driver instance.

    Usage::

        pool = Neo4jPool(uri, user, password)
        result = await pool.execute_read("MATCH (n) RETURN n")
        await pool.close()
    """

    def __init__(self, uri: str, user: str, password: str, max_connections: int = 50):
        self._driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connections,
        )
        logger.info("Neo4jPool created (uri=%s, pool_size=%d)", uri, max_connections)

    async def close(self) -> None:
        """Close the driver and release all connections."""
        await self._driver.close()
        logger.info("Neo4jPool closed")

    async def execute_read(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        """Execute a read-only Cypher query and return the result as a list of dicts."""
        try:
            async with self._driver.session() as session:
                result = await session.execute_read(
                    lambda tx: tx.run(query, parameters=params or {}).data()
                )
                return result
        except Exception:
            logger.exception("Neo4j read query failed: %s", query[:120])
            raise

    async def execute_write(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
        """Execute a write Cypher query and return the result as a list of dicts."""
        try:
            async with self._driver.session() as session:
                result = await session.execute_write(
                    lambda tx: tx.run(query, parameters=params or {}).data()
                )
                return result
        except Exception:
            logger.exception("Neo4j write query failed: %s", query[:120])
            raise
