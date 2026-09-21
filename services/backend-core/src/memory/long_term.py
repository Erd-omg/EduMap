"""Long-term memory — PostgreSQL-backed episodic + semantic memory store."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from src.memory.models import EpisodicEntry, EventType, MemoryType, SemanticEntry

if TYPE_CHECKING:
    from src.memory.db import MemoryDBPool

logger = logging.getLogger(__name__)


class LongTermMemory:
    """PostgreSQL-backed long-term memory for persistent episodic and semantic storage.

    Episodic memory stores individual interaction records.
    Semantic memory stores distilled knowledge about users and the domain.
    """

    def __init__(self, db: MemoryDBPool) -> None:
        self._db = db

    # ── Episodic Memory ──────────────────────────────────────

    async def write_episodic(self, entry: EpisodicEntry) -> str:
        """Write an episodic memory record. Returns the new record ID."""
        async with self._db.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO episodic_memory
                    (user_id, session_id, event_type, input, output, metadata, importance_score)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                entry.user_id,
                entry.session_id,
                entry.event_type,
                entry.input,
                entry.output,
                json.dumps(entry.metadata),
                entry.importance_score,
            )
            record_id = str(row["id"])
            logger.debug("Wrote episodic memory %s for user %s", record_id, entry.user_id)
            return record_id

    async def recall_episodic(
        self,
        user_id: str,
        event_types: list[EventType | str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodicEntry]:
        """Recall recent episodic memories for a user, optionally filtered by type."""
        async with self._db.pool.acquire() as conn:
            if event_types:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, created_at
                    FROM episodic_memory
                    WHERE user_id = $1 AND event_type = ANY($2::varchar[])
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    user_id,
                    [t.value if isinstance(t, EventType) else t for t in event_types],
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, created_at
                    FROM episodic_memory
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_id,
                    limit,
                    offset,
                )
            return [_row_to_episodic(r) for r in rows]

    async def update_episodic_importance(
        self, record_id: str, importance: float,
    ) -> None:
        """Update importance score for an episodic record."""
        async with self._db.pool.acquire() as conn:
            await conn.execute(
                "UPDATE episodic_memory SET importance_score = $1 WHERE id = $2",
                importance, record_id,
            )

    async def delete_episodic(self, record_id: str) -> None:
        """Delete an episodic memory record."""
        async with self._db.pool.acquire() as conn:
            await conn.execute("DELETE FROM episodic_memory WHERE id = $1", record_id)

    # ── Semantic Memory ──────────────────────────────────────

    async def write_semantic(self, entry: SemanticEntry) -> str:
        """Upsert a semantic memory entry. Returns the record ID."""
        async with self._db.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO semantic_memory
                    (user_id, memory_type, key, value, confidence)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, memory_type, key)
                DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence,
                    updated_at = NOW()
                RETURNING id
                """,
                entry.user_id,
                entry.memory_type.value if isinstance(entry.memory_type, MemoryType) else entry.memory_type,
                entry.key,
                json.dumps(entry.value),
                entry.confidence,
            )
            return str(row["id"])

    async def recall_semantic(
        self,
        user_id: str,
        keys: list[str] | None = None,
        memory_types: list[str] | None = None,
    ) -> list[SemanticEntry]:
        """Recall semantic memories for a user."""
        async with self._db.pool.acquire() as conn:
            conditions = ["user_id = $1"]
            params: list[Any] = [user_id]
            param_idx = 2

            if keys:
                conditions.append(f"key = ANY(${param_idx}::varchar[])")
                params.append(keys)
                param_idx += 1
            if memory_types:
                conditions.append(f"memory_type = ANY(${param_idx}::varchar[])")
                params.append(memory_types)
                param_idx += 1

            query = f"""
                SELECT id, user_id, memory_type, key, value, confidence, created_at, updated_at
                FROM semantic_memory
                WHERE {' AND '.join(conditions)}
                ORDER BY updated_at DESC
            """
            rows = await conn.fetch(query, *params)
            return [_row_to_semantic(r) for r in rows]

    async def delete_semantic(self, record_id: str) -> None:
        """Delete a semantic memory entry."""
        async with self._db.pool.acquire() as conn:
            await conn.execute("DELETE FROM semantic_memory WHERE id = $1", record_id)

    async def delete_all_by_user(self, user_id: str) -> int:
        """Delete ALL episodic and semantic memory records for a user.

        Returns the total number of deleted rows.
        """
        total = 0
        async with self._db.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM episodic_memory WHERE user_id = $1",
                user_id,
            )
            total += _parse_delete_count(result)
            result = await conn.execute(
                "DELETE FROM semantic_memory WHERE user_id = $1",
                user_id,
            )
            total += _parse_delete_count(result)
        logger.info("Deleted %d memory records for user %s", total, user_id)
        return total


# ── Helpers ──────────────────────────────────────────────────

def _row_to_episodic(row) -> EpisodicEntry:
    return EpisodicEntry(
        id=str(row["id"]),
        user_id=row["user_id"],
        session_id=row["session_id"],
        event_type=row["event_type"],
        input=row["input"],
        output=row["output"],
        metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        importance_score=row["importance_score"],
        created_at=row["created_at"],
    )


def _parse_delete_count(result: str) -> int:
    """Parse the count from an asyncpg DELETE result string like 'DELETE 3'."""
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


def _row_to_semantic(row) -> SemanticEntry:
    return SemanticEntry(
        id=str(row["id"]),
        user_id=row["user_id"],
        memory_type=row["memory_type"],
        key=row["key"],
        value=row["value"] if isinstance(row["value"], dict) else json.loads(row["value"] or "{}"),
        confidence=row["confidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
