"""Long-term memory — PostgreSQL-backed episodic + semantic memory store."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Sequence

from src.memory import recall_scoring
from src.memory.models import EpisodicEntry, EventType, MemoryType, SemanticEntry

if TYPE_CHECKING:
    from src.memory.db import MemoryDBPool

logger = logging.getLogger(__name__)

# How many rows to pull from SQL per requested result before ranking.
# Ranking happens in Python (see recall_scoring), so the candidate window
# must be wider than ``limit`` — otherwise the newest ``limit`` rows are
# chosen before scoring and an older-but-important entry can never win.
# 5× was chosen so that in a 20-result recall an entry up to ~100 positions
# back in time can still surface; the cost is a bounded, indexed fetch.
_RECALL_CANDIDATE_FACTOR = 5


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
                    (user_id, session_id, event_type, input, output, metadata,
                     importance_score, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                entry.user_id,
                entry.session_id,
                entry.event_type,
                entry.input,
                entry.output,
                json.dumps(entry.metadata),
                entry.importance_score,
                json.dumps(entry.embedding) if entry.embedding is not None else None,
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
        """Recall episodic memories, ranked by recency × importance × relevance.

        Historically this was a plain ``ORDER BY created_at DESC``.  That
        ignored ``importance_score`` (which was already being persisted) and
        so surfaced "most recent" rather than "most worth recalling".  See
        :mod:`src.memory.recall_scoring` for the formula and rationale.

        Because the ranking now happens in Python, the SQL fetches a wider
        window than ``limit`` — otherwise the newest ``limit`` rows would be
        selected before scoring, and an older-but-important entry could
        never win.  The window is capped by ``_RECALL_CANDIDATE_FACTOR``.

        Paging note: the SQL query deliberately does **not** apply ``offset``.
        The ranking happens here, so paging must too — applying ``OFFSET`` at
        the SQL level and then taking ``ranked[offset:offset+limit]`` would
        overlap pages (rows ``offset..offset+candidate_limit`` are scored, and
        the top of that window is returned again on the next page) while
        skipping rows entirely. The window is grown to cover ``offset + limit``
        instead.
        """
        # Fetch enough rows to cover the requested page *and* leave room for
        # ranking to promote older entries.
        window = max(limit, (offset + limit) * _RECALL_CANDIDATE_FACTOR)

        async with self._db.pool.acquire() as conn:
            if event_types:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, embedding, created_at
                    FROM episodic_memory
                    WHERE user_id = $1 AND event_type = ANY($2::varchar[])
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    [t.value if isinstance(t, EventType) else t for t in event_types],
                    window,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, embedding, created_at
                    FROM episodic_memory
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    window,
                )
            candidates = [_row_to_episodic(r) for r in rows]

        ranked = recall_scoring.score_episodic_entries(candidates)
        entries = [entry for entry, _score in ranked]
        return entries[offset : offset + limit]

    async def recall_relevant(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        event_types: list[EventType | str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicEntry]:
        """Recall episodic memories ranked by **semantic relevance** to a query.

        Distinct from :meth:`recall_episodic`, which serves *conversation
        history* and therefore wants chronological continuity.  This method
        answers a different question — "which past interactions are related to
        what the user is asking now?" — and so weights relevance equally with
        recency and importance.

        Entries whose embedding is NULL (written before the ``embedding``
        column existed) score relevance 0 and can still be recalled on
        recency + importance alone.

        Args:
            user_id: whose history to search.
            query_embedding: vector of the current query/prompt.
            event_types: optional filter, as in ``recall_episodic``.
            limit: number of entries to return.
        """
        candidate_limit = max(limit, limit * _RECALL_CANDIDATE_FACTOR)

        async with self._db.pool.acquire() as conn:
            if event_types:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, embedding, created_at
                    FROM episodic_memory
                    WHERE user_id = $1 AND event_type = ANY($2::varchar[])
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    [t.value if isinstance(t, EventType) else t for t in event_types],
                    candidate_limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, session_id, event_type, input, output,
                           metadata, importance_score, embedding, created_at
                    FROM episodic_memory
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    candidate_limit,
                )
            candidates = [_row_to_episodic(r) for r in rows]

        # index entry embeddings by id so the scorer can pair them up
        embeddings = {
            e.id: e.embedding for e in candidates if e.id and e.embedding
        }
        ranked = recall_scoring.score_episodic_entries(
            candidates,
            query_embedding=query_embedding,
            entry_embeddings=embeddings,
        )
        return [entry for entry, _score in ranked[:limit]]

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

def _parse_embedding(raw) -> list[float] | None:
    """Decode an ``embedding`` column value.

    asyncpg returns JSONB as a str (or already-decoded list, depending on
    codec registration), so both shapes are accepted. A malformed or absent
    value yields ``None`` — callers then score that row as relevance 0 rather
    than failing the whole recall.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(parsed, list):
        return [float(x) for x in parsed]
    return None


def _row_to_episodic(row) -> EpisodicEntry:
    # ``row`` is either an asyncpg Record or a plain dict (tests). Record
    # supports `in` / indexing but not ``.get``, so probe membership instead
    # of relying on a dict-only API.
    raw_embedding = row["embedding"] if "embedding" in row else None
    return EpisodicEntry(
        id=str(row["id"]),
        user_id=row["user_id"],
        session_id=row["session_id"],
        event_type=row["event_type"],
        input=row["input"],
        output=row["output"],
        metadata=row["metadata"] if isinstance(row["metadata"], dict) else json.loads(row["metadata"] or "{}"),
        importance_score=row["importance_score"],
        embedding=_parse_embedding(raw_embedding),
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
