"""ResourceRepository — PostgreSQL-backed CRUD for resource management.

Replaces the in-memory ``_resources: dict`` in ``router.py``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from uuid import UUID
from typing import Any

from src.resources.models import ResourceMetadata

logger = logging.getLogger(__name__)


class ResourceRepository:
    """Async PostgreSQL repository for the ``resources`` table."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    async def create(self, resource: ResourceMetadata) -> ResourceMetadata:
        """INSERT a new resource and return it with DB-assigned fields."""
        resource_id = resource.id or str(uuid.uuid4())
        # asyncpg requires a datetime for TIMESTAMPTZ columns; the model
        # carries created_at as an ISO string for JSON serialization.
        created_at = (
            datetime.fromisoformat(resource.created_at)
            if isinstance(resource.created_at, str) and resource.created_at
            else resource.created_at
        )
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO resources
                    (id, user_id, name, type, source, kp_id, kp_name,
                     file_size, file_path, description, created_at,
                     parse_status, parse_stats)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    kp_id = EXCLUDED.kp_id,
                    kp_name = EXCLUDED.kp_name,
                    parse_status = EXCLUDED.parse_status,
                    parse_stats = EXCLUDED.parse_stats,
                    updated_at = NOW()
                RETURNING id, user_id, name, type, source, kp_id, kp_name,
                          file_size, file_path, description, created_at,
                          parse_status, parse_stats
                """,
                resource_id,
                resource.user_id,
                resource.name,
                resource.type,
                resource.source,
                resource.kp_id,
                resource.kp_name,
                resource.file_size,
                resource.file_path,
                resource.description,
                created_at,
                resource.parse_status,
                _serialize_stats(resource.parse_stats),
            )
        return _row_to_metadata(row)

    async def get(self, resource_id: str) -> ResourceMetadata | None:
        """SELECT a single resource by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM resources WHERE id = $1",
                resource_id,
            )
        return _row_to_metadata(row) if row else None

    async def list(
        self,
        user_id: str | None = None,
        type_filter: str | None = None,
        source_filter: str | None = None,
        kp_id: str | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[ResourceMetadata], int]:
        """SELECT resources with optional filters, returning (items, total)."""
        conditions: list[str] = []
        params: list[Any] = []
        idx = 1

        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(user_id)
            idx += 1
        if type_filter:
            conditions.append(f"type = ${idx}")
            params.append(type_filter)
            idx += 1
        if source_filter:
            conditions.append(f"source = ${idx}")
            params.append(source_filter)
            idx += 1
        if kp_id:
            conditions.append(f"kp_id = ${idx}")
            params.append(kp_id)
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        async with self._pool.acquire() as conn:
            # Total count
            count_row = await conn.fetchval(
                f"SELECT COUNT(*) FROM resources WHERE {where_clause}",
                *params,
            )
            total = count_row or 0

            # Paginated results
            rows = await conn.fetch(
                f"""
                SELECT * FROM resources
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params,
                limit,
                skip,
            )

        return [_row_to_metadata(r) for r in rows], total

    async def delete(self, resource_id: str) -> ResourceMetadata | None:
        """DELETE a resource by ID and return its metadata (before deletion)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM resources WHERE id = $1 RETURNING *",
                resource_id,
            )
        return _row_to_metadata(row) if row else None

    async def sync_batch(self, resources: list[ResourceMetadata]) -> int:
        """Batch upsert of system-generated resources. Returns count synced."""
        count = 0
        async with self._pool.acquire() as conn:
            for r in resources:
                await conn.execute(
                    """
                    INSERT INTO resources
                        (id, user_id, name, type, source, kp_id, kp_name,
                         description, created_at, parse_status, parse_stats)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    r.id,
                    r.user_id,
                    r.name,
                    r.type,
                    r.source,
                    r.kp_id,
                    r.kp_name,
                    r.description,
                    r.created_at,
                    r.parse_status,
                    _serialize_stats(r.parse_stats),
                )
                count += 1
        return count

    async def delete_all_by_user(self, user_id: str) -> int:
        """Delete all resources for a user. Returns count of deleted rows."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM resources WHERE user_id = $1",
                user_id,
            )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def update_parse_status(
        self,
        resource_id: str,
        parse_status: str,
        parse_stats: dict | None = None,
    ) -> None:
        """Update parse status and stats for a resource."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE resources
                SET parse_status = $1, parse_stats = $2, updated_at = NOW()
                WHERE id = $3
                """,
                parse_status,
                _serialize_stats(parse_stats),
                resource_id,
            )


# ── Helpers ────────────────────────────────────────────────────────────────


def _serialize_stats(stats: dict | None) -> str | None:
    """Serialize parse_stats to JSON string for PostgreSQL JSONB."""
    if stats is None:
        return None
    import json
    return json.dumps(stats)


def _row_to_metadata(row) -> ResourceMetadata:
    """Convert a DB row (Record) to a ResourceMetadata instance."""
    data = dict(row)
    # Parse JSONB parse_stats
    raw_stats = data.get("parse_stats")
    if isinstance(raw_stats, str):
        import json
        try:
            data["parse_stats"] = json.loads(raw_stats)
        except (json.JSONDecodeError, TypeError):
            data["parse_stats"] = {}
    elif hasattr(raw_stats, "__getitem__"):
        # asyncpg Record or dict-like
        data["parse_stats"] = dict(raw_stats) if raw_stats else {}
    else:
        data["parse_stats"] = raw_stats or {}

    # Format datetime fields
    for key in ("created_at", "updated_at"):
        val = data.get(key)
        if val and hasattr(val, "isoformat"):
            data[key] = val.isoformat()

    # asyncpg returns uuid.UUID objects for uuid columns — model expects str
    if isinstance(data.get("id"), UUID):
        data["id"] = str(data["id"])

    # Strip binary/file columns from response model
    data.pop("updated_at", None)
    file_path = data.pop("file_path", None)

    result = ResourceMetadata(**data)
    # Restore file_path in the model (it's filtered at API layer)
    object.__setattr__(result, "file_path", file_path)
    return result
