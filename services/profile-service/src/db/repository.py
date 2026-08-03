import json
from datetime import datetime, timezone
from typing import Any

from src.db.database import DatabasePool


class ProfileRepository:
    def __init__(self, db: DatabasePool) -> None:
        self.db = db

    async def get_profile(self, user_id: str) -> dict | None:
        """Get profile by user_id. Returns None if not found."""
        row = await self.db.fetchrow(
            "SELECT up.user_id, up.profile_data, up.version, up.updated_at "
            "FROM user_profiles up WHERE up.user_id = $1",
            user_id,
        )
        if row is None:
            return None
        profile_data = row["profile_data"]
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)
        return {
            "user_id": row["user_id"],
            "profile": profile_data,
            "version": row["version"],
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def upsert_profile(self, user_id: str, profile_data: dict) -> dict:
        """Insert or update profile. Increment version on update."""
        now = datetime.now(timezone.utc)
        row = await self.db.fetchrow(
            """INSERT INTO user_profiles (user_id, profile_data, version, updated_at)
               VALUES ($1, $2::jsonb, 1, $3)
               ON CONFLICT (user_id) DO UPDATE
               SET profile_data = EXCLUDED.profile_data,
                   version = user_profiles.version + 1,
                   updated_at = EXCLUDED.updated_at
               RETURNING user_id, version, updated_at""",
            user_id,
            json.dumps(profile_data),
            now,
        )
        return {
            "user_id": row["user_id"],
            "profile": profile_data,
            "version": row["version"],
            "updated_at": row["updated_at"].isoformat(),
        }

    async def list_profiles(self, skip: int = 0, limit: int = 100) -> list[dict]:
        rows = await self.db.fetch(
            "SELECT user_id, profile_data, version, updated_at FROM user_profiles "
            "ORDER BY updated_at DESC LIMIT $1 OFFSET $2",
            limit,
            skip,
        )
        result: list[dict] = []
        for row in rows:
            pd = row["profile_data"]
            if isinstance(pd, str):
                pd = json.loads(pd)
            result.append({
                "user_id": row["user_id"],
                "profile": pd,
                "version": row["version"],
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            })
        return result

    async def delete_profile(self, user_id: str) -> bool:
        result = await self.db.execute(
            "DELETE FROM user_profiles WHERE user_id = $1", user_id
        )
        return result != "DELETE 0"
