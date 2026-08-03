"""Short-term memory — Redis-backed session store with TTL.

Replaces the in-memory ``_sessions`` dict in orchestrator/router.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.memory.models import ConversationTurn, SessionMemory

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_SESSION_TTL = timedelta(hours=1)
_MAX_CONVERSATION_TURNS = 50


class ShortTermMemory:
    """Redis-backed short-term memory for active sessions.

    Each session is stored as a JSON blob with a TTL of 1 hour,
    automatically extended on every read/write.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._prefix = "session:"

    # ── Session lifecycle ────────────────────────────────────

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMemory:
        """Create a new session and store it in Redis."""
        session = SessionMemory(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {},
        )
        await self._save(session)
        logger.debug("Created session %s for user %s", session_id, user_id)
        return session

    async def get_session(self, session_id: str) -> SessionMemory | None:
        """Retrieve a session by ID. Returns None if not found or expired."""
        raw = await self._redis.get(f"{self._prefix}{session_id}")
        if not raw:
            return None
        try:
            data = json.loads(raw)
            session = SessionMemory(**data)
            # Extend TTL on access
            await self._redis.expire(f"{self._prefix}{session_id}", int(_SESSION_TTL.total_seconds()))
            return session
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Corrupt session data for %s: %s", session_id, exc)
            return None

    async def delete_session(self, session_id: str) -> None:
        """Remove a session from Redis."""
        await self._redis.delete(f"{self._prefix}{session_id}")
        logger.debug("Deleted session %s", session_id)

    async def delete_all_sessions_by_user(self, user_id: str) -> int:
        """Delete all sessions belonging to a user.

        Uses Redis SCAN to find matching sessions.  For MVP this is a
        best-effort scan — on large datasets it may not be fully atomic.
        Returns the count of deleted sessions.
        """
        deleted = 0
        cursor = 0
        pattern = f"{self._prefix}*"
        try:
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                if not keys:
                    break
                # Check each session's user_id
                for key in keys:
                    raw = await self._redis.get(key)
                    if raw is None:
                        continue
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and data.get("user_id") == user_id:
                            await self._redis.delete(key)
                            deleted += 1
                    except (json.JSONDecodeError, TypeError):
                        continue
                if cursor == 0:
                    break
        except Exception as exc:
            logger.warning("Failed to scan/delete sessions for %s: %s", user_id, exc)
        if deleted:
            logger.info("Deleted %d Redis sessions for user %s", deleted, user_id)
        return deleted

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists and is not expired."""
        return await self._redis.exists(f"{self._prefix}{session_id}") > 0

    # ── Conversation management ──────────────────────────────

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> SessionMemory | None:
        """Append a conversation turn and persist."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.conversation.append(
            ConversationTurn(role=role, content=content)
        )
        # Trim to max turns
        if len(session.conversation) > _MAX_CONVERSATION_TURNS:
            session.conversation = session.conversation[-_MAX_CONVERSATION_TURNS:]

        session.updated_at = datetime.utcnow()
        await self._save(session)
        return session

    async def get_conversation(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[ConversationTurn]:
        """Get recent conversation turns."""
        session = await self.get_session(session_id)
        if not session:
            return []
        turns = session.conversation
        if last_n:
            turns = turns[-last_n:]
        return turns

    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
    ) -> SessionMemory | None:
        """Merge metadata into the session."""
        session = await self.get_session(session_id)
        if not session:
            return None
        session.metadata.update(metadata)
        session.updated_at = datetime.utcnow()
        await self._save(session)
        return session

    # ── Internal ─────────────────────────────────────────────

    async def _save(self, session: SessionMemory) -> None:
        key = f"{self._prefix}{session.session_id}"
        raw = session.model_dump_json()
        await self._redis.setex(key, int(_SESSION_TTL.total_seconds()), raw)

    async def clear_all(self) -> None:
        """Clear all sessions (for testing)."""
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor, match=f"{self._prefix}*")
            if keys:
                await self._redis.delete(*keys)
            if cursor == 0:
                break


    @classmethod
    def _in_memory_fallback(cls) -> ShortTermMemory:
        """Create a fallback ShortTermMemory with no Redis backend.

        Uses a plain dict. Not persistent across restarts — for development only.
        """
        import uuid
        _fallback_store: dict[str, str] = {}
        _fallback_ttl: dict[str, datetime] = {}

        _fake_redis_type = type('_FakeRedis', (), {})

        class _FakeRedis:
            async def get(self, key):
                if key in _fallback_ttl and datetime.utcnow() > _fallback_ttl[key]:
                    _fallback_store.pop(key, None)
                    _fallback_ttl.pop(key, None)
                    return None
                return _fallback_store.get(key)

            async def setex(self, key, ttl, value):
                _fallback_store[key] = value
                _fallback_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)

            async def expire(self, key, ttl):
                if key in _fallback_store:
                    _fallback_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)

            async def delete(self, *keys):
                for key in keys:
                    _fallback_store.pop(key, None)
                    _fallback_ttl.pop(key, None)

            async def exists(self, key):
                return 1 if key in _fallback_store else 0

            async def scan(self, cursor, match="*"):
                import fnmatch
                matching = [k for k in _fallback_store if fnmatch.fnmatch(k, match)]
                return 0, matching

        return cls(_FakeRedis())
