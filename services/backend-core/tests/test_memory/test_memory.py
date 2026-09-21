"""Comprehensive tests for the three-tier memory system.

Covers:
  - Data models (EventType, MemoryType, EpisodicEntry, SemanticEntry, …)
  - ShortTermMemory via a dict-based FakeRedis
  - LongTermMemory via AsyncMock for asyncpg pool
  - MemoryOperations with mocked short + long term components
  - MemoryDBPool DSN normalization
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.db import MemoryDBPool
from src.memory.long_term import LongTermMemory, _row_to_episodic, _row_to_semantic
from src.memory.models import (
    ConversationTurn,
    EpisodicEntry,
    EventType,
    MemoryContext,
    MemoryType,
    SemanticEntry,
    SensoryInput,
    SessionMemory,
)
from src.memory.operations import MemoryOperations
from src.memory.short_term import ShortTermMemory


# ===========================================================================
# FakeRedis — dict-based mock that mimics the Redis async interface used by
# ShortTermMemory (setex / get / delete / exists / expire / scan).
# ===========================================================================

class _FakeRedis:
    """In-memory dict-based mock of the async Redis interface.

    Tracks TTL via an internal expiration dict.  Expired keys return None on
    ``get`` and are excluded from ``exists`` / ``scan``.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._ttl: dict[str, datetime] = {}

    def _evict_expired(self) -> None:
        now = datetime.utcnow()
        expired = [k for k, t in self._ttl.items() if now > t]
        for k in expired:
            self._store.pop(k, None)
            self._ttl.pop(k, None)

    async def get(self, key: str) -> str | None:
        self._evict_expired()
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value
        self._ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)

    async def expire(self, key: str, ttl: int) -> None:
        if key in self._store:
            self._ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self._store.pop(key, None)
            self._ttl.pop(key, None)

    async def exists(self, key: str) -> int:
        self._evict_expired()
        return 1 if key in self._store else 0

    async def scan(self, cursor: int = 0, match: str = "*") -> tuple[int, list[str]]:
        import fnmatch
        self._evict_expired()
        matching = [k for k in self._store if fnmatch.fnmatch(k, match)]
        return 0, matching


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def fake_redis() -> _FakeRedis:
    return _FakeRedis()


@pytest.fixture
def short_term(fake_redis: _FakeRedis) -> ShortTermMemory:
    return ShortTermMemory(redis=fake_redis)  # type: ignore[arg-type]


# ── Helper: build a MagicMock that acts like an asyncpg pool.
#    pool.acquire() returns a context manager that yields conn_mock.

def _make_pool(conn_mock: AsyncMock | None = None) -> MagicMock:
    """Return a MagicMock exposing `acquire()` as an async context manager.

    The context manager's ``__aenter__`` yields *conn_mock* (or a fresh
    ``AsyncMock`` if omitted).
    """
    if conn_mock is None:
        conn_mock = AsyncMock()
    pool = MagicMock()
    # pool.acquire → context manager
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn_mock)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=cm)
    return pool


@pytest.fixture
def pool() -> MagicMock:
    """Basic unconfigured pool mock (conn is auto-created)."""
    return _make_pool()


@pytest.fixture
def pool_with_conn() -> tuple[MagicMock, AsyncMock]:
    """Return (pool_mock, conn_mock) for per-test modifications."""
    conn = AsyncMock()
    return _make_pool(conn), conn


@pytest.fixture
def short_term(fake_redis: _FakeRedis) -> ShortTermMemory:
    return ShortTermMemory(redis=fake_redis)  # type: ignore[arg-type]


@pytest.fixture
def long_term(pool: MagicMock) -> LongTermMemory:
    db = MagicMock(spec=MemoryDBPool)
    db.pool = pool
    return LongTermMemory(db=db)


@pytest.fixture
def mock_short_term() -> AsyncMock:
    return AsyncMock(spec=ShortTermMemory)


@pytest.fixture
def mock_long_term() -> AsyncMock:
    return AsyncMock(spec=LongTermMemory)


@pytest.fixture
def memory_ops(
    mock_short_term: AsyncMock,
    mock_long_term: AsyncMock,
) -> MemoryOperations:
    return MemoryOperations(short_term=mock_short_term, long_term=mock_long_term)


# ===========================================================================
# 1. Data Model Tests
# ===========================================================================

class TestEventType:
    """EventType enum — valid members and string values."""

    def test_members(self) -> None:
        assert EventType.MENTOR_QUERY.value == "mentor_query"
        assert EventType.QUIZ_ANSWER.value == "quiz_answer"
        assert EventType.GENERATION.value == "generation"
        assert EventType.REVIEW.value == "review"
        assert EventType.PROFILE_UPDATE.value == "profile_update"

    def test_all_members_covered(self) -> None:
        expected = {"MENTOR_QUERY", "QUIZ_ANSWER", "GENERATION", "REVIEW", "PROFILE_UPDATE"}
        assert {m.name for m in EventType} == expected

    def test_accepts_string_value(self) -> None:
        # EpisodicEntry.event_type is `EventType | str`, so a plain string is valid
        entry = EpisodicEntry(user_id="u1", session_id="s1", event_type="custom_event")
        assert entry.event_type == "custom_event"


class TestMemoryType:
    """MemoryType enum — valid members and string values."""

    def test_members(self) -> None:
        assert MemoryType.USER_PREFERENCE.value == "user_preference"
        assert MemoryType.DOMAIN_KNOWLEDGE.value == "domain_knowledge"
        assert MemoryType.SKILL_SUMMARY.value == "skill_summary"
        assert MemoryType.LEARNING_STYLE.value == "learning_style"

    def test_all_members_covered(self) -> None:
        expected = {"USER_PREFERENCE", "DOMAIN_KNOWLEDGE", "SKILL_SUMMARY", "LEARNING_STYLE"}
        assert {m.name for m in MemoryType} == expected


class TestEpisodicEntry:
    """EpisodicEntry model — creation, defaults, serialization."""

    def test_minimal_creation(self) -> None:
        entry = EpisodicEntry(user_id="u1", session_id="s1", event_type=EventType.MENTOR_QUERY)
        assert entry.user_id == "u1"
        assert entry.session_id == "s1"
        assert entry.event_type == EventType.MENTOR_QUERY
        assert entry.id is None
        assert entry.input is None
        assert entry.output is None
        assert entry.metadata == {}
        assert entry.importance_score == 0.5
        assert entry.created_at is None

    def test_full_creation(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0)
        entry = EpisodicEntry(
            id="rec-1",
            user_id="u1",
            session_id="s1",
            event_type="quiz_answer",
            input="What is 2+2?",
            output="4",
            metadata={"score": 100},
            importance_score=0.9,
            created_at=ts,
        )
        assert entry.id == "rec-1"
        assert entry.importance_score == 0.9
        assert entry.created_at == ts

    def test_serialize_roundtrip(self) -> None:
        entry = EpisodicEntry(
            user_id="u1",
            session_id="s1",
            event_type=EventType.REVIEW,
            input="review input",
            importance_score=0.75,
        )
        raw = entry.model_dump_json()
        restored = EpisodicEntry(**json.loads(raw))
        assert restored.user_id == "u1"
        assert restored.event_type == "review"
        assert restored.importance_score == 0.75


class TestSemanticEntry:
    """SemanticEntry model — creation, defaults, upsert-compatible fields."""

    def test_minimal_creation(self) -> None:
        entry = SemanticEntry(
            user_id="u1",
            memory_type=MemoryType.USER_PREFERENCE,
            key="user_profile",
            value={"name": "Alice"},
        )
        assert entry.confidence == 1.0
        assert entry.id is None
        assert entry.created_at is None
        assert entry.updated_at is None

    def test_confidence_default(self) -> None:
        entry = SemanticEntry(
            user_id="u1",
            memory_type=MemoryType.DOMAIN_KNOWLEDGE,
            key="topic:math",
            value={},
        )
        assert entry.confidence == 1.0

    def test_value_serialization(self) -> None:
        entry = SemanticEntry(
            user_id="u1",
            memory_type=MemoryType.LEARNING_STYLE,
            key="style",
            value={"visual": True, "auditory": False},
            confidence=0.85,
        )
        raw = entry.model_dump_json()
        restored = SemanticEntry(**json.loads(raw))
        assert restored.value == {"visual": True, "auditory": False}
        assert restored.confidence == 0.85


class TestConversationTurn:
    """ConversationTurn model — role, content, auto-timestamp."""

    def test_creation(self) -> None:
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert isinstance(turn.timestamp, datetime)

    def test_timestamp_auto_generated(self) -> None:
        turn1 = ConversationTurn(role="assistant", content="Hi")
        turn2 = ConversationTurn(role="assistant", content="Hi")
        # Timestamps are datetimes (not None)
        assert isinstance(turn1.timestamp, datetime)
        assert isinstance(turn2.timestamp, datetime)


class TestSessionMemory:
    """SessionMemory model — session state in short-term memory."""

    def test_defaults(self) -> None:
        session = SessionMemory(session_id="s1", user_id="u1")
        assert session.conversation == []
        assert session.metadata == {}
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_with_conversation(self) -> None:
        turns = [ConversationTurn(role="user", content="Hi")]
        session = SessionMemory(session_id="s1", user_id="u1", conversation=turns)
        assert len(session.conversation) == 1
        assert session.conversation[0].content == "Hi"

    def test_serialize_roundtrip(self) -> None:
        session = SessionMemory(
            session_id="s1",
            user_id="u1",
            metadata={"topic": "math"},
        )
        raw = session.model_dump_json()
        restored = SessionMemory(**json.loads(raw))
        assert restored.session_id == "s1"
        assert restored.metadata == {"topic": "math"}


class TestSensoryInput:
    """SensoryInput model — per-request input context."""

    def test_creation(self) -> None:
        si = SensoryInput(user_id="u1", session_id="s1", query="hello")
        assert si.user_id == "u1"
        assert si.session_id == "s1"
        assert si.query == "hello"
        assert si.extra == {}
        assert isinstance(si.timestamp, datetime)

    def test_with_extra(self) -> None:
        si = SensoryInput(user_id="u1", query="test", extra={"source": "web"})
        assert si.extra == {"source": "web"}

    def test_optional_session_id(self) -> None:
        si = SensoryInput(user_id="u1", query="no-session")
        assert si.session_id is None


class TestMemoryContext:
    """MemoryContext model — combined context from all three tiers."""

    def test_defaults(self) -> None:
        ctx = MemoryContext()
        assert ctx.sensory is None
        assert ctx.conversation_history == []
        assert ctx.user_profile == {}
        assert ctx.session_memory is None
        assert ctx.semantic_recall == []

    def test_full_creation(self) -> None:
        sensory = SensoryInput(user_id="u1", query="math help")
        profile = {"skill_level": "beginner"}
        session = SessionMemory(session_id="s1", user_id="u1")

        ctx = MemoryContext(
            sensory=sensory,
            conversation_history=[EpisodicEntry(user_id="u1", session_id="s1", event_type="mentor_query")],
            user_profile=profile,
            session_memory=session,
            semantic_recall=[SemanticEntry(user_id="u1", memory_type="user_preference", key="k", value={})],
        )
        assert ctx.sensory is not None
        assert ctx.sensory.query == "math help"
        assert ctx.user_profile == profile
        assert len(ctx.conversation_history) == 1
        assert len(ctx.semantic_recall) == 1


# ===========================================================================
# 2. ShortTermMemory Tests (FakeRedis backend)
# ===========================================================================

class TestShortTermMemory:
    """ShortTermMemory — session lifecycle, conversation, metadata."""

    # -- Session lifecycle ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_session(self, short_term: ShortTermMemory) -> None:
        session = await short_term.create_session("s1", "u1", metadata={"lang": "zh"})
        assert session.session_id == "s1"
        assert session.user_id == "u1"
        assert session.metadata == {"lang": "zh"}

        exists = await short_term.session_exists("s1")
        assert exists is True

    @pytest.mark.asyncio
    async def test_get_session_found(self, short_term: ShortTermMemory) -> None:
        await short_term.create_session("s1", "u1")
        session = await short_term.get_session("s1")
        assert session is not None
        assert session.session_id == "s1"
        assert session.user_id == "u1"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, short_term: ShortTermMemory) -> None:
        session = await short_term.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_session_returns_none_on_corrupt_data(
        self, fake_redis: _FakeRedis, short_term: ShortTermMemory,
    ) -> None:
        # Write invalid JSON into Redis directly
        await fake_redis.setex("session:corrupt", 3600, "{invalid json")
        session = await short_term.get_session("corrupt")
        assert session is None

    @pytest.mark.asyncio
    async def test_delete_session(self, short_term: ShortTermMemory) -> None:
        await short_term.create_session("s1", "u1")
        await short_term.delete_session("s1")
        exists = await short_term.session_exists("s1")
        assert exists is False

    @pytest.mark.asyncio
    async def test_session_exists_false_for_missing(
        self, short_term: ShortTermMemory,
    ) -> None:
        exists = await short_term.session_exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_session_exists_after_ttl_expiry(
        self, fake_redis: _FakeRedis, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1")
        # Manually expire the key
        fake_redis._ttl["session:s1"] = datetime.utcnow() - timedelta(seconds=1)
        exists = await short_term.session_exists("s1")
        assert exists is False

    # -- Conversation management --------------------------------------------

    @pytest.mark.asyncio
    async def test_add_turn_appends_conversation(
        self, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1")
        session = await short_term.add_turn("s1", "user", "Hello")
        assert session is not None
        assert len(session.conversation) == 1
        assert session.conversation[0].role == "user"
        assert session.conversation[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_add_turn_returns_none_for_missing_session(
        self, short_term: ShortTermMemory,
    ) -> None:
        session = await short_term.add_turn("nonexistent", "user", "Hi")
        assert session is None

    @pytest.mark.asyncio
    async def test_add_turn_trims_to_max(
        self, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1")
        # The limit is 50; add 55 turns
        for i in range(55):
            await short_term.add_turn("s1", "user", f"turn-{i}")
        session = await short_term.get_session("s1")
        assert session is not None
        assert len(session.conversation) == 50
        # The oldest turns should be the last 50, i.e. turn-5..turn-54
        assert session.conversation[0].content == "turn-5"
        assert session.conversation[-1].content == "turn-54"

    @pytest.mark.asyncio
    async def test_get_conversation_returns_all(
        self, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1")
        await short_term.add_turn("s1", "user", "A")
        await short_term.add_turn("s1", "assistant", "B")
        turns = await short_term.get_conversation("s1")
        assert len(turns) == 2
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_get_conversation_with_last_n(
        self, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1")
        for i in range(10):
            await short_term.add_turn("s1", "user", str(i))
        turns = await short_term.get_conversation("s1", last_n=3)
        assert len(turns) == 3
        assert [t.content for t in turns] == ["7", "8", "9"]

    @pytest.mark.asyncio
    async def test_get_conversation_empty_for_missing_session(
        self, short_term: ShortTermMemory,
    ) -> None:
        turns = await short_term.get_conversation("nonexistent")
        assert turns == []

    # -- Metadata ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_metadata_merges(
        self, short_term: ShortTermMemory,
    ) -> None:
        await short_term.create_session("s1", "u1", metadata={"lang": "zh"})
        session = await short_term.update_metadata("s1", {"topic": "math"})
        assert session is not None
        assert session.metadata == {"lang": "zh", "topic": "math"}

    @pytest.mark.asyncio
    async def test_update_metadata_returns_none_for_missing(
        self, short_term: ShortTermMemory,
    ) -> None:
        result = await short_term.update_metadata("nonexistent", {"key": "val"})
        assert result is None

    # -- Clear all -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_clear_all(self, short_term: ShortTermMemory) -> None:
        await short_term.create_session("s1", "u1")
        await short_term.create_session("s2", "u2")
        await short_term.clear_all()
        assert await short_term.session_exists("s1") is False
        assert await short_term.session_exists("s2") is False

    # -- In-memory fallback (short_term._in_memory_fallback) ----------------

    def test_in_memory_fallback_creates_instance(self) -> None:
        stm = ShortTermMemory._in_memory_fallback()
        assert isinstance(stm, ShortTermMemory)
        # Verify it uses a _FakeRedis-like instance (duck-typing)
        assert hasattr(stm._redis, "get")
        assert hasattr(stm._redis, "setex")

    @pytest.mark.asyncio
    async def test_in_memory_fallback_basic_ops(self) -> None:
        stm = ShortTermMemory._in_memory_fallback()
        session = await stm.create_session("f1", "u1")
        assert session.session_id == "f1"
        fetched = await stm.get_session("f1")
        assert fetched is not None
        assert fetched.user_id == "u1"


# ===========================================================================
# 3. LongTermMemory Tests (AsyncMock asyncpg pool)
# ===========================================================================

class TestLongTermMemory:
    """LongTermMemory — episodic and semantic memory operations,
    each test builds its own pool/conn mock via _make_pool()."""

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_lt(pool_mock: MagicMock) -> LongTermMemory:
        db = MagicMock(spec=MemoryDBPool)
        db.pool = pool_mock
        return LongTermMemory(db=db)

    # -- Episodic: write -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_write_episodic_returns_id(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=MagicMock(**{"__getitem__.return_value": 42}))
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        entry = EpisodicEntry(
            user_id="u1",
            session_id="s1",
            event_type=EventType.MENTOR_QUERY,
            input="question",
            output="answer",
            metadata={"key": "val"},
            importance_score=0.8,
        )
        record_id = await lt.write_episodic(entry)
        assert record_id == "42"
        conn.fetchrow.assert_called_once()
        call_args = conn.fetchrow.call_args[0]
        assert "INSERT INTO episodic_memory" in call_args[0]
        assert call_args[1] == "u1"

    @pytest.mark.asyncio
    async def test_write_episodic_minimal(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=MagicMock(**{"__getitem__.return_value": 99}))
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        entry = EpisodicEntry(user_id="u2", session_id="s2", event_type="test_event")
        record_id = await lt.write_episodic(entry)
        assert record_id == "99"

    # -- Episodic: recall ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_recall_episodic_without_filter(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {
                "id": 1,
                "user_id": "u1",
                "session_id": "s1",
                "event_type": "mentor_query",
                "input": "q1",
                "output": "a1",
                "metadata": "{}",
                "importance_score": 0.5,
                "created_at": datetime(2025, 6, 1),
            },
        ])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_episodic("u1", limit=5)
        assert len(results) == 1
        assert results[0].user_id == "u1"
        assert results[0].input == "q1"

    @pytest.mark.asyncio
    async def test_recall_episodic_with_event_types(self) -> None:
        conn = AsyncMock()
        # Note: in recall_episodic, conn.fetch is used (not fetchrow)
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_episodic(
            "u1", event_types=[EventType.MENTOR_QUERY, EventType.QUIZ_ANSWER],
        )
        assert results == []
        # Verify SQL uses ANY($2::varchar[])
        sql = conn.fetch.call_args[0][0]
        assert "event_type = ANY(" in sql

    @pytest.mark.asyncio
    async def test_recall_episodic_with_string_event_types(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.recall_episodic("u1", event_types=["mentor_query"])
        params = conn.fetch.call_args[0]
        assert "mentor_query" in str(params)

    @pytest.mark.asyncio
    async def test_recall_episodic_empty_result(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_episodic("u_unknown")
        assert results == []

    # -- Episodic: update importance -----------------------------------------

    @pytest.mark.asyncio
    async def test_update_episodic_importance(self) -> None:
        conn = AsyncMock()
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.update_episodic_importance("rec-1", 0.95)
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "UPDATE episodic_memory" in sql
        assert "importance_score" in sql

    # -- Episodic: delete ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_episodic(self) -> None:
        conn = AsyncMock()
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.delete_episodic("rec-1")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM episodic_memory" in sql

    # -- Semantic: write (upsert) -------------------------------------------

    @pytest.mark.asyncio
    async def test_write_semantic_upsert(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=MagicMock(**{"__getitem__.return_value": 7}))
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        entry = SemanticEntry(
            user_id="u1",
            memory_type=MemoryType.USER_PREFERENCE,
            key="user_profile",
            value={"name": "Alice"},
            confidence=0.9,
        )
        record_id = await lt.write_semantic(entry)
        assert record_id == "7"
        sql = conn.fetchrow.call_args[0][0]
        assert "INSERT INTO semantic_memory" in sql
        assert "ON CONFLICT" in sql

    @pytest.mark.asyncio
    async def test_write_semantic_with_string_memory_type(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=MagicMock(**{"__getitem__.return_value": 8}))
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        entry = SemanticEntry(
            user_id="u1",
            memory_type="domain_knowledge",
            key="topic:dsa",
            value={"difficulty": "hard"},
        )
        await lt.write_semantic(entry)
        call_args = conn.fetchrow.call_args[0]
        assert "domain_knowledge" in str(call_args)

    # -- Semantic: recall ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_recall_semantic_no_filters(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_semantic("u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_semantic_with_keys_filter(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.recall_semantic("u1", keys=["user_profile"])
        sql = conn.fetch.call_args[0][0]
        assert "key = ANY($" in sql

    @pytest.mark.asyncio
    async def test_recall_semantic_with_memory_types_filter(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.recall_semantic("u1", memory_types=["user_preference"])
        sql = conn.fetch.call_args[0][0]
        assert "memory_type = ANY($" in sql

    @pytest.mark.asyncio
    async def test_recall_semantic_with_all_filters(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {
                "id": 10,
                "user_id": "u1",
                "memory_type": "user_preference",
                "key": "user_profile",
                "value": '{"name":"Alice"}',
                "confidence": 0.9,
                "created_at": datetime(2025, 6, 15),
                "updated_at": datetime(2025, 6, 15),
            },
        ])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_semantic(
            "u1",
            keys=["user_profile"],
            memory_types=["user_preference"],
        )
        assert len(results) == 1
        assert results[0].value == {"name": "Alice"}
        assert results[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_recall_semantic_parses_json_value(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[
            {
                "id": 11,
                "user_id": "u1",
                "memory_type": "user_preference",
                "key": "profile",
                "value": '{"lang":"zh"}',
                "confidence": 0.8,
                "created_at": datetime(2025, 1, 1),
                "updated_at": datetime(2025, 1, 1),
            },
        ])
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        results = await lt.recall_semantic("u1")
        assert results[0].value == {"lang": "zh"}

    # -- Semantic: delete ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_semantic(self) -> None:
        conn = AsyncMock()
        pool = _make_pool(conn)
        lt = self._make_lt(pool)

        await lt.delete_semantic("sem-1")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM semantic_memory" in sql


# ===========================================================================
# 4. Row-to-model helper tests (_row_to_episodic, _row_to_semantic)
# ===========================================================================

class TestRowToModelHelpers:
    """Internal helpers that convert asyncpg rows to Pydantic models.

    The helpers access ``row["column_name"]`` via ``__getitem__`` (as
    asyncpg ``Record`` objects do).  We provide a simple dict-backed fake.
    """

    @staticmethod
    def _make_row(**kwargs: object) -> MagicMock:
        """Return a MagicMock whose ``__getitem__`` dispatches on key."""
        row = MagicMock()
        row.__getitem__ = MagicMock(side_effect=kwargs.get)
        for k, v in kwargs.items():
            setattr(row, k, v)
        return row

    def test_row_to_episodic(self) -> None:
        ts = datetime(2025, 7, 1)
        row = self._make_row(
            id=1,
            user_id="u1",
            session_id="s1",
            event_type="mentor_query",
            input="q",
            output="a",
            metadata='{"key":"val"}',
            importance_score=0.6,
            created_at=ts,
        )
        entry = _row_to_episodic(row)
        assert entry.id == "1"
        assert entry.metadata == {"key": "val"}
        assert entry.created_at == ts

    def test_row_to_episodic_parses_str_metadata(self) -> None:
        row = self._make_row(
            id=1,
            user_id="u1",
            session_id="s1",
            event_type="test",
            input=None,
            output=None,
            metadata='{"nested":{"a":1}}',
            importance_score=0.5,
            created_at=None,
        )
        entry = _row_to_episodic(row)
        assert entry.metadata == {"nested": {"a": 1}}

    def test_row_to_episodic_handles_dict_metadata(self) -> None:
        """If metadata is already a dict (some drivers), pass through."""
        row = self._make_row(
            id=2,
            user_id="u1",
            session_id="s1",
            event_type="test",
            input=None,
            output=None,
            metadata={"direct": "dict"},
            importance_score=0.5,
            created_at=None,
        )
        entry = _row_to_episodic(row)
        assert entry.metadata == {"direct": "dict"}

    def test_row_to_semantic(self) -> None:
        ts = datetime(2025, 7, 1)
        row = self._make_row(
            id=5,
            user_id="u1",
            memory_type="user_preference",
            key="profile",
            value='{"name":"Bob"}',
            confidence=0.95,
            created_at=ts,
            updated_at=ts,
        )
        entry = _row_to_semantic(row)
        assert entry.id == "5"
        assert entry.value == {"name": "Bob"}

    def test_row_to_semantic_handles_dict_value(self) -> None:
        row = self._make_row(
            id=6,
            user_id="u1",
            memory_type="skill_summary",
            key="algos",
            value={"topics": ["sorting"]},
            confidence=0.7,
            created_at=None,
            updated_at=None,
        )
        entry = _row_to_semantic(row)
        assert entry.value == {"topics": ["sorting"]}


# ===========================================================================
# 5. MemoryOperations Tests (mocked short + long term)
# ===========================================================================

class TestMemoryOperations:
    """MemoryOperations — context building, interaction recording, profile save."""

    # -- build_context -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_build_context_with_existing_session(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        session = SessionMemory(session_id="s1", user_id="u1")
        mock_short_term.get_session = AsyncMock(return_value=session)
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context("u1", session_id="s1", query="help")

        assert ctx.sensory is not None
        assert ctx.sensory.query == "help"
        assert ctx.session_memory is not None
        assert ctx.session_memory.session_id == "s1"
        mock_short_term.create_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_context_creates_session_when_missing(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_short_term.get_session = AsyncMock(return_value=None)
        new_session = SessionMemory(session_id="s1", user_id="u1")
        mock_short_term.create_session = AsyncMock(return_value=new_session)
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context("u1", session_id="s1", query="hello")
        mock_short_term.create_session.assert_called_once_with("s1", "u1")
        assert ctx.session_memory is not None

    @pytest.mark.asyncio
    async def test_build_context_no_session_id(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context("u1", query="no-session")

        assert ctx.sensory is not None
        assert ctx.session_memory is None
        mock_short_term.get_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_context_recalls_episodic_history(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        entry = EpisodicEntry(user_id="u1", session_id="s1", event_type="mentor_query")
        mock_short_term.get_session = AsyncMock(return_value=SessionMemory(session_id="s1", user_id="u1"))
        mock_long_term.recall_episodic = AsyncMock(return_value=[entry])
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context("u1", session_id="s1")
        assert len(ctx.conversation_history) == 1
        mock_long_term.recall_episodic.assert_called_with(
            user_id="u1",
            event_types=[EventType.MENTOR_QUERY],
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_build_context_recalls_user_profile(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        profile = {"name": "Alice", "level": "beginner"}
        mock_short_term.get_session = AsyncMock(return_value=SessionMemory(session_id="s1", user_id="u1"))
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(return_value=[
            SemanticEntry(
                user_id="u1", memory_type="user_preference",
                key="user_profile", value=profile,
            ),
        ])

        ctx = await memory_ops.build_context("u1", session_id="s1")
        assert ctx.user_profile == profile

    @pytest.mark.asyncio
    async def test_build_context_handles_episodic_error(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_short_term.get_session = AsyncMock(return_value=SessionMemory(session_id="s1", user_id="u1"))
        mock_long_term.recall_episodic = AsyncMock(side_effect=Exception("DB error"))
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context("u1", session_id="s1")
        # Should not raise — error is logged and context built with defaults
        assert ctx.conversation_history == []

    @pytest.mark.asyncio
    async def test_build_context_handles_semantic_error(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_short_term.get_session = AsyncMock(return_value=SessionMemory(session_id="s1", user_id="u1"))
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(side_effect=Exception("Semantic error"))

        ctx = await memory_ops.build_context("u1", session_id="s1")
        assert ctx.semantic_recall == []
        assert ctx.user_profile == {}

    @pytest.mark.asyncio
    async def test_build_context_extra_passthrough(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_short_term.get_session = AsyncMock(return_value=SessionMemory(session_id="s1", user_id="u1"))
        mock_long_term.recall_episodic = AsyncMock(return_value=[])
        mock_long_term.recall_semantic = AsyncMock(return_value=[])

        ctx = await memory_ops.build_context(
            "u1", session_id="s1", query="test", extra={"source": "web"},
        )
        assert ctx.sensory is not None
        assert ctx.sensory.extra == {"source": "web"}

    # -- record_interaction --------------------------------------------------

    @pytest.mark.asyncio
    async def test_record_interaction_basic(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        await memory_ops.record_interaction(
            user_id="u1",
            session_id="s1",
            event_type=EventType.MENTOR_QUERY,
            input_text="hello",
            output_text="world",
            importance=0.7,
            metadata={"topic": "greeting"},
        )

        # Verify episodic write
        mock_long_term.write_episodic.assert_called_once()
        entry: EpisodicEntry = mock_long_term.write_episodic.call_args[0][0]
        assert entry.user_id == "u1"
        assert entry.importance_score == 0.7
        assert entry.metadata == {"topic": "greeting"}

        # Verify short-term turns
        mock_short_term.add_turn.assert_any_call("s1", "user", "hello")
        mock_short_term.add_turn.assert_any_call("s1", "assistant", "world")

    @pytest.mark.asyncio
    async def test_record_interaction_without_session_skips_short_term(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        await memory_ops.record_interaction(
            user_id="u1",
            input_text="hello",
            output_text="world",
        )
        mock_long_term.write_episodic.assert_called_once()
        mock_short_term.add_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_interaction_without_text_skips_short_term(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        await memory_ops.record_interaction(
            user_id="u1",
            session_id="s1",
        )
        mock_long_term.write_episodic.assert_called_once()
        mock_short_term.add_turn.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_interaction_handles_episodic_error(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_long_term.write_episodic = AsyncMock(side_effect=Exception("Write failed"))

        # Should not raise — error is logged
        await memory_ops.record_interaction(
            user_id="u1",
            session_id="s1",
            input_text="hi",
            output_text="bye",
        )
        mock_short_term.add_turn.assert_called()  # still called

    @pytest.mark.asyncio
    async def test_record_interaction_handles_short_term_error(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_short_term.add_turn = AsyncMock(side_effect=Exception("Redis error"))
        mock_long_term.write_episodic = AsyncMock()

        # Should not raise
        await memory_ops.record_interaction(
            user_id="u1",
            session_id="s1",
            input_text="hi",
        )

    # -- save_user_profile ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_user_profile(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        profile = {"name": "Alice", "skill": "beginner"}
        await memory_ops.save_user_profile("u1", profile, confidence=0.85)

        mock_long_term.write_semantic.assert_called_once()
        entry: SemanticEntry = mock_long_term.write_semantic.call_args[0][0]
        assert entry.user_id == "u1"
        assert entry.key == "user_profile"
        assert entry.value == profile
        assert entry.confidence == 0.85
        assert entry.memory_type == MemoryType.USER_PREFERENCE

    @pytest.mark.asyncio
    async def test_save_user_profile_default_confidence(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        await memory_ops.save_user_profile("u1", {"name": "Bob"})
        entry: SemanticEntry = mock_long_term.write_semantic.call_args[0][0]
        assert entry.confidence == 0.8

    @pytest.mark.asyncio
    async def test_save_user_profile_handles_error(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        mock_long_term.write_semantic = AsyncMock(side_effect=Exception("Write failed"))
        # Should not raise
        await memory_ops.save_user_profile("u1", {"name": "Alice"})

    # -- clear_session -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_clear_session(
        self, memory_ops: MemoryOperations,
        mock_short_term: AsyncMock, mock_long_term: AsyncMock,
    ) -> None:
        await memory_ops.clear_session("s1")
        mock_short_term.delete_session.assert_called_once_with("s1")


# ===========================================================================
# 6. MemoryDBPool Tests
# ===========================================================================

class TestMemoryDBPool:
    """MemoryDBPool — DSN normalization, create / close lifecycle."""

    def test_dsn_normalization(self) -> None:
        """DSN with postgresql+asyncpg:// scheme is normalized."""
        db = MemoryDBPool(dsn="postgresql+asyncpg://user:pass@localhost:5432/db")
        assert db._dsn == "postgresql://user:pass@localhost:5432/db"

    def test_dsn_unchanged_when_no_normalization_needed(self) -> None:
        original = "postgresql://user:pass@localhost:5432/db"
        db = MemoryDBPool(dsn=original)
        assert db._dsn == original

    def test_dsn_short_form(self) -> None:
        """Standard postgres:// DSN also works."""
        db = MemoryDBPool(dsn="postgres://user:pass@localhost:5432/db")
        assert db._dsn == "postgres://user:pass@localhost:5432/db"

    @pytest.mark.asyncio
    async def test_create_and_close(self) -> None:
        """create() and close() lifecycle with mocked asyncpg.create_pool."""
        mock_pool_obj = AsyncMock()
        with patch("src.memory.db.asyncpg.create_pool", new_callable=AsyncMock) as mock_create_pool:
            mock_create_pool.return_value = mock_pool_obj
            db = MemoryDBPool(dsn="postgresql://u:p@localhost/db")

            await db.create()
            assert db.pool is mock_pool_obj
            mock_create_pool.assert_called_once_with(
                dsn="postgresql://u:p@localhost/db",
                min_size=2,
                max_size=10,
            )

            await db.close()
            mock_pool_obj.close.assert_called_once()
            assert db.pool is None

    @pytest.mark.asyncio
    async def test_close_when_no_pool(self) -> None:
        """close() is a no-op when pool was never created."""
        db = MemoryDBPool(dsn="postgresql://u:p@localhost/db")
        # Should not raise
        await db.close()

    @pytest.mark.asyncio
    async def test_close_when_pool_none(self) -> None:
        """close() is a no-op when pool is explicitly None."""
        db = MemoryDBPool(dsn="postgresql://u:p@localhost/db")
        db.pool = None
        await db.close()  # should not raise


class TestRedisUnavailableFallbackSelection:
    """The fallback must be *selected* when Redis is unreachable.

    ``test_in_memory_fallback_*`` above proves the fallback object works, but
    not that it gets chosen — and the orchestrator depends on that choice being
    automatic (src/agents/orchestrator/router.py:109 notes it "should not
    happen" that short-term memory is missing).  This drives the real
    ``_init_memory`` with an unreachable Redis.
    """

    async def test_init_memory_falls_back_and_stays_usable(self) -> None:
        from unittest.mock import patch

        from src import main as main_module

        class _FakeApp:
            def __init__(self):
                self.state = type("S", (), {})()

        class _FakePool:
            async def create(self):
                return None

        app = _FakeApp()
        # Port 1 is reliably closed, so redis ping() fails fast.
        with patch.object(main_module.settings, "redis_url", "redis://127.0.0.1:1/0"):
            with patch("src.memory.db.MemoryDBPool", return_value=_FakePool()):
                ops = await main_module._init_memory(app)

        st = ops.short_term
        # Fallback engaged, and it is actually usable for the session
        # create/read cycle the orchestrator performs.
        await st.create_session(
            session_id="s1", user_id="u1", metadata={"k": "v"}
        )
        session = await st.get_session("s1")
        assert session is not None
        assert session.metadata.get("k") == "v"
