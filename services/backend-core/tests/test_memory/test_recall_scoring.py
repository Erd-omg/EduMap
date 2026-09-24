"""Tests for episodic-memory recall ranking (I-2).

Background — why this file exists:

``recall_episodic`` used to be a plain ``ORDER BY created_at DESC``.  The
existing tests in ``test_memory.py`` assert only that the SQL contains
``event_type = ANY(`` and that the row count matches.  Those assertions do
**not** exercise the ordering contract at all, so they would pass unchanged
if the ranking were silently removed.  The tests below drive the real
``recall_episodic`` entry point and assert which entry wins, so that
deleting the ranking fails them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.long_term import LongTermMemory, _RECALL_CANDIDATE_FACTOR
from src.memory.models import EventType
from src.memory.recall_scoring import (
    RECENCY_DECAY,
    cosine_similarity,
    recency_score,
    score_episodic_entries,
)


NOW = datetime(2026, 9, 23, 12, 0, 0)


def _row(
    id: int,
    *,
    hours_ago: float,
    importance: float = 0.5,
    event_type: str = "mentor_query",
    input_text: str = "q",
) -> dict:
    """Build a fake asyncpg row as ``_row_to_episodic`` expects."""
    return {
        "id": id,
        "user_id": "u1",
        "session_id": "s1",
        "event_type": event_type,
        "input": input_text,
        "output": "a",
        "metadata": "{}",
        "importance_score": importance,
        "created_at": NOW - timedelta(hours=hours_ago),
    }


def _make_lt(rows: list[dict]) -> tuple[LongTermMemory, AsyncMock]:
    """Return (service, conn) with ``conn.fetch`` returning *rows*.

    ``LongTermMemory`` takes a ``MemoryDBPool`` and accesses ``self._db.pool``,
    so the pool mock must be nested under a db object. Passing the pool
    directly makes ``self._db.pool`` auto-generate a *different*, empty mock —
    ``fetch`` then returns a Mock (not the rows) and every recall silently
    looks empty while raising nothing.
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=cm)

    db = MagicMock()
    db.pool = pool
    return LongTermMemory(db), conn


# ===========================================================================
# Pure scoring functions
# ===========================================================================

class TestRecencyScore:
    def test_zero_age_is_one(self) -> None:
        assert recency_score(NOW, now=NOW) == pytest.approx(1.0)

    def test_decay_constant_is_the_paper_value(self) -> None:
        """Pin the constant itself, so changing it fails loudly.

        Asserting ``recency_score(t) == RECENCY_DECAY ** hours`` would be a
        tautology — it re-reads the same constant and passes for *any*
        value. The number must be asserted literally.
        """
        assert RECENCY_DECAY == pytest.approx(0.995)

    def test_ten_hours_matches_hand_computed_value(self) -> None:
        """0.995^10 ≈ 0.9511 — computed independently of the implementation."""
        assert recency_score(NOW - timedelta(hours=10), now=NOW) == pytest.approx(
            0.9511101304657719, abs=1e-12
        )

    def test_one_week_is_still_above_half(self) -> None:
        """Sanity-check the decay rate's *shape*, not just its constant.

        At 0.995/hour, 168h (1 week) retains 0.995^168 ≈ 0.431 — an entry
        from a week ago is heavily discounted but not annihilated. A decay
        constant of 0.9 (a plausible typo) would leave ≈ 2e-8, i.e. zero.
        """
        week = recency_score(NOW - timedelta(hours=168), now=NOW)
        assert 0.3 < week < 0.6, f"1-week recency {week} implies the wrong decay rate"

    def test_none_timestamp_treated_as_now(self) -> None:
        """An undateable entry is 'fresh', not 'ancient' — else it gets buried."""
        assert recency_score(None, now=NOW) == pytest.approx(1.0)

    def test_naive_datetime_treated_as_utc(self) -> None:
        """asyncpg can hand back naive datetimes; they must not raise."""
        naive = datetime(2026, 9, 23, 11, 0, 0)  # 1h before NOW, tz-naive
        assert recency_score(naive, now=NOW) == pytest.approx(
            0.995, abs=1e-12
        )

    def test_naive_and_aware_agree(self) -> None:
        """The tz-coercion must not change the computed age."""
        aware = NOW - timedelta(hours=3)
        naive = aware.replace(tzinfo=None)
        assert recency_score(naive, now=NOW) == pytest.approx(
            recency_score(aware, now=NOW)
        )

    def test_future_timestamp_clamped_to_one(self) -> None:
        """Clock skew must not produce recency > 1."""
        assert recency_score(NOW + timedelta(hours=5), now=NOW) == pytest.approx(1.0)


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_negative_clamped_to_zero(self) -> None:
        """Anti-relevance and irrelevance carry the same information here."""
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    def test_missing_or_mismatched_vectors_are_zero(self) -> None:
        assert cosine_similarity(None, [1.0]) == 0.0
        assert cosine_similarity([1.0], None) == 0.0
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector_is_zero_not_nan(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestScoreEpisodicEntries:
    def _entry(self, id_: str, *, hours_ago: float, importance: float):
        from src.memory.models import EpisodicEntry

        return EpisodicEntry(
            id=id_,
            user_id="u1",
            session_id="s1",
            event_type="mentor_query",
            importance_score=importance,
            created_at=NOW - timedelta(hours=hours_ago),
        )

    def test_empty_input(self) -> None:
        assert score_episodic_entries([]) == []

    def test_importance_can_outrank_recency(self) -> None:
        """The whole point: an older but important entry beats a fresh trivial one."""
        old_important = self._entry("important", hours_ago=48, importance=1.0)
        new_trivial = self._entry("trivial", hours_ago=0, importance=0.0)

        ranked = score_episodic_entries([new_trivial, old_important], now=NOW)

        assert ranked[0][0].id == "important", (
            "an entry with importance=1.0 and age 48h must outrank one with "
            "importance=0.0 written now — otherwise importance is not wired in"
        )

    def test_ordering_is_descending(self) -> None:
        entries = [
            self._entry("a", hours_ago=1, importance=0.2),
            self._entry("b", hours_ago=1, importance=0.9),
            self._entry("c", hours_ago=100, importance=0.5),
        ]
        scores = [s for _e, s in score_episodic_entries(entries, now=NOW)]
        assert scores == sorted(scores, reverse=True)

    def test_relevance_breaks_ties(self) -> None:
        """With age and importance equal, the query-matching entry must win."""
        e1 = self._entry("far", hours_ago=10, importance=0.5)
        e2 = self._entry("near", hours_ago=10, importance=0.5)

        ranked = score_episodic_entries(
            [e1, e2],
            query_embedding=[1.0, 0.0],
            entry_embeddings={"far": [0.0, 1.0], "near": [1.0, 0.0]},
            now=NOW,
        )
        assert ranked[0][0].id == "near"

    def test_no_embeddings_does_not_reorder_by_relevance(self) -> None:
        """Without a query embedding the relevance term must be inert."""
        e1 = self._entry("a", hours_ago=1, importance=0.9)
        e2 = self._entry("b", hours_ago=2, importance=0.1)

        with_embeds = score_episodic_entries([e2, e1], now=NOW)
        assert [e.id for e, _ in with_embeds][0] == "a"

    def test_stable_ties_preserve_input_order(self) -> None:
        """Equal scores keep the caller's (newest-first) order."""
        e1 = self._entry("first", hours_ago=5, importance=0.5)
        e2 = self._entry("second", hours_ago=5, importance=0.5)

        ranked = score_episodic_entries([e1, e2], now=NOW)
        assert [e.id for e, _ in ranked] == ["first", "second"]


# ===========================================================================
# recall_episodic — the real entry point
# ===========================================================================

class TestRecallEpisodicRanking:
    @pytest.mark.asyncio
    async def test_old_important_entry_is_returned_first(self) -> None:
        """End-to-end: SQL order is newest-first, output must be re-ranked."""
        rows = [
            _row(1, hours_ago=0, importance=0.0, input_text="trivial-fresh"),
            _row(2, hours_ago=200, importance=1.0, input_text="important-old"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_episodic("u1", limit=2)

        assert results[0].input == "important-old", (
            "recall must rank by importance×recency, not return raw "
            "created_at DESC order"
        )

    @pytest.mark.asyncio
    async def test_limit_is_applied_after_ranking(self) -> None:
        """The old-but-important row must survive truncation to limit=1."""
        rows = [
            _row(1, hours_ago=0, importance=0.0, input_text="trivial-fresh"),
            _row(2, hours_ago=200, importance=1.0, input_text="important-old"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_episodic("u1", limit=1)

        assert len(results) == 1
        assert results[0].input == "important-old"

    @pytest.mark.asyncio
    async def test_sql_over_fetches_beyond_limit(self) -> None:
        """Candidates must exceed limit, or ranking can never promote anyone."""
        rows = [_row(i, hours_ago=i) for i in range(10)]
        lt, conn = _make_lt(rows)

        await lt.recall_episodic("u1", limit=2)

        # params are positional: for the unfiltered query, limit is the 2nd
        # positional arg after the SQL string.
        passed_limit = conn.fetch.call_args[0][2]
        assert passed_limit == 2 * _RECALL_CANDIDATE_FACTOR, (
            "SQL must fetch a wider candidate window than `limit`; otherwise "
            "the newest `limit` rows are chosen before scoring"
        )

    @pytest.mark.asyncio
    async def test_event_type_filter_still_applies(self) -> None:
        rows = [_row(1, hours_ago=1, event_type="quiz_answer")]
        lt, conn = _make_lt(rows)

        results = await lt.recall_episodic("u1", event_types=["quiz_answer"])

        assert len(results) == 1
        sql = conn.fetch.call_args[0][0]
        assert "event_type = ANY(" in sql

    @pytest.mark.asyncio
    async def test_filtered_branch_also_over_fetches(self) -> None:
        """The filtered branch has an extra param, so the window sits at index 3.

        Asserting only the unfiltered branch would let the filtered one
        quietly regress to ``LIMIT = limit``.
        """
        rows = [_row(1, hours_ago=1, event_type="quiz_answer")]
        lt, conn = _make_lt(rows)

        await lt.recall_episodic(
            "u1", event_types=[EventType.QUIZ_ANSWER], limit=3
        )

        # args: (sql, user_id, event_types, candidate_limit, offset)
        assert conn.fetch.call_args[0][3] == 3 * _RECALL_CANDIDATE_FACTOR

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        lt, _conn = _make_lt([])
        assert await lt.recall_episodic("u1") == []

    @pytest.mark.asyncio
    async def test_limit_larger_than_candidates(self) -> None:
        """Asking for more than exist must not raise or pad."""
        lt, _conn = _make_lt([_row(1, hours_ago=1)])
        results = await lt.recall_episodic("u1", limit=50)
        assert len(results) == 1


# ===========================================================================
# recall_relevant — semantic relevance ranking (distinct from history recall)
# ===========================================================================

def _row_emb(
    id: int,
    *,
    hours_ago: float,
    importance: float = 0.5,
    embedding=None,
    input_text: str = "q",
) -> dict:
    row = _row(id, hours_ago=hours_ago, importance=importance, input_text=input_text)
    row["embedding"] = embedding
    return row


class TestRecallRelevant:
    """``recall_relevant`` answers "what relates to the current query?",
    which is a different question from ``recall_episodic``'s "what did we
    just talk about?" — hence a separate method rather than a flag."""

    @pytest.mark.asyncio
    async def test_semantically_closest_entry_wins(self) -> None:
        """An older entry matching the query must beat a recent unrelated one."""
        rows = [
            _row_emb(1, hours_ago=1, importance=0.5, embedding=[0.0, 1.0],
                     input_text="recent-unrelated"),
            _row_emb(2, hours_ago=500, importance=0.5, embedding=[1.0, 0.0],
                     input_text="old-but-relevant"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_relevant("u1", [1.0, 0.0], limit=2)

        assert results[0].input == "old-but-relevant"

    @pytest.mark.asyncio
    async def test_rows_without_embedding_are_still_recallable(self) -> None:
        """Legacy rows (embedding NULL) must not be dropped from results."""
        rows = [
            _row_emb(1, hours_ago=1, importance=0.8, embedding=None,
                     input_text="legacy-no-vector"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_relevant("u1", [1.0, 0.0], limit=5)

        assert len(results) == 1
        assert results[0].input == "legacy-no-vector"

    @pytest.mark.asyncio
    async def test_embedding_decoded_from_json_string(self) -> None:
        """asyncpg may hand JSONB back as a str — it must still be parsed."""
        rows = [
            _row_emb(1, hours_ago=1, embedding="[1.0, 0.0]", input_text="from-json"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_relevant("u1", [1.0, 0.0], limit=1)

        assert results[0].embedding == [1.0, 0.0]

    @pytest.mark.asyncio
    async def test_malformed_embedding_does_not_break_recall(self) -> None:
        """A corrupt vector must degrade to relevance 0, not raise."""
        rows = [
            _row_emb(1, hours_ago=1, embedding="{not json", input_text="corrupt"),
        ]
        lt, _conn = _make_lt(rows)

        results = await lt.recall_relevant("u1", [1.0, 0.0], limit=1)

        assert len(results) == 1
        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_event_type_filter_passed_through(self) -> None:
        rows = [_row_emb(1, hours_ago=1, embedding=[1.0, 0.0])]
        lt, conn = _make_lt(rows)

        await lt.recall_relevant(
            "u1", [1.0, 0.0], event_types=["mentor_query"], limit=3
        )

        sql = conn.fetch.call_args[0][0]
        assert "event_type = ANY(" in sql
        # window sits after (sql, user_id, event_types)
        assert conn.fetch.call_args[0][3] == 3 * _RECALL_CANDIDATE_FACTOR

    @pytest.mark.asyncio
    async def test_empty_history(self) -> None:
        lt, _conn = _make_lt([])
        assert await lt.recall_relevant("u1", [1.0, 0.0]) == []

    @pytest.mark.asyncio
    async def test_does_not_use_offset(self) -> None:
        """recall_relevant ranks globally; paging would defeat the ranking."""
        rows = [_row_emb(1, hours_ago=1, embedding=[1.0, 0.0])]
        lt, conn = _make_lt(rows)

        await lt.recall_relevant("u1", [1.0, 0.0], limit=2)

        # args: (sql, user_id, candidate_limit) — no offset parameter
        assert len(conn.fetch.call_args[0]) == 3
