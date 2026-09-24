"""Tests for episodic-embedding write path (recall_relevant's data source).

``recall_relevant`` can only rank semantically if embeddings were stored at
write time. ``MemoryOperations.record_interaction`` is the single write path,
and it receives an optional encoder via ``embed_fn``.

Design constraints asserted here:

* Embedding must never block the interaction from being recorded. The encoder
  is an enhancement, so a raising / empty / absent encoder degrades to "no
  vector" rather than losing the memory.
* The vector is stored on the entry that reaches ``write_episodic`` — asserting
  only on ``write_episodic`` being called would not notice the field being
  dropped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.memory.models import EventType
from src.memory.operations import MemoryOperations


def _ops(embed_fn=None) -> tuple[MemoryOperations, AsyncMock, AsyncMock]:
    short = AsyncMock()
    long = AsyncMock()
    return MemoryOperations(short_term=short, long_term=long, embed_fn=embed_fn), short, long


class TestEmbedOnWrite:
    @pytest.mark.asyncio
    async def test_encoder_output_is_stored_on_the_entry(self) -> None:
        ops, _short, long = _ops(embed_fn=lambda t: [0.1, 0.2, 0.3])

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.MENTOR_QUERY,
            input_text="什么是红黑树", output_text="一种自平衡二叉查找树",
        )

        entry = long.write_episodic.call_args[0][0]
        assert entry.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_encodes_input_and_output_together(self) -> None:
        """One topic = one vector; splitting Q and A would fragment the signal."""
        seen: list[str] = []

        def _capture(text: str) -> list[float]:
            seen.append(text)
            return [1.0]

        ops, _short, long = _ops(embed_fn=_capture)

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.MENTOR_QUERY,
            input_text="Q", output_text="A",
        )

        assert len(seen) == 1
        assert "Q" in seen[0] and "A" in seen[0]
        assert long.write_episodic.call_args[0][0].embedding == [1.0]

    @pytest.mark.asyncio
    async def test_no_encoder_writes_none(self) -> None:
        ops, _short, long = _ops(embed_fn=None)

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.MENTOR_QUERY, input_text="Q",
        )

        assert long.write_episodic.call_args[0][0].embedding is None

    @pytest.mark.asyncio
    async def test_raising_encoder_is_non_fatal(self) -> None:
        """A broken encoder must not lose the memory."""
        def _boom(_t: str) -> list[float]:
            raise RuntimeError("model exploded")

        ops, _short, long = _ops(embed_fn=_boom)

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.MENTOR_QUERY, input_text="Q",
        )

        long.write_episodic.assert_called_once()
        assert long.write_episodic.call_args[0][0].embedding is None

    @pytest.mark.asyncio
    async def test_empty_vector_is_stored_as_none(self) -> None:
        """An encoder returning [] must not store a useless empty list."""
        ops, _short, long = _ops(embed_fn=lambda _t: [])

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.MENTOR_QUERY, input_text="Q",
        )

        assert long.write_episodic.call_args[0][0].embedding is None

    @pytest.mark.asyncio
    async def test_blank_text_skips_encoder(self) -> None:
        """Whitespace-only input should not waste an encode call."""
        calls: list[str] = []

        def _count(text: str) -> list[float]:
            calls.append(text)
            return [1.0]

        ops, _short, _long = _ops(embed_fn=_count)

        await ops.record_interaction(
            user_id="u1", session_id="s1",
            event_type=EventType.QUIZ_ANSWER, input_text="   ",
        )

        assert calls == []
