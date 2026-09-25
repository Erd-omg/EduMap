"""MemoryOperations — unified interface across all three memory tiers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from src.memory.long_term import LongTermMemory
from src.memory.models import (
    EpisodicEntry,
    EventType,
    MemoryContext,
    MemoryType,
    SemanticEntry,
    SensoryInput,
    SessionMemory,
)
from src.memory.short_term import ShortTermMemory

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MemoryOperations:
    """Coordinator across sensory / short-term / long-term memory.

    Provides a single entry point for all memory operations used by agents
    and the orchestrator.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        # Optional text→vector encoder. When supplied, newly recorded
        # interactions are stored with an embedding so that
        # ``long_term.recall_relevant`` can rank them semantically. When None,
        # rows are written without a vector and remain recallable on
        # recency + importance alone.
        self._embed_fn = embed_fn

    def _embed(self, text: str | None) -> list[float] | None:
        """Encode *text*, tolerating any encoder failure.

        Embedding is an enhancement, not a correctness requirement: a failure
        here must not stop the interaction from being recorded. Returns None
        on any error, which ``recall_relevant`` treats as relevance 0.
        """
        if not self._embed_fn or not text or not text.strip():
            return None
        try:
            vector = self._embed_fn(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to embed episodic text (non-fatal): %s", exc)
            return None
        if not vector:
            return None
        return list(vector)

    # ── Context building (orchestrator START node) ───────────

    async def build_context(
        self,
        user_id: str,
        session_id: str | None = None,
        query: str = "",
        extra: dict[str, Any] | None = None,
    ) -> MemoryContext:
        """Build a MemoryContext from all three tiers for a pipeline run.

        Called at the START of the orchestrator graph.
        """
        sensory = SensoryInput(
            user_id=user_id,
            session_id=session_id,
            query=query,
            extra=extra or {},
        )

        # Short-term: get or create session
        session_memory: SessionMemory | None = None
        if session_id:
            session_memory = await self.short_term.get_session(session_id)
            if not session_memory:
                session_memory = await self.short_term.create_session(
                    session_id, user_id,
                )

        # Long-term: recall last 10 episodic entries for conversation history
        conversation_history: list[EpisodicEntry] = []
        try:
            conversation_history = await self.long_term.recall_episodic(
                user_id=user_id,
                event_types=[EventType.MENTOR_QUERY],
                limit=10,
            )
        except Exception as exc:
            logger.warning("Failed to recall episodic memory: %s", exc)

        # Long-term: recall semantic entries
        semantic_recall: list[SemanticEntry] = []
        user_profile: dict[str, Any] = {}
        try:
            semantic_recall = await self.long_term.recall_semantic(
                user_id=user_id,
            )
            # Extract user_profile from semantic recall
            for entry in semantic_recall:
                if entry.key == "user_profile":
                    user_profile = entry.value
                    break
        except Exception as exc:
            logger.warning("Failed to recall semantic memory: %s", exc)

        return MemoryContext(
            sensory=sensory,
            conversation_history=conversation_history,
            user_profile=user_profile,
            session_memory=session_memory,
            semantic_recall=semantic_recall,
        )

    # ── Recording (orchestrator END / mentor post-answer) ────

    async def record_interaction(
        self,
        user_id: str,
        session_id: str | None = None,
        event_type: EventType | str = EventType.MENTOR_QUERY,
        input_text: str | None = None,
        output_text: str | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction to both short-term and long-term memory."""
        # Episodic (long-term)
        entry = EpisodicEntry(
            user_id=user_id,
            session_id=session_id or "",
            event_type=event_type,
            input=input_text,
            output=output_text,
            metadata=metadata or {},
            importance_score=importance,
            # Embed the interaction so recall_relevant can rank semantically.
            # Encodes input+output together: a question and its answer describe
            # one topic, and splitting them would fragment that signal.
            embedding=self._embed(
                " ".join(t for t in (input_text, output_text) if t)
            ),
        )
        try:
            await self.long_term.write_episodic(entry)
        except Exception as exc:
            logger.warning("Failed to write episodic memory: %s", exc)

        # Short-term (conversation turn)
        if session_id and input_text:
            try:
                await self.short_term.add_turn(session_id, "user", input_text)
            except Exception as exc:
                logger.warning("Failed to add user turn to short-term: %s", exc)

        if session_id and output_text:
            try:
                await self.short_term.add_turn(session_id, "assistant", output_text)
            except Exception as exc:
                logger.warning("Failed to add assistant turn to short-term: %s", exc)

    async def save_user_profile(
        self,
        user_id: str,
        profile: dict[str, Any],
        confidence: float = 0.8,
    ) -> None:
        """Save or update user profile in semantic memory."""
        entry = SemanticEntry(
            user_id=user_id,
            memory_type=MemoryType.USER_PREFERENCE,
            key="user_profile",
            value=profile,
            confidence=confidence,
        )
        try:
            await self.long_term.write_semantic(entry)
        except Exception as exc:
            logger.warning("Failed to save user profile: %s", exc)

    async def clear_session(self, session_id: str) -> None:
        """Clear a session from short-term memory."""
        await self.short_term.delete_session(session_id)
