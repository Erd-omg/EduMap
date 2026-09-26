"""Server-side store for generated quiz questions and their answer keys.

Why this exists
---------------
``POST /quiz/generate`` used to hand the client the full ``QuizQuestion``
model — ``correct_answer`` included — and ``POST /quiz/grade`` then accepted
the *questions themselves* in the request body and graded against whatever
answer key that body carried.  The client supplied both the exam paper and the
answer key, and the server's only contribution was to stamp it "graded".

That made the score, and the mastery estimate derived from it, self-certified.
A learner could read ``correct_answer`` out of devtools; anyone at all could
POST a synthetic quiz whose keys they chose and receive a perfect score, which
``POST /progress`` then recorded as ``status='completed'`` — enough to unlock
the dependent knowledge points in the path (``PathService._determine_status``),
without any mastery evidence at all.

The fix is to keep the answer key where the client cannot reach it.  Generation
stores the questions here under an opaque id; grading takes only that id plus
the learner's answers, and looks the key up server-side.

Storage choice
--------------
In-process :class:`~src.utils.ttl_cache.TTLLRUCache`.  The deployment runs a
single uvicorn process (no ``--workers`` in the Dockerfile or compose file), so
a process-local cache is correct; Redis would add serialisation overhead for a
value that only lives 15 minutes, and Postgres has no TTL/cleanup precedent in
this repo (nothing in ``scripts/db/init`` expires rows).
"""

from __future__ import annotations

import logging
import uuid

from src.agents.models import QuizQuestion
from src.utils.ttl_cache import TTLLRUCache

logger = logging.getLogger(__name__)

# Long enough for a learner to finish reading 1-3 questions, short enough that
# a leaked id is useless by the time anyone finds it.
DEFAULT_TTL_SECONDS = 900.0
DEFAULT_MAXSIZE = 256

# Marks a consumed quiz.  ``TTLLRUCache`` has no single-key delete, so a read
# that must not be repeatable overwrites the entry with this sentinel instead:
# a replay finds something truthy but is refused.  Storing a sentinel rather
# than deleting also keeps the emptiness unambiguous — "expired or never
# existed" and "already used" are distinct outcomes we may want to tell apart.
_CONSUMED = object()


class QuizStore:
    """Short-lived, single-use store of quiz questions keyed by an opaque id."""

    def __init__(
        self,
        maxsize: int = DEFAULT_MAXSIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        self._cache: TTLLRUCache[str, object] = TTLLRUCache(
            maxsize=maxsize, ttl_seconds=ttl_seconds
        )

    def save(self, questions: list[QuizQuestion]) -> str:
        """Store ``questions`` and return the opaque id that retrieves them.

        The id is a random UUID4 — 122 bits, consistent with the session-id
        convention in ``agents/orchestrator/router.py``.  It is a bearer
        capability: whoever holds it can grade against these questions, so it
        must not be guessable from the knowledge point alone.
        """
        quiz_id = str(uuid.uuid4())
        self._cache.put(quiz_id, list(questions))
        return quiz_id

    def consume(self, quiz_id: str) -> list[QuizQuestion] | None:
        """Return the questions for ``quiz_id`` and burn the entry.

        Returns ``None`` when the id is unknown, expired, or has already been
        consumed — the caller cannot distinguish those, and should not need to:
        all three mean "generate a fresh quiz".

        Marking consumed on *read* (rather than after grading succeeds) is
        deliberate.  Grading is a pure function of (questions, answers) and
        cannot fail once the questions are in hand, so there is no meaningful
        "graded but should stay retryable" state to preserve.
        """
        entry = self._cache.get(quiz_id)
        if entry is None:
            return None
        if entry is _CONSUMED:
            logger.warning("Quiz id %s replayed after being consumed", quiz_id)
            return None
        self._cache.put(quiz_id, _CONSUMED)
        return entry  # type: ignore[return-value]

    def peek(self, quiz_id: str) -> list[QuizQuestion] | None:
        """Look up ``quiz_id`` *without* consuming it.

        For tests and debugging only — never call this on the grading path, or
        the single-use guarantee is lost.
        """
        entry = self._cache.get(quiz_id)
        if entry is None or entry is _CONSUMED:
            return None
        return entry  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._cache)
