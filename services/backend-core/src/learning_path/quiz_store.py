"""Redis-backed store for generated quiz questions and their answer keys.

Why this exists
---------------
``POST /quiz/generate`` used to hand the client the full ``QuizQuestion``
model — ``correct_answer`` included — and ``POST /quiz/grade`` then accepted
the *questions themselves* in the request body and graded against whatever
answer key that body carried.  The client supplied both the exam paper and the
answer key, and the server's only contribution was to stamp it "graded".

The fix keeps the key where the client cannot reach it: generation stores the
questions under an opaque id; grading takes only that id plus the learner's
answers, and looks the key up server-side.

Why Redis rather than a process-local cache
-------------------------------------------
This store started as an ``TTLLRUCache`` on ``app.state``.  That made the
answer key (and the ``/quiz/generate`` → ``/quiz/grade`` handshake) depend on
both calls landing in the same process — one of the blockers to running more
than one uvicorn worker.  Redis gives the handshake three properties a local
cache cannot:

* **cross-process** — generate on worker A, grade on worker B;
* **TTL** — a leaked id expires by itself (``EXPIRE``), no sweep loop;
* **atomic single-use** — ``GETDEL`` in :meth:`consume` means two concurrent
  submissions of the same id cannot both grade against it.

Scope
-----
Keys live under ``quiz:``; the answer key is visible to anyone who can read
Redis — the same trust boundary as the database.  The client cannot.

Async API
---------
The endpoints are async; :meth:`save` / :meth:`consume` / :meth:`peek` are
coroutines so the store can move to a different backend later without touching
call sites.  A synchronous facade (``SyncQuizStore``) exists for tests and
scripts that do not run an event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from redis.asyncio import Redis

from src.agents.models import QuizQuestion

logger = logging.getLogger(__name__)

# Long enough for a learner to finish reading 1-3 questions, short enough that
# a leaked id is useless by the time anyone finds it.
DEFAULT_TTL_SECONDS = 900

# How long a *consumed* id is remembered so a replay gets a distinct log line
# ("replayed") instead of looking like a typo.  It does not extend access: the
# entry is a tombstone, not the questions.
_CONSUMED_MARK = '{"consumed": true}'


class QuizStore:
    """Short-lived, single-use store of quiz questions keyed by an opaque id."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = "quiz:",
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def _key(self, quiz_id: str) -> str:
        return f"{self._prefix}{quiz_id}"

    async def save(self, questions: list[QuizQuestion]) -> str:
        """Store ``questions`` and return the opaque id that retrieves them.

        The id is a random UUID4 — 122 bits, consistent with the session-id
        convention in ``agents/orchestrator/router.py``.  It is a bearer
        capability: whoever holds it can grade against these questions, so it
        must not be guessable from the knowledge point alone.
        """
        quiz_id = str(uuid.uuid4())
        payload = json.dumps([q.model_dump() for q in questions])
        await self._redis.set(self._key(quiz_id), payload, ex=self._ttl)
        return quiz_id

    async def consume(self, quiz_id: str) -> list[QuizQuestion] | None:
        """Return the questions for ``quiz_id`` and burn the entry.

        Returns ``None`` when the id is unknown, expired, or has already been
        consumed — the caller cannot distinguish those, and should not need to:
        all three mean "generate a fresh quiz".

        ``GETDEL`` is a single Redis operation, so the read-and-burn is atomic:
        two simultaneous submissions of the same id cannot both receive the
        questions.  The tombstone is written back only so a later replay is
        logged as a replay; it never contains the questions.

        Burning on *read* (rather than after grading succeeds) is deliberate.
        Grading is a pure function of (questions, answers) and cannot fail once
        the questions are in hand, so there is no meaningful "graded but should
        stay retryable" state to preserve.
        """
        raw = await self._redis.getdel(self._key(quiz_id))
        if raw is None:
            if await self._redis.exists(self._key(quiz_id)):
                logger.warning("Quiz id %s replayed after being consumed", quiz_id)
            return None
        if raw == _CONSUMED_MARK:
            logger.warning("Quiz id %s replayed after being consumed", quiz_id)
            return None
        await self._redis.set(self._key(quiz_id), _CONSUMED_MARK, ex=self._ttl)
        return [QuizQuestion(**item) for item in json.loads(raw)]

    async def peek(self, quiz_id: str) -> list[QuizQuestion] | None:
        """Look up ``quiz_id`` *without* consuming it.

        For tests and debugging only — never call this on the grading path, or
        the single-use guarantee is lost.
        """
        raw = await self._redis.get(self._key(quiz_id))
        if raw is None or raw == _CONSUMED_MARK:
            return None
        return [QuizQuestion(**item) for item in json.loads(raw)]


class SyncQuizStore:
    """Blocking facade over :class:`QuizStore` for tests and scripts.

    Runs each operation on a private event loop.  Not for request handlers —
    they use the async store directly.
    """

    def __init__(self, redis: Redis, **kwargs: Any) -> None:
        self._inner = QuizStore(redis, **kwargs)

    @staticmethod
    def _run(coro: Any) -> Any:
        return asyncio.run(coro)

    def save(self, questions: list[QuizQuestion]) -> str:
        return self._run(self._inner.save(questions))

    def consume(self, quiz_id: str) -> list[QuizQuestion] | None:
        return self._run(self._inner.consume(quiz_id))

    def peek(self, quiz_id: str) -> list[QuizQuestion] | None:
        return self._run(self._inner.peek(quiz_id))


def peek_sync(store: QuizStore, quiz_id: str) -> list[QuizQuestion] | None:
    """Blocking ``peek`` for tests that drive the app through ``TestClient``.

    Kept next to the store (not in the test file) so the single event-loop
    bridge lives beside the async API it wraps.
    """
    return asyncio.run(store.peek(quiz_id))
