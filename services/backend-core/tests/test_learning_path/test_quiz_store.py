"""Redis-backed ``QuizStore`` — real store code over an in-memory Redis.

``fakeredis`` is a real Redis server implementation in-process: ``GETDEL`` is
implemented the same way, so the atomic single-use path is exercised rather
than mocked.  What is *not* exercised here is latency/network behaviour — that
is the integration job in CI.
"""

from __future__ import annotations

import asyncio
import logging


from src.agents.models import QuizQuestion
from src.learning_path.quiz_store import QuizStore
from tests.support.fake_redis import new_async_redis


def _questions() -> list[QuizQuestion]:
    return [
        QuizQuestion(
            id="q1", type="choice", content="1+1?",
            correct_answer="2", knowledge_point_id="kp-x",
        ),
        QuizQuestion(
            id="q2", type="true_false", content="1+1=3?",
            correct_answer="错误", knowledge_point_id="kp-x",
        ),
    ]


class TestQuizStoreOnRedis:
    """``save`` / ``consume`` / ``peek`` over a shared Redis."""

    def test_questions_survive_into_another_store_instance(self):
        """The whole point of Redis: the instance that saved is not the one that reads."""
        redis = new_async_redis()
        writer = QuizStore(redis)
        reader = QuizStore(redis)  # another process / worker

        async def scenario():
            quiz_id = await writer.save(_questions())
            return await reader.consume(quiz_id)

        got = asyncio.run(scenario())
        assert got is not None
        assert [q.id for q in got] == ["q1", "q2"]
        assert {q.id: q.correct_answer for q in got}["q2"] == "错误"

    def test_consume_is_single_use(self):
        async def scenario():
            store = QuizStore(new_async_redis())
            quiz_id = await store.save(_questions())
            first = await store.consume(quiz_id)
            second = await store.consume(quiz_id)
            return first, second

        first, second = asyncio.run(scenario())
        assert first is not None
        assert second is None

    def test_concurrent_consumers_cannot_both_win(self):
        """``GETDEL`` — exactly one of two simultaneous grades gets the questions.

        With a read-then-write the interleaving below delivers to both.
        """
        async def scenario():
            store = QuizStore(new_async_redis())
            quiz_id = await store.save(_questions())
            results = await asyncio.gather(store.consume(quiz_id), store.consume(quiz_id))
            return results

        results = asyncio.run(scenario())
        assert sum(r is not None for r in results) == 1

    def test_peek_does_not_burn(self):
        async def scenario():
            store = QuizStore(new_async_redis())
            quiz_id = await store.save(_questions())
            peeked = await store.peek(quiz_id)
            consumed = await store.consume(quiz_id)
            return peeked, consumed

        peeked, consumed = asyncio.run(scenario())
        assert peeked is not None and consumed is not None

    def test_unknown_id_is_none(self):
        assert asyncio.run(QuizStore(new_async_redis()).consume("nope")) is None

    def test_ttl_is_set_on_the_key(self):
        """A leaked id must expire by itself — no sweeper process."""
        async def scenario():
            redis = new_async_redis()
            store = QuizStore(redis, ttl_seconds=120)
            quiz_id = await store.save(_questions())
            return await redis.ttl(f"quiz:{quiz_id}")

        assert 0 < asyncio.run(scenario()) <= 120

    def test_expired_quiz_is_refused(self):
        """TTL is the only thing that closes a never-submitted quiz."""
        async def scenario():
            redis = new_async_redis()
            store = QuizStore(redis, ttl_seconds=1)
            quiz_id = await store.save(_questions())
            await redis.delete(f"quiz:{quiz_id}")  # what expiry does, deterministically
            return await store.consume(quiz_id)

        assert asyncio.run(scenario()) is None

    def test_questions_are_gone_from_redis_after_consume(self):
        """Burning must leave no answer key reachable under the quiz key.

        NOTE ON SCOPE: this asserts the *end state* after ``consume`` returns —
        the key holds the tombstone, not the questions.  It cannot distinguish
        ``getdel`` from ``get`` at the call site, because ``_arm_tombstone``
        overwrites the payload either way; the mutation that swaps them is
        therefore annotated equivalent in ``scripts/mutation_check.py``.

        The property ``getdel`` actually buys is atomicity: with ``get`` the
        payload survives for the duration between the read and the tombstone
        write.  No test here can observe that window — ``fakeredis`` is
        in-process and its commands are not coroutines, so they never yield and
        ``gather`` cannot interleave them (verified by running the
        ``getdel``/``get`` pair under ``asyncio.gather``).  A real-Redis
        integration test would be needed.
        """

        async def scenario():
            redis = new_async_redis()
            store = QuizStore(redis)
            quiz_id = await store.save(_questions())
            await store.consume(quiz_id)
            return await redis.get(f"quiz:{quiz_id}")

        raw = asyncio.run(scenario())
        assert raw == '{"consumed": true}'
        assert "correct_answer" not in raw


class TestReplayIsLogged:
    """A replayed id is refused **and** logged as a replay.

    The distinction matters operationally: "unknown id" is usually a typo or a
    stale tab, while "replayed" means someone is retrying a spent attempt —
    which is the signal the single-use rule exists to produce.
    """

    def test_replay_logs_a_warning(self, caplog):
        async def scenario():
            store = QuizStore(new_async_redis())
            quiz_id = await store.save(_questions())
            await store.consume(quiz_id)
            with caplog.at_level("WARNING", logger="src.learning_path.quiz_store"):
                await store.consume(quiz_id)

        asyncio.run(scenario())
        assert any("replayed after being consumed" in r.message for r in caplog.records)

    def test_unknown_id_does_not_log_a_replay(self, caplog):
        async def scenario():
            store = QuizStore(new_async_redis())
            with caplog.at_level("WARNING", logger="src.learning_path.quiz_store"):
                await store.consume("never-issued")

        asyncio.run(scenario())
        assert not any("replayed" in r.message for r in caplog.records)


class TestNoRedisOption:
    """Redis down ⇒ no store ⇒ the endpoints fail closed.

    There is deliberately no in-process fallback: a fallback would silently
    reintroduce the single-process handshake this store exists to remove, and
    would make a multi-worker deployment answer correctly *sometimes*.

    These tests cover the *store* half (no entries after a lost Redis).  The
    wiring half — "main never attaches a store without Redis" — is in
    ``test_quiz_store_signature.py``.
    """

    def test_a_lost_redis_means_no_quiz_survives(self):
        """A restart of Redis is a restart of the quiz state, not a fallback."""
        async def scenario():
            redis = new_async_redis()
            store = QuizStore(redis)
            quiz_id = await store.save(_questions())
            await redis.flushall()                 # what a Redis restart does
            return await store.consume(quiz_id)

        assert asyncio.run(scenario()) is None

    def test_main_does_not_create_a_store_without_redis(self):
        """The Redis wiring itself is pinned by ``test_quiz_store_signature.py``.

        It lives in its own file because it constrains ``main.lifespan``, not the
        store, and because it was first written here with ``import main`` — which
        passes on a dev machine (the backend dir is on ``sys.path``) and fails in
        CI (only the repo root is), the "test call pattern ≠ production" trap.
        """


class TestReplayIsBounded:
    """重放不能成为「续命」或刷屏的手段。

    - 每次重放都写一个墓碑，但只用 ``NX``：已存在的墓碑保持旧过期时间，
      否则客户端只要不停重试，id 就永远不过期。
    - 同一个 id 的重放日志最多 ``_REPLAY_LOG_LIMIT`` 条，之后静默；
      计数器本身也带 TTL，不会成为第二份可增长的存储。
    """

    def test_repeated_replays_do_not_extend_the_tombstone_ttl(self):
        async def scenario():
            redis = new_async_redis()
            store = QuizStore(redis, ttl_seconds=60)
            quiz_id = await store.save(_questions())
            await store.consume(quiz_id)
            key = f"quiz:{quiz_id}"

            await asyncio.sleep(1.1)                 # let the clock advance
            await store.consume(quiz_id)             # replay re-arms the tombstone
            rearmed = await redis.ttl(key)

            await asyncio.sleep(1.1)
            await store.consume(quiz_id)             # and again
            again = await redis.ttl(key)
            return rearmed, again

        rearmed, again = asyncio.run(scenario())
        assert rearmed <= 59, f"re-arm must use the remaining time, got {rearmed}s"
        assert again < rearmed, "each replay must move the expiry *closer*, never reset it"

    def test_replay_logging_goes_quiet(self, caplog):
        # Wrap the coroutine, not the store: caplog's handler is thread/context
        # local and never sees records emitted from a fresh ``asyncio.run`` loop.
        async def scenario():
            store = QuizStore(new_async_redis())
            quiz_id = await store.save(_questions())
            await store.consume(quiz_id)
            for _ in range(10):
                await store.consume(quiz_id)

        # A handler attached on the same thread's loop; caplog's own handler
        # is context-local and misses records from another ``asyncio.run`` loop.
        logger = logging.getLogger("src.learning_path.quiz_store")
        with caplog.at_level("WARNING", logger="src.learning_path.quiz_store"):
            asyncio.run(scenario())
        assert logger.propagate, "test requires the logger to propagate to caplog"
        replayed = [r for r in caplog.records if "replayed" in r.message]
        assert len(replayed) == 4, (
            "expected 3 replays + 1 'going quiet' notice; an unbounded loop must not keep logging"
        )
