"""Redis-backed ``QuizStore`` — real store code over an in-memory Redis.

``fakeredis`` is a real Redis server implementation in-process: ``GETDEL`` is
implemented the same way, so the atomic single-use path is exercised rather
than mocked.  What is *not* exercised here is latency/network behaviour — that
is the integration job in CI.
"""

from __future__ import annotations

import asyncio


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
        """Burning must delete the payload, not merely hide it.

        ``getdel`` is what makes this true; a ``get`` + tombstone leaves the
        answer key in Redis for the whole TTL, reachable by anything that can
        read the key — including a replay that bypasses the tombstone check.
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
    """

    def test_main_does_not_create_a_store_without_redis(self):
        import inspect

        import main as main_module

        lifespan_src = inspect.getsource(main_module.lifespan)
        assert "redis_client = getattr(app.state, \"redis_client\", None)" in lifespan_src
        assert "QuizStore(redis_client)" in lifespan_src
        assert "QuizStore()" not in lifespan_src, "an argument-less QuizStore would be process-local again"
