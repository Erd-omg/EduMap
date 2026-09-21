"""Tests for the multi-path fused intent classifier (src/analysis/intent.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.intent import (
    INTENT_CACHE,
    IntentResult,
    LLMIntentOutput,
    classify_intent,
    contains_profile_intent,
    is_question,
)


def _make_request(llm=None, embedding_model=None):
    state = SimpleNamespace()
    if llm is not None:
        state.llm_adapter = llm
    if embedding_model is not None:
        state._embedding_model = embedding_model
    return SimpleNamespace(app=SimpleNamespace(state=state))


class _FakeLLM:
    def __init__(self, output: LLMIntentOutput | None = None, error: Exception | None = None):
        self.output = output
        self.error = error
        self.calls = 0

    async def generate_structured(self, prompt, schema, system_prompt=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output


class _FakeEmbeddingModel:
    """Deterministic hash-based embeddings — enough to exercise the path."""

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.array(
            [[float(hash(t) % 97), float((hash(t) // 97) % 89)] for t in texts],
            dtype="float32",
        )
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


# ── Rule heuristics ──────────────────────────────────────────────────────


def test_contains_profile_intent_positive():
    assert contains_profile_intent("我是计算机专业大三的学生")
    assert contains_profile_intent("我学过Python和一点数据结构")


def test_contains_profile_intent_negative_for_knowledge_questions():
    # Ordinary knowledge questions must NOT look profile-oriented.
    assert not contains_profile_intent("学习链表之前我应该先掌握什么？")
    assert not contains_profile_intent("什么是时间复杂度？")


def test_is_question():
    assert is_question("链表和数组有什么区别？")
    assert is_question("为什么需要虚拟内存")


# ── Rule-path regressions (from the n=72 labeled eval) ────────────────────
# These lock in three fixes that took cold-start accuracy from 75.0% to 91.7%
# on src/rag/evaluation/datasets/intent_labeled.json.


def test_topic_mention_in_self_disclosure_is_profile_not_mixed():
    """A subject mention inside a self-disclosure is not a question.

    ``has_topic`` used to force ``mixed`` for any self-disclosure naming a
    subject, which flipped 8/72 labeled rows from profile to mixed.
    """
    from src.analysis.intent import _rule_vote

    assert _rule_vote("我学过 Python 和一点数据结构", has_topic=True) == ("profile", 0.80)
    assert _rule_vote("我熟悉 Java 后端开发", has_topic=True) == ("profile", 0.80)


def test_broadened_disclosure_detection():
    """Disclosures phrased as 我只会/我平时/我今年 must count as profile."""
    assert contains_profile_intent("我只会写 SQL 查询")
    assert contains_profile_intent("我平时用 Java 比较多")
    assert contains_profile_intent("我今年大三")
    # …and must not swallow plain questions.
    assert not contains_profile_intent("为什么数组的随机访问是 O(1)？")


def test_disclosure_plus_question_is_mixed():
    from src.analysis.intent import _rule_vote

    assert _rule_vote("我只会写 SQL 查询，怎么开始学数据库优化？", has_topic=True) == ("mixed", 0.80)
    assert _rule_vote("我对图论完全不熟，最短路径算法该怎么理解？", has_topic=True) == ("mixed", 0.80)


def test_pure_question_still_fast_paths():
    from src.analysis.intent import _rule_vote

    assert _rule_vote("什么是时间复杂度？", has_topic=False) == ("question", 0.90)
    assert not is_question("我是大二学生，学过C语言")


# ── Rule fast path (no model calls) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rule_fast_path_question():
    llm = _FakeLLM()
    request = _make_request(llm=llm)
    result = await classify_intent(
        "什么是虚拟内存？", request, user_id="u1", use_cache=False
    )
    assert isinstance(result, IntentResult)
    assert result.intent == "question"
    assert result.path == "rule"
    assert llm.calls == 0  # fast path: no LLM call


@pytest.mark.asyncio
async def test_rule_fast_path_profile():
    llm = _FakeLLM()
    request = _make_request(llm=llm)
    result = await classify_intent(
        "我是数学专业的大一新生，线性代数学得很吃力",
        request,
        user_id="u1",
        has_topic=False,
        use_cache=False,
    )
    assert result.intent == "profile"
    assert result.path == "rule"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_rule_fast_path_mixed():
    llm = _FakeLLM()
    request = _make_request(llm=llm)
    result = await classify_intent(
        "我学过Python基础，接下来应该学什么？",
        request,
        user_id="u1",
        has_topic=True,
        use_cache=False,
    )
    assert result.intent == "mixed"
    assert result.path == "rule"


# ── Fused path ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fused_llm_vote_wins_on_ambiguity():
    # No disclosure keyword, no question marker → rule path abstains,
    # LLM + embedding paths vote.
    llm = _FakeLLM(LLMIntentOutput(intent="mixed", confidence=0.9, reason="both"))
    request = _make_request(llm=llm, embedding_model=_FakeEmbeddingModel())
    result = await classify_intent(
        "帮我看看这个", request, user_id="u2", use_cache=False
    )
    assert result.path == "fused"
    assert result.intent == "mixed"
    assert "llm" in result.votes
    assert "embedding" in result.votes
    # LLM carries weight 2.0, embedding 1.2 — LLM should dominate.
    assert result.confidence > 0.5


@pytest.mark.asyncio
async def test_fused_falls_back_to_question_without_models():
    # Ambiguous message, no LLM and no embedding model available.
    request = _make_request()
    result = await classify_intent(
        "看看这个", request, user_id="u3", use_cache=False
    )
    assert result.intent == "question"
    assert result.confidence == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_llm_failure_degrades_gracefully():
    llm = _FakeLLM(error=RuntimeError("boom"))
    request = _make_request(llm=llm, embedding_model=_FakeEmbeddingModel())
    result = await classify_intent(
        "看看这个", request, user_id="u4", use_cache=False
    )
    # LLM vote skipped; embedding vote (or fallback) still yields an intent.
    assert result.intent in {"profile", "question", "mixed"}
    assert "llm" not in result.votes


# ── LRU/TTL cache ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hits_on_repeat_message():
    INTENT_CACHE._store.clear()
    llm = _FakeLLM(LLMIntentOutput(intent="question", confidence=0.9))
    request = _make_request(llm=llm)

    first = await classify_intent(
        "帮我看看这个", request, user_id="u5", use_cache=True
    )
    assert first.path == "fused"

    second = await classify_intent(
        "帮我看看这个", request, user_id="u5", use_cache=True
    )
    assert second.path == "cache"
    assert second.intent == first.intent
    assert llm.calls == 1  # second call served from cache


@pytest.mark.asyncio
async def test_cache_evicts_on_lru_overflow():
    INTENT_CACHE._store.clear()
    small_cache = INTENT_CACHE  # maxsize 512 — exercise eviction via direct API
    for i in range(600):
        small_cache.put((f"user", f"msg{i}"), "x")
    assert len(small_cache._store) <= 512


# ── Real endpoint: GET /api/v1/analysis/stream/{user_id} ────────────────
#
# Every test above calls classify_intent() with a hand-built request and a
# hand-supplied `has_topic`.  That leaves the wiring in the endpoint untested:
# src/analysis/router.py:349 computes `has_topic = bool(_extract_topics(msg))`
# and passes it down, then emits the classification as an SSE `intent` event.
# A divergence between _extract_topics and the rule vote — or a regression in
# the SSE contract the frontend reads — is invisible to the unit tests.


class TestAnalysisStreamEndpoint:
    """Drive the real SSE endpoint, not classify_intent() directly."""

    @staticmethod
    def _collect_sse_events(message: str) -> list[dict]:
        """GET the stream endpoint and parse its SSE frames."""
        import json

        from fastapi.testclient import TestClient

        from tests.test_api.conftest import _create_test_app

        app = _create_test_app()
        with TestClient(app) as client:
            with client.stream(
                "GET",
                "/api/v1/analysis/stream/u-test",
                params={"message": message},
            ) as resp:
                assert resp.status_code == 200
                events, current = [], {}
                for raw in resp.iter_lines():
                    line = raw if isinstance(raw, str) else raw.decode()
                    if line.startswith("event: "):
                        current["event"] = line[len("event: "):]
                    elif line.startswith("data: "):
                        current["data"] = json.loads(line[len("data: "):])
                    elif line == "" and current:
                        events.append(current)
                        current = {}
        return events

    def test_intent_event_is_emitted_first(self) -> None:
        """The stream must lead with an `intent` event (frontend contract)."""
        events = self._collect_sse_events("什么是时间复杂度？")
        assert events, "no SSE events emitted"
        assert events[0]["event"] == "intent"

    def test_intent_event_carries_classification_fields(self) -> None:
        """SSE payload shape the frontend's IntentBadge reads."""
        events = self._collect_sse_events("什么是时间复杂度？")
        payload = events[0]["data"]
        for field in ("intent", "confidence", "path", "votes"):
            assert field in payload, f"missing {field!r} in intent event"

    def test_pure_question_routes_to_mentor(self) -> None:
        """A clear knowledge question must classify as `question`."""
        events = self._collect_sse_events("什么是时间复杂度？")
        assert events[0]["data"]["intent"] == "question"

    def test_self_disclosure_routes_to_profile(self) -> None:
        """Pure background disclosure must classify as `profile`."""
        events = self._collect_sse_events("我是计算机专业大三的学生")
        assert events[0]["data"]["intent"] == "profile"

    def test_disclosure_plus_question_routes_to_mixed(self) -> None:
        """THE regression the fused classifier was built for.

        Previously mis-routed to profile-only, which swallowed the question.
        """
        events = self._collect_sse_events(
            "我学过链表，请问什么是二叉树？"
        )
        assert events[0]["data"]["intent"] == "mixed"

    def test_has_topic_is_derived_from_message_not_assumed(self) -> None:
        """End-to-end: `has_topic` comes from _extract_topics on the real text.

        A self-disclosure naming a subject must NOT become `mixed` just
        because a topic keyword appears — `mixed` requires an interrogative.
        This is the link the unit tests bypass by passing has_topic by hand.
        """
        events = self._collect_sse_events("我学过 Python 和一点数据结构")
        assert events[0]["data"]["intent"] == "profile"

    def test_stream_terminates_with_complete(self) -> None:
        """The stream must end, not hang (terminal-event invariant)."""
        events = self._collect_sse_events("什么是时间复杂度？")
        assert events[-1]["event"] in ("complete", "error")
