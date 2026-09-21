"""Multi-path fused intent recognition.

Fuses three voting paths to classify a user message as one of:

- ``profile`` — self-disclosure about background / goals / ability →
  profile analysis only
- ``question`` — knowledge question → Mentor RAG answer
- ``mixed`` — both → profile update + mentor answer in one stream

Paths:

1. **Rule path** — keyword/regex heuristics.  Decisive (returned directly)
   on clear-cut messages; otherwise contributes a weaker vote.
2. **LLM semantic path** — structured LLM classification, only invoked when
   the rule path is ambiguous.  Guarded by a timeout and skipped silently
   when the adapter is unavailable or returns garbage.
3. **Embedding path** — cosine similarity against few-shot labeled examples
   (uses the app-wide SentenceTransformer model when loaded).

Results are fused by weighted voting.  A small LRU+TTL cache keyed on
``(user_id, normalized_text)`` avoids re-inferring high-frequency duplicate
messages.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.utils.ttl_cache import TTLLRUCache

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)

# ── Intent space ─────────────────────────────────────────────────────────

IntentName = Literal["profile", "question", "mixed"]

# ── Rule path: keyword heuristics (moved from router.py) ─────────────────

# Strong background-disclosure signals.  Deliberately excludes single
# characters / generic verbs like 我, 学习, 掌握, 之前 — those appear in
# ordinary knowledge questions ("学习链表之前我应该先掌握什么？") and must
# not mark a message as profile-oriented.
BACKGROUND_SIGNALS = [
    "我的", "专业", "年级", "学校", "学过", "熟悉", "擅长",
    "弱项", "困难", "不懂", "背景", "基础", "水平", "能力",
    "掌握程度", "感兴趣",
]

# Interrogative markers — a message containing any of these is a *question*
# and should reach the Mentor (RAG) answer path rather than being swallowed
# by profile analysis.
QUESTION_MARKERS = [
    "什么", "怎么", "如何", "为什么", "为啥", "哪些", "哪个", "怎样",
    "吗", "呢", "对不对", "是不是", "有没有", "可不可以", "该不该",
    "区别", "对比", "原理", "含义", "怎么办", "步骤",
    "先学", "先掌握", "需要掌握",
]

_PROFILE_DISCLOSURE_RE = re.compile(
    r"(我是|我叫|我在学|我正在|我目前|我学过|我以前|我打算|我想学|我准备)"
)
# Broader first-person state/ability disclosure.  These phrasings describe the
# learner ("我只会写 SQL" / "我平时用 Java" / "我今年大三" / "我对图论完全不熟")
# and were previously missed, causing genuine ``mixed`` messages to be routed
# to ``question`` because no disclosure was detected.
#
# The first-person subject and the state marker are separated by an optional
# short noun phrase (``我对图论完全不熟`` — "about graph theory, not familiar"),
# hence ``.{0,8}?``.  Kept deliberately narrow: it still requires a first-person
# subject, so ordinary questions ("为什么数组随机访问是 O(1)") don't match.
_STATE_MARKERS = (
    r"只会|只学过|只写过|平时|一般|今年|已经学|比较擅|擅长|不太好|不太会"
    r"|没系统学|没学过|完全没有|完全不熟|完全不懂|不熟|不懂|很感兴趣"
    r"|(?:基础|底子|数学|英语|算法|编程)(?:比较|很|挺|还|不)?(?:差|好|弱|薄弱|一般|不好|行)"
)
_PROFILE_STATE_RE = re.compile(rf"我.{{0,8}}?(?:{_STATE_MARKERS})")
_PROFILE_SELF_RE = re.compile(
    r"(我的|自己的)(背景|水平|基础|情况|目标|专业|能力|弱点|困难|学习风格)"
)


def contains_profile_intent(text: str) -> bool:
    """Heuristic: does the message look profile-oriented?

    Requires an explicit self-disclosure (我是/我在学/我的背景 …) or a
    strong background-signal word.  Generic tokens like 我/学习/掌握 no
    longer count on their own.
    """
    if _PROFILE_DISCLOSURE_RE.search(text):
        return True
    if _PROFILE_SELF_RE.search(text):
        return True
    if _PROFILE_STATE_RE.search(text):
        return True
    return any(kw in text for kw in BACKGROUND_SIGNALS)


def is_question(text: str) -> bool:
    """Heuristic: does the message look like a question (not a statement)?"""
    if "?" in text or "？" in text:
        return True
    return any(marker in text for marker in QUESTION_MARKERS)


# ── LLM path: structured classification ──────────────────────────────────

LLM_TIMEOUT_SECONDS = 5.0

_INTENT_SYSTEM_PROMPT = """\
你是 EduMap 学习助手的消息意图分类器。将用户消息分类为以下三种意图之一：

- profile: 用户在陈述个人背景、学习经历、能力水平、目标或学习偏好，
  目的是让系统了解自己（例如自我介绍、说明基础、表达目标）。
- question: 用户在提出计算机/数学等知识类问题，期望获得知识讲解。
- mixed: 消息中既有个人背景陈述，又包含需要回答的知识问题，
  应同时更新画像并给出知识解答。

注意：即使消息提到了学习计划或学习路径（"我想学X应该怎么学"），
只要其核心是请求建议/解答，就归为 question 或 mixed；
只有纯背景陈述且不含任何提问，才归为 profile。
只输出 JSON，不要输出其他内容。"""


class LLMIntentOutput(BaseModel):
    """Structured output contract for the LLM intent path."""

    intent: IntentName
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = ""


# ── Embedding path: few-shot similarity voting ───────────────────────────

INTENT_EXAMPLES: dict[str, list[str]] = {
    "profile": [
        "我是计算机专业大三的学生",
        "我学过Python和一点数据结构",
        "我的数学基础比较薄弱",
        "我熟悉Java后端开发",
        "我打算考研，需要复习408",
        "我以前是做测试的，想转开发",
        "我对前端很感兴趣，但没系统学过",
        "我英语还行，看文档没什么问题",
    ],
    "question": [
        "什么是时间复杂度？",
        "链表和数组有什么区别？",
        "为什么需要虚拟内存？",
        "解释一下二分查找的原理",
        "怎么实现一个LRU缓存？",
        "TCP三次握手的过程是什么",
        "动态规划和贪心算法的区别",
        "什么是数据库事务的ACID特性",
    ],
    "mixed": [
        "我学过Python基础，接下来应该学什么？",
        "我是前端新手，React的状态管理该怎么理解？",
        "我数学不好，学机器学习前需要补哪些知识？",
        "我会一点C语言，指针和内存那块总是搞不懂，能讲讲吗？",
        "我是大二学生，学操作系统之前要先掌握什么？",
        "我只会写SQL查询，怎么开始学数据库优化？",
    ],
}


def _cosine_similarity(a, b) -> float:
    """Cosine similarity for 1-D numpy arrays (normalized inputs → dot)."""
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Example-embedding cache keyed by id(model) so a reloaded model recomputes.
_EXAMPLE_EMB_CACHE: dict[int, dict[str, list]] = {}


def _embedding_vote(text: str, model) -> tuple[str, float] | None:
    """Vote by cosine similarity against few-shot examples.

    Returns ``(intent, confidence)`` or ``None`` when the model is missing
    or encoding fails.
    """
    if model is None:
        return None
    try:
        cache_key = id(model)
        example_embs = _EXAMPLE_EMB_CACHE.get(cache_key)
        if example_embs is None:
            example_embs = {
                intent: model.encode(
                    examples, show_progress_bar=False, normalize_embeddings=True
                ).tolist()
                for intent, examples in INTENT_EXAMPLES.items()
            }
            _EXAMPLE_EMB_CACHE[cache_key] = example_embs

        query_emb = model.encode(
            [text], show_progress_bar=False, normalize_embeddings=True
        )[0].tolist()

        scores: dict[str, float] = {}
        for intent, embs in example_embs.items():
            sims = sorted(
                (_cosine_similarity(query_emb, emb) for emb in embs), reverse=True
            )
            # Top-3 mean: robust to a single lucky example.
            scores[intent] = sum(sims[:3]) / min(3, len(sims))

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values())
        confidence = scores[best] / total if total > 0 else 0.0
        return best, round(confidence, 4)
    except Exception as exc:
        logger.debug("Embedding intent vote skipped: %s", exc)
        return None


# ── Rule path scoring ────────────────────────────────────────────────────


def _rule_vote(text: str, has_topic: bool) -> tuple[str, float] | None:
    """Derive a rule-path intent vote.

    Returns ``None`` when the heuristics are ambiguous — the caller should
    escalate to the LLM / embedding paths.

    Tuning notes (driven by ``benchmark_results/intent_eval_*.json``, n=72):

    - ``has_topic`` alone must NOT produce ``mixed``.  A pure self-disclosure
      that happens to mention a subject ("我学过 Python 和一点数据结构")
      contains no question, so ``mixed`` requires an actual interrogative.
    - Background disclosure is broader than ``_PROFILE_DISCLOSURE_RE``:
      phrasings like "我只会写 SQL 查询" / "我平时用 Java" / "我今年大三" are
      disclosures too, and used to fall through to ``question``.
    - A confident ``question`` verdict is only safe when *no* disclosure is
      present; otherwise the message is ``mixed`` and must not be short-circuited.
    """
    profile_score = 0.05
    if (
        _PROFILE_DISCLOSURE_RE.search(text)
        or _PROFILE_SELF_RE.search(text)
        or _PROFILE_STATE_RE.search(text)
    ):
        profile_score = 0.85
    elif any(kw in text for kw in BACKGROUND_SIGNALS):
        profile_score = 0.65

    question_score = 0.05
    if "?" in text or "？" in text:
        question_score = 0.80
    elif any(marker in text for marker in QUESTION_MARKERS):
        question_score = 0.65

    has_disclosure = profile_score >= 0.6
    has_question = question_score >= 0.6

    # Both present → mixed.  Note ``has_topic`` is intentionally NOT a
    # question signal any more: a subject mention inside a self-disclosure
    # ("我学过链表") is not a question, and treating it as one flipped 8 of
    # the 72 labeled rows from ``profile`` to ``mixed``.
    if has_disclosure and has_question:
        return "mixed", 0.80
    if has_disclosure:
        return "profile", 0.80
    if has_question:
        return "question", 0.90
    return None


# ── LRU + TTL cache ──────────────────────────────────────────────────────

# The cache implementation lives in src/utils/ttl_cache.py so the RAG
# retrieval cache and the tool-result cache can share it.
INTENT_CACHE: TTLLRUCache = TTLLRUCache(maxsize=512, ttl_seconds=600.0)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ── Public API ───────────────────────────────────────────────────────────


@dataclass
class IntentResult:
    """Fused intent classification result, with per-path details."""

    intent: IntentName
    confidence: float
    path: str  # "cache" | "rule" | "fused"
    votes: dict = field(default_factory=dict)


async def classify_intent(
    text: str,
    request: "Request",
    *,
    user_id: str = "",
    has_topic: bool = False,
    use_cache: bool = True,
) -> IntentResult:
    """Classify a message into profile / question / mixed.

    Fast path: LRU/TTL cache hit → zero inference cost.
    Rule path: clear-cut messages resolve without any model call.
    Fused path: ambiguous messages vote across rule + LLM + embedding.
    """
    cache_key = (user_id, _normalize(text))
    if use_cache:
        cached = INTENT_CACHE.get(cache_key)
        if cached is not None:
            return IntentResult(
                intent=cached.intent,
                confidence=cached.confidence,
                path="cache",
                # Same shape as the other paths ({name: {intent, confidence}})
                # plus ``cached_from`` naming the path that produced the entry,
                # so consumers can render it uniformly.
                votes={
                    "cache": {
                        "intent": cached.intent,
                        "confidence": cached.confidence,
                        "cached_from": cached.path,
                    }
                },
            )

    rule_vote = _rule_vote(text, has_topic)

    # ── Fast path: rules are decisive ────────────────────────────────
    if rule_vote is not None:
        intent, confidence = rule_vote
        result = IntentResult(
            intent=intent, confidence=confidence, path="rule",
            votes={"rule": {"intent": intent, "confidence": confidence}},
        )
        INTENT_CACHE.put(cache_key, result)
        return result

    # ── Fused path: ambiguous message ────────────────────────────────
    votes: dict[str, dict] = {}

    # LLM semantic vote (weight 2.0 — the strongest signal when available)
    llm = getattr(request.app.state, "llm_adapter", None)
    if llm is not None:
        try:
            output = await asyncio.wait_for(
                llm.generate_structured(
                    f"用户消息：{text}",
                    LLMIntentOutput,
                    system_prompt=_INTENT_SYSTEM_PROMPT,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            votes["llm"] = {
                "intent": output.intent,
                "confidence": output.confidence,
                "weight": 2.0,
                "reason": output.reason,
            }
        except asyncio.TimeoutError:
            logger.warning("Intent LLM classification timed out")
        except Exception as exc:
            logger.debug("Intent LLM classification failed: %s", exc)

    # Embedding few-shot vote (weight 1.2)
    embed_model = getattr(request.app.state, "_embedding_model", None)
    emb_vote = _embedding_vote(text, embed_model)
    if emb_vote is not None:
        emb_intent, emb_conf = emb_vote
        votes["embedding"] = {
            "intent": emb_intent,
            "confidence": emb_conf,
            "weight": 1.2,
        }

    # Fallback default when no model path produced a vote: treat as question
    # (mentor answer is the safe default — it also streams a useful reply).
    if not votes:
        result = IntentResult(
            intent="question", confidence=0.4, path="fused",
            votes={"fallback": {"intent": "question", "confidence": 0.4}},
        )
        INTENT_CACHE.put(cache_key, result)
        return result

    # Weighted fusion
    totals: dict[str, float] = {}
    for vote in votes.values():
        totals[vote["intent"]] = (
            totals.get(vote["intent"], 0.0)
            + vote["weight"] * vote["confidence"]
        )

    best = max(totals, key=totals.get)  # type: ignore[arg-type]
    total = sum(totals.values())
    confidence = round(totals[best] / total, 4) if total > 0 else 0.0

    result = IntentResult(
        intent=best, confidence=confidence, path="fused", votes=votes
    )
    INTENT_CACHE.put(cache_key, result)
    logger.info(
        "Intent fused: %r → %s (conf=%.2f, votes=%s)",
        text[:50], best, confidence, votes,
    )
    return result
