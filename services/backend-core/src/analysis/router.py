"""Unified analysis + mentor SSE endpoint.

Replaces the separate profile-service (port 8001) by providing an SSE endpoint
on the main backend-core (port 8000).  Messages are classified as:

- **Profile-oriented** — messages about the user's background, goals, or
  learning style → builds/updates the learner profile and yields
  ``profile_update`` events.
- **Mentor-oriented** — knowledge questions → delegates to the MentorAgent
  for RAG-grounded answers.
- **Mixed** — short profile extraction + mentor answer in a single SSE stream.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

# ── Profile keywords and topics ─────────────────────────────────────────

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

PROFILE_TOPICS = {
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "c语言": "C",
    "javascript": "JavaScript",
    "数据结构": "数据结构",
    "算法": "算法",
    "数据库": "数据库",
    "操作系统": "操作系统",
    "网络": "计算机网络",
    "前端": "前端开发",
    "后端": "后端开发",
    "机器学习": "机器学习",
    "深度学习": "深度学习",
    "数学": "数学",
    "线性代数": "线性代数",
    "概率": "概率论",
    "统计": "统计学",
}


def _contains_profile_intent(text: str) -> bool:
    """Heuristic: does the message look profile-oriented?

    Requires an explicit self-disclosure (我是/我在学/我的背景 …) or a
    strong background-signal word.  Generic tokens like 我/学习/掌握 no
    longer count on their own.
    """
    if re.search(r"(我是|我叫|我在学|我正在|我目前|我学过|我以前|我打算|我想学|我准备)", text):
        return True
    if re.search(r"(我的|自己的)(背景|水平|基础|情况|目标|专业|能力|弱点|困难|学习风格)", text):
        return True
    return any(kw in text for kw in BACKGROUND_SIGNALS)


def _is_question(text: str) -> bool:
    """Heuristic: does the message look like a question (not a statement)?"""
    if "?" in text or "？" in text:
        return True
    return any(marker in text for marker in QUESTION_MARKERS)


def _extract_topics(text: str) -> dict[str, float]:
    """Extract mentioned topics and assign a confidence score (0.3-0.9)."""
    topics: dict[str, float] = {}
    lowered = text.lower()
    for keyword, topic_name in PROFILE_TOPICS.items():
        if keyword in lowered:
            match = re.search(re.escape(keyword), lowered)
            if match:
                start = max(0, match.start() - 20)
                end = min(len(lowered), match.end() + 20)
                context = lowered[start:end]
                if any(w in context for w in ["学过", "掌握", "熟悉", "了解", "知道", "会用", "擅长"]):
                    topics[topic_name] = 0.7
                elif any(w in context for w in ["精通", "熟练", "深入", "专业"]):
                    topics[topic_name] = 0.9
                elif any(w in context for w in ["想学", "了解", "兴趣", "感兴趣"]):
                    topics[topic_name] = 0.3
                else:
                    topics[topic_name] = 0.5
    return topics


def _estimate_learning_ability(text: str) -> dict[str, float]:
    """Rough estimate of learning ability dimensions from text patterns."""
    ability = {
        "understanding_speed": 0.5,
        "problem_solving": 0.5,
        "critical_thinking": 0.5,
        "memory_retention": 0.5,
        "analytical_ability": 0.5,
    }
    lowered = text.lower()
    if any(w in lowered for w in ["理解", "明白", "懂了", "原来如此", "所以"]):
        ability["understanding_speed"] = 0.65
    if any(w in lowered for w in ["为什么", "怎么", "如何", "区别", "比较", "分析"]):
        ability["critical_thinking"] = 0.7
        ability["analytical_ability"] = 0.7
    if any(w in lowered for w in ["记得", "记住", "回忆", "之前学过"]):
        ability["memory_retention"] = 0.6
    if any(w in lowered for w in ["解决", "实现", "写代码", "调试", "bug"]):
        ability["problem_solving"] = 0.7
    return ability


def _estimate_motivation(text: str) -> dict[str, float]:
    """Estimate motivation dimensions."""
    mot = {
        "intrinsic_interest": 0.5,
        "goal_oriented": 0.5,
        "extrinsic_motivation": 0.5,
    }
    lowered = text.lower()
    if any(w in lowered for w in ["兴趣", "喜欢", "有趣", "好奇", "有意思"]):
        mot["intrinsic_interest"] = 0.8
    if any(w in lowered for w in ["目标", "计划", "打算", "想成为", "未来"]):
        mot["goal_oriented"] = 0.75
    if any(w in lowered for w in ["考试", "作业", "项目", "工作需要", "面试"]):
        mot["extrinsic_motivation"] = 0.7
    return mot


def _build_profile_update(
    text: str,
    existing: dict | None,
) -> tuple[dict, dict[str, float]]:
    """Given a message and existing profile, produce an updated profile + confidence."""
    topics = _extract_topics(text)
    ability = _estimate_learning_ability(text)
    motivation = _estimate_motivation(text)

    profile = existing or {
        "knowledge_base": {},
        "learning_ability": {"understanding_speed": 0.5, "problem_solving": 0.5,
                             "critical_thinking": 0.5, "memory_retention": 0.5,
                             "analytical_ability": 0.5},
        "learning_motivation": {"intrinsic_interest": 0.5, "goal_oriented": 0.5,
                                "extrinsic_motivation": 0.5},
        "knowledge_coverage": {"mastered": [], "learning": [], "not_started": []},
        "interaction_style": {"visual": 0.5, "textual": 0.5,
                              "interactive": 0.5, "auditory": 0.3},
        "focus_characteristics": {"avg_focus_duration_min": 25,
                                  "distraction_frequency": 0.5,
                                  "recommended_session_length": 30},
    }

    # Merge knowledge_base
    for topic, score in topics.items():
        current = profile["knowledge_base"].get(topic, 0.0)
        weight = 0.4 if current == 0.0 else 0.2
        profile["knowledge_base"][topic] = round(current * (1 - weight) + score * weight, 2)

    # Merge learning_ability (smooth update)
    for k, v in ability.items():
        current = profile["learning_ability"].get(k, 0.5)
        profile["learning_ability"][k] = round(current * 0.7 + v * 0.3, 2)

    # Merge motivation
    for k, v in motivation.items():
        current = profile["learning_motivation"].get(k, 0.5)
        profile["learning_motivation"][k] = round(current * 0.7 + v * 0.3, 2)

    # Confidence scores
    total_topics = len(profile["knowledge_base"])
    confidence = {
        "knowledge_base": min(0.9, 0.3 + total_topics * 0.1),
        "learning_ability": min(0.8, 0.4 + sum(1 for v in ability.values() if v != 0.5) * 0.08),
        "learning_motivation": min(0.8, 0.4 + sum(1 for v in motivation.values() if v != 0.5) * 0.08),
    }

    return profile, confidence


# ── Async generators ────────────────────────────────────────────────────


async def _gen_profile_analysis(
    text: str,
    user_id: str,
    request: Request,
) -> AsyncIterator[dict]:
    """Generate SSE events for a profile-oriented message."""
    profiles: dict[str, dict] = _get_profiles(request)
    existing = profiles.get(user_id)

    profile, confidence = _build_profile_update(text, existing)
    profiles[user_id] = profile

    topics_found = [t for t, s in profile["knowledge_base"].items() if s > 0.3]
    if topics_found:
        topics_str = "、".join(topics_found[:5])
        msg = f"了解到你在 {topics_str} 方面的背景，我会根据你的情况调整学习建议。"
    else:
        msg = "已记录你的学习信息，你可以继续分享更多背景，或者提出具体的学习问题。"
        if not _contains_profile_intent(text):
            msg = ""

    yield {
        "event": "profile_update",
        "data": json.dumps({
            "profile": profile,
            "confidence_scores": confidence,
        }, ensure_ascii=False),
    }

    if msg:
        for chunk in _chunk_text(msg, size=4):
            yield {
                "event": "token",
                "data": json.dumps({"content": chunk}, ensure_ascii=False),
            }

    yield {
        "event": "complete",
        "data": json.dumps({"sources": []}, ensure_ascii=False),
    }


async def _gen_mentor_answer(
    text: str,
    user_id: str,
    request: Request,
) -> AsyncIterator[dict]:
    """Generate SSE events by delegating to the MentorAgent."""
    agent = getattr(request.app.state, "mentor_agent", None)
    if not agent:
        msg = "AI 导师暂不可用，请确认后端服务已正确启动。"
        for chunk in _chunk_text(msg, size=6):
            yield {
                "event": "token",
                "data": json.dumps({"content": chunk}, ensure_ascii=False),
            }
        yield {
            "event": "complete",
            "data": json.dumps({"sources": []}, ensure_ascii=False),
        }
        return

    # Load conversation history from memory system
    conversation_history = None
    try:
        memory_ops = getattr(request.app.state, "memory_ops", None)
        if memory_ops:
            from src.memory.models import EventType
            episodic = await memory_ops.long_term.recall_episodic(
                user_id=user_id,
                event_types=[EventType.MENTOR_QUERY],
                limit=10,
            )
            if episodic:
                conversation_history = [
                    {"input": e.input, "output": e.output}
                    for e in episodic
                ]
    except Exception as exc:
        logger.debug("Failed to load conversation history: %s", exc)

    try:
        async for event in agent.answer_stream(
            query=text,
            user_id=user_id,
            conversation_history=conversation_history,
        ):
            yield {
                "event": event["type"],
                "data": json.dumps(event["data"], ensure_ascii=False),
            }

        # Record this interaction to episodic memory
        try:
            if memory_ops:
                await memory_ops.record_interaction(
                    user_id=user_id,
                    event_type="mentor_query",
                    input_text=text,
                    output_text="(streamed)",
                    importance=0.5,
                )
        except Exception as exc:
            logger.debug("Failed to record mentor interaction: %s", exc)

    except Exception as exc:
        logger.exception("Mentor stream error in analysis endpoint")
        err_msg = f"生成回答时出现异常：{exc}"
        for chunk in _chunk_text(err_msg, size=6):
            yield {
                "event": "token",
                "data": json.dumps({"content": chunk}, ensure_ascii=False),
            }
        yield {
            "event": "complete",
            "data": json.dumps({"sources": []}, ensure_ascii=False),
        }


async def _gen_mixed(
    text: str,
    user_id: str,
    request: Request,
) -> AsyncIterator[dict]:
    """Both profile analysis and mentor answer."""
    profiles = _get_profiles(request)
    existing = profiles.get(user_id)

    # 1. Profile update first
    profile, confidence = _build_profile_update(text, existing)
    profiles[user_id] = profile

    yield {
        "event": "profile_update",
        "data": json.dumps({
            "profile": profile,
            "confidence_scores": confidence,
        }, ensure_ascii=False),
    }

    # 2. Mentor answer (iterating the async generator, not awaiting it)
    async for event in _gen_mentor_answer(text, user_id, request):
        yield event


# ── Route ───────────────────────────────────────────────────────────────


@router.get("/stream/{user_id}")
async def stream_analysis(
    user_id: str,
    message: str = Query(...),
    request: Request = None,
):
    """Unified analysis + mentor SSE endpoint.

    Smart-routes messages:
    - Profile-oriented (self-introduction, background, goals) → profile analysis
    - Knowledge questions → Mentor RAG answer
    - Mixed → both

    Events::

        event: profile_update
        data: {"profile": {...}, "confidence_scores": {...}}

        event: source
        data: {"sources": [...]}

        event: token
        data: {"content": "..."}

        event: complete
        data: {"sources": [...], "confidence": 0.85}
    """
    _ensure_profiles(request)

    is_profile = _contains_profile_intent(message)
    is_question = _is_question(message)
    has_topic = bool(_extract_topics(message))

    # Questions about knowledge ("学习链表之前我应该先掌握什么？") must reach
    # the Mentor RAG answer path — the profile keyword heuristic used to
    # swallow them with the canned "已记录你的学习信息…" message.
    if is_profile and has_topic:
        generator = _gen_mixed(message, user_id, request)
    elif is_question and not is_profile:
        generator = _gen_mentor_answer(message, user_id, request)
    elif is_profile:
        generator = _gen_profile_analysis(message, user_id, request)
    else:
        generator = _gen_mentor_answer(message, user_id, request)

    return EventSourceResponse(generator)


# ── Helpers ─────────────────────────────────────────────────────────────


def _ensure_profiles(request: Request) -> None:
    """Initialize profiles storage if needed."""
    if not hasattr(request.app.state, "analysis_profiles"):
        request.app.state.analysis_profiles = {}


def _get_profiles(request: Request) -> dict[str, dict]:
    """Get or create the profiles dict."""
    if not hasattr(request.app.state, "analysis_profiles"):
        request.app.state.analysis_profiles = {}
    return request.app.state.analysis_profiles


def _chunk_text(text: str, size: int = 4) -> list[str]:
    """Split text into small chunks for streaming simulation."""
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]
