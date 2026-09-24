"""LLM-as-judge evaluation for RAG answer quality.

Provides LLM-based alternatives to the token-overlap heuristics in
``metrics.py``.  These require an LLM adapter but produce much more
accurate faithfulness and relevancy scores for Chinese text.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.llm_adapter import BaseLLMAdapter

logger = logging.getLogger(__name__)

_FAITHFULNESS_JUDGE_PROMPT = """你是一个严格的审核员。判断以下回答中的每一句话是否被"参考资料"支持。

参考资料：
{context}

回答：
{answer}

逐句分析。返回 JSON：
{{
  "supported_sentences": [
    "被支持的句子1",
    "被支持的句子2"
  ],
  "unsupported_sentences": [
    "不支持的句子1"
  ],
  "faithfulness_score": 0.75,
  "reasoning": "简要分析"
}}

要求：
- faithfulness_score 范围 0-1，= 支持句数 / 总句数
- 句子要原文提取
- 找不到参考资料时（context 为空），直接给 0"""

_RELEVANCY_JUDGE_PROMPT = """你是一个问答质量评估员。判断以下回答是否准确回答了用户的问题。

问题：{query}

回答：{answer}

返回 JSON：
{{
  "relevancy_score": 0.85,
  "reasons": ["回答直接针对问题", "包含了关键信息"],
  "missing_points": ["没有提及时间复杂度比较"]
}}

要求：
- relevancy_score 范围 0-1
- 回答完全切题且准确 = 0.9-1.0
- 回答了部分内容 = 0.5-0.8
- 完全不相关 = 0.0-0.3"""


async def llm_faithfulness(
    answer: str,
    context: str,
    llm: BaseLLMAdapter,
) -> dict[str, Any]:
    """Evaluate faithfulness using LLM-as-judge.

    Args:
        answer: The generated answer text.
        context: Retrieved context used for generation.
        llm: LLM adapter to use as judge.

    Returns:
        Dict with keys: ``faithfulness_score`` (0-1), ``supported_sentences``,
        ``unsupported_sentences``, ``reasoning``.
    """
    prompt = _FAITHFULNESS_JUDGE_PROMPT.format(
        context=context[:3000] if context else "（无参考资料）",
        answer=answer[:2000],
    )

    try:
        from pydantic import BaseModel, Field

        class FaithfulnessSchema(BaseModel):
            supported_sentences: list[str] = Field(default_factory=list)
            unsupported_sentences: list[str] = Field(default_factory=list)
            faithfulness_score: float = 0.0
            reasoning: str = ""

        result = await llm.generate_structured(
            prompt=prompt,
            schema=FaithfulnessSchema,  # type: ignore[arg-type]
        )
        return {
            "faithfulness_score": result.faithfulness_score,
            "supported_sentences": result.supported_sentences,
            "unsupported_sentences": result.unsupported_sentences,
            "reasoning": result.reasoning,
            "used_judge": True,
        }
    except Exception as exc:
        logger.warning("LLM faithfulness judge failed, falling back: %s", exc)
        # Fallback: token-overlap heuristic.
        #
        # `used_judge: False` is the important part. The fallback keeps the
        # function total (callers get a number rather than an exception), but a
        # heuristic result must not be *reported* as an LLM-judge result: the
        # two are independent methods, and conflating them destroys the
        # cross-validation they exist for — if both "legs" are the same
        # heuristic, a disagreement (the signal that something is wrong) can
        # never appear.
        from src.rag.evaluation.metrics import faithfulness

        fallback = faithfulness(answer, [context])
        return {
            "faithfulness_score": fallback["faithfulness"],
            "supported_sentences": [],
            "unsupported_sentences": fallback["unsupported"],
            "reasoning": "fallback_heuristic",
            "used_judge": False,
        }


async def llm_answer_relevancy(
    query: str,
    answer: str,
    llm: BaseLLMAdapter,
) -> dict[str, Any]:
    """Evaluate answer relevancy using LLM-as-judge.

    Args:
        query: The user's question.
        answer: The generated answer.
        llm: LLM adapter.

    Returns:
        Dict with keys: ``relevancy_score``, ``reasons``, ``missing_points``.
    """
    prompt = _RELEVANCY_JUDGE_PROMPT.format(
        query=query[:500],
        answer=answer[:2000],
    )

    try:
        from pydantic import BaseModel, Field

        class RelevancySchema(BaseModel):
            relevancy_score: float = 0.5
            reasons: list[str] = Field(default_factory=list)
            missing_points: list[str] = Field(default_factory=list)

        result = await llm.generate_structured(
            prompt=prompt,
            schema=RelevancySchema,  # type: ignore[arg-type]
        )
        return {
            "relevancy_score": result.relevancy_score,
            "reasons": result.reasons,
            "missing_points": result.missing_points,
        }
    except Exception as exc:
        logger.warning("LLM relevancy judge failed, falling back: %s", exc)
        from src.rag.evaluation.metrics import answer_relevancy
        return {
            "relevancy_score": answer_relevancy(answer, query),
            "reasons": [],
            "missing_points": [],
        }
