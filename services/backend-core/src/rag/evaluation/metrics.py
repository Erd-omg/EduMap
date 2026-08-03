"""Evaluation metrics for RAG quality measurement."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# Try to load jieba for Chinese tokenization; gracefully fall back to str.split().
try:
    import jieba as _jieba_mod
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

# Chinese + English stop words that carry little semantic meaning
_STOP_WORDS: set[str] = {
    # Chinese stop words
    "的", "了", "是", "在", "有", "和", "就", "都", "而", "且", "也", "很",
    "到", "去", "能", "可以", "会", "要", "对", "等", "并", "或", "与",
    "及", "之", "被", "把", "从", "向", "以", "为", "于", "但", "但是",
    "因为", "所以", "如果", "虽然", "而且", "或者", "然后", "那么", "这",
    "那", "哪", "什么", "怎么", "如何", "为什么", "是否", "没有", "不是",
    "一个", "这个", "那个", "这些", "那些", "的", "地", "得", "着", "过",
    "了", "吗", "呢", "吧", "啊", "哦", "嗯", "哈", "呀",
    # English stop words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "or", "but",
    "not", "no", "if", "because", "so", "than", "that", "this", "these",
    "those", "it", "its", "they", "them", "their", "we", "you", "your",
    "he", "she", "his", "her", "my", "me", "our", "what", "which", "who",
    "whom", "when", "where", "why", "how", "all", "each", "every", "both",
    "few", "more", "most", "some", "any", "such", "only", "own", "same",
    "very", "just", "about", "up", "out", "also", "than", "then", "now",
    "here", "there",
}


def _tokenize(text: str, remove_stop: bool = True) -> set[str]:
    """Tokenize text into a set of lowercased content tokens.

    For segments containing CJK characters (U+4E00–U+9FFF), uses ``jieba``
    for proper Chinese word segmentation.  For pure English/Latin text, falls
    back to ``str.split()``.  When ``remove_stop=True`` (default), common
    Chinese and English stop words are filtered out.
    """
    tokens: set[str] = set()
    for part in text.split():
        if _HAS_JIEBA and any('一' <= c <= '鿿' for c in part):
            for word in _jieba_mod.cut(part, cut_all=False):
                w = word.strip().lower()
                if not w or (remove_stop and w in _STOP_WORDS):
                    continue
                tokens.add(w)
        else:
            w = part.strip().lower()
            if not w or (remove_stop and w in _STOP_WORDS):
                continue
            tokens.add(w)
    return tokens


def _tokenize_weighted(text: str) -> dict[str, float]:
    """Tokenize and return a dict of token → estimated importance weight.

    Longer content words (2+ chars) are weighted higher than short
    function words.  This is a simplified IDF proxy when a full corpus
    is not available.
    """
    tokens = _tokenize(text, remove_stop=True)
    weights: dict[str, float] = {}
    for t in tokens:
        # Length-based importance: longer words carry more meaning
        base = 1.0
        if len(t) >= 4:
            base = 2.0  # technical terms (e.g. "时间复杂度", "二叉搜索树")
        elif len(t) >= 2:
            base = 1.5  # meaningful words (e.g. "数组", "插入")
        weights[t] = base
    return weights


def _concept_overlap(query_tokens: set[str], answer_tokens: set[str]) -> float:
    """Compute concept-weighted overlap between query and answer.

    Focuses on semantically meaningful overlap: technical terms
    matching in the answer carry more weight than common words.
    """
    if not query_tokens or not answer_tokens:
        return 0.0

    # Weight each matched concept by its "importance" (length-based proxy)
    q_weights = _tokenize_weighted(" ".join(query_tokens))

    total_weight = sum(q_weights.values()) or 1.0
    matched_weight = sum(
        w for t, w in q_weights.items() if t in answer_tokens
    )

    return round(matched_weight / total_weight, 4)


def precision_at_k(
    retrieved: list[Any],
    relevant: set[str],
    k: int,
) -> float:
    """Precision@K = (# of relevant items in top-K) / K."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if _get_id(item) in relevant)
    return hits / k


def recall_at_k(
    retrieved: list[Any],
    relevant: set[str],
    k: int,
) -> float:
    """Recall@K = (# of relevant items in top-K) / total relevant."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if _get_id(item) in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(
    retrieved: list[Any],
    relevant: set[str],
) -> float:
    """MRR = 1 / rank of first relevant item, or 0 if none found."""
    for i, item in enumerate(retrieved, 1):
        if _get_id(item) in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: list[Any],
    relevant: set[str],
    k: int,
    relevance_scores: dict[str, float] | None = None,
) -> float:
    """Normalized Discounted Cumulative Gain @K.

    Uses binary relevance (1 if relevant, 0 otherwise) unless
    ``relevance_scores`` is provided.
    """
    if k <= 0 or not retrieved:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(retrieved[:k], 1):
        item_id = _get_id(item)
        if relevance_scores and item_id in relevance_scores:
            rel = relevance_scores[item_id]
        else:
            rel = 1.0 if item_id in relevant else 0.0
        dcg += (2**rel - 1) / math.log2(i + 1)

    # Ideal DCG (sort by relevance descending)
    def _get_rel(item_iter: Any) -> float:
        item_id = _get_id(item_iter)
        if relevance_scores and item_id in relevance_scores:
            return relevance_scores[item_id]
        return 1.0 if item_id in relevant else 0.0

    ideal_rels = sorted(
        [_get_rel(item) for item in retrieved[:k]],
        reverse=True,
    )
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, 1))

    return dcg / idcg if idcg > 0 else 0.0


def faithfulness(
    generated_answer: str,
    context_sentences: list[str],
) -> dict[str, float]:
    """Estimate faithfulness — whether the answer is grounded in the context.

    Uses a lexical overlap heuristic with Chinese-aware tokenization (jieba).
    - Proportion of answer sentences that have significant overlap with context.
    - Higher = more faithful to source material.

    For production use, an NLI-based faithfulness model or the
    LLM-as-judge (``llm_judge.llm_faithfulness``) is recommended.

    Returns:
        Dict with keys: ``faithfulness`` (0-1), ``supported_sentences``,
        ``total_sentences``, ``unsupported``.
    """
    # Split answer into sentences
    sentences = re.split(r'(?<=[。！？.!?])\s*', generated_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return {"faithfulness": 1.0, "supported_sentences": 0, "total_sentences": 0, "unsupported": []}

    # Build context token set (Chinese-aware via _tokenize, with stop words)
    context_tokens: set[str] = set()
    for ctx in context_sentences:
        context_tokens.update(_tokenize(ctx, remove_stop=False))

    supported = []
    unsupported = []
    for sent in sentences:
        sent_tokens = _tokenize(sent, remove_stop=True)
        if len(sent_tokens) == 0:
            continue
        # Adaptive threshold: lower when context is sparse
        context_density = len(context_tokens) / max(len(context_sentences), 1)
        threshold = 0.15 if context_density < 30 else 0.25
        # Calculate token overlap ratio (concept-weighted)
        sent_weights = _tokenize_weighted(" ".join(sent_tokens))
        total_weight = sum(sent_weights.values()) or 1.0
        matched_weight = sum(
            w for t, w in sent_weights.items() if t in context_tokens
        )
        overlap_ratio = matched_weight / total_weight
        if overlap_ratio >= threshold:
            supported.append(sent)
        else:
            unsupported.append(sent)

    total = len(supported) + len(unsupported)
    score = len(supported) / total if total > 0 else 1.0
    return {
        "faithfulness": round(score, 4),
        "supported_sentences": len(supported),
        "total_sentences": total,
        "unsupported": unsupported,
    }


def answer_relevancy(
    generated_answer: str,
    query: str,
) -> float:
    """Estimate answer relevancy — whether the answer addresses the query.

    Uses concept-weighted overlap with stop-word removal, which is more
    robust for Chinese text than pure Jaccard.  Technical terms (e.g.
    "数组", "时间复杂度") receive higher weight.

    For production, use the LLM-based ``llm_judge.llm_answer_relevancy``.

    Returns:
        Score from 0 to 1.
    """
    query_tokens = _tokenize(query, remove_stop=True)
    answer_tokens = _tokenize(generated_answer, remove_stop=True)

    if not query_tokens or not answer_tokens:
        return 0.5  # neutral

    # Primary: concept-weighted overlap (emphasizes key technical terms)
    weighted = _concept_overlap(query_tokens, answer_tokens)

    # Secondary: TF/IDF-style coverage (what fraction of query concepts appear?)
    coverage = len(query_tokens & answer_tokens) / max(len(query_tokens), 1)

    # Blend: weighted overlap (0.7) + coverage (0.3)
    score = 0.7 * weighted + 0.3 * coverage
    return round(min(score, 1.0), 4)


def hit_rate_at_k(
    retrieved: list[Any],
    relevant: set[str],
    k: int,
) -> float:
    """Hit Rate@K = 1 if at least one relevant item is in top-K, else 0.

    This is the most intuitive user-facing metric — "did we find anything
    useful in the top K results?"
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    return 1.0 if any(_get_id(item) in relevant for item in top_k) else 0.0


def context_coverage(
    answer_sentences: list[str],
    context_sentences: list[str],
) -> float:
    """Estimate what fraction of the answer's information is covered by context.

    For each answer sentence, checks whether its content words have a match
    in any context sentence using Chinese-aware tokenization (jieba).
    This is a heuristic — for production use the LLM-based LLM-based
    ``llm_judge.llm_faithfulness`` instead.

    Returns:
        Score 0-1.
    """
    if not answer_sentences:
        return 1.0
    if not context_sentences:
        return 0.0

    # Build a set of all content-bearing tokens from the context
    context_words: set[str] = set()
    for cs in context_sentences:
        for w in _tokenize(cs):
            if len(w) > 1:  # skip single-char tokens (punctuation, etc.)
                context_words.add(w)

    supported = 0
    for sentence in answer_sentences:
        tokens = [w for w in _tokenize(sentence) if len(w) > 1]
        if not tokens:
            continue
        overlap = sum(1 for w in tokens if w in context_words)
        if overlap / len(tokens) >= 0.3:
            supported += 1

    return supported / len(answer_sentences)


def citation_accuracy(
    answer: str,
    sources: dict[str, str],
) -> float:
    """Check what fraction of ``[N]`` citations in the answer reference valid sources.

    This is a structural check — it only verifies the citation number falls
    within the range of provided sources, not that the content actually matches.

    Args:
        answer: Generated answer text containing ``[1]``, ``[2]``, etc.
        sources: Dict mapping source index (str) to summary text.

    Returns:
        Fraction of citations that have a valid source (0-1).
    """
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    if not citations:
        return 1.0  # no citations = vacuously correct

    valid = sum(1 for c in citations if c in sources)
    return valid / len(citations)


def semantic_faithfulness(
    generated_answer: str,
    context_sentences: list[str],
    embedding_model: Any | None = None,
    threshold: float = 0.55,
) -> dict[str, float]:
    """Estimate faithfulness using semantic similarity (sentence embeddings).

    Unlike the token-overlap heuristic, this method uses the same
    sentence-transformer model that powers the RAG system to compare
    answer sentences against context at the *semantic* level.
    This captures paraphrases and conceptual overlap that token matching
    would miss.

    Args:
        generated_answer: The LLM's answer text.
        context_sentences: List of context text strings (from RAG retrieval).
        embedding_model: A SentenceTransformer model (or None to lazy-load).
        threshold: Cosine similarity threshold (0.45 default for Chinese).

    Returns:
        Dict with keys: ``faithfulness`` (0-1), ``supported_sentences``,
        ``total_sentences``, ``unsupported``, ``avg_similarity``.
    """
    # Lazy-load embedding model
    model = embedding_model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception:
            # Fall back to token-overlap faithfulness
            return faithfulness(generated_answer, context_sentences)

    # Split answer into sentences
    sentences = re.split(r'(?<=[。！？.!?])\s*', generated_answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences or not context_sentences:
        return {"faithfulness": 1.0 if not sentences else 0.0,
                "supported_sentences": 0, "total_sentences": len(sentences),
                "unsupported": [], "avg_similarity": 0.0}

    # Encode answer sentences and context sentences
    try:
        sent_embs = model.encode(sentences, show_progress_bar=False)
        ctx_embs = model.encode(context_sentences, show_progress_bar=False)
    except Exception:
        return faithfulness(generated_answer, context_sentences)

    import numpy as np

    def _cos_sim(a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    supported = []
    unsupported = []
    similarities = []

    for i, sent in enumerate(sentences):
        sent_emb = sent_embs[i]
        # Find max similarity to any context sentence
        max_sim = max(
            (_cos_sim(sent_emb, ctx_emb) for ctx_emb in ctx_embs),
            default=0.0,
        )
        similarities.append(max_sim)
        if max_sim >= threshold:
            supported.append(sent)
        else:
            unsupported.append(sent)

    total = len(supported) + len(unsupported)
    score = len(supported) / total if total > 0 else 1.0
    avg_sim = sum(similarities) / max(len(similarities), 1)

    return {
        "faithfulness": round(score, 4),
        "supported_sentences": len(supported),
        "total_sentences": total,
        "unsupported": unsupported,
        "avg_similarity": round(avg_sim, 4),
    }


# ── Helpers ──────────────────────────────────────────────────

def _get_id(item: Any, default_id: str = "") -> str:
    """Extract a string ID from various result types."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("id", item.get("source_id", default_id)))
    if hasattr(item, "source_id"):
        return item.source_id
    if hasattr(item, "id"):
        return item.id
    return str(item)
