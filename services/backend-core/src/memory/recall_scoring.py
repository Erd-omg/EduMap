"""Episodic-memory recall scoring — recency × importance × relevance.

Replaces the previous plain ``ORDER BY created_at DESC`` recall with the
generative-agents retrieval formula (Park et al., arXiv 2304.03442)::

    score = α·recency + β·importance + γ·relevance

with ``α = β = γ = 1`` (equal weights, as in the paper) and::

    recency    = RECENCY_DECAY ** hours_since_created     # 0.995^hours
    importance = importance_score                          # already [0, 1]
    relevance  = cosine(query_embedding, entry_embedding)  # [0, 1]

All three components are already on a [0, 1] scale, so they are summed
**raw** — deliberately *not* min-max normalised across the candidate set.

Why not min-max normalise (the obvious-looking choice):

    Per-candidate normalisation rescales the largest value to 1.0 and the
    smallest to 0.0, which *destroys the magnitude information* that makes
    the weights meaningful.  With two candidates it degenerates completely:
    a fresh-but-worthless entry (recency 1.0, importance 0.0) and an
    old-but-important one (recency 0.79, importance 1.0) both score exactly
    1.5 and tie.  Because the paper's constants are already commensurate,
    summing raw values preserves the intended trade-off.  (This is the same
    failure mode as RRF discarding score magnitudes — see the RAG fusion
    note in ``RAGRetrievalService``.)

Why this matters: ``EpisodicEntry.importance_score`` was already persisted
but never used for ordering — recall returned the most *recent* records
rather than the most *worth-recalling* ones.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.memory.models import EpisodicEntry

# 0.995 ** hours — the decay constant from Park et al. (arXiv 2304.03442).
# Interpretation: an entry loses ~0.5% of its recency weight per hour.
RECENCY_DECAY = 0.995

# Equal weights, per the paper (α = β = γ = 1).
WEIGHT_RECENCY = 1.0
WEIGHT_IMPORTANCE = 1.0
WEIGHT_RELEVANCE = 1.0


def _hours_since(created_at: datetime | None, *, now: datetime | None = None) -> float:
    """Hours elapsed since ``created_at``.

    A missing timestamp is treated as "just now" (recency 1.0) rather than
    "infinitely old" — an entry we cannot date is more likely freshly written
    than ancient, and treating it as ancient would silently bury it.

    Both operands are coerced to aware-UTC before subtraction: asyncpg may
    return naive datetimes, and tests/callers may pass a naive ``now``.
    Mixing a naive and an aware datetime raises ``TypeError``, so both sides
    are normalised rather than only ``created_at``.
    """
    if created_at is None:
        return 0.0

    now = now or datetime.now(timezone.utc)
    # Defensive: asyncpg may hand back naive datetimes; assume UTC.
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    delta = (now - created_at).total_seconds() / 3600.0
    # Guard against clock skew producing a negative age.
    return max(0.0, delta)


def recency_score(created_at: datetime | None, *, now: datetime | None = None) -> float:
    """``0.995 ** hours_since_created`` — in (0, 1]."""
    return RECENCY_DECAY ** _hours_since(created_at, now=now)


def cosine_similarity(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    """Cosine similarity of two vectors, clamped to [0, 1].

    Returns 0.0 when either vector is missing or degenerate, so a caller
    without embeddings degrades to recency+importance only instead of
    raising. Negative similarities are clamped to 0: for this scoring
    formula "anti-relevant" and "unrelated" carry the same information.
    """
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(0.0, min(1.0, sim))


def score_episodic_entries(
    entries: Sequence[EpisodicEntry],
    *,
    query_embedding: Sequence[float] | None = None,
    entry_embeddings: dict[str, Sequence[float]] | None = None,
    now: datetime | None = None,
) -> list[tuple[EpisodicEntry, float]]:
    """Rank episodic entries by the generational-agents retrieval formula.

    Components are summed raw (each is already on a [0, 1] scale) so that
    the *magnitude* of each term survives to influence the ordering.  See
    the module docstring for why min-max normalisation must not be used.

    Args:
        entries: candidate entries (any order; this function re-ranks them).
        query_embedding: embedding of the current query, if available.
            Pass ``None`` (or omit ``entry_embeddings``) to score on
            recency + importance only.
        entry_embeddings: ``{entry_id: embedding}`` for the candidates.
            Entries without an embedding contribute relevance 0.
        now: injectable clock for deterministic tests.

    Returns:
        ``[(entry, score), ...]`` sorted by score descending. Ties keep the
        input order (Python's sort is stable), which — given the caller
        fetches newest-first — means newer entries win ties.
    """
    if not entries:
        return []

    embeds = entry_embeddings or {}
    use_relevance = query_embedding is not None and bool(embeds)

    scored: list[tuple[EpisodicEntry, float]] = []
    for entry in entries:
        recency = recency_score(entry.created_at, now=now)
        importance = float(entry.importance_score or 0.0)
        score = WEIGHT_RECENCY * recency + WEIGHT_IMPORTANCE * importance

        if use_relevance:
            relevance = cosine_similarity(
                query_embedding, embeds.get(entry.id or "")
            )
            score += WEIGHT_RELEVANCE * relevance

        scored.append((entry, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
