"""Dataset loader for RAG evaluation queries.

Loads manually curated or LLM-generated query sets and converts them
to the format expected by ``RAGEvaluator.evaluate_retrieval()``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATASETS_DIR = Path(__file__).parent


def load_sample_queries() -> list[dict[str, Any]]:
    """Load the manually curated student query dataset.

    Returns:
        List of dicts with keys ``query``, ``relevant_ids`` (set of str),
        ``difficulty``, and ``expected_answer_points``.
    """
    path = _DATASETS_DIR / "sample_queries.json"
    if not path.exists():
        logger.warning("Sample queries file not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    result = []
    for q in queries:
        result.append({
            "query": q["query"],
            "relevant_ids": set(q.get("relevant_kp_ids", [])),
            "difficulty": q.get("difficulty", "basic"),
            "expected_answer_points": q.get("expected_answer_points", []),
        })

    logger.info(
        "Loaded %d evaluation queries from %s",
        len(result),
        path.name,
    )
    return result


def load_expanded_queries() -> list[dict[str, Any]]:
    """Load the expanded 200-query evaluation dataset.

    This dataset covers all 22 KPs with diverse query types (factual,
    comparative, procedural, analytical, multi-KP).  Falls back to
    ``load_sample_queries()`` if the expanded file is not available.

    Returns:
        List of query dicts in the same format as ``load_sample_queries()``.
    """
    path = _DATASETS_DIR / "expanded_queries.json"
    if not path.exists():
        logger.warning("Expanded queries file not found: %s — using sample", path)
        return load_sample_queries()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    queries = data.get("queries", [])
    result = []
    for q in queries:
        result.append({
            "query": q["query"],
            "relevant_ids": set(q.get("relevant_kp_ids", [])),
            "difficulty": q.get("difficulty", "basic"),
            "expected_answer_points": q.get("expected_answer_points", []),
        })

    logger.info("Loaded %d evaluation queries from %s", len(result), path.name)
    return result


def load_queries_by_difficulty(
    difficulty: str = "basic",
) -> list[dict[str, Any]]:
    """Load only queries matching the given difficulty level.

    Args:
        difficulty: One of ``"basic"``, ``"intermediate"``, ``"advanced"``.

    Returns:
        Filtered list of query dicts.
    """
    all_queries = load_expanded_queries()
    return [q for q in all_queries if q.get("difficulty") == difficulty]
