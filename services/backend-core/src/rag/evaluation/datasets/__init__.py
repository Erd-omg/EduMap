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

# Course the two original datasets were authored for.  Kept as the default so
# every existing caller keeps loading exactly what it loaded before.
DEFAULT_COURSE_ID = "cs201"


def _rows_from_file(path: Path) -> list[dict[str, Any]]:
    """Read a query file and map rows to the evaluator's shape."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "query": q["query"],
            "relevant_ids": set(q.get("relevant_kp_ids", [])),
            "difficulty": q.get("difficulty", "basic"),
            "expected_answer_points": q.get("expected_answer_points", []),
        }
        for q in data.get("queries", [])
    ]


def _file_for(kind: str, course_id: str) -> Path:
    """Resolve the dataset path for a course.

    ``cs201`` keeps the historical bare filenames (``expanded_queries.json``)
    so existing invocations and bookmarks do not break; any other course uses
    the namespaced form ``<kind>_queries_<course_id>.json``.
    """
    if course_id == DEFAULT_COURSE_ID:
        return _DATASETS_DIR / f"{kind}_queries.json"
    return _DATASETS_DIR / f"{kind}_queries_{course_id}.json"


def _course_of(path: Path) -> str | None:
    """The course a dataset file belongs to, or ``None`` if it is not one.

    Derived from the ``_meta.course_id`` the file declares, falling back to the
    filename — *not* from the filename alone.  Globbing names alone would treat
    an archival file (``expanded_queries_old.json``, a ``*_v2`` draft) as a
    loadable course, and ``--course old`` would then benchmark a stray file and
    report the numbers as if that course existed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f).get("_meta", {})
    except (OSError, ValueError):
        return None
    declared = meta.get("course_id")
    if isinstance(declared, str) and declared:
        return declared
    # No declaration: only accept the historical bare name.
    return DEFAULT_COURSE_ID if path.stem.endswith("_queries") else None


def load_sample_queries(course_id: str = DEFAULT_COURSE_ID) -> list[dict[str, Any]]:
    """Load the manually curated student query dataset.

    Args:
        course_id: Which course's query set to load.  Defaults to ``cs201``,
            the corpus the original file was written for.

    Returns:
        List of dicts with keys ``query``, ``relevant_ids`` (set of str),
        ``difficulty``, and ``expected_answer_points``.  Empty list (with a
        warning) if the course has no file.
    """
    path = _file_for("sample", course_id)
    if not path.exists():
        logger.warning("Sample queries file not found for %s: %s", course_id, path)
        return []

    result = _rows_from_file(path)
    logger.info(
        "Loaded %d evaluation queries from %s", len(result), path.name,
    )
    return result


def load_expanded_queries(course_id: str = DEFAULT_COURSE_ID) -> list[dict[str, Any]]:
    """Load the expanded 200-query evaluation dataset.

    The ``cs201`` set covers all 22 KPs with diverse query types (factual,
    comparative, procedural, analytical, multi-KP).  Falls back to
    ``load_sample_queries()`` if the expanded file is not available.

    Args:
        course_id: Which course's query set to load.  The fusion-default
            evidence in ``src/config.py`` was measured on ``cs201`` only —
            pass another course to test whether a conclusion generalises.

    Returns:
        List of query dicts in the same format as ``load_sample_queries()``.
    """
    path = _file_for("expanded", course_id)
    if not path.exists():
        logger.warning(
            "Expanded queries file not found for %s: %s — using sample",
            course_id, path,
        )
        return load_sample_queries(course_id)

    result = _rows_from_file(path)
    logger.info("Loaded %d evaluation queries from %s", len(result), path.name)
    return result


def load_queries_by_difficulty(
    difficulty: str = "basic",
    course_id: str = DEFAULT_COURSE_ID,
) -> list[dict[str, Any]]:
    """Load only queries matching the given difficulty level.

    Args:
        difficulty: One of ``"basic"``, ``"intermediate"``, ``"advanced"``.
        course_id: Which course's query set to load.

    Returns:
        Filtered list of query dicts.
    """
    all_queries = load_expanded_queries(course_id)
    return [q for q in all_queries if q.get("difficulty") == difficulty]


def available_courses() -> list[str]:
    """List course ids that have a query dataset on disk.

    Only files that *declare* a ``_meta.course_id`` (or carry the historical
    bare name) count — see :func:`_course_of` for why the filename alone is not
    trusted.
    """
    courses = set()
    for path in _DATASETS_DIR.glob("*_queries*.json"):
        course = _course_of(path)
        if course is not None:
            courses.add(course)
    return sorted(courses)


def load_intent_labeled() -> list[dict[str, Any]]:
    """Load the hand-labeled intent-classification evaluation set.

    Unlike the retrieval datasets, these rows carry a ``label`` from the
    intent space (``profile`` / ``question`` / ``mixed``) rather than a set
    of relevant knowledge-point ids.

    Returns:
        List of dicts with keys ``text``, ``label``, ``difficulty``, ``note``.
        Returns an empty list (with a warning) if the file is missing.
    """
    path = _DATASETS_DIR / "intent_labeled.json"
    if not path.exists():
        logger.warning("Intent dataset not found: %s", path)
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = [
        {
            "text": q["text"],
            "label": q["label"],
            "difficulty": q.get("difficulty", "basic"),
            "note": q.get("note", ""),
        }
        for q in data.get("queries", [])
    ]
    logger.info("Loaded %d labeled intent queries from %s", len(rows), path.name)
    return rows
