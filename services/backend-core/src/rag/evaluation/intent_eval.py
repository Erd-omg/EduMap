"""Intent-classification evaluation harness.

Measures the accuracy of ``src.analysis.intent.classify_intent`` against a
hand-labeled dataset, breaking results down by the *path* that produced each
verdict (rule / fused / cache).  The path breakdown is the whole point: it
shows how often the cheap rule path resolves a message with zero model calls,
which is the design's cost argument.

Design notes that matter for honest numbers:

- **Cache hygiene.** ``classify_intent`` writes to a module-level singleton
  (``INTENT_CACHE``) keyed on ``(user_id, normalized_text)``.  A non-cold run
  would score mostly ``path="cache"`` hits and measure nothing.  We therefore
  use a unique ``user_id`` per run and, in cold mode, clear the cache before
  every row.
- **The embedding model must be present.** ``_embedding_vote`` returns ``None``
  when the model is missing, which would silently turn a 3-path classifier into
  a 2-path one.  ``assert_models_present`` fails loudly instead.
- **``has_topic`` is computed with production code** (``_extract_topics`` from
  the analysis router) rather than being hard-coded in the dataset, so the
  measurement reflects the real call path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.analysis.intent import INTENT_CACHE, classify_intent
from src.analysis.router import _extract_topics

logger = logging.getLogger(__name__)

INTENT_LABELS = ("profile", "question", "mixed")


@dataclass
class IntentEvalMetrics:
    """Aggregate metrics for one intent-eval run."""

    n_queries: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    # confusion[true][predicted]
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    path_counts: dict[str, int] = field(default_factory=dict)
    # path -> {"n": int, "correct": int}
    accuracy_by_path: dict[str, dict[str, int]] = field(default_factory=dict)
    misclassified: list[dict[str, Any]] = field(default_factory=list)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    cold: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "confusion": self.confusion,
            "per_class": self.per_class,
            "path_counts": self.path_counts,
            "accuracy_by_path": self.accuracy_by_path,
            "misclassified": self.misclassified,
            "by_difficulty": self.by_difficulty,
            "cold": self.cold,
        }


def assert_models_present(request: Any) -> None:
    """Fail loudly if the classifier would silently lose a voting path.

    ``classify_intent`` degrades quietly when ``llm_adapter`` or
    ``_embedding_model`` is missing, so an eval run without them would report
    the accuracy of a *different* classifier than the one in production.
    """
    state = request.app.state
    if getattr(state, "llm_adapter", None) is None:
        raise SystemExit(
            "llm_adapter missing on the request stub — the LLM voting path "
            "would be silently skipped, so this run would not measure the "
            "3-path classifier."
        )
    if getattr(state, "_embedding_model", None) is None:
        raise SystemExit(
            "embedding model missing on the request stub — the embedding "
            "voting path would be silently skipped, so this run would not "
            "measure the 3-path classifier."
        )


def _prf(confusion: dict[str, dict[str, int]], label: str) -> dict[str, float]:
    """Precision / recall / F1 for one class from the confusion matrix."""
    tp = confusion.get(label, {}).get(label, 0)
    predicted = sum(confusion.get(t, {}).get(label, 0) for t in INTENT_LABELS)
    actual = sum(confusion.get(label, {}).values())
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": actual,
    }


async def evaluate_intent(
    rows: list[dict[str, Any]],
    request: Any,
    *,
    cold: bool = True,
    run_id: str = "0",
) -> IntentEvalMetrics:
    """Classify every labeled row and aggregate accuracy by path.

    Args:
        rows: Labeled rows with ``text`` and ``label``.
        request: Request stub exposing ``app.state.llm_adapter`` and
            ``app.state._embedding_model``.
        cold: When True, clear the intent cache before every row so the
            reported accuracy is the classifier's real (first-pass) accuracy
            rather than a cache-hit ratio.
        run_id: Unique per run; forms the cache key's ``user_id`` so runs
            never share cached verdicts.
    """
    metrics = IntentEvalMetrics(n_queries=len(rows), cold=cold)
    metrics.confusion = {t: {p: 0 for p in INTENT_LABELS} for t in INTENT_LABELS}

    for row in rows:
        text, expected = row["text"], row["label"]

        if cold:
            INTENT_CACHE._store.clear()  # type: ignore[attr-defined]

        has_topic = bool(_extract_topics(text))
        result = await classify_intent(
            text, request, user_id=f"eval-{run_id}", has_topic=has_topic
        )

        predicted = result.intent
        metrics.confusion[expected][predicted] += 1

        bucket = metrics.accuracy_by_path.setdefault(
            result.path, {"n": 0, "correct": 0}
        )
        bucket["n"] += 1

        correct = predicted == expected
        if correct:
            metrics.n_correct += 1
            bucket["correct"] += 1
        else:
            metrics.misclassified.append({
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": result.confidence,
                "path": result.path,
                "has_topic": has_topic,
                "votes": result.votes,
                "note": row.get("note", ""),
            })

        diff = row.get("difficulty", "basic")
        dbucket = metrics.by_difficulty.setdefault(
            diff, {"n": 0, "correct": 0}
        )
        dbucket["n"] += 1
        if correct:
            dbucket["correct"] += 1

    n = metrics.n_queries or 1
    metrics.accuracy = round(metrics.n_correct / n, 4)
    metrics.path_counts = {
        path: v["n"] for path, v in metrics.accuracy_by_path.items()
    }
    metrics.per_class = {
        label: _prf(metrics.confusion, label) for label in INTENT_LABELS
    }
    for dbucket in metrics.by_difficulty.values():
        dbucket["accuracy"] = round(dbucket["correct"] / (dbucket["n"] or 1), 4)

    return metrics


def print_report(metrics: IntentEvalMetrics) -> None:
    """Print a formatted intent-accuracy report."""
    sep = "=" * 72
    mode = "冷启动（每行清缓存）" if metrics.cold else "含缓存命中"
    print(f"\n{sep}")
    print(f"  意图识别评测 — {mode}")
    print(f"  样本数: {metrics.n_queries}   准确率: {metrics.accuracy:.4f} "
          f"({metrics.n_correct}/{metrics.n_queries})")
    print(sep)

    print("\n  混淆矩阵 (行=真实, 列=预测):")
    header = "  " + " " * 12 + "".join(f"{p:>10}" for p in INTENT_LABELS)
    print(header)
    for t in INTENT_LABELS:
        row = "".join(f"{metrics.confusion[t][p]:>10}" for p in INTENT_LABELS)
        print(f"  {t:<12}{row}")

    print("\n  每类指标:")
    print(f"  {'类':<10}{'P':>10}{'R':>10}{'F1':>10}{'support':>10}")
    for label, m in metrics.per_class.items():
        print(f"  {label:<10}{m['precision']:>10.4f}{m['recall']:>10.4f}"
              f"{m['f1']:>10.4f}{m['support']:>10}")

    print("\n  路径分布 (哪种路径产出了判定):")
    total = metrics.n_queries or 1
    for path, v in sorted(metrics.accuracy_by_path.items()):
        acc = v["correct"] / (v["n"] or 1)
        print(f"     {path:<10} n={v['n']:<5} ({v['n']/total:>5.1%})  准确率={acc:.4f}")

    if metrics.by_difficulty:
        print("\n  按难度:")
        for diff, v in sorted(metrics.by_difficulty.items()):
            print(f"     {diff:<14} n={v['n']:<5} 准确率={v['accuracy']:.4f}")

    if metrics.misclassified:
        print(f"\n  错判样本 ({len(metrics.misclassified)} 条):")
        for m in metrics.misclassified:
            print(f"     [{m['expected']} → {m['predicted']}] "
                  f"(conf={m['confidence']}, path={m['path']}) {m['text'][:44]}")
            if m.get("note"):
                print(f"        注: {m['note']}")
    print(f"\n{sep}\n")
