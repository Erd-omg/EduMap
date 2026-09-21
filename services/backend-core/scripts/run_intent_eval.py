#!/usr/bin/env python3
"""Run the intent-classification accuracy evaluation.

Classifies a hand-labeled dataset with the production fused classifier
(``src.analysis.intent.classify_intent``) and reports accuracy, a confusion
matrix, per-class P/R/F1, and — most importantly — the *path distribution*:
how often the cheap rule path resolved a message with zero model calls versus
how often an ambiguous message had to escalate to the LLM + embedding vote.

Usage:
    cd services/backend-core
    python scripts/run_intent_eval.py [--no-cache] [--output benchmark_results]

Flags:
    --no-cache   Clear the intent cache before every row and report the
                 classifier's true first-pass accuracy (the headline number).
                 Without it, repeated messages hit the cache and the reported
                 accuracy is inflated by cache hits.
    --sample N   Evaluate only the first N rows (smoke runs).

Requires a reachable LLM (for the fused path) and the local embedding model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.rag.evaluation.datasets import load_intent_labeled  # noqa: E402
from src.rag.evaluation.intent_eval import (  # noqa: E402
    assert_models_present,
    evaluate_intent,
    print_report,
)
from src.utils.llm_adapter import create_llm  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("intent_eval")


def _bypass_proxy_for_localhost() -> None:
    """Keep local infra traffic off the system proxy (see run_strategy_comparison)."""
    local = "localhost,127.0.0.1,::1"
    existing = os.environ.get("no_proxy", "")
    if "localhost" not in existing:
        os.environ["no_proxy"] = f"{existing},{local}".strip(",") if existing else local
    os.environ["NO_PROXY"] = os.environ["no_proxy"]


def build_request_stub():
    """Build a minimal request object carrying the models the classifier reads.

    ``classify_intent`` reaches into ``request.app.state`` for the LLM adapter
    and the embedding model, so a script run needs a stub with the same shape.
    """
    _bypass_proxy_for_localhost()

    llm = create_llm(settings)
    model = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.llm_embedding_model)
        logger.warning("Embedding model loaded: %s", settings.llm_embedding_model)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("Embedding model failed to load: %s", exc)

    state = SimpleNamespace(llm_adapter=llm, _embedding_model=model)
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="每行前清空意图缓存，测冷启动真实准确率（头条数字）",
    )
    parser.add_argument("--sample", type=int, default=0, help="只评测前 N 条")
    parser.add_argument(
        "--output", type=str, default="benchmark_results", help="结果输出目录"
    )
    args = parser.parse_args()

    rows = load_intent_labeled()
    if not rows:
        logger.error("评测集为空 — 检查 datasets/intent_labeled.json")
        return 1
    # Shuffle before sampling: the dataset is grouped by class, so taking a
    # naive prefix for --sample would evaluate a single class only.
    random.Random(0).shuffle(rows)
    if args.sample:
        rows = rows[: args.sample]
    logger.warning("加载意图标注集: %d 条", len(rows))

    request = build_request_stub()
    assert_models_present(request)

    cold = args.no_cache
    run_id = uuid.uuid4().hex[:8]
    metrics = await evaluate_intent(rows, request, cold=cold, run_id=run_id)

    print_report(metrics)

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"intent_eval_{timestamp}.json"
    payload = {
        "report_name": "intent_eval",
        "generated_at": timestamp,
        "config": {
            "cold": cold,
            "run_id": run_id,
            "embedding_model": settings.llm_embedding_model,
            "llm_model": settings.llm_model,
            "n_rows": len(rows),
        },
        "metrics": metrics.to_dict(),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.warning("结果已保存: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
