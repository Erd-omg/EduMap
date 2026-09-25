#!/usr/bin/env python3
"""Compare forgetting-curve models on a common scale.

Scores three candidates on the same histories with the same metrics
(LogLoss / RMSE / AUC) that the public ``srs-benchmark`` uses:

1. **edumap**  — the shipped model: ``S = 24 × BetaPosteriorMean × log2(n+1)``,
   ``R = exp(-t/S)``.  Exponential curve, scalar strength.
2. **moving-avg** — a **0-parameter** baseline.  Predicts the mean of the
   learner's prior scores, ignoring elapsed time entirely.
   This is the baseline that matters: on srs-benchmark a moving average
   scores LogLoss 0.3369 versus FSRS-6's 0.3460, i.e. it *beats* a
   21-parameter model.  Any model that cannot clear this bar is not earning
   its complexity.
3. **fsrs-shaped** — a power-law curve with a fixed (untuned) decay, to show
   what the curve *shape* alone buys.  This is deliberately NOT a real FSRS
   implementation: real FSRS requires per-item parameter fitting on a large
   review corpus, which this project does not have.  It isolates the shape
   variable so the comparison is interpretable.

Usage
-----
    cd services/backend-core
    python scripts/run_forgetting_eval.py [--items 300] [--reviews 12]
        [--seed 20260923] [--output benchmark_results]
    python scripts/run_forgetting_eval.py --self-check   # harness sanity
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learning_path.forgetting_eval import (  # noqa: E402
    ModelReport,
    RecallModel,
    evaluate_model,
)
from src.learning_path.forgetting_curve import (  # noqa: E402
    S_BASE,
    S_MAX,
    S_MIN,
)
from src.learning_path.forgetting_synth import generate_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("forgetting_eval")

# Below this many real observations the fit is noise. Stated explicitly so the
# decision "is there enough data yet?" has an answer that does not depend on
# someone's judgement in the moment.
_MIN_REAL_OBSERVATIONS = 500

# ── Candidate models ─────────────────────────────────────────────────────
#
# Each is a plain callable (elapsed_hours, prior_count, mean_prior_score) -> p.
# Keeping them decoupled from the production classes means the harness scores
# the *formula*, and a refactor of the service cannot silently change the
# benchmark.
#
# S_BASE / S_MIN / S_MAX are imported from the production module (see the
# import block above) rather than redefined here, so the legacy comparison
# below uses the same constants the shipped model does.


def edumap_model(elapsed_hours: float, prior_count: int, mean_prior: float) -> float:
    """The curve the service actually uses.

    Delegates to the production functions rather than re-implementing the
    formula: a benchmark that scores a *copy* of the maths can silently drift
    from the shipped behaviour, which would make the whole measurement
    worthless.  ``forgetting_curve`` is the single source of truth.
    """
    from src.learning_path.forgetting_curve import (
        recall_probability,
        stability_after_review,
    )

    stability = stability_after_review(prior_count, mean_prior)
    return recall_probability(elapsed_hours, stability)


def legacy_edumap_model(elapsed_hours: float, prior_count: int, mean_prior: float) -> float:
    """The PREVIOUS curve, kept so the fix can be measured against it.

    ``S = 24 · mean_prior · log2(n+1)`` with ``R = exp(-t/S)``.  Retained to
    document what was wrong and to prove the replacement is better — without
    it, a future reader cannot tell whether the change helped.
    """
    stability = S_BASE * mean_prior * math.log2(prior_count + 1)
    stability = min(max(stability, S_MIN), S_MAX)
    if elapsed_hours <= 0:
        return 1.0
    return math.exp(-elapsed_hours / stability)


def moving_average_model(elapsed_hours: float, prior_count: int, mean_prior: float) -> float:
    """0-parameter baseline: your past average predicts your next result.

    Deliberately ignores elapsed time.  It is the "do nothing clever" control
    that any proposed memory model must beat to justify itself.
    """
    return mean_prior


def fsrs_shaped_model(elapsed_hours: float, prior_count: int, mean_prior: float) -> float:
    """Power-law curve with untuned parameters (shape-only comparison).

    ``R = (1 + factor·t/S)^(-decay)`` — the FSRS v4+ form.  Stability grows
    with successful repetition (the spacing effect).  Parameters are the
    documented FSRS defaults, NOT fitted to this data, so this measures what
    the curve shape buys before any fitting.
    """
    if elapsed_hours <= 0:
        return 1.0
    # FSRS-6 default w0..w2 ≈ 0.212, 1.2931, 2.3065; decay w20 ≈ 0.1542
    decay = 0.1542
    stability = 0.212 + 1.2931 * prior_count * (mean_prior ** 2.3065)
    stability = max(stability, 0.01)
    factor = 0.9 ** (-1.0 / decay) - 1.0
    p = (1.0 + factor * elapsed_hours / stability) ** (-decay)
    return min(1.0, max(0.0, p))


CANDIDATES: dict[str, RecallModel] = {
    "edumap": edumap_model,
    "legacy": legacy_edumap_model,
    "moving-avg": moving_average_model,
    "fsrs-shaped": fsrs_shaped_model,
}


# ── Self-check ───────────────────────────────────────────────────────────

def self_check() -> int:
    """Assert the harness can tell a good model from a deliberately bad one.

    Without this, a "winner" on real data could just mean the metric is
    insensitive.  A model shaped like the ground truth and its exact inverse
    must land on opposite sides of the AUC scale.

    Thresholds are deliberately loose (0.65 / 0.35 rather than 0.8 / 0.2):
    the generator adds guess/slip noise, so an oracle that knows the true
    curve still cannot score perfectly.  What must hold is the *separation*,
    not an absolute ceiling.
    """
    dataset = generate_dataset(n_items=80, n_reviews=10, seed=7)

    def oracle(elapsed: float, n: int, mean_prior: float) -> float:
        # The ground-truth generator's own shape, with its default stability.
        if elapsed <= 0:
            return 1.0
        stability = 24.0 * (1.0 + n) ** 0.6
        return min(1.0, max(0.0, (1.0 + elapsed / (9.0 * stability)) ** (-1.0)))

    def inverted(elapsed: float, n: int, mean_prior: float) -> float:
        return 1.0 - oracle(elapsed, n, mean_prior)

    oracle_report = evaluate_model("oracle", oracle, dataset)
    inverted_report = evaluate_model("inverted", inverted, dataset)

    print(f"self-check  oracle   AUC={oracle_report.auc:.4f}  LogLoss={oracle_report.log_loss:.4f}")
    print(f"self-check  inverted AUC={inverted_report.auc:.4f}  LogLoss={inverted_report.log_loss:.4f}")

    problems: list[str] = []
    if not oracle_report.auc > 0.65:
        problems.append(f"oracle AUC {oracle_report.auc:.4f} should exceed 0.65")
    if not inverted_report.auc < 0.35:
        problems.append(f"inverted AUC {inverted_report.auc:.4f} should be below 0.35")
    if not oracle_report.log_loss < inverted_report.log_loss:
        problems.append("oracle LogLoss should be lower than inverted")
    # The separation itself is the property under test.
    gap = oracle_report.auc - inverted_report.auc
    if not gap > 0.3:
        problems.append(f"AUC separation {gap:.4f} too small — metrics are insensitive")

    for p in problems:
        print(f"  FAIL {p}")
    if problems:
        return 1
    print(f"self-check OK — harness discriminates (AUC gap {gap:.4f}).")
    return 0


# ── Reporting ────────────────────────────────────────────────────────────

def _print_table(reports: dict[str, ModelReport]) -> None:
    header = f"{'model':14} {'n':>6} {'LogLoss↓':>10} {'RMSE↓':>9} {'AUC↑':>8}"
    print()
    print("遗忘曲线模型对比（合成数据，指标口径同 srs-benchmark）")
    print(header)
    print("-" * len(header))
    for name, r in reports.items():
        print(
            f"{name:14} {r.n:>6} {r.log_loss:>10.4f} {r.rmse:>9.4f} {r.auc:>8.4f}"
        )


def _verdict(reports: dict[str, ModelReport]) -> list[str]:
    """Turn the table into statements, since that is what a reader needs."""
    lines: list[str] = []
    base = reports.get("moving-avg")
    shipped = reports.get("edumap")
    legacy = reports.get("legacy")
    shaped = reports.get("fsrs-shaped")

    if base and shipped:
        delta = shipped.log_loss - base.log_loss
        if delta < 0:
            lines.append(
                f"✅ edumap 的 LogLoss 比 0 参数基线低 {abs(delta):.4f} —— "
                "**超越了朴素基线**，其复杂度是有理由的。"
            )
        else:
            lines.append(
                f"⚠️ edumap 的 LogLoss 比 0 参数基线**高** {delta:.4f} —— "
                "**仍未打赢「用历史均值预测下一次」这个朴素策略**。"
            )

    if legacy and shipped:
        improvement = legacy.log_loss - shipped.log_loss
        lines.append(
            f"相对旧公式（log2 增长 + 指数曲线）的 LogLoss 改善：{improvement:+.4f}"
            f"（{legacy.log_loss:.4f} → {shipped.log_loss:.4f}）。"
        )

    if base and shaped:
        lines.append(
            f"仅换曲线形状（幂律 vs 指数，未拟合）相对基线的 LogLoss 差为 "
            f"{shaped.log_loss - base.log_loss:+.4f}。"
        )

    best = min(reports.values(), key=lambda r: r.log_loss)
    lines.append(f"LogLoss 最优：{best.name}")
    return lines


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=300)
    parser.add_argument("--reviews", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", default="benchmark_results")
    parser.add_argument(
        "--source", default="synthetic", choices=["synthetic", "real"],
        help=(
            "数据来源：synthetic（默认，自我校验用）或 real —— 后者从 "
            "forgetting_review_log 读取真实复习历史。**真实数据才是决策依据**；"
            "合成数据只用于验证评测框架本身。"
        ),
    )
    parser.add_argument(
        "--user-id", default=None,
        help="--source real 时限定某个用户（默认取全部用户）",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="只跑框架自检（确认指标能区分好坏模型），不出对比表",
    )
    args = parser.parse_args()

    if args.self_check:
        return self_check()

    if args.source == "real":
        dataset = await _load_real_dataset(args.user_id)
        n_obs = sum(max(len(h) - 1, 0) for h in dataset)

        if not dataset:
            logger.error(
                "forgetting_review_log 里没有可用的复习历史。\n"
                "  评测需要 (间隔, 成绩) 序列：同一用户在同一知识点上至少复习两次，\n"
                "  且需要至少 %d 个观测点才值得据此调参（当前 0）。\n"
                "  管道本身是通的 —— 缺的是真实使用产生的数据。\n"
                "  注意：**不要用合成数据上的最优值去改生产参数**，那是拟合自己的假设。",
                _MIN_REAL_OBSERVATIONS,
            )
            return 2

        # Margin gate. Fitting exponents to a handful of observations would
        # produce a number that looks like a result but is noise; the public
        # benchmark this project compares against uses ~350M reviews across
        # 10k users, so a few hundred observations is already a stretch. The
        # threshold is not a validity claim — it is the point below which the
        # output should not be acted on at all.
        if n_obs < _MIN_REAL_OBSERVATIONS:
            logger.warning(
                "真实数据 %d 个观测点，低于建议下限 %d —— 结果仅供参考，**不要**据此改"
                "生产参数（会把噪声当结论）。继续跑只是为了看数据形态。",
                n_obs, _MIN_REAL_OBSERVATIONS,
            )
        logger.info("真实数据集: %d 条历史 / %d 个观测点", len(dataset), n_obs)
    else:
        dataset = generate_dataset(
            n_items=args.items, n_reviews=args.reviews, seed=args.seed
        )
        n_obs = sum(max(len(h) - 1, 0) for h in dataset)
        logger.info("合成数据集: %d 条历史 / %d 个观测点", len(dataset), n_obs)

    reports = {
        name: evaluate_model(name, model, dataset)
        for name, model in CANDIDATES.items()
    }
    _print_table(reports)

    print()
    for line in _verdict(reports):
        print(f"· {line}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"forgetting_eval_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "kind": args.source,
                "n_items": args.items,
                "n_reviews": args.reviews,
                "n_observations": n_obs,
                "seed": args.seed,
                "models": {n: r.to_dict() for n, r in reports.items()},
                "calibration": {n: r.calibration for n, r in reports.items()},
                "warning": (
                    "合成数据仅用于验证评测框架与建立量级参照，"
                    "不能替代真实复习历史上的验证。"
                ) if args.source == "synthetic" else (
                    "真实复习历史。仍受样本量限制 —— 记录用户数与观测点数，"
                    "样本过小时结论不稳。"
                ),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("结果已写入 %s", out_path)
    return 0


async def _load_real_dataset(user_id: str | None) -> list[list[tuple[float, float]]]:
    """Read real review histories from forgetting_review_log.

    Returns the same shape the synthetic generator produces, so the evaluation
    harness is identical for both sources — which is what makes the two
    comparable at all.

    Requires at least two reviews on a (user, kp) pair: with one review there is
    no interval to predict from, and ``predict_all`` skips such histories
    entirely (it would otherwise contribute nothing but still inflate the item
    count).
    """
    from src.config import settings
    from src.memory.db import MemoryDBPool
    from src.learning_path.forgetting_curve import ForgettingCurveService

    db = MemoryDBPool(dsn=settings.database_url)
    await db.create()
    try:
        service = ForgettingCurveService(db_pool=db.pool)
        if user_id:
            histories = await service.load_review_history(user_id)
        else:
            histories = await _load_all_users_history(db.pool)

        return [
            hist for hist in histories.values()
            if len(hist) >= 2  # need a measurable interval
        ]
    finally:
        await db.close()


async def _load_all_users_history(pool) -> dict[str, list[tuple[float, float]]]:
    """Every user's histories, keyed by ``{user}::{kp}``.

    ``load_review_history`` is scoped to one user, so the multi-user case reads
    the log directly and groups by (user_id, kp_id).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT user_id, kp_id, score, reviewed_at
            FROM forgetting_review_log
            ORDER BY user_id, kp_id, reviewed_at ASC
            """
        )

    out: dict[str, list[tuple[float, float]]] = {}
    previous: dict[tuple[str, str], object] = {}
    for row in rows:
        key = (row["user_id"], row["kp_id"])
        reviewed = row["reviewed_at"]
        prev = previous.get(key)
        elapsed = 0.0 if prev is None else (reviewed - prev).total_seconds() / 3600.0
        previous[key] = reviewed
        out.setdefault(f"{key[0]}::{key[1]}", []).append(
            (round(elapsed, 4), float(row["score"]))
        )
    return out


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
