#!/usr/bin/env python3
"""Scripted mutation testing for EduMap's core modules.

Why this exists instead of mutmut
---------------------------------
mutmut 3.8 cannot mutate this project.  Its trampoline asserts that a mutated
module's name must not start with ``src.``:

    mutmut/stats.py:151
    assert not name.startswith("src."), "Failed trampoline hit..."

This repo's import root *is* ``src`` (``src/__init__.py`` exists and tests do
``from src.agents... import ...``), so every mutant trips that assertion and
the stats phase aborts.  The restriction is hard-coded and not configurable, so
rather than patch a third-party library we apply the same technique ourselves.

What it does
------------
For each configured source file it applies a list of **source-level mutations**
(flip a comparison, negate a boolean, drop an ``await``, swap a branch body,
remove a guard).  For each mutant it runs the targeted test subset and records
whether the tests noticed.  A mutant that survives means the test suite cannot
distinguish the mutated code from the original — i.e. that behaviour is
unverified.

Usage
-----
    python scripts/mutation_check.py                # all configured targets
    python scripts/mutation_check.py --file src/harness/retry.py
    python scripts/mutation_check.py --list          # show targets, run nothing

Exit code is 1 when any mutant survives, so it can be wired into CI later.
Treat survivors as a *triage list*, not a hard gate: some are equivalent
mutations that no test could ever distinguish.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Test subset that exercises the mutated modules.  Deliberately narrow: running
# the whole suite per mutant takes minutes each and most tests are irrelevant
# to any given module.
TEST_TARGETS = [
    "tests/test_agents/",
    "tests/test_harness/",
    "tests/test_tools/",
    "tests/test_utils/",
    "tests/test_rag/test_chunking.py",
    "tests/test_memory/test_recall_scoring.py",
    "tests/test_memory/test_episodic_embedding.py",
    "tests/test_agents/test_irt.py",
    "tests/test_agents/test_grading.py",
    "tests/test_agents/test_assessment_agent.py",
    "tests/test_agents/test_checkpointing.py",
    "tests/test_agents/test_retention.py",
    "tests/test_learning_path/test_forgetting_eval.py",
    "tests/test_learning_path/test_forgetting_curve.py",
    "tests/test_learning_path/test_review_log.py",
    "tests/test_resources/",
]

# Default mutation targets: the modules where a silent logic error reaches
# users, and where test gaps have actually hidden real bugs.
DEFAULT_TARGETS = [
    "src/agents/orchestrator/graph.py",
    "src/agents/orchestrator/state.py",
    "src/harness/retry.py",
    "src/harness/base.py",
    "src/tools/registry.py",
    "src/utils/llm_adapter.py",
    "src/main.py",
    "src/rag/chunking/semantic_chunker.py",
    "src/memory/recall_scoring.py",
    "src/agents/assessment/irt.py",
    "src/agents/assessment/grading.py",
    "src/learning_path/forgetting_eval.py",
    "src/learning_path/forgetting_curve.py",
    "src/utils/db_pool.py",
    "src/agents/assessment/agent.py",
    "src/resources/provenance.py",
    "src/agents/orchestrator/checkpointing.py",
    "src/agents/orchestrator/retention.py",
]


@dataclass
class Mutation:
    """One source-level edit to apply."""

    name: str
    pattern: str          # regex; first match is replaced
    replacement: str
    # Only apply if the pattern occurs exactly this many times (guards against
    # mutating an unexpected site after a refactor).  None disables the check.
    expected_count: int | None = 1


@dataclass
class Result:
    mutation: Mutation
    file: str
    killed: bool          # True = tests caught it (good)
    detail: str = ""


# ── Mutation catalogue ──────────────────────────────────────────────────
#
# Hand-written rather than generic AST mutation: the goal is to probe the
# *specific* invariants this codebase relies on (reducer semantics, failure
# attribution, retry boundaries), not to churn out hundreds of equivalent
# mutants.

MUTATIONS: dict[str, list[Mutation]] = {
    "src/agents/orchestrator/state.py": [
        Mutation(
            "resources-reducer: empty no longer resets",
            r"    if not right:\n        return \[\]",
            "    if False:\n        return []",
        ),
        Mutation(
            "resources-reducer: append dropped",
            r"    return list\(left or \[\]\) \+ list\(right\)",
            "    return list(right)",
        ),
        Mutation(
            "agent_results: merge becomes last-write-wins",
            r"    merged = dict\(left or \{\}\)\n    merged\.update\(right or \{\}\)\n    return merged",
            "    return dict(right or {})",
        ),
    ],
    "src/agents/orchestrator/graph.py": [
        Mutation(
            "router: fan-out reduced to one branch",
            r'return \["designer", "coder"\]',
            'return ["designer"]',
        ),
        Mutation(
            "retry: budget never exhausts",
            r"    if retries < max_retries:\n        return \"retry_generate\"",
            "    if True:\n        return \"retry_generate\"",
        ),
        Mutation(
            "retry_prep: counter not incremented",
            r'        "generation_retry_count": retries,',
            '        "generation_retry_count": 0,',
        ),
        Mutation(
            "assessment: prior status whitelist broken",
            # Appears in BOTH assessment_node and assessment_degraded_node;
            # mutate the first (assessment_node) only.
            r'        if prior_status in \("failed", "degraded", "cancelled"\)',
            '        if prior_status in ("", "processing")',
            expected_count=2,
        ),
        Mutation(
            "designer: echoes accumulated list instead of own delta",
            r"    designer_resources: list\[dict\] = \[\]",
            "    designer_resources: list[dict] = list(state.get('generated_resources', []))",
        ),
    ],
    "src/harness/retry.py": [
        Mutation(
            "retry: loop runs once",
            r"        for attempt in range\(max_retries\):",
            "        for attempt in range(1):",
        ),
        Mutation(
            "circuit: never records failure",
            r"            if circuit_record_failure:\n                circuit_record_failure\(\)",
            "            if False:\n                circuit_record_failure()",
        ),
        Mutation(
            "circuit: open check ignored",
            r"        if circuit_check and circuit_check\(\):",
            "        if False:",
        ),
    ],
    "src/harness/base.py": [
        Mutation(
            "report: tool_calls not surfaced",
            r"            if tool_calls:\n                report.tool_calls = list\(tool_calls\)",
            "            if False:\n                report.tool_calls = list(tool_calls)",
        ),
        Mutation(
            "breaker: attribution dropped",
            r"self\._llm\._record_failure\(self\._agent_name\)",
            "self._llm._record_failure()",
        ),
    ],
    "src/utils/llm_adapter.py": [
        Mutation(
            "breaker: attribution dropped",
            r"        if source:\n            self\._failure_sources\[source\] = self\._failure_sources\.get\(source, 0\) \+ 1\n            self\._circuit_source = source",
            "        if False:\n            self._failure_sources[source] = self._failure_sources.get(source, 0) + 1\n            self._circuit_source = source",
        ),
        Mutation(
            "breaker: open_count never increments",
            r"            self\._circuit_open_count \+= 1",
            "            self._circuit_open_count += 0",
        ),
        Mutation(
            "breaker: success does not clear attribution",
            r"        if self\._failure_sources:\n            self\._failure_sources\.clear\(\)",
            "        if False:\n            self._failure_sources.clear()",
        ),
        Mutation(
            "breaker: state reports closed always",
            r'            "open": state == "open",',
            '            "open": False,',
        ),
        Mutation(
            "half-open: probe slot not gated (all callers admitted)",
            r"        if self\._half_open_probe_in_flight:\n            return False",
            "        if self._half_open_probe_in_flight:\n            return True",
        ),
        Mutation(
            "half-open: probe failure does not re-open",
            r"        if was_half_open or self\._consecutive_failures >= self\._circuit_threshold:",
            "        if self._consecutive_failures >= self._circuit_threshold:",
        ),
        Mutation(
            "expiry: half-open collapses back to closed",
            r'        if time\.monotonic\(\) > self\._circuit_open_until:\n            return "half_open"',
            '        if time.monotonic() > self._circuit_open_until:\n            return "closed"',
        ),
    ],
    "src/main.py": [
        Mutation(
            "readiness: circuit breaker block dropped",
            r'        "circuit_breaker": circuit_breaker,',
            '        "circuit_breaker": None,',
        ),
    ],
    "src/rag/chunking/semantic_chunker.py": [
        Mutation(
            "chunker: explicit None-model branch disabled",
            r"        if self\._model is None:\n            return self\._paragraph_fallback\(text\)",
            "        if False:\n            return self._paragraph_fallback(text)",
        ),
        Mutation(
            "chunker: model reloaded even when already set",
            r"        if self\._model is not None:\n            return",
            "        if False:\n            return",
        ),
    ],
    "src/tools/registry.py": [
        Mutation(
            "tool: timeout removed",
            r"            return await asyncio\.wait_for\(\n                tool\.execute\(\*\*kwargs\), timeout=self\._timeout\n            \)",
            "            return await tool.execute(**kwargs)",
        ),
        Mutation(
            "tool: exceptions propagate instead of degrading",
            r'            return ToolResult\(success=False, error=str\(exc\)\)\n\n    def build_prompt_block',
            '            raise\n\n    def build_prompt_block',
        ),
    ],
    "src/memory/recall_scoring.py": [
        Mutation(
            "recency: decay constant changed away from the paper value",
            r"RECENCY_DECAY = 0\.995",
            "RECENCY_DECAY = 0.9",
        ),
        Mutation(
            "score: importance term dropped",
            r"        score = WEIGHT_RECENCY \* recency \+ WEIGHT_IMPORTANCE \* importance",
            "        score = WEIGHT_RECENCY * recency",
        ),
        Mutation(
            "score: relevance term dropped",
            r"            score \+= WEIGHT_RELEVANCE \* relevance",
            "            pass",
        ),
        Mutation(
            "recency: undateable entries treated as ancient",
            r"    if created_at is None:\n        return 0\.0",
            "    if created_at is None:\n        return float('inf')",
        ),
        Mutation(
            "cosine: negative similarity not clamped",
            r"    return max\(0\.0, min\(1\.0, sim\)\)",
            "    return sim",
        ),
        Mutation(
            "ranking: ascending instead of descending",
            r"    scored\.sort\(key=lambda pair: pair\[1\], reverse=True\)",
            "    scored.sort(key=lambda pair: pair[1])",
        ),
        Mutation(
            "cosine: missing vector scores 1.0 instead of 0.0",
            r"    if not a or not b:\n        return 0\.0",
            "    if not a or not b:\n        return 1.0",
        ),
    ],
    "src/agents/assessment/irt.py": [
        Mutation(
            "ability: prior term dropped (MLE instead of MAP)",
            r"        return _log_likelihood\(responses, theta\) - \(\n            prior_weight \* \(theta - prior\) \*\* 2 / 2\.0\n        \)",
            "        return _log_likelihood(responses, theta)",
        ),
        Mutation(
            "mastery: not monotone in ability",
            r"    return _sigmoid\(ability\)\n\n\ndef _log_likelihood",
            "    return 1.0 - _sigmoid(ability)\n\n\ndef _log_likelihood",
        ),
        Mutation(
            "probability: discrimination sign flipped",
            r"    return _sigmoid\(discrimination \* \(ability - difficulty\)\)",
            "    return _sigmoid(discrimination * (difficulty - ability))",
        ),
    ],
    "src/agents/assessment/grading.py": [
        Mutation(
            "grade: blank answer treated as correct",
            r"    submitted = _normalise\(answer\)\n    if not submitted:\n        return False",
            "    submitted = _normalise(answer)\n    if not submitted:\n        return True",
        ),
        Mutation(
            "difficulty: 1-5 mapping reversed",
            r"    return \(value - midpoint\) / \(midpoint - _DIFFICULTY_MIN\) \* \(_DIFFICULTY_LOGIT_SPAN / 2\)",
            "    return -(value - midpoint) / (midpoint - _DIFFICULTY_MIN) * (_DIFFICULTY_LOGIT_SPAN / 2)",
        ),
        Mutation(
            "delta: sign flipped",
            r"    delta = graded\.mastery - baseline",
            "    delta = baseline - graded.mastery",
        ),
    ],
    "src/learning_path/forgetting_eval.py": [
        Mutation(
            "auc: ties counted as all-positive",
            r"    u = pos_rank_sum - n_pos \* \(n_pos \+ 1\) / 2",
            "    u = pos_rank_sum",
        ),
        Mutation(
            "rmse: observed rate forced to prediction (zero error)",
            r"        total \+= \(mean_pred - observed\) \*\* 2",
            "        total += 0.0",
        ),
        Mutation(
            "recalled: threshold inverted",
            r"                        recalled=score >= 0\.6,",
            "                        recalled=score < 0.6,",
        ),
    ],
    "src/learning_path/forgetting_curve.py": [
        Mutation(
            "curve: exponential instead of power law",
            r"    return \(1\.0 \+ FACTOR \* elapsed_hours / stability\) \*\* \(-DECAY\)",
            "    return math.exp(-elapsed_hours / stability)",
        ),
        Mutation(
            "curve: growth exponent dropped (no spacing effect)",
            r"    growth = \(\(prior_count \+ 1\) / 2\.0\) \*\* GROWTH_EXP",
            "    growth = 1.0",
        ),
        Mutation(
            "curve: score factor normalisation removed",
            r"    score_factor = \(max\(posterior_mean, MIN_POSTERIOR\) / NEUTRAL_POSTERIOR\) \*\* SCORE_EXP",
            "    score_factor = posterior_mean ** SCORE_EXP",
        ),
        Mutation(
            "curve: stability clamp removed",
            r"    return max\(S_MIN, min\(S_MAX, raw\)\)",
            "    return raw",
        ),
        Mutation(
            "curve: zero elapsed no longer short-circuits to certainty",
            r"    if elapsed_hours <= 0:\n        return 1\.0\n    if stability <= 0:",
            "    if elapsed_hours <= 0:\n        return 0.0\n    if stability <= 0:",
        ),
        Mutation(
            "curve: zero stability no longer returns zero recall",
            r"    if stability <= 0:\n        return 0\.0",
            "    if stability <= -1.0:\n        return 0.0",
        ),
    ],
    "src/agents/orchestrator/retention.py": [
        Mutation(
            "retention: undatable thread treated as stale (destructive guess)",
            r"        if last_activity is None:\n            # Either the thread vanished or its timestamp is unreadable. Both\n            # mean \"cannot date it\" → keep.\n            return False",
            "        if last_activity is None:\n            return True",
        ),
        Mutation(
            "retention: stale threads never deleted",
            r"    for tid in stale:\n        if await delete_thread\(saver, tid\):\n            report.threads_deleted \+= 1",
            "    for tid in stale:\n        if False:\n            report.threads_deleted += 1",
        ),
        Mutation(
            "retention: stale-set argument replaces discovery",
            r"        stale = \[tid for tid in thread_ids if tid in precomputed\]",
            "        stale = list(precomputed)",
        ),
        Mutation(
            "retention: quoted JSON timestamp not unquoted",
            r"""    cleaned = raw\.strip\(\)\.strip\('\"'\)""",
            "    cleaned = raw.strip()",
        ),
        Mutation(
            "retention: naive timestamp left unlocalised",
            r"        parsed = parsed\.replace\(tzinfo=timezone\.utc\)",
            "        pass",
        ),
    ],
    "src/agents/orchestrator/checkpointing.py": [
        Mutation(
            "checkpoint: 255-char guard removed",
            r'    if len\(session_id\) > 255:\n        raise ValueError\(\n            f"thread_id must be < 255 chars',
            '    if False:\n        raise ValueError(\n            f"thread_id must be < 255 chars',
        ),
        Mutation(
            "checkpoint: DSN scheme not stripped",
            r'    if dsn\.startswith\("postgresql\+asyncpg://"\):\n        return dsn\.replace\("postgresql\+asyncpg://", "postgresql://", 1\)',
            '    if False:\n        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)',
        ),
        Mutation(
            "checkpoint: tables never created",
            r"                await self\._saver\.setup\(\)",
            "                pass",
        ),
        Mutation(
            "checkpoint: pool not opened explicitly",
            r"            await self\._pool\.open\(wait=True, timeout=10\)",
            "            pass",
        ),
        Mutation(
            "checkpoint: autocommit dropped (langgraph requires it)",
            r'                kwargs=\{"autocommit": True, "prepare_threshold": 0\},',
            '                kwargs={"autocommit": False, "prepare_threshold": 0},',
        ),
    ],
    "src/resources/provenance.py": [
        Mutation(
            "provenance: unlocated chunk given offset 0 instead of None",
            r"            spans\.append\(ChunkSpan\(text=chunk, char_start=None, char_end=None, index=index\)\)\n            continue\n\n        end = start \+ len\(stripped\)",
            "            spans.append(ChunkSpan(text=chunk, char_start=0, char_end=0, index=index))\n            continue\n\n        end = start + len(stripped)",
        ),
        Mutation(
            "provenance: whitespace-tolerant search disabled",
            r"    match = re\.compile\(pattern\)\.search\(haystack, start\)\n    return match\.start\(\) if match else -1",
            "    return -1",
        ),
        Mutation(
            "provenance: cursor never advances (overlaps re-found)",
            r"        cursor = end",
            "        cursor = 0",
        ),
        Mutation(
            "provenance: forward search replaced by global search",
            r"        start = _search_from\(source, stripped, cursor\)",
            "        start = _search_from(source, stripped, 0)",
        ),
    ],
    "src/utils/db_pool.py": [
        Mutation(
            "pool: wrapper no longer unwrapped",
            r"    inner = getattr\(db_pool, \"pool\", None\)\n    if inner is not None and hasattr\(inner, \"acquire\"\):\n        logger\.debug\(\n            \"%s: unwrapped a pool wrapper \(%s\) to its \.pool\",\n            owner,\n            type\(db_pool\)\.__name__,\n        \)\n        return inner",
            "    return db_pool",
        ),
        Mutation(
            "pool: unusable argument silently accepted",
            r'    raise TypeError\(\n        f"\{owner\}: db_pool must expose acquire\(\)',
            '    return None\n    raise TypeError(\n        f"{owner}: db_pool must expose acquire()',
        ),
    ],
    "src/agents/assessment/agent.py": [
        Mutation(
            "mastery: LLM self-report allowed through again",
            r'                if data\.get\("estimated_mastery_delta"\):\n                    logger\.debug\(',
            '                if data.get("estimated_mastery_delta"):\n                    mastery_delta.update(data["estimated_mastery_delta"])\n                    logger.debug(',
        ),
        Mutation(
            "questions: difficulty dropped",
            r"                            difficulty=knowledge_unit\.difficulty,",
            "                            difficulty=None,",
        ),
    ],
}


def apply_mutation(text: str, mutation: Mutation) -> str | None:
    """Return mutated text, or None when the pattern doesn't match cleanly.

    Returning None (rather than mutating the wrong site) is deliberate: after a
    refactor the patterns can go stale, and a stale mutation silently testing
    nothing is the exact failure mode this tool is meant to catch.
    """
    matches = re.findall(mutation.pattern, text)
    if not matches:
        return None
    if mutation.expected_count is not None and len(matches) != mutation.expected_count:
        return None
    return re.sub(mutation.pattern, lambda _m: mutation.replacement, text, count=1)


def run_tests() -> tuple[bool, str]:
    """Run the targeted subset.  Returns (passed, tail_of_output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *TEST_TARGETS,
         "-q", "-x", "-p", "no:randomly", "--no-header", "--timeout=120"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, "\n".join(out[-3:])


def check_file(rel_path: str, mutations: list[Mutation], verbose: bool) -> list[Result]:
    path = ROOT / rel_path
    if not path.exists():
        print(f"  ! {rel_path} not found — skipped")
        return []

    original = path.read_text()
    backup = path.with_suffix(path.suffix + ".mutbak")
    shutil.copy2(path, backup)
    results: list[Result] = []

    try:
        for mutation in mutations:
            mutated = apply_mutation(original, mutation)
            if mutated is None:
                results.append(Result(
                    mutation, rel_path, killed=False,
                    detail="PATTERN DID NOT MATCH (stale mutation — update it)",
                ))
                print(f"  ~ {rel_path}::{mutation.name}: pattern did not match")
                continue

            path.write_text(mutated)
            try:
                passed, tail = run_tests()
            except subprocess.TimeoutExpired:
                passed, tail = False, "test run timed out (counted as killed)"

            killed = not passed
            results.append(Result(mutation, rel_path, killed, "" if killed else tail))
            print(f"  {'KILLED ' if killed else 'SURVIVED'} {rel_path}::{mutation.name}")
            if verbose and not killed:
                print(f"      {tail}")
    finally:
        # Always restore, even on exception — leaving a mutant in the tree
        # would be exactly the kind of silent corruption this script exists to
        # prevent.
        shutil.copy2(backup, path)
        backup.unlink(missing_ok=True)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", help="only check this file")
    parser.add_argument("--list", action="store_true", help="list targets and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show test output for survivors")
    args = parser.parse_args()

    targets = DEFAULT_TARGETS if not args.file else [args.file]

    if args.list:
        for t in targets:
            n = len(MUTATIONS.get(t, []))
            print(f"{t}  ({n} mutations)")
        return 0

    print("Scripted mutation check — see module docstring for why not mutmut.\n")
    all_results: list[Result] = []
    for target in targets:
        if target not in MUTATIONS:
            print(f"! no mutations defined for {target}")
            continue
        print(f"{target}")
        all_results.extend(check_file(target, MUTATIONS[target], args.verbose))

    if not all_results:
        print("\nNo mutations ran.")
        return 1

    killed = [r for r in all_results if r.killed]
    survived = [r for r in all_results if not r.killed]
    print(f"\n{'=' * 62}")
    print(f"killed {len(killed)}/{len(all_results)}  "
          f"({len(killed) / len(all_results):.0%} mutation score)")
    if survived:
        print(f"\nSURVIVORS — these behaviours are not verified by any test:")
        for r in survived:
            print(f"  - {r.file}::{r.mutation.name}")
            if r.detail:
                for line in r.detail.splitlines():
                    print(f"      {line}")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
