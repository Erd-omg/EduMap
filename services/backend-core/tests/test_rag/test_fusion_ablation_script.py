"""Tests for the fusion-ablation script's methodological guards.

Why this file exists: the ablation is a *measurement* script, and a
measurement script that silently measures the wrong thing is worse than no
script. Two failure modes have actually bitten this repo before:

1. **Cache contamination.** An earlier "hybrid latency ×0.64" claim turned
   out to be a retrieval-result-cache artefact. The ablation must therefore
   call ``search(use_cache=False)`` on every query.
2. **Empty-index measurement.** ``VectorIndex`` silently degrades to an
   in-memory dict when ChromaDB is unreachable, so a run would "succeed"
   while measuring nothing. The script must refuse to start in that case.

These are asserted against the real script source, not a reimplementation,
so that deleting the guard fails the test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
ABLATION = SCRIPTS_DIR / "run_fusion_ablation.py"


def _load_source() -> str:
    assert ABLATION.exists(), f"ablation script missing: {ABLATION}"
    return ABLATION.read_text(encoding="utf-8")


class TestScriptGuards:
    def test_script_exists_and_parses(self) -> None:
        ast.parse(_load_source())

    def test_search_is_called_with_cache_bypassed(self) -> None:
        """The evaluation loop must pass use_cache=False."""
        src = _load_source()
        assert "use_cache=False" in src, (
            "ablation must bypass the retrieval-result cache; comparing "
            "latency with the cache on produced a fake result before"
        )

    def test_refuses_empty_vector_index(self) -> None:
        """A degraded VectorIndex must abort the run, not silently score 0."""
        src = _load_source()
        assert "_client" in src and "SystemExit" in src, (
            "ablation must detect VectorIndex's in-memory fallback and abort"
        )

    def test_allows_overriding_the_fusion_method(self) -> None:
        """The whole point: the same corpus scored under several strategies."""
        src = _load_source()
        assert "service.fusion_method = method" in src
        for method in ("rrf", "minmax", "score"):
            assert method in src, f"strategy {method} not offered"


class TestEvaluateMethod:
    """Drive the real evaluation helper with a stub service."""

    @pytest.mark.asyncio
    async def test_metrics_reflect_returned_order(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("fusion_ablation", ABLATION)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)

        from src.rag.models import RAGResult

        def _res(source_id: str, score: float) -> RAGResult:
            return RAGResult(
                content=f"c-{source_id}",
                source_type="chroma",
                source_id=source_id,
                source_name=source_id,
                score=score,
            )

        service = AsyncMock()
        service.search = AsyncMock(
            return_value=[_res("hit", 0.9), _res("miss", 0.5)]
        )

        queries = [{"query": "q1", "relevant_ids": {"hit"}}]
        out = await mod.evaluate_method(
            service, queries, "rrf", top_k=5, k_values=[1, 3]
        )

        assert out["n_queries"] == 1
        # relevant doc is ranked first
        assert out["mrr"] == pytest.approx(1.0)
        assert out["precision"]["1"] == pytest.approx(1.0)
        assert out["hit_rate"]["1"] == pytest.approx(1.0)
        # the fusion method was actually applied to the service
        assert service.fusion_method == "rrf"

    @pytest.mark.asyncio
    async def test_empty_query_set_does_not_divide_by_zero(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("fusion_ablation", ABLATION)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)

        service = AsyncMock()
        out = await mod.evaluate_method(service, [], "score", top_k=5, k_values=[1])
        assert out["n_queries"] == 0
        assert out["mrr"] == 0.0
