"""Tests for ``src.config.Settings`` validation.

Focus: ``hybrid_fusion_method`` must reject unrecognised values at
construction time rather than letting them degrade silently at request time.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.rag.rag_service import RAGRetrievalService


class TestFusionMethodValidation:
    """``HYBRID_FUSION_METHOD`` is validated on load, not on first query."""

    def test_default_is_score(self, monkeypatch) -> None:
        """The *declared default*, independent of the ambient environment.

        ``Settings`` reads the process env and the repo ``.env``; without
        clearing ``HYBRID_FUSION_METHOD`` this asserts whatever the developer
        happens to have exported, and fails for a correct checkout when the
        (documented, supported) rollback override is in play.
        """
        monkeypatch.delenv("HYBRID_FUSION_METHOD", raising=False)
        assert Settings(_env_file=None).hybrid_fusion_method == "score"

    @pytest.mark.parametrize("value", ["rrf", "minmax", "score"])
    def test_accepts_every_implemented_strategy(self, value: str) -> None:
        """The allow-list must match what ``_merge_and_rank`` dispatches on."""
        assert Settings(hybrid_fusion_method=value).hybrid_fusion_method == value

    def test_allow_list_matches_the_dispatch_table(self) -> None:
        """``config`` and ``rag_service`` must not drift apart."""
        from src.rag.rag_service import FUSION_METHODS

        for method in FUSION_METHODS:
            assert Settings(hybrid_fusion_method=method).hybrid_fusion_method == method

    def test_normalises_case_and_padding(self) -> None:
        assert Settings(hybrid_fusion_method="  RRF ").hybrid_fusion_method == "rrf"

    def test_rejects_a_typo(self) -> None:
        """The whole point: ``scoer`` used to fall through to rrf, silently.

        ``_merge_and_rank``'s dispatch ends in ``return _merge_rrf(...)``, so
        an unrecognised value was served as RRF with nothing logged.  A
        benchmark run against that deployment would report numbers for a
        strategy nobody selected.
        """
        with pytest.raises(ValidationError) as exc:
            Settings(hybrid_fusion_method="scoer")
        assert "hybrid_fusion_method" in str(exc.value)

    def test_rejects_an_empty_value(self) -> None:
        with pytest.raises(ValidationError):
            Settings(hybrid_fusion_method="")

    def test_env_var_is_the_rollback_path(self, monkeypatch) -> None:
        """Overriding by environment must work without touching code.

        This is the mechanism the issue assumed did not exist.
        """
        monkeypatch.setenv("HYBRID_FUSION_METHOD", "rrf")
        assert Settings().hybrid_fusion_method == "rrf"


class TestAssignmentIsAlsoValidated:
    """``Settings`` is not the only way in — the attribute must guard too.

    ``main.py`` and three benchmark scripts assign ``svc.fusion_method``
    directly.  A validator on ``Settings`` alone leaves those paths unchecked,
    so the guard lives on the attribute itself.
    """

    def test_assigning_a_valid_method_works(self) -> None:
        svc = RAGRetrievalService()
        svc.fusion_method = "RRF"
        assert svc.fusion_method == "rrf"

    def test_assigning_an_unknown_method_raises(self) -> None:
        svc = RAGRetrievalService()
        with pytest.raises(ValueError):
            svc.fusion_method = "scoer"

    def test_assigning_an_empty_method_raises(self) -> None:
        svc = RAGRetrievalService()
        with pytest.raises(ValueError):
            svc.fusion_method = ""


class TestDispatchFallbackIsLoud:
    """The residual fallback path warns — once, not once per query."""

    @staticmethod
    def _result(source_id: str, source_type: str, score: float):
        from src.rag.models import RAGResult

        return RAGResult(
            source_id=source_id,
            source_type=source_type,  # type: ignore[arg-type]
            source_name="测试来源",
            content="c",
            score=score,
        )

    def test_unknown_method_on_the_dispatch_path_warns(self, caplog) -> None:
        """Passing ``method=`` directly is the remaining unguarded route.

        It bypasses the property setter (the argument never touches the
        attribute), so the dispatch still has to say something.
        """
        import logging

        RAGRetrievalService._warned_unknown_fusion = False
        svc = RAGRetrievalService()
        with caplog.at_level(logging.WARNING, logger="src.rag.rag_service"):
            svc._merge_and_rank([self._result("d1", "chroma", 0.9)], "q", method="bogus")
        messages = [r.getMessage() for r in caplog.records]
        assert any("bogus" in m and "falling back" in m for m in messages), messages

    def test_the_warning_fires_only_once_per_process(self, caplog) -> None:
        """A per-call warning would emit one identical line per query.

        ``_merge_and_rank`` runs once per retrieval; on a 200-query benchmark
        that is 200 lines for a condition that is constant for the process
        lifetime — the signal disappears into its own volume.
        """
        import logging

        RAGRetrievalService._warned_unknown_fusion = False
        svc = RAGRetrievalService()
        with caplog.at_level(logging.WARNING, logger="src.rag.rag_service"):
            for _ in range(5):
                svc._merge_and_rank(
                    [self._result("d1", "chroma", 0.9)], "q", method="bogus"
                )
        warnings = [r for r in caplog.records if "falling back" in r.getMessage()]
        assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"

    def test_known_method_does_not_warn(self, caplog) -> None:
        """No noise on the normal path."""
        import logging

        RAGRetrievalService._warned_unknown_fusion = False
        svc = RAGRetrievalService()
        with caplog.at_level(logging.WARNING, logger="src.rag.rag_service"):
            svc._merge_and_rank([self._result("d1", "chroma", 0.9)], "q", method="score")
        assert not [r for r in caplog.records if "falling back" in r.getMessage()]
