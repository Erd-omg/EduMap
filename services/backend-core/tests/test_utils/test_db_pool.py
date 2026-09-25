"""Tests for db_pool normalisation.

The trap being guarded: ``MemoryDBPool`` (the wrapper) has no ``acquire()`` —
only ``.pool`` does. Every persistence call site wraps acquisition in
``except Exception`` so a DB failure cannot break the live feature, which means
passing the wrapper made persistence a **silent no-op** while the in-memory
path kept working. Nothing surfaced until someone noticed state vanishing on
restart.

So the contract under test is: accept both shapes, and for anything else fail
loudly *at construction* rather than quietly per-query.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.db_pool import unwrap_pool


class _Wrapper:
    """Stands in for MemoryDBPool: no acquire(), has .pool."""

    def __init__(self, pool):
        self.pool = pool


class TestUnwrapPool:
    def test_none_stays_none(self) -> None:
        assert unwrap_pool(None) is None

    def test_raw_pool_passes_through(self) -> None:
        pool = MagicMock()
        pool.acquire = MagicMock()
        assert unwrap_pool(pool) is pool

    def test_wrapper_is_unwrapped_to_its_pool(self) -> None:
        inner = MagicMock()
        inner.acquire = MagicMock()
        assert unwrap_pool(_Wrapper(inner)) is inner

    def test_wrapper_without_usable_inner_pool_raises(self) -> None:
        bad = MagicMock()
        bad.pool = None
        del bad.acquire  # ensure the attribute-probe path is exercised
        with pytest.raises(TypeError):
            unwrap_pool(bad)

    def test_object_with_neither_shape_raises(self) -> None:
        with pytest.raises(TypeError) as exc:
            unwrap_pool(object())
        # the message must say what to pass, not just "invalid"
        assert "acquire()" in str(exc.value)

    def test_error_names_the_owner(self) -> None:
        """A construction failure must identify which service broke."""
        with pytest.raises(TypeError) as exc:
            unwrap_pool(42, owner="ForgettingCurveService")
        assert "ForgettingCurveService" in str(exc.value)

    def test_deeply_nested_wrapper_is_not_chased(self) -> None:
        """One level of unwrapping only — a wrapper of a wrapper is a bug."""
        inner = MagicMock()
        inner.acquire = MagicMock()
        nested = _Wrapper(_Wrapper(inner))
        # The inner value is itself a wrapper (no acquire), so this must raise
        with pytest.raises(TypeError):
            unwrap_pool(nested)


class TestServicesAcceptBothShapes:
    """The three services that take a db_pool must all normalise it."""

    def test_forgetting_curve_unwraps_wrapper(self) -> None:
        from src.learning_path.forgetting_curve import ForgettingCurveService

        inner = MagicMock()
        inner.acquire = MagicMock()
        svc = ForgettingCurveService(db_pool=_Wrapper(inner))
        assert svc._db_pool is inner

    def test_forgetting_curve_rejects_junk_at_construction(self) -> None:
        from src.learning_path.forgetting_curve import ForgettingCurveService

        with pytest.raises(TypeError):
            ForgettingCurveService(db_pool="postgres://not-a-pool")

    def test_path_service_unwraps_wrapper(self) -> None:
        from src.learning_path.path_service import PathService

        inner = MagicMock()
        inner.acquire = MagicMock()
        svc = PathService(
            kp_repo=MagicMock(), edge_repo=MagicMock(), db_pool=_Wrapper(inner)
        )
        assert svc._db_pool is inner

    def test_anti_gaming_unwraps_wrapper(self) -> None:
        from src.agents.assessment.anti_gaming import AntiGamingService

        inner = MagicMock()
        inner.acquire = MagicMock()
        svc = AntiGamingService(db_pool=_Wrapper(inner))
        assert svc._db_pool is inner

    @pytest.mark.asyncio
    async def test_wrapper_path_actually_persists(self) -> None:
        """End-to-end: a wrapped pool must still reach the database.

        This is the regression the guard exists for — under the old behaviour
        the service constructed fine and then silently never wrote anything.
        """
        from src.learning_path.forgetting_curve import ForgettingCurveService

        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetchrow = AsyncMock(return_value=None)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=None)
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=cm)

        svc = ForgettingCurveService(db_pool=_Wrapper(pool))
        await svc.update_after_quiz("kp1", "u1", 0.9)

        executed = [str(c.args[0]) for c in conn.execute.call_args_list if c.args]
        assert any("forgetting_curve_state" in s for s in executed), (
            "state must be persisted even when a wrapped pool was passed"
        )
        assert any("forgetting_review_log" in s for s in executed)
