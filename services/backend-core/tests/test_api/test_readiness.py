"""Readiness endpoint contract — drives the REAL handler.

``readiness()`` in main.py closes over the module-level ``app`` (main.py:371),
so the ``client`` fixture's test app cannot reach it: its own ``/health/ready``
stub shadows the real one, and delegating would return nulls because the handler
reads production app state.

So this file calls ``main.readiness()`` directly and sets state on the *same*
app object the handler closes over.  That is the real entry point for this
endpoint — not a copy of it.
"""
from __future__ import annotations

import json

import pytest

from src import main as main_module
from src.utils.llm_adapter import OpenAICompatibleAdapter


@pytest.fixture
def restore_app_state():
    """Snapshot/restore the module app's state around each test.

    Skips underscore-prefixed keys: Starlette's ``State`` holds its backing
    dict in ``_state``, which must not be deleted.
    """
    saved = {k: v for k, v in vars(main_module.app.state).items()}
    yield main_module.app
    state = main_module.app.state
    for key in list(vars(state)):
        if not key.startswith("_"):
            delattr(state, key)
    for key, value in saved.items():
        setattr(state, key, value)


@pytest.fixture
def tripped_adapter(restore_app_state):
    """An adapter whose breaker is open, attributed to 'coder'."""
    adapter = OpenAICompatibleAdapter(
        api_key="k", circuit_breaker_threshold=2, circuit_breaker_recovery_s=60,
    )
    adapter._record_failure("designer")
    adapter._record_failure("coder")
    restore_app_state.state.llm_adapter = adapter
    return adapter


class TestReadinessCircuitBreaker:
    """The readiness payload must make a suspended LLM surface visible."""

    async def test_circuit_breaker_block_present(
        self, tripped_adapter, restore_app_state,
    ) -> None:
        body = await main_module.readiness()
        assert "circuit_breaker" in body
        assert body["circuit_breaker"] is not None

    async def test_open_breaker_is_reported_as_open(
        self, tripped_adapter, restore_app_state,
    ) -> None:
        cb = (await main_module.readiness())["circuit_breaker"]
        assert cb["open"] is True
        assert cb["consecutive_failures"] == 2
        assert cb["threshold"] == 2
        assert cb["open_count"] == 1

    async def test_attribution_is_exposed(
        self, tripped_adapter, restore_app_state,
    ) -> None:
        """Which agent tripped it must be answerable without reading logs.

        This is the whole point of the attribution work: one adapter is shared
        by all agents, so a bare counter cannot identify the culprit.
        """
        cb = (await main_module.readiness())["circuit_breaker"]
        assert cb["last_failing_source"] == "coder"
        assert cb["failure_sources"] == {"designer": 1, "coder": 1}

    async def test_payload_is_json_serialisable(
        self, tripped_adapter, restore_app_state,
    ) -> None:
        body = await main_module.readiness()
        json.dumps(body)  # health endpoints must serialise cleanly

    async def test_missing_adapter_yields_null_not_error(
        self, restore_app_state,
    ) -> None:
        """No adapter (demo mode / not yet wired) must not break readiness."""
        restore_app_state.state.llm_adapter = None
        body = await main_module.readiness()
        assert body["circuit_breaker"] is None
        assert body["status"] == "ok"

    async def test_adapter_without_breaker_block_yields_null(
        self, restore_app_state,
    ) -> None:
        """An adapter lacking the breaker API degrades to null, not a crash."""
        class _Bare:
            async def health_check(self):
                return {"reachable": True, "detail": "ok"}

        restore_app_state.state.llm_adapter = _Bare()
        body = await main_module.readiness()
        assert body["circuit_breaker"] is None
