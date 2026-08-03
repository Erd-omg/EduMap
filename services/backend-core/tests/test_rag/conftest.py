"""Test fixtures for RAG + Mentor Agent tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _load_prompt_registry() -> None:
    """Load PromptRegistry once per test session.

    The PromptRegistry is a class-level singleton.  In production it is
    loaded during ``main.py``'s ``lifespan``; here we load it explicitly
    so that any code path that calls ``PromptRegistry.get(...)`` works.
    """
    from src.prompts import PromptRegistry
    PromptRegistry.load()
