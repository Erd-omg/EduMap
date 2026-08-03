"""Global test fixtures for EduMap backend-core tests."""

from __future__ import annotations

from typing import Any

import pytest

from tests.mocks.llm_adapter import MockLLMAdapter


@pytest.fixture
def mock_llm() -> MockLLMAdapter:
    """Return a MockLLMAdapter that returns a fixed Chinese response."""
    return MockLLMAdapter(response="这是一个模拟回答，用于测试。来源[1]显示数组插入复杂度为O(n)。")


@pytest.fixture
def mock_llm_empty() -> MockLLMAdapter:
    """Return a MockLLMAdapter with empty response."""
    return MockLLMAdapter(response="")

