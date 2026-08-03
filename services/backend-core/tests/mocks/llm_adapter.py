"""Mock LLM adapter for testing — returns controlled responses."""
from __future__ import annotations

from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel


class MockResponse:
    """Simple mock response object."""

    def __init__(self, content: str = "模拟回答", model: str = "mock") -> None:
        self.content = content
        self.model = model
        self.usage: dict[str, int] | None = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


class MockLLMAdapter:
    """Mock LLM adapter that returns configurable responses.

    Tracks the last prompt received for test assertions.
    """

    def __init__(self, response: str = "这是一个模拟回答", stream_chunk_size: int = 4):
        self._response = response
        self._stream_chunk_size = stream_chunk_size
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self._last_usage: dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> MockResponse:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return MockResponse(content=self._response)

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        for i in range(0, len(self._response), self._stream_chunk_size):
            yield self._response[i:i + self._stream_chunk_size]

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        from src.harness.structured import OutputSchema as _OutputSchema
        schema_obj = schema if isinstance(schema, _OutputSchema) else _OutputSchema(schema)
        return schema_obj.parse(self._response)
