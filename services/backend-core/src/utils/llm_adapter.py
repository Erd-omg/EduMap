"""LLM Adapter — abstract interface + factory for pluggable LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[dict] = None


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> AsyncIterator[str]:
        pass
        yield  # pragma: no cover


class SparkAdapter(BaseLLMAdapter):
    """Placeholder for iFlyTek Spark implementation."""

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        raise NotImplementedError("Spark adapter not implemented yet")

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> AsyncIterator[str]:
        raise NotImplementedError("Spark adapter not implemented yet")


class OpenAICompatibleAdapter(BaseLLMAdapter):
    """Placeholder for OpenAI-compatible API."""

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        raise NotImplementedError("OpenAI adapter not implemented yet")

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> AsyncIterator[str]:
        raise NotImplementedError("OpenAI adapter not implemented yet")


def create_llm(config) -> BaseLLMAdapter:
    """Factory method: returns the appropriate LLM adapter based on config."""
    if config.llm_model == "spark":
        return SparkAdapter()
    return OpenAICompatibleAdapter()
