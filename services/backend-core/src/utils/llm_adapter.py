"""LLM Adapter — abstract interface + factory for pluggable LLM backends."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger(__name__)


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
    """LLM adapter for OpenAI-compatible chat completion APIs.

    Supports both ``generate()`` (full response) and ``generate_stream()``
    (SSE-chunked) via httpx.  Retries on transient failures with exponential
    backoff.  Falls back to a mock response when ``api_key`` is empty.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        model: str = "gpt-4o-mini",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    # ── Public API ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LLM and return the full response."""
        messages = self._build_messages(prompt, system_prompt)
        text = await self._call_llm(messages, **kwargs)
        return LLMResponse(content=text, model=self.model)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream token chunks from the LLM."""
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self._call_llm_stream(messages, **kwargs):
            yield chunk

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def _call_llm(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            return json.dumps({"note": "Mock response — no LLM API key configured"}, ensure_ascii=False)

        url = f"{self.api_base}/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        url,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                if resp.status_code in (429, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("LLM transient error %d, retrying in %ds", resp.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("LLM request failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
                    await asyncio.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_exc

    async def _call_llm_stream(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            yield json.dumps({"note": "Mock response — no LLM API key configured"}, ensure_ascii=False)
            return

        url = f"{self.api_base}/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.timeout + 60) as client:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue


def create_llm(config) -> BaseLLMAdapter:
    """Factory method: returns the appropriate LLM adapter based on config."""
    if config.llm_model == "spark":
        return SparkAdapter()
    return OpenAICompatibleAdapter(
        api_key=config.llm_api_key,
        api_base=config.llm_api_base,
        model=config.llm_model,
        timeout=getattr(config, "llm_timeout", 60),
        max_retries=getattr(config, "llm_max_retries", 3),
    )
