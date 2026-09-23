"""LLM Adapter — abstract interface + factory for pluggable LLM backends."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import httpx
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from src.harness.structured import OutputSchema

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

    async def generate_structured(
        self,
        prompt: str,
        schema: OutputSchema | type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured response matching the provided schema.

        Primary mode: function-calling (tool_use) API.
        Fallback mode: prompt-injected JSON schema + Pydantic parsing.

        Subclasses that support function-calling should override.
        The default implementation uses prompt-injection mode.
        """
        from src.harness.structured import OutputSchema as _OutputSchema

        schema_obj = schema if isinstance(schema, _OutputSchema) else _OutputSchema(schema)
        instructions = schema_obj.to_prompt_instructions()
        combined = f"{system_prompt or ''}\n\n{instructions}".strip()
        response = await self.generate(prompt, system_prompt=combined, **kwargs)
        return schema_obj.parse(response.content)


class SparkAdapter(BaseLLMAdapter):
    """Placeholder for iFlyTek Spark implementation.

    Falls back to OpenAI-compatible adapter until Spark support is added.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        model: str = "spark",
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        logger.info(
            "SparkAdapter: using OpenAI-compatible fallback. "
            "Set LLM_API_KEY and LLM_API_BASE to an OpenAI-compatible endpoint."
        )
        self._fallback = OpenAICompatibleAdapter(
            api_key=api_key,
            api_base=api_base,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        return await self._fallback.generate(prompt, system_prompt, **kwargs)

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> AsyncIterator[str]:
        async for chunk in self._fallback.generate_stream(prompt, system_prompt, **kwargs):
            yield chunk

    async def generate_structured(
        self,
        prompt: str,
        schema: OutputSchema | type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        return await self._fallback.generate_structured(prompt, schema, system_prompt, **kwargs)


class OpenAICompatibleAdapter(BaseLLMAdapter):
    """LLM adapter for OpenAI-compatible chat completion APIs.

    Supports both ``generate()`` (full response) and ``generate_stream()``
    (SSE-chunked) via httpx.  Retries on transient failures with exponential
    backoff.  Falls back to a mock response when ``api_key`` is empty.

    Includes a simple circuit breaker: after ``circuit_breaker_threshold``
    consecutive failures the adapter stops calling the API for
    ``circuit_breaker_recovery_s`` seconds, returning a degraded response
    instead.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        model: str = "gpt-4o-mini",
        timeout: int = 60,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_s: int = 60,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: float = 0.0
        self._circuit_threshold = circuit_breaker_threshold
        self._circuit_recovery = circuit_breaker_recovery_s
        # Observability for the breaker.  One adapter instance is shared by
        # every agent + Mentor, so a bare counter cannot answer "which agent
        # opened it?" — these attributes make the breaker attributable and
        # queryable instead of a silent log line.
        #
        # ``_circuit_source`` records who fed the failure that tripped it;
        # ``_circuit_open_count`` counts how many times it has tripped, so a
        # flap-loop (open → expire → immediately re-open) is visible as a
        # rising count rather than indistinguishable from a single outage.
        self._circuit_source: str = ""
        self._circuit_open_count: int = 0
        self._failure_sources: dict[str, int] = {}
        # Half-open probe gate.  After the recovery window expires the breaker
        # enters HALF_OPEN: exactly ONE caller is allowed through to test the
        # upstream, and everyone else keeps getting the degraded response.
        #
        # Without this the breaker fully closed on expiry and needed
        # ``threshold`` fresh failures to re-open, so a sustained outage let
        # `threshold` real calls through every recovery window.  Measured with
        # threshold=3/recovery=2s: 9 dead-upstream calls in 10s instead of 3.
        #
        # A bare flag is sufficient because this object is only ever touched
        # from one event loop (every caller is async), so check-and-set below
        # is atomic with respect to the loop.  Guarding with a threading.Lock
        # would be wrong here, not just unnecessary: the flag is held across an
        # ``await`` (the probe request), and a lock cannot be held across awaits.
        self._half_open_probe_in_flight: bool = False
        # Token usage tracking (extracted from API responses)
        self._last_usage: dict[str, int] = {}
        # Rate limiter — max 10 concurrent LLM calls per adapter instance
        self._rate_limit_semaphore = asyncio.Semaphore(10)

    def _circuit_state(self) -> str:
        """Return the breaker state: ``closed`` | ``open`` | ``half_open``.

        Pure read — never mutates.  ``half_open`` means the recovery window has
        elapsed but the upstream is still unverified; a caller may pass only if
        it wins the probe (see :meth:`_try_acquire_slot`).
        """
        import time
        if self._circuit_open_until == 0.0:
            return "closed"
        if time.monotonic() > self._circuit_open_until:
            return "half_open"
        return "open"

    def _is_circuit_open(self) -> bool:
        """Whether calls are currently suspended.  Pure read, no side effects.

        Deliberately does NOT consume the half-open probe slot: this is called
        from read-only paths (``circuit_breaker_state``, ``health_check``) which
        must not steal the single probe from a real request.
        """
        return self._circuit_state() == "open"

    def _try_acquire_slot(self) -> bool:
        """Whether this caller may make an LLM call.  Consumes the probe slot.

        - ``closed``    → always allowed.
        - ``open``      → denied.
        - ``half_open`` → allowed only for the FIRST caller; that caller
          becomes the probe.  Everyone else is denied until the probe resolves
          (via :meth:`_record_success` or :meth:`_record_failure`).

        The check-and-set is not interrupted by an await, so with a single
        event loop exactly one caller can win.
        """
        state = self._circuit_state()
        if state == "closed":
            return True
        if state == "open":
            return False
        # half_open
        if self._half_open_probe_in_flight:
            return False
        self._half_open_probe_in_flight = True
        logger.info(
            "LLM circuit breaker HALF_OPEN — releasing one probe request "
            "(open #%d, was open for %ds)",
            self._circuit_open_count,
            self._circuit_recovery,
        )
        return True

    def _record_failure(self, source: str = "") -> None:
        """Record a failure; open the circuit if the threshold is reached.

        ``source`` names the caller (agent name) so the open event and the
        ``circuit_breaker`` health report can attribute the failure.  Optional
        so existing callers that only know "the HTTP call failed" still work.

        In HALF_OPEN a single failure re-opens immediately — the probe proved
        the upstream is still down, so there is nothing to learn from letting
        more calls through.
        """
        import time
        was_half_open = self._circuit_state() == "half_open"
        self._half_open_probe_in_flight = False
        self._consecutive_failures += 1
        if source:
            self._failure_sources[source] = self._failure_sources.get(source, 0) + 1
            self._circuit_source = source

        if was_half_open or self._consecutive_failures >= self._circuit_threshold:
            # Re-arm the full recovery window from now.
            self._circuit_open_until = time.monotonic() + self._circuit_recovery
            self._circuit_open_count += 1
            # Log the attribution prominently: with a shared adapter this is
            # the only place that says WHICH agent suspended LLM traffic.
            logger.warning(
                "LLM circuit breaker OPEN (#%d) — %s; %d consecutive failures, "
                "suspending ALL LLM calls for %ds; failing source=%s, "
                "failure_sources=%s",
                self._circuit_open_count,
                "probe failed while HALF_OPEN" if was_half_open else "threshold reached",
                self._consecutive_failures,
                self._circuit_recovery,
                self._circuit_source or "unknown",
                dict(self._failure_sources),
            )

    def _release_probe_slot(self) -> None:
        """Release the half-open probe slot WITHOUT recording an outcome.

        Used on an exceptional exit (or cancellation) where the request never
        produced a verdict.  The failure/success counters are left untouched —
        we learned nothing about the upstream — but the slot is freed so a
        later caller can probe again.  Without this, an unexpected exception
        would strand the breaker in HALF_OPEN permanently.
        """
        self._half_open_probe_in_flight = False

    def _record_success(self) -> None:
        """Reset the breaker on a successful call.

        A success while HALF_OPEN is the probe succeeding, so the circuit
        closes fully; a success while CLOSED just clears the failure streak.
        """
        self._half_open_probe_in_flight = False
        if self._circuit_open_until != 0.0:
            logger.info(
                "LLM circuit breaker CLOSED — probe succeeded after %d open(s)",
                self._circuit_open_count,
            )
        self._circuit_open_until = 0.0
        self._consecutive_failures = 0
        if self._failure_sources:
            self._failure_sources.clear()
        self._circuit_source = ""

    def circuit_breaker_state(self) -> dict:
        """Snapshot the breaker for health endpoints and debugging.

        Returns a plain dict (JSON-serialisable) rather than logging, so an
        operator can answer "is the LLM degraded right now, and who caused it"
        without grepping logs.  Pure read — safe to call from health checks.
        """
        import time
        state = self._circuit_state()
        remaining = 0.0
        if self._circuit_open_until:
            remaining = max(0.0, self._circuit_open_until - time.monotonic())
        return {
            "state": state,                      # closed | open | half_open
            "open": state == "open",
            "consecutive_failures": self._consecutive_failures,
            "threshold": self._circuit_threshold,
            "recovery_seconds": self._circuit_recovery,
            "seconds_until_retry": round(remaining, 2),
            # A count > 1 means the breaker has re-tripped after recovering,
            # which points at an ongoing upstream outage rather than a blip.
            "open_count": self._circuit_open_count,
            "probe_in_flight": self._half_open_probe_in_flight,
            "last_failing_source": self._circuit_source,
            "failure_sources": dict(self._failure_sources),
        }

    async def health_check(self) -> dict:
        """Check if the LLM endpoint is reachable and authenticating correctly.

        Returns:
            Dict with keys ``reachable`` (bool) and ``detail`` (str).
        """
        if not self.api_key:
            return {"reachable": True, "detail": "mock_mode"}
        if self._is_circuit_open():
            return {"reachable": False, "detail": "circuit_open"}
        url = f"{self.api_base}/models" if self.api_base else "https://api.openai.com/v1/models"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if resp.status_code == 200:
                return {"reachable": True, "detail": "ok"}
            return {"reachable": False, "detail": f"http_{resp.status_code}"}
        except httpx.RequestError as exc:
            return {"reachable": False, "detail": str(exc)}

    # ── Public API ──────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call LLM and return the full response with usage info."""
        messages = self._build_messages(prompt, system_prompt)
        text, usage = await self._call_llm(messages, **kwargs)
        self._last_usage = usage
        return LLMResponse(content=text, model=self.model, usage=usage)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream token chunks from the LLM, tracking last usage."""
        self._last_usage = {}
        messages = self._build_messages(prompt, system_prompt)
        async for chunk in self._call_llm_stream(messages, **kwargs):
            yield chunk

    async def generate_structured(
        self,
        prompt: str,
        schema: OutputSchema | type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate structured output with dual-mode strategy.

        1. Function-calling mode (preferred) — uses ``tools`` API param.
        2. Prompt-injection mode (fallback) — injects schema into prompt.
        """
        from src.harness.structured import OutputSchema as _OutputSchema

        schema_obj = schema if isinstance(schema, _OutputSchema) else _OutputSchema(schema)

        # Strategy 1: Function-calling mode
        try:
            return await self._generate_structured_via_fc(
                prompt, schema_obj, system_prompt, **kwargs,
            )
        except (NotImplementedError, Exception) as fc_exc:
            logger.debug(
                "Function-calling mode failed for %s, falling back: %s",
                schema_obj.schema_id, fc_exc,
            )

        # Strategy 2: Prompt-injection mode
        instructions = schema_obj.to_prompt_instructions()
        combined = f"{system_prompt or ''}\n\n{instructions}".strip()
        response = await self.generate(prompt, system_prompt=combined, **kwargs)
        return schema_obj.parse(response.content)

    async def _generate_structured_via_fc(
        self,
        prompt: str,
        schema: OutputSchema,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Use OpenAI function-calling API for structured output."""
        if not self.api_key:
            raise NotImplementedError("No API key configured for function calling")

        from src.harness.structured import OutputSchema as _OutputSchema

        if isinstance(schema, _OutputSchema):
            schema_obj = schema
        else:
            schema_obj = _OutputSchema(schema)

        messages = self._build_messages(prompt, system_prompt)
        tool_def = schema_obj.to_function_tool_def()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [tool_def],
            "tool_choice": {
                "type": "function",
                "function": {"name": schema_obj.schema_id},
            },
            "temperature": kwargs.get("temperature", 0.3),
        }

        url = f"{self.api_base}/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        resp.raise_for_status()
        data = resp.json()

        try:
            tool_calls = data["choices"][0]["message"]["tool_calls"]
            args_str = tool_calls[0]["function"]["arguments"]
            parsed = json.loads(args_str)
            # Track token usage from structured output calls
            usage_data = data.get("usage", {}) or {}
            self._last_usage = {
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }
            return schema_obj.model.model_validate(parsed)
        except (KeyError, IndexError, json.JSONDecodeError, ValidationError) as exc:
            from src.harness.errors import StructuredOutputError
            raise StructuredOutputError(
                f"Failed to parse function-calling output: {exc}",
                raw_output=json.dumps(data, ensure_ascii=False),
            )

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def _call_with_retry_and_cb(
        self,
        method: str,
        url: str,
        payload: dict[str, Any],
    ) -> dict:
        """Unified LLM call with: circuit breaker → rate limit → retry loop → failure tracking.

        Returns the parsed JSON response dict.  On repeated failure, opens the
        circuit and either returns a degraded response dict (containing a
        ``_degraded`` key) or raises ``httpx.HTTPStatusError`` as a last resort.
        """
        if not self._try_acquire_slot():
            logger.warning(
                "LLM circuit breaker %s — returning degraded response",
                self._circuit_state(),
            )
            return {"_degraded": True}

        try:
            return await self._call_with_slot(method, url, payload)
        except BaseException:
            # The probe slot must be released on ANY exit, including exception
            # types the retry loop does not catch (it only catches httpx
            # errors) and cancellation.  Leaking the slot would strand the
            # breaker in HALF_OPEN forever — no further probe could ever be
            # granted, so the service would never recover even after the
            # upstream came back.
            self._release_probe_slot()
            raise

    async def _call_with_slot(
        self,
        method: str,
        url: str,
        payload: dict[str, Any],
    ) -> dict:
        """The retry loop itself — split out so the caller can guarantee slot release."""
        async with self._rate_limit_semaphore:
            last_exc: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.request(
                            method,
                            url,
                            json=payload,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                        )
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning("LLM transient error %d, retrying in %ds", resp.status_code, wait)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    self._record_success()
                    return resp.json()
                except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as exc:
                    last_exc = exc
                    if attempt < self.max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning("LLM request failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, exc)
                        await asyncio.sleep(wait)
                        continue

            self._record_failure("http")
            logger.error("LLM call failed after %d retries: %s", self.max_retries, last_exc)
            return {"_degraded": True}

    async def _call_llm(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> tuple[str, dict[str, int]]:
        if not self.api_key:
            return self._mock_response(), {}

        url = f"{self.api_base}/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        data = await self._call_with_retry_and_cb("POST", url, payload)
        if data.get("_degraded"):
            return (
                "⚠️ AI 服务暂时不可用，请稍后再试。\n\n"
                "您可以继续浏览已生成的学习资源、查看知识图谱和学习路径。",
                {},
            )

        # Extract token usage from API response
        usage_data = data.get("usage", {}) or {}
        usage: dict[str, int] = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }
        return data["choices"][0]["message"]["content"], usage

    async def _call_llm_stream(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if not self.api_key:
            msg = self._mock_response()
            CHUNK_SIZE = 12
            for i in range(0, len(msg), CHUNK_SIZE):
                yield msg[i:i + CHUNK_SIZE]
            return

        # For streaming, check circuit breaker + rate limit without calling
        if not self._try_acquire_slot():
            logger.warning(
                "LLM circuit breaker %s — returning degraded stream response",
                self._circuit_state(),
            )
            degraded = "⚠️ AI 服务暂时不可用，请稍后再试。\n\n您可以继续浏览已生成的学习资源、查看知识图谱和学习路径。"
            for i in range(0, len(degraded), 12):
                yield degraded[i:i + 12]
            return

        url = f"{self.api_base}/chat/completions" if self.api_base else "https://api.openai.com/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": True,
        }

        # Streaming holds the probe slot for the lifetime of the generator.  A
        # consumer that abandons the stream (breaks out, disconnects) triggers
        # GeneratorExit, and an upstream error can raise — both must release
        # the slot or the breaker is stranded in HALF_OPEN.
        slot_released = False

        def _release_once() -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                self._release_probe_slot()

        try:
            async with self._rate_limit_semaphore:
                async with httpx.AsyncClient(timeout=self.timeout + 60) as client:
                    async with client.stream(
                        "POST", url,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    ) as resp:
                        try:
                            resp.raise_for_status()
                        except httpx.HTTPStatusError:
                            self._record_failure("http_stream")
                            logger.error("LLM stream returned HTTP %d", resp.status_code)
                            return

                        last_data: dict | None = None
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    last_data = chunk
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content")
                                    if content is not None:
                                        yield content
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue

                if last_data:
                    usage_data = last_data.get("usage", {}) or last_data.get("x-usage", {}) or {}
                    self._last_usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                    }
                slot_released = True   # _record_success frees the slot
                self._record_success()
        finally:
            # Covers the paths that never reached _record_success/_record_failure:
            # GeneratorExit (consumer abandoned the stream), cancellation, and
            # any transport error that escapes the inner handlers.
            _release_once()

    @staticmethod
    def _mock_response() -> str:
        """Return a friendly message for mock (no-API-key) mode."""
        return (
            "💡 系统当前以演示模式运行（未配置 LLM_API_KEY）。\n\n"
            "您可以正常使用以下功能：\n"
            "• 上传学习资料到资源库（系统会解析并索引）\n"
            "• 查看知识图谱和学习路径\n"
            "• 浏览已生成的资源\n\n"
            "如需完整的 AI 对话和资源生成功能，请在 .env 文件中设置 LLM_API_KEY。"
        )


def create_llm(config) -> BaseLLMAdapter:
    """Factory method: returns the appropriate LLM adapter based on config."""
    if config.llm_model == "spark":
        return SparkAdapter(
            api_key=getattr(config, "llm_api_key", ""),
            api_base=getattr(config, "llm_api_base", ""),
            model="spark",
            timeout=getattr(config, "llm_timeout", 60),
            max_retries=getattr(config, "llm_max_retries", 3),
        )
    return OpenAICompatibleAdapter(
        api_key=config.llm_api_key,
        api_base=config.llm_api_base,
        model=config.llm_model,
        timeout=getattr(config, "llm_timeout", 60),
        max_retries=getattr(config, "llm_max_retries", 3),
    )
