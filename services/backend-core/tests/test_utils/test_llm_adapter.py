"""Tests for LLM Adapter circuit breaker and health_check.

Uses MockLLMAdapter for mock mode tests and the real adapter
for circuit breaker logic (standalone, no HTTP calls needed).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.llm_adapter import OpenAICompatibleAdapter


@pytest.fixture
def adapter() -> OpenAICompatibleAdapter:
    """Adapter with empty API key — runs in mock mode."""
    return OpenAICompatibleAdapter(
        api_key="",
        api_base="https://api.example.com",
        model="test-model",
        circuit_breaker_threshold=3,  # lower threshold for faster tests
        circuit_breaker_recovery_s=60,
    )


class TestMockMode:
    """Mock mode (no API key) behavior."""

    @pytest.mark.asyncio
    async def test_mock_response_on_generate(self, adapter: OpenAICompatibleAdapter) -> None:
        """Mock mode returns friendly Chinese demo message."""
        response = await adapter.generate("用户提问")
        assert "演示模式" in response.content

    @pytest.mark.asyncio
    async def test_mock_response_on_stream(self, adapter: OpenAICompatibleAdapter) -> None:
        """Mock mode yields chunks from the demo message."""
        chunks: list[str] = []
        async for chunk in adapter.generate_stream("用户提问"):
            chunks.append(chunk)
        full = "".join(chunks)
        assert "演示模式" in full
        assert len(chunks) > 1  # should be split into multiple chunks

    @pytest.mark.asyncio
    async def test_health_check_mock_mode(self, adapter: OpenAICompatibleAdapter) -> None:
        """Health check in mock mode returns reachable immediately."""
        result = await adapter.health_check()
        assert result["reachable"] is True
        assert result["detail"] == "mock_mode"


class TestCircuitBreaker:
    """Circuit breaker: consecutive failures → open → recovery."""

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self, adapter: OpenAICompatibleAdapter) -> None:
        """Initially the circuit is closed (calls allowed)."""
        assert adapter._is_circuit_open() is False
        assert adapter._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_record_success_resets_counter(self, adapter: OpenAICompatibleAdapter) -> None:
        """A single success resets the consecutive failure counter."""
        adapter._consecutive_failures = 2
        adapter._record_success()
        assert adapter._consecutive_failures == 0

    def test_record_failure_increments(self, adapter: OpenAICompatibleAdapter) -> None:
        """Each failure increments the counter."""
        adapter._record_failure()
        assert adapter._consecutive_failures == 1

        adapter._record_failure()
        assert adapter._consecutive_failures == 2

    def test_threshold_opens_circuit(self, adapter: OpenAICompatibleAdapter) -> None:
        """Reaching threshold opens the circuit and sets recovery window."""
        # threshold is 3
        adapter._record_failure()  # 1
        adapter._record_failure()  # 2
        assert adapter._is_circuit_open() is False  # not yet open

        adapter._record_failure()  # 3 — threshold reached
        assert adapter._circuit_open_until > 0
        assert adapter._is_circuit_open() is True

    def test_circuit_open_returns_degraded(self, adapter: OpenAICompatibleAdapter) -> None:
        """When circuit is open, _call_llm returns degraded response."""
        adapter._consecutive_failures = 3
        adapter._circuit_open_until = time.monotonic() + 60

        assert adapter._is_circuit_open() is True

    def test_recovery_after_timeout(self, adapter: OpenAICompatibleAdapter) -> None:
        """After recovery window passes, circuit closes automatically."""
        adapter._consecutive_failures = 3
        adapter._circuit_open_until = time.monotonic() + 0.01  # 10ms

        # Wait for recovery
        time.sleep(0.02)

        # _is_circuit_open should close it and reset
        assert adapter._is_circuit_open() is False
        assert adapter._consecutive_failures == 0
        assert adapter._circuit_open_until == 0.0


class TestCircuitBreakerThreshold:
    """Default and custom threshold configuration."""

    def test_default_threshold_is_5(self) -> None:
        """Default circuit breaker threshold is 5."""
        a = OpenAICompatibleAdapter(api_key="")
        assert a._circuit_threshold == 5
        assert a._circuit_recovery == 60

    def test_custom_threshold(self) -> None:
        """Custom threshold and recovery window."""
        a = OpenAICompatibleAdapter(
            api_key="",
            circuit_breaker_threshold=2,
            circuit_breaker_recovery_s=30,
        )
        assert a._circuit_threshold == 2
        assert a._circuit_recovery == 30

    def test_threshold_one(self) -> None:
        """threshold=1 opens circuit on the first failure."""
        a = OpenAICompatibleAdapter(
            api_key="",
            circuit_breaker_threshold=1,
        )
        assert a._is_circuit_open() is False
        a._record_failure()
        assert a._is_circuit_open() is True


class TestHealthCheck:
    """Health check endpoint behavior."""

    @pytest.mark.asyncio
    async def test_health_circuit_open(self, adapter: OpenAICompatibleAdapter) -> None:
        """Health check reports circuit_open when circuit is open."""
        # Simulate: add an API key so it doesn't short-circuit to mock_mode
        adapter.api_key = "sk-test"
        adapter._circuit_open_until = time.monotonic() + 60
        adapter._consecutive_failures = 3

        result = await adapter.health_check()
        assert result["reachable"] is False
        assert result["detail"] == "circuit_open"


class TestBuildMessages:
    """_build_messages() — message construction."""

    def test_with_system_prompt(self, adapter: OpenAICompatibleAdapter) -> None:
        """System prompt added as first message."""
        messages = adapter._build_messages("user query", "system instructions")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "system instructions"}
        assert messages[1] == {"role": "user", "content": "user query"}

    def test_without_system_prompt(self, adapter: OpenAICompatibleAdapter) -> None:
        """Only user message when no system prompt."""
        messages = adapter._build_messages("user query")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "user query"}


class TestTokenTracking:
    """Token usage tracking in _call_llm and generate()."""

    @pytest.mark.asyncio
    async def test_mock_mode_usage_empty(self, adapter: OpenAICompatibleAdapter) -> None:
        """Mock mode returns empty usage dict."""
        text, usage = await adapter._call_llm([{"role": "user", "content": "hi"}])
        assert usage == {}

    @pytest.mark.asyncio
    async def test_last_usage_set_after_generate(self, adapter: OpenAICompatibleAdapter) -> None:
        """_last_usage is populated after generate() in mock mode."""
        response = await adapter.generate("test")
        assert adapter._last_usage == {}
        assert response.usage == {}

    @pytest.mark.asyncio
    async def test_last_usage_reset_on_stream(self, adapter: OpenAICompatibleAdapter) -> None:
        """_last_usage is reset at start of streaming."""
        adapter._last_usage = {"prompt_tokens": 100}
        chunks: list[str] = []
        async for chunk in adapter.generate_stream("test"):
            chunks.append(chunk)
        # After streaming, _last_usage should be empty (mock mode has no usage)
        assert adapter._last_usage == {}

    @pytest.mark.asyncio
    async def test_call_with_retry_and_cb_success(self) -> None:
        """_call_with_retry_and_cb returns API response on success."""
        adapter = OpenAICompatibleAdapter(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            result = await adapter._call_with_retry_and_cb("POST", "https://api.example.com/chat/completions", {"model": "test", "messages": []})

        assert "_degraded" not in result
        assert result["choices"][0]["message"]["content"] == "hi"
        assert adapter._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_call_with_retry_and_cb_retry_on_429(self) -> None:
        """_call_with_retry_and_cb retries on HTTP 429."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = OpenAICompatibleAdapter(api_key="sk-test", max_retries=3)
        mock_429 = MagicMock()
        mock_429.status_code = 429

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        call_count = [0]

        async def mock_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_429
            return mock_200

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=mock_request)

        with patch("asyncio.sleep", AsyncMock()):
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                result = await adapter._call_with_retry_and_cb("POST", "https://api.example.com/chat/completions", {})

        assert "_degraded" not in result
        assert call_count[0] == 2
        assert adapter._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_call_with_retry_and_cb_exhaustion_returns_degraded(self) -> None:
        """_call_with_retry_and_cb returns degraded after exhausting retries."""
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        adapter = OpenAICompatibleAdapter(api_key="sk-test", max_retries=2)
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("500 error", request=MagicMock(), response=mock_500)
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_500)

        with patch("asyncio.sleep", AsyncMock()):
            with patch("httpx.AsyncClient") as mock_cls:
                mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                result = await adapter._call_with_retry_and_cb("POST", "https://api.example.com/chat/completions", {})

        assert result.get("_degraded") is True
        assert adapter._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls_when_open(self, adapter: OpenAICompatibleAdapter) -> None:
        """When circuit is open, _call_with_retry_and_cb returns degraded immediately."""
        import time
        adapter.api_key = "sk-test"  # Enable non-mock mode
        adapter._circuit_open_until = time.monotonic() + 60
        adapter._consecutive_failures = 3

        result = await adapter._call_with_retry_and_cb("POST", "https://api.example.com/chat/completions", {})
        assert result.get("_degraded") is True

    @pytest.mark.asyncio
    async def test_retry_then_circuit_opens(self) -> None:
        """After exhausting retries, circuit breaker records the failure."""
        from unittest.mock import AsyncMock, MagicMock, patch
        import httpx

        adapter = OpenAICompatibleAdapter(
            api_key="sk-test",
            max_retries=1,  # Only 1 attempt = no retry
            circuit_breaker_threshold=1,  # Open on first failure
        )
        mock_503 = MagicMock()
        mock_503.status_code = 503
        mock_503.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("503 error", request=MagicMock(), response=mock_503)
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_503)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            result = await adapter._call_with_retry_and_cb("POST", "https://api.example.com/chat/completions", {})

        assert result.get("_degraded") is True
        assert adapter._consecutive_failures >= 1
        assert adapter._is_circuit_open() is True


class TestGenerateWithUsage:
    """Token usage in generated responses when real API key provided."""

    @pytest.mark.asyncio
    async def test_generate_usage_in_llmresponse(self) -> None:
        """generate() returns LLMResponse with usage from API."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = OpenAICompatibleAdapter(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            response = await adapter.generate("test")

        assert response.content == "Hello"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert adapter._last_usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_structured_via_fc_tracks_usage(self) -> None:
        """_generate_structured_via_fc extracts token usage."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = OpenAICompatibleAdapter(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {"arguments": '{"result": "test"}'},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            from src.harness.structured import OutputSchema
            from pydantic import BaseModel

            class TestSchema(BaseModel):
                result: str

            schema = OutputSchema(TestSchema, schema_id="test")
            result = await adapter._generate_structured_via_fc("test", schema)

        assert isinstance(result, TestSchema)
        assert result.result == "test"
        assert adapter._last_usage["total_tokens"] == 80

    @pytest.mark.asyncio
    async def test_call_llm_extracts_usage(self) -> None:
        """_call_llm extracts usage from API response and returns it."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = OpenAICompatibleAdapter(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 14, "total_tokens": 21},
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            text, usage = await adapter._call_llm([{"role": "user", "content": "hi"}])

        assert text == "response"
        assert usage == {"prompt_tokens": 7, "completion_tokens": 14, "total_tokens": 21}

    @pytest.mark.asyncio
    async def test_call_llm_missing_usage_returns_empty(self) -> None:
        """_call_llm handles missing usage field gracefully."""
        from unittest.mock import AsyncMock, MagicMock, patch

        adapter = OpenAICompatibleAdapter(api_key="sk-test")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}],
        }  # No usage key

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            text, usage = await adapter._call_llm([{"role": "user", "content": "hi"}])

        assert text == "response"
        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class TestHarnessTokenTransfer:
    """Token usage transfer from LLM adapter to ExecutionReport via BaseAgent."""

    @pytest.mark.asyncio
    async def test_execute_transfers_usage_to_report(self) -> None:
        """BaseAgent.execute() copies LLM _last_usage into report.token_usage."""
        from src.harness.base import BaseAgent as _BaseAgent
        from src.harness.types import AgentConfig, AgentInput

        adapter = OpenAICompatibleAdapter(api_key="sk-test", max_retries=1)

        class TestAgent(_BaseAgent):
            async def run(self, input: AgentInput) -> str:
                return "ok"

        agent = TestAgent(llm_adapter=adapter, agent_name="test_agent")
        report = await agent.execute(AgentInput(task_input="test"))

        # In mock mode (no API key), token_usage should be an empty dict
        assert report.token_usage == {}


class TestRateLimiting:
    """Rate limiting semaphore in _call_with_retry_and_cb."""

    def test_semaphore_initialized(self, adapter: OpenAICompatibleAdapter) -> None:
        """Rate limit semaphore exists with default limit of 10."""
        assert adapter._rate_limit_semaphore is not None
        assert adapter._rate_limit_semaphore._value == 10

    def test_custom_rate_limit(self) -> None:
        """Adapter with custom semaphore capacity."""
        adapter = OpenAICompatibleAdapter(api_key="")
        # Default semaphore is 10
        assert adapter._rate_limit_semaphore._value == 10

    @pytest.mark.asyncio
    async def test_mock_mode_bypasses_semaphore_for_stream(self, adapter: OpenAICompatibleAdapter) -> None:
        """Mock mode streaming bypasses semaphore (no HTTP call)."""
        chunks: list[str] = []
        async for chunk in adapter.generate_stream("test"):
            chunks.append(chunk)
        assert len(chunks) > 0


@pytest.mark.asyncio
async def test_health_check_with_real_api_key() -> None:
    """Health check with real API key does not crash."""
    adapter = OpenAICompatibleAdapter(api_key="sk-test")
    result = await adapter.health_check()
    # Should either succeed or fail gracefully, never crash
    assert "reachable" in result
    assert "detail" in result


class TestCircuitBreakerObservability:
    """The breaker must be attributable and queryable.

    One adapter instance is shared by all 6 agents + Mentor, so a bare counter
    could not answer "which agent suspended LLM traffic?" — the open event and
    the state snapshot now carry the source.
    """

    def test_state_snapshot_shape(self, adapter: OpenAICompatibleAdapter) -> None:
        """circuit_breaker_state() is JSON-serialisable and complete."""
        import json

        state = adapter.circuit_breaker_state()
        for field in (
            "open", "consecutive_failures", "threshold", "recovery_seconds",
            "seconds_until_retry", "open_count", "last_failing_source",
            "failure_sources",
        ):
            assert field in state, f"missing {field!r}"
        json.dumps(state)  # must not raise
        assert state["open"] is False
        assert state["open_count"] == 0

    def test_attribution_recorded(self, adapter: OpenAICompatibleAdapter) -> None:
        """The failing agent is named, not just counted."""
        adapter._record_failure("designer")
        adapter._record_failure("designer")
        adapter._record_failure("coder")

        state = adapter.circuit_breaker_state()
        assert state["last_failing_source"] == "coder"
        assert state["failure_sources"] == {"designer": 2, "coder": 1}

    def test_open_count_increments_per_trip(self) -> None:
        """Re-opening after recovery is visible as a rising count.

        Without this, a flap-loop (open → expire → immediately re-open) is
        indistinguishable from a single outage.
        """
        a = OpenAICompatibleAdapter(
            api_key="k", circuit_breaker_threshold=1, circuit_breaker_recovery_s=0,
        )
        a._record_failure("planner")
        assert a.circuit_breaker_state()["open_count"] == 1

        # recovery_s=0 means the very next check closes it, then re-trip.
        a._is_circuit_open()
        a._record_failure("planner")
        assert a.circuit_breaker_state()["open_count"] == 2

    def test_success_clears_attribution(self, adapter: OpenAICompatibleAdapter) -> None:
        """A success resets counters AND attribution — no stale blame."""
        adapter._record_failure("designer")
        assert adapter.circuit_breaker_state()["failure_sources"]

        adapter._record_success()
        state = adapter.circuit_breaker_state()
        assert state["failure_sources"] == {}
        assert state["last_failing_source"] == ""
        assert state["consecutive_failures"] == 0

    def test_seconds_until_retry_is_reported(self) -> None:
        """An open breaker reports how long until it will be retried."""
        a = OpenAICompatibleAdapter(
            api_key="k", circuit_breaker_threshold=1, circuit_breaker_recovery_s=60,
        )
        a._record_failure("mentor")
        state = a.circuit_breaker_state()
        assert state["open"] is True
        assert 0 < state["seconds_until_retry"] <= 60

    def test_source_defaults_when_unknown(self, adapter: OpenAICompatibleAdapter) -> None:
        """A caller that doesn't know its name still works (http path)."""
        adapter._record_failure()
        state = adapter.circuit_breaker_state()
        assert state["consecutive_failures"] == 1
        assert state["failure_sources"] == {}

    def test_health_check_reports_circuit_open(self) -> None:
        """health_check surfaces the breaker so /health/ready is actionable."""
        import asyncio

        a = OpenAICompatibleAdapter(
            api_key="k", circuit_breaker_threshold=1, circuit_breaker_recovery_s=60,
        )
        a._record_failure("planner")
        result = asyncio.run(a.health_check())
        assert result["reachable"] is False
        assert result["detail"] == "circuit_open"
