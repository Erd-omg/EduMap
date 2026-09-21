"""Tests for RetryHandler — exponential backoff with configurable retry.

Pure logic tests — httpx exceptions are used as standard Python objects
(no actual HTTP requests are made).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.harness.retry import RetryHandler


class TestRetrySuccess:
    """Tests where the function eventually/natively succeeds."""

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(self) -> None:
        """No retry needed on first success."""
        func = AsyncMock(return_value="success")
        result = await RetryHandler.with_retry(func, max_retries=3)
        assert result == "success"
        assert func.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self) -> None:
        """Fails once, succeeds on retry."""
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("first attempt timed out")
            return "recovered"

        result = await RetryHandler.with_retry(flaky_func, max_retries=3)
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_retries_then_succeed(self) -> None:
        """Fails 2 times, succeeds on 3rd."""
        call_count = 0

        async def flaky_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.TimeoutException("request timed out")
            return "ok"

        result = await RetryHandler.with_retry(flaky_func, max_retries=5)
        assert result == "ok"
        assert call_count == 3


class TestRetryExhaustion:
    """Tests where all retries are consumed."""

    @pytest.mark.asyncio
    async def test_raises_after_exhausted_retries(self) -> None:
        """All retries consumed, last exception is raised."""
        func = AsyncMock(side_effect=TimeoutError("always fails"))
        with pytest.raises(TimeoutError, match="always fails"):
            await RetryHandler.with_retry(func, max_retries=3)
        assert func.await_count == 3

    @pytest.mark.asyncio
    async def test_single_attempt_no_retry(self) -> None:
        """max_retries=1 means no retry at all."""
        func = AsyncMock(side_effect=TimeoutError("fail"))
        with pytest.raises(TimeoutError):
            await RetryHandler.with_retry(func, max_retries=1)
        assert func.await_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_caught(self) -> None:
        """Exceptions not in the retryable list propagate immediately."""
        func = AsyncMock(side_effect=ValueError("not retryable"))
        with pytest.raises(ValueError):
            await RetryHandler.with_retry(func, max_retries=3)
        assert func.await_count == 1

    @pytest.mark.asyncio
    async def test_custom_retryable_list(self) -> None:
        """Only specified exception types trigger retry."""
        call_count = 0

        async def func() -> str:
            nonlocal call_count
            call_count += 1
            raise KeyError("custom error")

        # KeyError is NOT in default retryable list
        with pytest.raises(KeyError):
            await RetryHandler.with_retry(func, max_retries=3)
        assert call_count == 1

        # KeyError IS in custom retryable list
        call_count = 0
        with pytest.raises(KeyError):
            await RetryHandler.with_retry(
                func, max_retries=3,
                retryable=(KeyError,),
            )
        assert call_count == 3


@patch("src.harness.retry.asyncio.sleep", return_value=None)
class TestBackoffDelay:
    """Exponential backoff delay calculation."""

    @pytest.mark.asyncio
    async def test_exponential_delay(self, mock_sleep: AsyncMock) -> None:
        """Delay doubles each attempt: 1s, 2s, 4s."""
        func = AsyncMock(side_effect=TimeoutError("fail"))

        with pytest.raises(TimeoutError):
            await RetryHandler.with_retry(func, max_retries=4, base_delay=1.0)

        assert mock_sleep.await_count == 3
        expected_delays = [1.0, 2.0, 4.0]
        for call, expected in zip(mock_sleep.await_args_list, expected_delays):
            (actual,) = call[0]
            assert actual == pytest.approx(expected, rel=0.1)

    @pytest.mark.asyncio
    async def test_custom_base_delay(self, mock_sleep: AsyncMock) -> None:
        """Base delay of 0.5 produces delays 0.5, 1.0, 2.0."""
        func = AsyncMock(side_effect=TimeoutError("fail"))

        with pytest.raises(TimeoutError):
            await RetryHandler.with_retry(func, max_retries=4, base_delay=0.5)

        expected_delays = [0.5, 1.0, 2.0]
        for call, expected in zip(mock_sleep.await_args_list, expected_delays):
            (actual,) = call[0]
            assert actual == pytest.approx(expected, rel=0.1)


class TestOnRetryCallback:
    """Custom on_retry callback is invoked on each retry."""

    @pytest.mark.asyncio
    async def test_on_retry_callback_invoked(self) -> None:
        """Callback receives attempt index and exception."""
        call_args: list[tuple[int, Exception]] = []

        def callback(attempt: int, exc: Exception) -> None:
            call_args.append((attempt, exc))

        func = AsyncMock(side_effect=TimeoutError("fail"))

        with pytest.raises(TimeoutError):
            await RetryHandler.with_retry(
                func, max_retries=3, on_retry=callback,
            )

        assert len(call_args) == 2
        assert call_args[0][0] == 0  # first retry (attempt 0)
        assert call_args[1][0] == 1  # second retry (attempt 1)
        assert isinstance(call_args[0][1], TimeoutError)


class TestEdgeCases:
    """Boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_args_passed(self) -> None:
        """Function called with no positional args."""
        func = AsyncMock(return_value=42)
        result = await RetryHandler.with_retry(func)
        assert result == 42

    @pytest.mark.asyncio
    async def test_kwargs_passed_through(self) -> None:
        """Keyword arguments are forwarded to the function."""
        func = AsyncMock(return_value="ok")
        result = await RetryHandler.with_retry(
            func, x=1, y="test",
        )
        func.assert_called_once_with(x=1, y="test")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_positional_args_passed_through(self) -> None:
        """Positional arguments are forwarded to the function."""
        func = AsyncMock(return_value="ok")
        result = await RetryHandler.with_retry(func, "arg1", "arg2")
        func.assert_called_once_with("arg1", "arg2")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_max_retries_one_is_no_retry(self) -> None:
        """max_retries=1 means one attempt, no retry."""
        func = AsyncMock(side_effect=TimeoutError("fail"))
        with pytest.raises(TimeoutError):
            await RetryHandler.with_retry(func, max_retries=1)
        assert func.await_count == 1


class TestWithCircuitBreaker:
    """Coverage for ``RetryHandler.with_circuit_breaker``.

    Previously untested — the harness's only circuit-breaker entry point.  The
    tests below pin the CURRENT behaviour so a change to it is deliberate.
    """

    async def test_open_circuit_raises_before_calling_func(self) -> None:
        """An already-open circuit short-circuits without invoking func."""
        from src.harness.errors import CircuitBreakerOpenError

        func = AsyncMock(return_value="ok")
        with pytest.raises(CircuitBreakerOpenError):
            await RetryHandler.with_circuit_breaker(
                func, circuit_check=lambda: True,
            )
        assert func.await_count == 0

    async def test_closed_circuit_calls_func(self) -> None:
        func = AsyncMock(return_value="ok")
        result = await RetryHandler.with_circuit_breaker(
            func, circuit_check=lambda: False,
        )
        assert result == "ok"

    async def test_retryable_failure_records_once(self) -> None:
        """A retryable error that exhausts retries records exactly one failure.

        Recording once per *call* (not per attempt) is what makes the breaker
        threshold mean "N failing calls" rather than "N failing attempts".
        """
        recorded = []
        func = AsyncMock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(httpx.ConnectError):
            await RetryHandler.with_circuit_breaker(
                func,
                max_retries=3,
                base_delay=0,
                circuit_record_failure=lambda: recorded.append(1),
            )
        assert func.await_count == 3
        assert len(recorded) == 1

    async def test_non_retryable_failure_also_records(self) -> None:
        """DOCUMENTS AN ASYMMETRY: non-retryable errors still count.

        ``with_circuit_breaker`` catches bare ``Exception`` for the failure
        record, while the retry loop only catches ``DEFAULT_RETRYABLE``.  So a
        logic bug (KeyError, ValidationError, malformed LLM output) that is
        never retried still advances the breaker toward opening — and because
        one adapter instance is shared by all agents, a single buggy agent can
        suspend LLM traffic fleet-wide for the recovery window.

        Pinned deliberately: if this becomes "does not record", that is an
        intentional behaviour change, not a silent regression.
        """
        recorded = []
        func = AsyncMock(side_effect=KeyError("bad_field"))
        with pytest.raises(KeyError):
            await RetryHandler.with_circuit_breaker(
                func,
                max_retries=3,
                base_delay=0,
                circuit_record_failure=lambda: recorded.append(1),
            )
        assert func.await_count == 1          # never retried
        assert len(recorded) == 1             # ...but still counted

    async def test_no_check_or_record_callbacks_is_safe(self) -> None:
        """Both callbacks are optional."""
        func = AsyncMock(return_value="ok")
        assert await RetryHandler.with_circuit_breaker(func) == "ok"

        failing = AsyncMock(side_effect=httpx.ConnectError("x"))
        with pytest.raises(httpx.ConnectError):
            await RetryHandler.with_circuit_breaker(
                failing, max_retries=1, base_delay=0,
            )
