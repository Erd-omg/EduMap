"""Standardized retry with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RetryHandler:
    """Retry a coroutine with exponential backoff on configurable exceptions."""

    DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
        TimeoutError,
        httpx.TimeoutException,
        httpx.RequestError,
        ConnectionError,
    )

    @staticmethod
    async def with_retry(
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        retryable: tuple[type[Exception], ...] | None = None,
        on_retry: Callable[[int, Exception], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute ``func(*args, **kwargs)`` with retry.

        Returns the first successful result. Re-raises the last exception
        when all retries are exhausted.
        """
        retryable = retryable or RetryHandler.DEFAULT_RETRYABLE
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except retryable as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    if on_retry:
                        on_retry(attempt, exc)
                    else:
                        logger.warning(
                            "Retry %d/%d in %.1fs: %s",
                            attempt + 1, max_retries, delay, exc,
                        )
                    await asyncio.sleep(delay)
                continue

        raise last_exc  # type: ignore[misc]

    @staticmethod
    async def with_circuit_breaker(
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        circuit_check: Callable[[], bool] | None = None,
        circuit_record_failure: Callable[[], None] | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        retryable: tuple[type[Exception], ...] | None = None,
        on_retry: Callable[[int, Exception], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute ``func(*args, **kwargs)`` with retry + circuit breaker awareness.

        - Before retrying, calls ``circuit_check()`` — if it returns True,
          raises ``CircuitBreakerOpenError`` immediately.
        - When all retries are exhausted, calls ``circuit_record_failure()``
          before re-raising.
        """
        if circuit_check and circuit_check():
            from src.harness.errors import CircuitBreakerOpenError
            raise CircuitBreakerOpenError("Circuit breaker is open — calls suspended")

        try:
            return await RetryHandler.with_retry(
                func, *args,
                max_retries=max_retries,
                base_delay=base_delay,
                retryable=retryable,
                on_retry=on_retry,
                **kwargs,
            )
        except Exception:
            if circuit_record_failure:
                circuit_record_failure()
            raise
