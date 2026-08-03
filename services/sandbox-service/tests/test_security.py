"""Tests for EduMap sandbox-service security and rate limiting.

Tests the security hardening, rate limiter, and configuration functions
of the sandbox service without requiring a Docker daemon.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear shared rate limiter state before each test."""
    from src.config import _IP_REQUEST_COUNTS

    _IP_REQUEST_COUNTS.clear()
    yield
    _IP_REQUEST_COUNTS.clear()


@pytest.fixture(autouse=True)
def _preserve_settings():
    """Save and restore settings modified during tests."""
    from src.config import settings

    saved = settings.rate_limit_per_minute
    yield
    settings.rate_limit_per_minute = saved


# ── Security kwargs ──────────────────────────────────────────────────────


class TestSecurityKwargs:
    """Container security hardening configuration."""

    def test_build_security_kwargs_returns_all_keys(self) -> None:
        """_build_security_kwargs returns expected security parameters."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert "cap_drop" in kwargs
        assert "security_opt" in kwargs
        assert "cpu_period" in kwargs
        assert "cpu_quota" in kwargs
        assert "pids_limit" in kwargs
        assert "user" in kwargs

    def test_cap_drop_all(self) -> None:
        """All capabilities are dropped."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert kwargs["cap_drop"] == ["ALL"]

    def test_security_opt_no_new_privileges(self) -> None:
        """Container runs with no-new-privileges."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert "no-new-privileges:true" in kwargs["security_opt"]

    def test_cpu_quota_half_core(self) -> None:
        """CPU quota limits to 0.5 core (50000/100000)."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert kwargs["cpu_period"] == 100000
        assert kwargs["cpu_quota"] == 50000

    def test_pids_limit_prevents_fork_bombs(self) -> None:
        """pids_limit prevents fork bomb attacks."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert kwargs["pids_limit"] == 50

    def test_runs_as_nobody(self) -> None:
        """Container runs as nobody user (non-root)."""
        from src.main import _build_security_kwargs

        kwargs = _build_security_kwargs()
        assert kwargs["user"] == "nobody"


# ── Language mapping ─────────────────────────────────────────────────────


class TestLanguageMapping:
    """Language -> image and command mappings."""

    def test_python_image(self) -> None:
        """Python maps to python:3.12-alpine."""
        from src.main import _IMAGE_MAP
        assert _IMAGE_MAP["python"] == "python:3.12-alpine"
        assert _IMAGE_MAP["python3"] == "python:3.12-alpine"
        assert _IMAGE_MAP["py"] == "python:3.12-alpine"

    def test_javascript_image(self) -> None:
        """JavaScript maps to node:20-alpine."""
        from src.main import _IMAGE_MAP
        assert _IMAGE_MAP["javascript"] == "node:20-alpine"
        assert _IMAGE_MAP["js"] == "node:20-alpine"
        assert _IMAGE_MAP["node"] == "node:20-alpine"

    def test_typescript_image(self) -> None:
        """TypeScript maps to node:20-alpine."""
        from src.main import _IMAGE_MAP
        assert _IMAGE_MAP["typescript"] == "node:20-alpine"
        assert _IMAGE_MAP["ts"] == "node:20-alpine"

    def test_python_command(self) -> None:
        """Python command is ['python', '-c']."""
        from src.main import _COMMAND_MAP
        assert _COMMAND_MAP["python"] == ["python", "-c"]

    def test_javascript_command(self) -> None:
        """JavaScript command is ['node', '-e']."""
        from src.main import _COMMAND_MAP
        assert _COMMAND_MAP["javascript"] == ["node", "-e"]

    def test_typescript_command(self) -> None:
        """TypeScript command is ['deno', 'eval']."""
        from src.main import _COMMAND_MAP
        assert _COMMAND_MAP["typescript"] == ["deno", "eval"]


# ── Rate limiter ─────────────────────────────────────────────────────────


class TestRateLimiter:
    """Per-IP rate limiter behavior."""

    @pytest.mark.asyncio
    async def test_first_request_allowed(self) -> None:
        """First request from an IP is allowed."""
        from src.main import _check_rate_limit
        from fastapi import Request

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.1"

        result = await _check_rate_limit(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_requests_within_limit_allowed(self) -> None:
        """Requests within rate limit are allowed."""
        from src.main import _check_rate_limit
        from fastapi import Request

        for _ in range(9):
            mock_request = MagicMock(spec=Request)
            mock_request.client.host = "10.0.0.2"
            result = await _check_rate_limit(mock_request)
            assert result is None

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_over_limit(self) -> None:
        """Requests beyond rate limit raise HTTPException 429."""
        from src.main import _check_rate_limit
        from src.config import settings
        from fastapi import Request, HTTPException

        settings.rate_limit_per_minute = 3

        for _ in range(3):
            mock_request = MagicMock(spec=Request)
            mock_request.client.host = "10.0.0.3"
            await _check_rate_limit(mock_request)

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.3"
        with pytest.raises(HTTPException) as exc_info:
            await _check_rate_limit(mock_request)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_different_ips_independent(self) -> None:
        """Different IPs have independent rate limit counters."""
        from src.main import _check_rate_limit
        from src.config import settings
        from fastapi import Request

        # Set a lower limit for faster test
        settings.rate_limit_per_minute = 2

        # Exhaust IP 10.0.0.4
        for _ in range(2):
            mock_request = MagicMock(spec=Request)
            mock_request.client.host = "10.0.0.4"
            await _check_rate_limit(mock_request)

        # IP 10.0.0.5 should still be allowed
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.5"
        result = await _check_rate_limit(mock_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_window_resets_after_timeout(self) -> None:
        """Rate limit window resets after 60 seconds."""
        from src.main import _check_rate_limit
        from src.config import settings, _IP_REQUEST_COUNTS
        from fastapi import Request, HTTPException
        import time

        settings.rate_limit_per_minute = 1

        # Consume the one allowed request
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.6"
        await _check_rate_limit(mock_request)

        # Should be blocked
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.6"
        with pytest.raises(HTTPException):
            await _check_rate_limit(mock_request)

        # Simulate window expiry by manipulating the timestamp
        ip_key = "10.0.0.6"
        old_count, old_window = _IP_REQUEST_COUNTS[ip_key]
        _IP_REQUEST_COUNTS[ip_key] = (old_count, old_window - 61)  # 61s in the past

        # Should be allowed again (window reset)
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.6"
        result = await _check_rate_limit(mock_request)
        assert result is None


# ── Concurrency semaphore ────────────────────────────────────────────────


class TestConcurrencySemaphore:
    """Concurrency semaphore behavior."""

    @pytest.fixture(autouse=True)
    def _reset_semaphore(self):
        """Reset semaphore before each test."""
        from src.config import _CONCURRENT_SEMAPHORE

        _CONCURRENT_SEMAPHORE = None
        yield
        _CONCURRENT_SEMAPHORE = None

    def test_semaphore_initialized_with_default(self) -> None:
        """Semaphore is created with max_concurrent_executions."""
        from src.main import _get_semaphore

        sem = _get_semaphore()
        assert sem is not None
        assert sem._value == 5

    def test_semaphore_reuses_instance(self) -> None:
        """Subsequent calls return the same semaphore instance."""
        from src.main import _get_semaphore

        sem1 = _get_semaphore()
        sem2 = _get_semaphore()
        assert sem1 is sem2

    def test_semaphore_value_from_config(self) -> None:
        """Semaphore capacity matches settings."""
        from src.main import _get_semaphore
        from src.config import settings

        sem = _get_semaphore()
        assert sem._value == settings.max_concurrent_executions


# ── Health endpoint ──────────────────────────────────────────────────────


class TestHealthEndpoint:
    """GET /health endpoint behavior."""

    def test_health_returns_ok(self) -> None:
        """Health endpoint returns status ok."""
        from src.main import app
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["service"] == "sandbox-service"
            assert "version" in data
