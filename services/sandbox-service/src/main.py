"""Sandbox service — secure code execution via Docker containers."""

from __future__ import annotations

import asyncio
import hashlib
import time
import logging
import uuid

import docker
from docker.errors import DockerException, ImageNotFound
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.auth import verify_api_key
# ``_CONCURRENT_SEMAPHORE`` looks unused to linters but is referenced by the
# ``global`` statement in ``_get_semaphore()`` below — do not remove.
from src.config import settings, _IP_REQUEST_COUNTS, _CONCURRENT_SEMAPHORE  # noqa: F401

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduMap Sandbox Service",
    version="0.1.0",
    description="Secure code execution sandbox with Docker isolation",
    dependencies=[Depends(verify_api_key)] if settings.api_key else [],
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────


class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout_seconds: int = Field(default=5, ge=1, le=30)
    memory_limit_mb: int = Field(default=128, ge=16, le=512)


class CodeExecutionResponse(BaseModel):
    status: str  # "success" | "timeout" | "error"
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time_ms: int = 0


# ── Language → image mapping ──────────────────────────────────────────

_IMAGE_MAP: dict[str, str] = {
    "python": "python:3.12-alpine",
    "python3": "python:3.12-alpine",
    "py": "python:3.12-alpine",
    "javascript": "node:20-alpine",
    "js": "node:20-alpine",
    "node": "node:20-alpine",
    "typescript": "node:20-alpine",
    "ts": "node:20-alpine",
}

_COMMAND_MAP: dict[str, list[str]] = {
    "python": ["python", "-c"],
    "python3": ["python3", "-c"],
    "py": ["python", "-c"],
    "javascript": ["node", "-e"],
    "js": ["node", "-e"],
    "node": ["node", "-e"],
    "typescript": ["deno", "eval"],
    "ts": ["deno", "eval"],
}

# ── Rate limiter (per-IP, in-memory) ──────────────────────────────────


async def _check_rate_limit(request: Request) -> None:
    """Enforce per-IP rate limit for sandbox executions."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = 60.0  # 1-minute window

    count, window_start = _IP_REQUEST_COUNTS.get(client_ip, (0, now))
    if now - window_start > window:
        # Reset window
        _IP_REQUEST_COUNTS[client_ip] = (1, now)
        return

    _IP_REQUEST_COUNTS[client_ip] = (count + 1, window_start)
    if count >= settings.rate_limit_per_minute:
        logger.warning("Rate limit exceeded for IP %s (%d/min)", client_ip, count + 1)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max "
                f"{settings.rate_limit_per_minute} requests/min per IP"
            ),
        )


# ── Concurrency semaphore ──────────────────────────────────────────────


def _get_semaphore() -> asyncio.Semaphore:
    global _CONCURRENT_SEMAPHORE
    if _CONCURRENT_SEMAPHORE is None:
        _CONCURRENT_SEMAPHORE = asyncio.Semaphore(settings.max_concurrent_executions)
    return _CONCURRENT_SEMAPHORE


# ── Container security helpers ─────────────────────────────────────────


def _build_security_kwargs() -> dict:
    """Build Docker container creation kwargs for sandbox security hardening."""
    return {
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "cpu_period": 100000,
        "cpu_quota": 50000,  # 0.5 CPU core
        "pids_limit": 50,  # Prevent fork bombs
        "user": "nobody",  # Run as non-root inside container
    }


# ── Docker client (lazy) ──────────────────────────────────────────────

_docker_client: docker.DockerClient | None = None


def get_docker() -> docker.DockerClient:
    global _docker_client
    if _docker_client is None:
        try:
            _docker_client = docker.from_env()
            _docker_client.ping()
            logger.info("Docker client connected")
        except DockerException as exc:
            logger.warning("Docker not available, sandbox will fail gracefully: %s", exc)
            _docker_client = None
    return _docker_client  # type: ignore[return-value]


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sandbox-service", "version": "0.1.0"}


@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest, req: Request) -> CodeExecutionResponse:
    # Enforce per-IP rate limit
    await _check_rate_limit(req)
    """Execute user code in an isolated Docker container and return the result."""
    client = get_docker()
    if client is None:
        return CodeExecutionResponse(
            status="error",
            stdout="",
            stderr="Sandbox unavailable — Docker not available in this environment",
            exit_code=1,
        )

    image = _IMAGE_MAP.get(request.language, "python:3.12-alpine")
    cmd_prefix = _COMMAND_MAP.get(request.language, ["python", "-c"])

    # Unique container name — use hash for dedup + uuid to avoid collisions
    code_hash = hashlib.md5(request.code.encode()).hexdigest()[:8]
    unique_suffix = uuid.uuid4().hex[:6]
    container_name = f"edumap-sandbox-{code_hash}-{unique_suffix}"

    # Build full command: prefix + [code]
    full_cmd = cmd_prefix + [request.code]

    try:
        # Ensure image is available
        try:
            client.images.get(image)
        except ImageNotFound:
            logger.info("Pulling image %s ...", image)
            client.images.pull(image)

        # Acquire concurrency slot
        sem = _get_semaphore()
        async with sem:  # type: ignore[arg-type]
            start = time.monotonic()
            sec_kwargs = _build_security_kwargs()
            container = client.containers.create(
                image,
                full_cmd,
                name=container_name,
                mem_limit=f"{request.memory_limit_mb}m",
                network_disabled=not settings.network_enabled,
                read_only=True,
                auto_remove=False,
                **sec_kwargs,
            )
        container.start()

        # Wait with timeout
        result = container.wait(timeout=request.timeout_seconds)
        elapsed = int((time.monotonic() - start) * 1000)

        logs_stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
        logs_stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

        container.remove(force=True)

        exit_code = result.get("StatusCode", -1)
        status = "success" if exit_code == 0 else "error"

        return CodeExecutionResponse(
            status=status,
            stdout=logs_stdout,
            stderr=logs_stderr,
            exit_code=exit_code,
            execution_time_ms=elapsed,
        )

    except docker.errors.ContainerError as exc:
        elapsed = int((time.monotonic() - start) * 1000) if 'start' in dir() else 0
        return CodeExecutionResponse(
            status="error",
            stdout=exc.stdout.decode() if exc.stdout else "",
            stderr=exc.stderr.decode() if exc.stderr else str(exc),
            exit_code=exc.exit_status or 1,
            execution_time_ms=elapsed,
        )

    except docker.errors.APIError as exc:
        # Container-level errors (timeout, OOM, etc.)
        elapsed = int((time.monotonic() - start) * 1000) if 'start' in dir() else 0
        return CodeExecutionResponse(
            status="timeout" if "timeout" in str(exc).lower() else "error",
            stdout="",
            stderr=f"Sandbox execution error: {exc}",
            exit_code=-1,
            execution_time_ms=elapsed,
        )

    except Exception as exc:
        logger.exception("Sandbox execution failed")
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
        except Exception:
            pass
        return CodeExecutionResponse(
            status="error",
            stdout="",
            stderr=f"Sandbox error: {exc}",
            exit_code=1,
        )
