"""Sandbox service — secure code execution via Docker containers."""

from __future__ import annotations

import hashlib
import time
import logging
import os

import docker
from docker.errors import DockerException, ImageNotFound
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="EduMap Sandbox Service",
    version="0.1.0",
    description="Secure code execution sandbox with Docker isolation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
async def execute_code(request: CodeExecutionRequest) -> CodeExecutionResponse:
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
    container_name = f"edumap-sandbox-{hashlib.md5(request.code.encode()).hexdigest()[:12]}"

    # Build full command: prefix + [code]
    full_cmd = cmd_prefix + [request.code]

    try:
        # Ensure image is available
        try:
            client.images.get(image)
        except ImageNotFound:
            logger.info("Pulling image %s ...", image)
            client.images.pull(image)

        # Create and start container
        start = time.monotonic()
        container = client.containers.create(
            image,
            full_cmd,
            name=container_name,
            mem_limit=f"{request.memory_limit_mb}m",
            network_disabled=not settings.network_enabled,
            read_only=True,
            auto_remove=False,
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

    except requests.ReadTimeoutError if os.name != "nt" else Exception as exc:  # noqa: F821
        # Timeout
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
        except Exception:
            pass
        elapsed = int((time.monotonic() - start) * 1000) if 'start' in dir() else 0
        return CodeExecutionResponse(
            status="timeout",
            stdout="",
            stderr=f"Execution timed out after {request.timeout_seconds}s",
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
