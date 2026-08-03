"""Observability helpers: Timer and execution report logging."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class Timer:
    """Simple wall-clock timer for agent execution duration."""

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self._start is None:
            return 0.0
        return (time.perf_counter() - self._start) * 1000

    def stop(self) -> float:
        ms = self.elapsed_ms
        self._start = None
        return ms


def log_execution_report(report: Any) -> None:
    """Log a structured execution report as JSON."""
    logger.info(
        "AgentReport agent=%s success=%s duration_ms=%.0f retries=%d tokens=%s",
        report.agent_name,
        report.success,
        report.duration_ms,
        report.retries,
        json.dumps(report.token_usage, ensure_ascii=False) if report.token_usage else "{}",
    )
