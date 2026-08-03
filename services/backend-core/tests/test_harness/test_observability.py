"""Tests for Observability — Timer and execution report logging.

Pure logic tests — no external dependencies.
"""

from __future__ import annotations

import logging
import time

import pytest

from src.harness.observability import Timer, log_execution_report
from src.harness.types import ExecutionReport


class TestTimer:
    """Timer — wall-clock duration measurement."""

    def test_timer_elapsed_ms_increases(self) -> None:
        """Timer.elapsed_ms returns elapsed time since start()."""
        timer = Timer()
        assert timer.elapsed_ms == 0.0  # not started yet
        timer.start()
        time.sleep(0.01)
        elapsed = timer.elapsed_ms
        assert elapsed > 0
        assert elapsed < 200.0  # sanity: shouldn't take >200ms

    def test_timer_stop_returns_elapsed(self) -> None:
        """Timer.stop() returns the elapsed duration in ms."""
        timer = Timer()
        timer.start()
        time.sleep(0.02)
        elapsed = timer.stop()
        assert elapsed > 10.0
        assert elapsed < 200.0

    def test_timer_start_resets(self) -> None:
        """Calling start() again resets the timer."""
        timer = Timer()
        timer.start()
        time.sleep(0.02)
        first = timer.stop()

        timer.start()
        time.sleep(0.01)
        second = timer.stop()

        assert first > second  # first should be longer
        assert second > 0

    def test_timer_stop_twice(self) -> None:
        """Stopping twice doesn't error, second call returns 0.0."""
        timer = Timer()
        timer.start()
        timer.stop()
        # Second stop should return 0.0 since _start is None
        assert timer.stop() == 0.0

    def test_timer_not_started_returns_zero(self) -> None:
        """Calling elapsed_ms before start() returns 0.0."""
        timer = Timer()
        assert timer.elapsed_ms == 0.0


class TestExecutionReport:
    """log_execution_report() — structured log output."""

    def test_success_report_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Success report is logged with agent name and metrics."""
        caplog.set_level(logging.INFO)
        report = ExecutionReport(
            agent_name="test_agent",
            success=True,
            output="completed",
            duration_ms=42.5,
            retries=1,
            memory_context_loaded=True,
        )
        log_execution_report(report)

        assert len(caplog.records) >= 1
        assert "test_agent" in caplog.records[0].message
        assert "true" in caplog.records[0].message.lower()

    def test_error_report_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Error report includes error details."""
        caplog.set_level(logging.INFO)
        report = ExecutionReport(
            agent_name="failing_agent",
            success=False,
            error="connection refused",
            error_type="ConnectionError",
            duration_ms=10.0,
        )
        log_execution_report(report)

        assert len(caplog.records) >= 1
        assert "failing_agent" in caplog.records[0].message

    def test_multiple_reports(self, caplog: pytest.LogCaptureFixture) -> None:
        """Multiple reports each produce a separate log entry."""
        caplog.set_level(logging.INFO)
        reports = [
            ExecutionReport(agent_name="agent_a", success=True, duration_ms=1.0),
            ExecutionReport(agent_name="agent_b", success=True, duration_ms=2.0),
        ]
        for r in reports:
            log_execution_report(r)

        assert len(caplog.records) >= 2
        assert "agent_a" in caplog.records[0].message
        assert "agent_b" in caplog.records[1].message

    def test_empty_output_report(self) -> None:
        """A report with no output is logged without error."""
        report = ExecutionReport(
            agent_name="empty_agent",
            success=True,
            output=None,
            duration_ms=0.0,
        )
        # Should not raise
        log_execution_report(report)

    def test_report_contains_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        """Duration is included in the log output."""
        caplog.set_level(logging.INFO)
        report = ExecutionReport(
            agent_name="perf_agent",
            success=True,
            duration_ms=123.45,
        )
        log_execution_report(report)

        assert "123" in caplog.records[0].message
