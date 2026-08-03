"""Logging middleware that sanitizes query parameters from access logs.

Prevents user messages (sent as GET query params to SSE endpoints) from
appearing in server access logs.

Usage in main.py::

    logger = logging.getLogger("uvicorn.access")
    logger.addFilter(SanitizeQueryFilter())
"""

from __future__ import annotations

import logging
import re

from starlette.datastructures import URL
from starlette.types import ASGIApp, Receive, Scope, Send

# Query parameters whose values should be redacted in access logs
_REDACTED_PARAMS = frozenset({"query", "message", "q", "token", "key", "password", "api_key"})

logger = logging.getLogger("uvicorn.access")


class SanitizeQueryFilter(logging.Filter):
    """Logging filter that redacts sensitive query parameters from access logs.

    Attach to ``uvicorn.access`` logger to prevent user messages (sent as
    GET query params to SSE endpoints) from appearing in server logs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Replace sensitive query strings in the log message."""
        if not hasattr(record, 'args') or not record.args:
            return True

        args = list(record.args) if isinstance(record.args, (list, tuple)) else list(record.args.values())
        for i, arg in enumerate(args):
            if isinstance(arg, str) and any(p in arg.lower() for p in _REDACTED_PARAMS):
                args[i] = _sanitize_query(arg)
        record.args = tuple(args)
        return True


def _sanitize_query(query_string: str) -> str:
    """Replace sensitive parameter values with 'REDACTED'."""
    params = query_string.split("&")
    sanitized = []
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)
            if key.lower() in _REDACTED_PARAMS:
                sanitized.append(f"{key}=REDACTED")
            else:
                sanitized.append(param)
        else:
            sanitized.append(param)
    return "&".join(sanitized)
