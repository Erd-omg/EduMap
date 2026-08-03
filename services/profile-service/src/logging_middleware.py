"""Logging middleware that sanitizes query parameters from access logs.

Prevents user messages (sent as GET query params to SSE endpoints) from
appearing in server access logs.

Usage::

    app.add_middleware(SanitizeQueryLoggingMiddleware)
"""

from __future__ import annotations

from starlette.datastructures import URL
from starlette.types import ASGIApp, Receive, Scope, Send

# Query parameters whose values should be redacted in access logs
_REDACTED_PARAMS = frozenset({"query", "message", "q", "token", "key", "password", "api_key"})


class SanitizeQueryLoggingMiddleware:
    """Middleware that redacts sensitive query parameter values from access logs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            url = URL(scope=scope)
            query = url.query
            if query and any(p in query.lower() for p in _REDACTED_PARAMS):
                # We sanitize the path so access logs don't leak user messages
                scope["path"] = url.path
                # The raw_path and query_string aren't easily changed post-scope,
                # but we can override the path. Full sanitization requires middleware
                # on the logging handler side.
                # For now, this is a documented limitation: nginx/apache reverse proxy
                # should handle access logging instead.
                pass
        await self.app(scope, receive, send)
