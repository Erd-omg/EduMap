"""Simple API key authentication for backend-core.

Supports both ``Authorization: Bearer`` header (for fetch-based endpoints)
and ``?api_key=xxx`` query parameter (for SSE EventSource which cannot
set custom headers).

Usage in routes::

    @router.get("/secure")
    async def secure_endpoint(_: None = Depends(verify_api_key)):
        ...
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings

_security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    api_key: str | None = Query(None, alias="api_key"),
) -> None:
    """Verify the API key from the Authorization header or ``?api_key=``
    query parameter.

    If ``settings.api_key`` is empty, authentication is skipped (open access).
    Uses constant-time comparison to prevent timing attacks.
    """
    if not settings.api_key:
        return  # Open access when no API key configured

    # Check query parameter first (for SSE EventSource compatibility)
    if api_key is not None and secrets.compare_digest(api_key, settings.api_key):
        return

    # Then check Authorization header
    if credentials is not None and secrets.compare_digest(credentials.credentials, settings.api_key):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key",
    )
