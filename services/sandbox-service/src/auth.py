"""Simple API key authentication for sandbox-service."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import settings

_security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> None:
    """Verify the API key from the Authorization header.

    If ``settings.api_key`` is empty, authentication is skipped (open access).
    """
    if not settings.api_key:
        return  # Open access when no API key configured

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
