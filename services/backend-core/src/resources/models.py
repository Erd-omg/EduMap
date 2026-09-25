"""Pydantic models for resource management."""

from __future__ import annotations

from pydantic import BaseModel


class ResourceMetadata(BaseModel):
    id: str
    user_id: str
    name: str
    type: str  # "upload" | "explanation" | "exercise" | "visualization" | "code"
    source: str  # "user_upload" | "system_generated"
    kp_id: str | None = None
    kp_name: str | None = None
    file_size: int | None = None
    file_path: str | None = None
    description: str | None = None
    created_at: str
    parse_status: str | None = None  # "pending" | "parsing" | "parsed" | "error"
    parse_stats: dict | None = None  # {"chunks": N, "chars": N, "error": "...", "chunk_previews": [...]}


class ResourceListResponse(BaseModel):
    resources: list[ResourceMetadata]
    total: int


class ChunkPreview(BaseModel):
    index: int
    text_preview: str
    char_count: int
    # Character span of this chunk in the source document, so the UI can
    # highlight the cited passage. ``None`` means "unknown" and must not be
    # rendered as 0 — that would highlight the document's first character.
    char_start: int | None = None
    char_end: int | None = None
    # 1-based page in the source document, when the format has pages.
    # None for plain text (no page concept) — must not be rendered as 1, which
    # would claim every chunk came from page 1.
    page_number: int | None = None


class ChunksResponse(BaseModel):
    resource_id: str
    total_chunks: int
    chunks: list[ChunkPreview]
