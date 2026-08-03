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


class ChunksResponse(BaseModel):
    resource_id: str
    total_chunks: int
    chunks: list[ChunkPreview]
