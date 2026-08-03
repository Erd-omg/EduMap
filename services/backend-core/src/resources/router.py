"""Resource management router — file upload, listing, deletion, and parsing.

Allows users to upload reference materials (PDF, MD, DOCX, PPT) and
browse system-generated resources alongside their uploads.  Uploaded files
are automatically parsed and indexed into ChromaDB for RAG retrieval.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request

from src.resources.models import ChunkPreview, ChunksResponse, ResourceListResponse, ResourceMetadata
from src.resources.parser import DocumentParser
from src.resources.repository import ResourceRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resources", tags=["resources"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx", ".txt", ".py", ".js", ".ts", ".html", ".csv"}
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_repo(request: Request) -> ResourceRepository | None:
    """Get the ResourceRepository from app state."""
    return getattr(request.app.state, "resource_repo", None)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/upload", response_model=ResourceMetadata)
async def upload_resource(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    kp_id: str | None = Form(None),
    kp_name: str | None = Form(None),
    description: str | None = Form(None),
    request: Request = None,
):
    """Upload a reference material file.

    Supported formats: PDF, MD, DOCX, PPT, TXT, PY, JS, TS, HTML, CSV
    After upload the file is automatically parsed and indexed into ChromaDB.
    """
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Check file size before reading
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB.",
        )

    # Generate unique filename
    resource_id = str(uuid.uuid4())
    safe_name = f"{resource_id}{ext}"
    file_path = UPLOAD_DIR / safe_name

    # Save file
    try:
        file_path.write_bytes(content)
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    file_size = len(content)

    metadata = ResourceMetadata(
        id=resource_id,
        user_id=user_id,
        name=file.filename or safe_name,
        type="upload",
        source="user_upload",
        kp_id=kp_id,
        kp_name=kp_name,
        file_size=file_size,
        file_path=str(file_path),
        description=description or f"用户上传的 {ext[1:].upper()} 文件",
        created_at=datetime.now(timezone.utc).isoformat(),
        parse_status="pending",
        parse_stats=None,
    )

    # Persist to PostgreSQL
    saved = await repo.create(metadata)
    logger.info("Resource uploaded: id=%s name=%s size=%d", resource_id, file.filename, file_size)

    # ── Auto-parse ─────────────────────────────────────────────────────────
    await repo.update_parse_status(resource_id, "parsing")
    vector_index = getattr(request.app.state, "vector_index", None) if request else None
    preloaded_model = getattr(request.app.state, "_embedding_model", None) if request else None
    parser = DocumentParser(preloaded_model=preloaded_model)
    parse_result = await parser.parse_and_index(
        file_path=file_path,
        resource_id=resource_id,
        resource_name=file.filename or safe_name,
        vector_index=vector_index,
        kp_id=kp_id,
        kp_name=kp_name,
    )

    if parse_result.get("error"):
        parse_status = "error"
    else:
        parse_status = "parsed"

    parse_stats: dict[str, Any] = {
        "chunks": parse_result.get("chunks", 0),
        "chars": parse_result.get("chars", 0),
        "indexed": parse_result.get("indexed", 0),
        "error": parse_result.get("error"),
    }

    # Store parsed chunk previews in parse_stats for /chunks endpoint
    in_memory_chunks = parse_result.get("in_memory_chunks", [])
    if in_memory_chunks:
        parse_stats["chunk_previews"] = in_memory_chunks

    await repo.update_parse_status(resource_id, parse_status, parse_stats)

    logger.info(
        "Resource parsed: id=%s status=%s chunks=%d chars=%d",
        resource_id,
        parse_status,
        parse_result.get("chunks", 0),
        parse_result.get("chars", 0),
    )

    # Re-fetch to get DB-assigned state
    saved = await repo.get(resource_id)
    if saved:
        return ResourceMetadata(**{k: v for k, v in saved.model_dump().items() if k != "file_path"})
    return ResourceMetadata(**{k: v for k, v in metadata.model_dump().items() if k != "file_path"})


@router.get("", response_model=ResourceListResponse)
async def list_resources(
    user_id: str = Query("anonymous"),
    type: str | None = Query(None, alias="type"),
    source: str | None = Query(None, alias="source"),
    kp_id: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    request: Request = None,
):
    """List resources, with optional filters."""
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    items, total = await repo.list(
        user_id=user_id or None,
        type_filter=type,
        source_filter=source,
        kp_id=kp_id,
        skip=skip,
        limit=limit,
    )

    # Strip file_path from response (don't expose internal paths)
    for item in items:
        object.__setattr__(item, "file_path", None)

    return ResourceListResponse(resources=items, total=total)


@router.get("/{resource_id}", response_model=ResourceMetadata)
async def get_resource(resource_id: str, request: Request = None):
    """Get a single resource by ID."""
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    meta = await repo.get(resource_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Resource not found")

    return ResourceMetadata(**{k: v for k, v in meta.model_dump().items() if k != "file_path"})


@router.delete("/{resource_id}", status_code=204)
async def delete_resource(resource_id: str, request: Request = None):
    """Delete a resource by ID."""
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    meta = await repo.delete(resource_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Delete file from disk
    file_path = meta.file_path
    if file_path and Path(file_path).exists():
        Path(file_path).unlink(missing_ok=True)

    logger.info("Resource deleted: id=%s", resource_id)


@router.post("/sync-generated")
async def sync_generated_resources(
    resources: list[ResourceMetadata],
    request: Request = None,
):
    """Sync system-generated resources into the resource store.

    Called by the orchestrator after generation completes, or by the
    frontend when it receives generated resources.
    """
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    count = await repo.sync_batch(resources)
    return {"synced": count}


@router.get("/{resource_id}/chunks", response_model=ChunksResponse)
async def get_resource_chunks(
    resource_id: str,
    request: Request = None,
    limit: int = Query(5, ge=1, le=50),
):
    """Get parsed text chunk previews for an uploaded resource.

    Returns the first N chunks that were extracted during document parsing.
    Prefers parse_stats.chunk_previews (always available after parsing);
    falls back to ChromaDB for system-generated or legacy resources.
    """
    repo = _get_repo(request)
    if not repo:
        raise HTTPException(status_code=503, detail="ResourceRepository not available")

    meta = await repo.get(resource_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Resource not found")

    # Try parse_stats.chunk_previews first (set during upload)
    parse_stats = meta.parse_stats or {}
    memory_chunks = parse_stats.get("chunk_previews", [])
    total_parsed = parse_stats.get("chunks", 0)

    if memory_chunks:
        previews = [
            ChunkPreview(
                index=c["index"],
                text_preview=c["text"],
                char_count=c["char_count"],
            )
            for c in memory_chunks[:limit]
        ]
        return ChunksResponse(
            resource_id=resource_id,
            total_chunks=total_parsed,
            chunks=previews,
        )

    # Fallback: try ChromaDB (for system-generated or legacy resources)
    vi = getattr(request.app.state, "vector_index", None) if request else None
    if not vi:
        # Show friendly message when parsed but not indexed
        indexed = parse_stats.get("indexed", 0)
        if total_parsed > 0 and indexed == 0:
            return ChunksResponse(
                resource_id=resource_id,
                total_chunks=total_parsed,
                chunks=[],
            )
        return ChunksResponse(resource_id=resource_id, total_chunks=0, chunks=[])

    try:
        client = vi._client  # chromadb.HttpClient
        collection = client.get_collection(name="resource_chunks")

        # Query chunks filtered by resource_id
        results = collection.get(
            where={"resource_id": resource_id},
            limit=limit,
        )

        chunks: list[ChunkPreview] = []
        metadatas = results.get("metadatas") or []
        documents = results.get("documents") or []
        if metadatas:
            for i, meta_item in enumerate(metadatas):
                doc_text = documents[i] if i < len(documents) and documents[i] else None
                text_preview = doc_text or meta_item.get("text_preview", "")
                chunks.append(ChunkPreview(
                    index=i,
                    text_preview=str(text_preview)[:200],
                    char_count=len(str(text_preview)),
                ))

        total = len(results.get("ids", [])) if results else 0
        return ChunksResponse(
            resource_id=resource_id,
            total_chunks=total,
            chunks=chunks,
        )
    except Exception as exc:
        logger.warning("Failed to retrieve chunks for resource %s: %s", resource_id, exc)
        return ChunksResponse(resource_id=resource_id, total_chunks=0, chunks=[])
