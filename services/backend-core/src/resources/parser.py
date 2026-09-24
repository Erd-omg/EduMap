"""Document parser — extracts text from uploaded files, chunks it, and indexes into ChromaDB.

Supports PDF, DOCX, PPTX via the ``unstructured`` library, and plain text formats
(MD, TXT, code) via built-in readers.  Chunks are embedded with sentence-transformers
and upserted into the VectorIndex (ChromaDB) for RAG retrieval.

The chunking strategy is configurable via ``chunking_strategy``:
- ``fixed``: Character-based fixed-size chunks with overlap (original behavior)
- ``recursive``: Recursive character split with separator hierarchy
- ``semantic``: Semantic boundary detection via embedding similarity
- ``auto``: Tries semantic, falls back to recursive, then fixed
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.kg.vector_index import VectorIndex

logger = logging.getLogger(__name__)

# ── Chunking constants ──────────────────────────────────────────────────────

COLLECTION_NAME = "resource_chunks"  # separate collection from kp_embeddings


class DocumentParser:
    """Parse uploaded documents, extract text, chunk, embed, and index.

    Usage::

        parser = DocumentParser()
        result = await parser.parse_and_index(
            file_path="/data/uploads/abc.pdf",
            resource_id="res-123",
            resource_name="Chapter 1.pdf",
            vector_index=vector_index_instance,
        )
    """

    def __init__(self, preloaded_model: Any | None = None) -> None:
        self._model: Any | None = preloaded_model
        self._vector_index: VectorIndex | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    async def parse_and_index(
        self,
        file_path: str | Path,
        resource_id: str,
        resource_name: str,
        vector_index: VectorIndex | None,
        kp_id: str | None = None,
        kp_name: str | None = None,
    ) -> dict[str, Any]:
        """Parse, chunk, embed, and index a document.

        Returns stats::

            {"chunks": int, "chars": int, "error": str | None}
        """
        path = Path(file_path)
        if not path.exists():
            return {"chunks": 0, "chars": 0, "error": f"File not found: {file_path}"}

        # 1. Extract text
        try:
            text = self._extract_text(path)
        except Exception as exc:
            logger.exception("Text extraction failed for %s", resource_name)
            return {"chunks": 0, "chars": 0, "error": str(exc)}

        if not text.strip():
            return {"chunks": 0, "chars": 0, "error": "No extractable text found"}

        # 2. Chunk
        from src.config import settings
        chunks = self._chunk_text(
            text,
            strategy=settings.chunking_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            return {"chunks": 0, "chars": len(text), "error": "Chunking produced no output"}

        # 2.5 Locate chunks in the source text.
        #
        # Done once and reused for both the vector-store metadata (below, in
        # _embed_and_index) and the chunk previews returned here — computing it
        # twice would be wasteful and could drift.
        from src.resources.provenance import locate_chunks

        spans = locate_chunks(text, chunks)

        # Embed and index (if VectorIndex available)
        indexed = 0
        if vector_index is not None:
            try:
                indexed = await self._embed_and_index(
                    chunks=chunks,
                    resource_id=resource_id,
                    resource_name=resource_name,
                    vector_index=vector_index,
                    kp_id=kp_id,
                    kp_name=kp_name,
                    spans=spans,
                )
            except Exception as exc:
                logger.exception("Embedding/indexing failed for %s", resource_name)
                return {
                    "chunks": len(chunks),
                    "chars": len(text),
                    "indexed": 0,
                    "error": f"Indexing failed: {exc}",
                }

        return {
            "chunks": len(chunks),
            "chars": len(text),
            "indexed": indexed,
            "error": None,
            # Character offsets travel with the previews so the UI can highlight
            # the cited passage rather than only naming the chunk. None (not 0)
            # when a chunk could not be located — 0 would point at the start of
            # the document and silently highlight the wrong text.
            "in_memory_chunks": [
                {
                    "index": i,
                    "text": chunk[:200],
                    "char_count": len(chunk),
                    "char_start": spans[i].char_start if i < len(spans) else None,
                    "char_end": spans[i].char_end if i < len(spans) else None,
                }
                for i, chunk in enumerate(chunks)
            ] if chunks else [],
        }

    # ── Text extraction ─────────────────────────────────────────────────────

    def _extract_text(self, path: Path) -> str:
        """Extract text from a file based on its extension."""
        ext = path.suffix.lower()

        if ext in (".pdf", ".docx", ".pptx"):
            return self._extract_with_unstructured(path)
        elif ext == ".md":
            return path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")
        elif ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".csv", ".json", ".yaml", ".yml"):
            return self._extract_code(path)
        else:
            # Fallback: try plain text
            return path.read_text(encoding="utf-8", errors="replace")

    def _extract_with_unstructured(self, path: Path) -> str:
        """Use the ``unstructured`` library to extract text from PDF/DOCX/PPTX."""
        try:
            from unstructured.partition.auto import partition  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "unstructured library not available — falling back to raw text for %s. "
                "Install with: pip install 'unstructured[pdf,docx,pptx]'",
                path.name,
            )
            return path.read_text(encoding="utf-8", errors="replace")

        elements = partition(str(path))
        return "\n\n".join(str(el) for el in elements)

    def _extract_code(self, path: Path) -> str:
        """Extract meaningful text from code files — reads as-is but strips excessive blank lines."""
        text = path.read_text(encoding="utf-8", errors="replace")
        # Collapse 3+ consecutive blank lines to 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text

    # ── Chunking ─────────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        strategy: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> list[str]:
        """Split text into chunks using the configured strategy.

        Args:
            text: Raw text to chunk.
            strategy: One of ``"auto"``, ``"fixed"``, ``"recursive"``, ``"semantic"``.
            chunk_size: Target character count per chunk (for fixed/recursive).
            chunk_overlap: Overlapping characters between chunks (for fixed/recursive).
        """
        if not text:
            return []
        text = re.sub(r"\n{3,}", "\n\n", text)

        strategy = self._resolve_strategy(strategy, text)
        logger.debug("Chunking with strategy=%s (size=%d, overlap=%d)", strategy, chunk_size, chunk_overlap)

        if strategy == "semantic":
            return self._chunk_semantic(text)
        elif strategy == "recursive":
            return self._chunk_recursive(text, chunk_size, chunk_overlap)
        else:  # "fixed"
            return self._chunk_fixed(text, chunk_size, chunk_overlap)

    def _resolve_strategy(self, strategy: str, text: str) -> str:
        """Resolve 'auto' to the best strategy for the given text."""
        if strategy != "auto":
            return strategy

        # Auto-detect: try semantic for well-structured text, fixed for short text
        if len(text) < 200:
            return "fixed"

        # Check if text has clear paragraph/section structure
        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 3:
            # Try semantic chunking
            try:
                from src.rag.chunking.semantic_chunker import SemanticChunker
                test_chunker = SemanticChunker()
                test_chunks = test_chunker.chunk(text[:2000])  # quick test on sample
                if len(test_chunks) > 1:
                    return "semantic"
            except Exception:
                pass

        return "recursive"

    def _chunk_fixed(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Original fixed-size character chunking with overlap."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk.strip())
            start += chunk_size - chunk_overlap
        return [c for c in chunks if c.strip()]

    def _chunk_recursive(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """Recursive chunking with separator hierarchy."""
        try:
            from src.rag.chunking.recursive_chunker import RecursiveChunker
            chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            return chunker.chunk(text)
        except Exception as exc:
            logger.warning("Recursive chunking failed, falling back to fixed: %s", exc)
            return self._chunk_fixed(text, chunk_size, chunk_overlap)

    def _chunk_semantic(self, text: str) -> list[str]:
        """Semantic boundary detection chunking."""
        try:
            from src.rag.chunking.semantic_chunker import SemanticChunker
            chunker = SemanticChunker()
            return chunker.chunk(text)
        except Exception as exc:
            logger.warning("Semantic chunking failed, falling back to recursive: %s", exc)
            return self._chunk_recursive(text, 512, 64)

    # ── Embedding & indexing ────────────────────────────────────────────────

    def _get_or_create_resource_collection(self, vector_index: VectorIndex) -> Any:
        """Get or create a dedicated ChromaDB collection for resource chunks."""
        # VectorIndex wraps collection "kp_embeddings", but we want a separate one
        # for resource chunks so they don't interfere with KP search.
        # Access the underlying client to create a new collection.
        client = getattr(vector_index, '_client', None)
        if client is None:
            # In-memory fallback mode — return None so caller skips ChromaDB indexing
            logger.info("VectorIndex in memory-fallback mode — resource chunks stored in-memory only")
            return None
        try:
            return client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            logger.exception("Failed to get/create collection '%s'", COLLECTION_NAME)
            raise

    async def _embed_and_index(
        self,
        chunks: list[str],
        resource_id: str,
        resource_name: str,
        vector_index: VectorIndex,
        kp_id: str | None = None,
        kp_name: str | None = None,
        spans: list | None = None,
    ) -> int:
        """Embed chunks and upsert into ChromaDB.

        ``spans`` are the located character offsets for each chunk (from
        ``locate_chunks``); when omitted the span fields are simply absent from
        the metadata rather than wrong.
        """
        collection = self._get_or_create_resource_collection(vector_index)
        if collection is None:
            # ChromaDB unavailable (in-memory fallback) — store in VectorIndex's
            # in-memory store using the KP embedding collection as best-effort.
            logger.info("Resource chunks stored in VectorIndex memory (no ChromaDB)")
            return 0

        # Lazy-load model
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

        # Generate embeddings
        embeddings = self._model.encode(chunks, show_progress_bar=False)
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Prepare metadata and IDs.
        #
        # Character offsets come in as ``spans`` (recovered by locate_chunks in
        # the caller) so that provenance works for every chunking strategy
        # without touching their splitting logic — see
        # src/resources/provenance.py for why.
        ids: list[str] = []
        metadatas: list[dict] = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{resource_id}::chunk::{i:04d}"
            ids.append(chunk_id)
            meta: dict = {
                "resource_id": resource_id,
                "resource_name": resource_name,
                "chunk_index": i,
                "source": "user_upload",
                "text_preview": chunk_text[:200],
            }
            # Span within the source document. None (not 0) when the chunk could
            # not be located, so a missing span is distinguishable from "starts
            # at the beginning" — defaulting to 0 would silently point every
            # such citation at the document's first character.
            if spans is not None and i < len(spans):
                meta["char_start"] = spans[i].char_start
                meta["char_end"] = spans[i].char_end
            if kp_id:
                meta["kp_id"] = kp_id
            if kp_name:
                meta["kp_name"] = kp_name
            metadatas.append(meta)

        # Upsert in batches to avoid oversized requests
        batch_size = 100
        total_upserted = 0
        for batch_start in range(0, len(ids), batch_size):
            # Clamp the end. Slicing would tolerate an overlong index (Python
            # truncates), but the *count* below is computed from these bounds —
            # so an unclamped batch_end reported a full batch_size even for a
            # partial one. A 19-chunk document was reported as "indexed: 100".
            batch_end = min(batch_start + batch_size, len(ids))
            collection.upsert(
                ids=ids[batch_start:batch_end],
                embeddings=embeddings_list[batch_start:batch_end],
                documents=chunks[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )
            total_upserted += (batch_end - batch_start)

        logger.info(
            "Indexed %d chunks for resource %s (collection: %s)",
            total_upserted,
            resource_id,
            COLLECTION_NAME,
        )
        return total_upserted
