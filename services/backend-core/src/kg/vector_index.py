"""Vector index with ChromaDB support and in-memory fallback."""

import json
import logging
from typing import Any

try:
    import chromadb  # noqa: F811
    _HAS_CHROMADB = True
except Exception:
    _HAS_CHROMADB = False

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "kp_embeddings"


class VectorIndex:
    """Vector index backed by ChromaDB (preferred) or in-memory dict (fallback).

    The in-memory fallback stores key-value pairs so that RAG/mentor
    functionality works even when ChromaDB is unavailable or version-incompatible.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self._host = host
        self._port = port
        self._memory_store: dict[str, dict[str, Any]] = {}  # id -> {"embedding": [...], "metadata": {...}}
        self._chroma_collection = None
        self._client: Any = None  # chromadb.HttpClient or None (in-memory fallback)

        # Try ChromaDB first, fall back to in-memory.  Retried a few times
        # because a transient DNS/socket failure at process start would
        # otherwise silently downgrade the whole index to an empty in-memory
        # dict — retrieval then returns nothing while looking healthy.
        if _HAS_CHROMADB:
            try:
                client = self._connect_with_retry(host, port)
                self._client = client
                self._chroma_collection = client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "VectorIndex initialised with ChromaDB (host=%s, port=%d, collection=%s)",
                    host, port, _COLLECTION_NAME,
                )
            except Exception as exc:
                # ``chromadb`` raises bare exceptions with an empty message, so
                # log the type and a traceback — otherwise this is impossible to
                # diagnose and the in-memory fallback silently fakes an empty index.
                logger.warning(
                    "ChromaDB unavailable (%s: %r) — using in-memory fallback for "
                    "VectorIndex. Any retrieval against this index is EMPTY.",
                    type(exc).__name__,
                    str(exc),
                    exc_info=True,
                )
        else:
            logger.warning("chromadb package not installed — using in-memory fallback")

    @staticmethod
    def _connect_with_retry(host: str, port: int, attempts: int = 3):
        """Create an ``HttpClient``, retrying transient connect failures.

        ChromaDB's client raises a bare ``ValueError`` (sometimes wrapping a
        ``httpcore.ConnectError`` such as "nodename nor servname provided")
        when the first connection races with process startup.  A short retry
        turns that into a successful connect instead of a silent downgrade to
        an empty in-memory index.
        """
        import time

        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return chromadb.HttpClient(host=host, port=port)
            except Exception as exc:  # noqa: BLE001 - chromadb raises bare exceptions
                last_exc = exc
                if attempt < attempts:
                    time.sleep(0.5 * attempt)
        assert last_exc is not None
        raise last_exc

    def upsert(
        self, kp_id: str, embedding: list[float], metadata: dict | None = None
    ) -> None:
        """Insert or update an embedding for a knowledge point."""
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.upsert(
                    ids=[kp_id],
                    embeddings=[embedding],
                    metadatas=[metadata or {}],
                )
                return
            except Exception:
                logger.exception("ChromaDB upsert failed, falling back to memory")

        self._memory_store[kp_id] = {
            "embedding": embedding,
            "metadata": metadata or {},
        }

    def search(
        self, embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """Search the index for the nearest neighbours.

        Returns a list of dicts with keys ``id``, ``distance``, and ``metadata``.
        """
        if self._chroma_collection is not None:
            try:
                results = self._chroma_collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    include=["metadatas", "distances"],
                )
                output: list[dict] = []
                ids = results.get("ids", [[]])[0]
                distances = results.get("distances", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                for i in range(len(ids)):
                    output.append({
                        "id": ids[i],
                        "distance": distances[i] if distances else None,
                        "metadata": metadatas[i] if metadatas else {},
                    })
                return output
            except Exception:
                logger.exception("ChromaDB search failed, falling back to memory")

        # In-memory fallback: naive cosine similarity (simplified)
        results = []
        for mem_id, mem_data in self._memory_store.items():
            mem_emb = mem_data.get("embedding", [])
            if mem_emb and len(mem_emb) == len(embedding):
                dot = sum(a * b for a, b in zip(embedding, mem_emb))
                norm_a = sum(a * a for a in embedding) ** 0.5
                norm_b = sum(b * b for b in mem_emb) ** 0.5
                sim = dot / (norm_a * norm_b + 1e-10)
                results.append({
                    "id": mem_id,
                    "distance": 1.0 - sim,
                    "metadata": mem_data.get("metadata", {}),
                })
        results.sort(key=lambda x: x["distance"])
        return results[:top_k]

    def collection_size(self) -> int:
        """Return the number of embedded items."""
        if self._chroma_collection is not None:
            try:
                return self._chroma_collection.count()
            except Exception:
                logger.exception("ChromaDB count failed")
        return len(self._memory_store)

    def delete(self, kp_id: str) -> None:
        """Remove an embedding entry by id."""
        if self._chroma_collection is not None:
            try:
                self._chroma_collection.delete(ids=[kp_id])
                return
            except Exception:
                logger.exception("ChromaDB delete failed, falling back to memory")
        self._memory_store.pop(kp_id, None)
