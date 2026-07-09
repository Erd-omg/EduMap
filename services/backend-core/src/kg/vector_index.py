"""ChromaDB-backed vector index for knowledge point embeddings."""

import logging

import chromadb

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "kp_embeddings"


class VectorIndex:
    """Vector index backed by ChromaDB for embedding-based similarity search.

    Usage::

        index = VectorIndex(host="localhost", port=8000)
        index.upsert("kp-abc", [0.1, 0.2, ...], {"name": "foo"})
        results = index.search([0.1, 0.2, ...], top_k=10)
        index.delete("kp-abc")
    """

    def __init__(self, host: str, port: int):
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorIndex initialised (host=%s, port=%d, collection=%s)",
            host,
            port,
            _COLLECTION_NAME,
        )

    def upsert(
        self, kp_id: str, embedding: list[float], metadata: dict | None = None
    ) -> None:
        """Insert or update an embedding for a knowledge point."""
        try:
            self._collection.upsert(
                ids=[kp_id],
                embeddings=[embedding],
                metadatas=[metadata or {}],
            )
        except Exception:
            logger.exception("VectorIndex upsert failed for id=%s", kp_id)
            raise

    def search(
        self, embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """Search the index for the nearest neighbours of the given embedding.

        Returns a list of dicts with keys ``id``, ``distance``, and ``metadata``.
        """
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
        except Exception:
            logger.exception("VectorIndex search failed")
            raise

        output: list[dict] = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for i in range(len(ids)):
            output.append(
                {
                    "id": ids[i],
                    "distance": distances[i] if distances else None,
                    "metadata": metadatas[i] if metadatas else {},
                }
            )
        return output

    def delete(self, kp_id: str) -> None:
        """Remove an embedding entry by knowledge point id."""
        try:
            self._collection.delete(ids=[kp_id])
        except Exception:
            logger.exception("VectorIndex delete failed for id=%s", kp_id)
            raise
