"""Tests for VectorIndex — in-memory fallback and mocked ChromaDB modes.

ChromaDB is installed but never reachable (no server), so the constructor
falls back to an in-memory dict store.  Tests verify upsert, search,
delete, and collection management against this fallback.  Additional
tests mock the ChromaDB client to exercise the ChromaDB code path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.kg.vector_index import VectorIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_index() -> VectorIndex:
    """Return a VectorIndex in in-memory fallback mode.

    ChromaDB is installed in the test env, so HttpClient will be called and
    will fail (no server).  The constructor falls back to in-memory.
    """
    return VectorIndex(host="127.0.0.1", port=1)  # unlikely port -> connection refused


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """VectorIndex initialization and mode detection."""

    def test_in_memory_fallback(self) -> None:
        """When ChromaDB connection fails, in-memory store is used."""
        idx = VectorIndex(host="127.0.0.1", port=1)
        assert idx._chroma_collection is None
        assert idx._client is None
        assert idx._memory_store == {}

    def test_in_memory_fallback_when_chromadb_missing(self) -> None:
        """When the chromadb package is absent, in-memory fallback is used."""
        with patch("src.kg.vector_index._HAS_CHROMADB", False):
            idx = VectorIndex()
        assert idx._chroma_collection is None
        assert idx._memory_store == {}

    def test_collection_name_default(self) -> None:
        """Collection name constant is 'kp_embeddings'."""
        from src.kg.vector_index import _COLLECTION_NAME
        assert _COLLECTION_NAME == "kp_embeddings"


# ---------------------------------------------------------------------------
# In-memory index operations
# ---------------------------------------------------------------------------

class TestInMemoryUpsert:
    """Upsert into in-memory store."""

    def test_upsert_stores_embedding(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:1", [0.1, 0.2, 0.3], {"name": "加法运算"})
        assert "kp:1" in in_memory_index._memory_store
        entry = in_memory_index._memory_store["kp:1"]
        assert entry["embedding"] == [0.1, 0.2, 0.3]
        assert entry["metadata"] == {"name": "加法运算"}

    def test_upsert_overwrites_existing(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:1", [0.1, 0.2], {"name": "original"})
        in_memory_index.upsert("kp:1", [0.9, 0.8], {"name": "updated"})
        entry = in_memory_index._memory_store["kp:1"]
        assert entry["embedding"] == [0.9, 0.8]
        assert entry["metadata"]["name"] == "updated"

    def test_upsert_without_metadata(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:2", [1.0, 0.0])
        assert in_memory_index._memory_store["kp:2"]["metadata"] == {}

    def test_multiple_upserts(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:1", [1.0, 0.0])
        in_memory_index.upsert("kp:2", [0.0, 1.0])
        in_memory_index.upsert("kp:3", [0.5, 0.5])
        assert in_memory_index.collection_size() == 3


class TestInMemorySearch:
    """Search against in-memory store with cosine similarity."""

    def test_search_returns_closest(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:1", [1.0, 0.0], {"name": "A"})
        in_memory_index.upsert("kp:2", [0.9, 0.1], {"name": "B"})
        in_memory_index.upsert("kp:3", [0.0, 1.0], {"name": "C"})

        results = in_memory_index.search([1.0, 0.0], top_k=2)
        assert len(results) == 2
        # kp:1 should be closest (identical vec → distance 0.0)
        assert results[0]["id"] == "kp:1"
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)
        assert results[0]["metadata"] == {"name": "A"}

    def test_search_top_k_limit(self, in_memory_index: VectorIndex) -> None:
        for i in range(10):
            in_memory_index.upsert(f"kp:{i}", [float(i) / 10, 0.0])

        results = in_memory_index.search([0.5, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_empty_store(self, in_memory_index: VectorIndex) -> None:
        results = in_memory_index.search([1.0, 0.0])
        assert results == []

    def test_search_returns_sorted_by_distance(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("far", [0.0, 1.0])
        in_memory_index.upsert("mid", [0.5, 0.5])
        in_memory_index.upsert("close", [0.9, 0.1])

        results = in_memory_index.search([1.0, 0.0])
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_search_dimension_mismatch_skipped(self, in_memory_index: VectorIndex) -> None:
        """Entries with different embedding dimensionality are skipped silently."""
        in_memory_index.upsert("kp:2d", [0.5, 0.5])  # 2D
        in_memory_index.upsert("kp:3d", [0.5, 0.5, 0.5])  # 3D

        results = in_memory_index.search([1.0, 0.0])  # query is 2D
        # Only kp:2d (same dimensionality) should be searchable
        assert len(results) == 1
        assert results[0]["id"] == "kp:2d"


class TestInMemoryDelete:
    """Delete operations against in-memory store."""

    def test_delete_removes_entry(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("kp:1", [1.0, 0.0])
        assert in_memory_index.collection_size() == 1

        in_memory_index.delete("kp:1")
        assert in_memory_index.collection_size() == 0
        assert "kp:1" not in in_memory_index._memory_store

    def test_delete_nonexistent_id(self, in_memory_index: VectorIndex) -> None:
        """Deleting a non-existent id does not raise."""
        in_memory_index.delete("nonexistent")  # should not raise

    def test_delete_from_empty_store(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.delete("anything")  # should not raise

    def test_collection_size_after_operations(self, in_memory_index: VectorIndex) -> None:
        in_memory_index.upsert("a", [1.0, 0.0])
        in_memory_index.upsert("b", [0.0, 1.0])
        assert in_memory_index.collection_size() == 2
        in_memory_index.delete("a")
        assert in_memory_index.collection_size() == 1
        in_memory_index.delete("b")
        assert in_memory_index.collection_size() == 0


# ---------------------------------------------------------------------------
# Mocked ChromaDB code path
# ---------------------------------------------------------------------------

class TestChromaDBMode:
    """Test VectorIndex when ChromaDB client is available (mocked)."""

    @pytest.fixture
    def mock_chroma_collection(self) -> MagicMock:
        collection = MagicMock()
        collection.count.return_value = 3
        collection.query.return_value = {
            "ids": [["kp:1", "kp:2"]],
            "distances": [[0.1, 0.5]],
            "metadatas": [[{"name": "A"}, {"name": "B"}]],
        }
        return collection

    @pytest.fixture
    def mock_client(self, mock_chroma_collection: MagicMock) -> MagicMock:
        client = MagicMock()
        client.get_or_create_collection.return_value = mock_chroma_collection
        return client

    def test_chromadb_upsert(self, mock_client: MagicMock, mock_chroma_collection: MagicMock) -> None:
        idx = VectorIndex(host="localhost", port=9999)
        idx._client = mock_client
        idx._chroma_collection = mock_chroma_collection

        idx.upsert("kp:1", [1.0, 0.0], {"name": "test"})
        mock_chroma_collection.upsert.assert_called_once_with(
            ids=["kp:1"],
            embeddings=[[1.0, 0.0]],
            metadatas=[{"name": "test"}],
        )

    def test_chromadb_search(self, mock_client: MagicMock, mock_chroma_collection: MagicMock) -> None:
        idx = VectorIndex(host="localhost", port=9999)
        idx._client = mock_client
        idx._chroma_collection = mock_chroma_collection

        results = idx.search([1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "kp:1"
        assert results[0]["distance"] == 0.1
        assert results[0]["metadata"] == {"name": "A"}
        mock_chroma_collection.query.assert_called_once()

    def test_chromadb_collection_size(
        self, mock_client: MagicMock, mock_chroma_collection: MagicMock,
    ) -> None:
        idx = VectorIndex(host="localhost", port=9999)
        idx._client = mock_client
        idx._chroma_collection = mock_chroma_collection

        assert idx.collection_size() == 3

    def test_chromadb_delete(self, mock_client: MagicMock, mock_chroma_collection: MagicMock) -> None:
        idx = VectorIndex(host="localhost", port=9999)
        idx._client = mock_client
        idx._chroma_collection = mock_chroma_collection

        idx.delete("kp:1")
        mock_chroma_collection.delete.assert_called_once_with(ids=["kp:1"])

    def test_chromadb_fallback_on_error(self, mock_client: MagicMock) -> None:
        """When ChromaDB upsert fails, fall back to in-memory store."""
        failing_collection = MagicMock()
        failing_collection.upsert.side_effect = Exception("ChromaDB error")
        mock_client.get_or_create_collection.return_value = failing_collection

        idx = VectorIndex(host="localhost", port=9999)
        idx._client = mock_client
        idx._chroma_collection = failing_collection

        idx.upsert("kp:1", [0.5, 0.5], {"name": "test"})
        # Should have fallen back to memory
        assert "kp:1" in idx._memory_store
        assert idx._memory_store["kp:1"]["embedding"] == [0.5, 0.5]
