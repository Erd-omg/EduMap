"""Tests for SemanticDedupService — pure logic + mocked dependencies.

Tests the pair generation algorithm, threshold comparison, and the
find_duplicates / merge flows with mocked repos and embedding model.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.kg.models import DedupResult, KnowledgePoint
from src.kg.semantic_dedup import SemanticDedupService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kp_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.list_all = AsyncMock(return_value=[])
    repo.get = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_edge_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def service(mock_kp_repo: AsyncMock, mock_edge_repo: AsyncMock) -> SemanticDedupService:
    return SemanticDedupService(kp_repo=mock_kp_repo, edge_repo=mock_edge_repo)


def make_kp(id_: str, name: str = "", description: str = "", difficulty: int = 1) -> KnowledgePoint:
    """Convenience factory for KnowledgePoint test instances."""
    return KnowledgePoint(
        id=id_,
        name=name or f"KP-{id_}",
        description=description or f"Description for {id_}",
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# _generate_pairs — pure logic, no side effects
# ---------------------------------------------------------------------------

class TestGeneratePairs:
    """Unit tests for _generate_pairs (static-like method)."""

    def test_empty_list(self, service: SemanticDedupService) -> None:
        assert service._generate_pairs([]) == []

    def test_single_node(self, service: SemanticDedupService) -> None:
        kp = make_kp("1")
        assert service._generate_pairs([kp]) == []

    def test_two_nodes(self, service: SemanticDedupService) -> None:
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        pairs = service._generate_pairs([kp1, kp2])
        assert len(pairs) == 1
        assert pairs[0] == (kp1, kp2)

    def test_three_nodes_yields_three_pairs(self, service: SemanticDedupService) -> None:
        nodes = [make_kp(str(i)) for i in range(3)]
        pairs = service._generate_pairs(nodes)
        assert len(pairs) == 3  # 3 choose 2 = 3

    def test_five_nodes_yields_ten_pairs(self, service: SemanticDedupService) -> None:
        nodes = [make_kp(str(i)) for i in range(5)]
        pairs = service._generate_pairs(nodes)
        assert len(pairs) == 10  # 5 choose 2 = 10

    def test_no_duplicate_pairs(self, service: SemanticDedupService) -> None:
        nodes = [make_kp(str(i)) for i in range(4)]
        pairs = service._generate_pairs(nodes)
        seen = set()
        for a, b in pairs:
            key = (a.id, b.id)
            assert key not in seen, f"Duplicate pair {key}"
            seen.add(key)


# ---------------------------------------------------------------------------
# find_duplicates — mocked model and repos
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    """Integration-style tests for find_duplicates with mocked model/repos."""

    @pytest.mark.asyncio
    async def test_no_candidates(self, service: SemanticDedupService) -> None:
        """When the repo has no nodes, the result is empty."""
        service.kp_repo.list_all.return_value = []
        with patch.object(service, "_get_model") as mock_get_model:
            results = await service.find_duplicates()
        assert results == []
        mock_get_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_node_no_pairs(self, service: SemanticDedupService) -> None:
        """One node in the graph generates zero pairs."""
        service.kp_repo.list_all.return_value = [make_kp("1")]
        with patch.object(service, "_get_model") as mock_get_model:
            results = await service.find_duplicates()
        assert results == []
        mock_get_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_similar_above_threshold(self, service: SemanticDedupService) -> None:
        """Pairs with cosine similarity >= 0.9 are reported as duplicates."""
        kp1 = make_kp("1", name="加法运算", description="基本加法")
        kp2 = make_kp("2", name="加法技巧", description="加法计算")
        kp3 = make_kp("3", name="减法运算", description="基本减法")
        service.kp_repo.list_all.return_value = [kp1, kp2, kp3]

        # Pairs: (1,2), (1,3), (2,3)
        #   text_a: [kp1, kp1, kp2]
        #   text_b: [kp2, kp3, kp3]
        # Want: cos(kp1,kp2)=0.95 >=0.9, cos(kp1,kp3)=0.5 <0.9, cos(kp2,kp3)=0.3 <0.9
        fake_embeddings = np.array([
            [1.0, 0.0],          # kp1_emb → texts_a[0]
            [1.0, 0.0],          # kp1_emb → texts_a[1]
            [0.95, 0.31225],     # kp2_emb → texts_a[2]
            [0.95, 0.31225],     # kp2_emb → texts_b[0] — cos(kp1,kp2) = 0.95
            [0.50, 0.86603],     # kp3_emb → texts_b[1] — cos(kp1,kp3) = 0.50
            [0.50, 0.86603],     # kp3_emb → texts_b[2] — cos(kp2,kp3) ≈ 0.745
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()

        # Only pair (kp1, kp2) should exceed the 0.9 threshold
        assert len(results) == 1
        assert results[0] == DedupResult(
            source_id="1", target_id="2",
            similarity=round(0.95, 4), merged=False,
        )

    @pytest.mark.asyncio
    async def test_results_sorted_by_similarity_descending(
        self, service: SemanticDedupService,
    ) -> None:
        """find_duplicates returns results sorted from most to least similar."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        kp3 = make_kp("3")
        service.kp_repo.list_all.return_value = [kp1, kp2, kp3]

        # Three pairs, all above 0.9 threshold with different sim values
        # cos(kp1,kp2)=0.99, cos(kp1,kp3)=0.95, cos(kp2,kp3)=0.92
        fake_embeddings = np.array([
            [1.0, 0.0],          # kp1
            [1.0, 0.0],          # kp1
            [0.99, 0.14107],     # kp2
            [0.99, 0.14107],     # kp2 → cos(kp1,kp2) = 0.99
            [0.95, 0.31225],     # kp3 → cos(kp1,kp3) = 0.95
            [0.92, 0.39192],     # kp3 → cos(kp2,kp3) = 0.92
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()

        assert len(results) == 3
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    @pytest.mark.asyncio
    async def test_find_duplicates_by_kp_id(
        self, service: SemanticDedupService,
    ) -> None:
        """When kp_id is specified, only pairs involving that node are checked."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        kp3 = make_kp("3")
        service.kp_repo.get = AsyncMock(return_value=kp1)
        service.kp_repo.list_all.return_value = [kp1, kp2, kp3]

        # Two pairs: (kp1,kp2), (kp1,kp3) — 4 texts total
        # Want cos(kp1,kp2) = 0.95 >= 0.9, cos(kp1,kp3) = 0.5 < 0.9
        angle_high = np.arccos(0.95)
        angle_low = np.arccos(0.5)
        fake_embeddings = np.array([
            [1.0, 0.0],              # kp1 (texts_a[0])
            [1.0, 0.0],              # kp1 (texts_a[1])
            [np.cos(angle_high), np.sin(angle_high)],  # kp2 (texts_b[0])
            [np.cos(angle_low), np.sin(angle_low)],    # kp3 (texts_b[1])
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates(kp_id="1")

        assert len(results) == 1
        assert results[0].source_id == "1"
        assert results[0].target_id == "2"

    @pytest.mark.asyncio
    async def test_kp_id_not_found_returns_empty(
        self, service: SemanticDedupService,
    ) -> None:
        """If kp_id does not exist, find_duplicates returns [].

        Does not call the model in this case.
        """
        service.kp_repo.get.return_value = None

        with patch.object(service, "_get_model") as mock_get_model:
            results = await service.find_duplicates(kp_id="nonexistent")

        assert results == []
        mock_get_model.assert_not_called()


# ---------------------------------------------------------------------------
# Threshold boundary tests
# ---------------------------------------------------------------------------

class TestSimilarityThreshold:
    """Boundary-value tests for the >= 0.9 similarity threshold."""

    @pytest.mark.asyncio
    async def test_exactly_threshold(self, service: SemanticDedupService) -> None:
        """Cosine similarity of exactly 0.9 qualifies (>=)."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        service.kp_repo.list_all.return_value = [kp1, kp2]

        # cos(kp1, kp2) = 0.9 (exact threshold)
        # Use np.arccos to get exact unit vectors at cos=0.9
        angle = np.arccos(0.9)
        fake_embeddings = np.array([
            [1.0, 0.0],                    # kp1 — unit vector
            [np.cos(angle), np.sin(angle)],  # kp2 — unit vector, cos = 0.9
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()
        assert len(results) == 1
        assert results[0].similarity == 0.9

    @pytest.mark.asyncio
    async def test_just_below_threshold(self, service: SemanticDedupService) -> None:
        """Cosine similarity of 0.8999 does not qualify."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        service.kp_repo.list_all.return_value = [kp1, kp2]

        # cos(kp1, kp2) ≈ 0.8999
        angle = np.arccos(0.8999)
        fake_embeddings = np.array([
            [1.0, 0.0],
            [np.cos(angle), np.sin(angle)],
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_max_similarity(self, service: SemanticDedupService) -> None:
        """Cosine similarity of 1.0 (identical vectors) qualifies."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        service.kp_repo.list_all.return_value = [kp1, kp2]

        fake_embeddings = np.array([
            [1.0, 2.0, 3.0],  # kp1
            [1.0, 2.0, 3.0],  # kp2 — identical → cos = 1.0
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()
        assert len(results) == 1
        assert results[0].similarity == 1.0

    @pytest.mark.asyncio
    async def test_negative_similarity(self, service: SemanticDedupService) -> None:
        """Negative cosine similarity does not qualify."""
        kp1 = make_kp("1")
        kp2 = make_kp("2")
        service.kp_repo.list_all.return_value = [kp1, kp2]

        # cos = -1.0 (opposite vectors)
        fake_embeddings = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float64)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings

        with patch.object(service, "_get_model", return_value=mock_model):
            results = await service.find_duplicates()
        assert len(results) == 0


# ---------------------------------------------------------------------------
# merge — input validation (pure logic, no DB)
# ---------------------------------------------------------------------------

class TestMergeValidation:
    """Test merge input validation without touching a real DB."""

    @pytest.mark.asyncio
    async def test_merge_self_raises(self, service: SemanticDedupService) -> None:
        """Merging a node into itself raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge a node into itself"):
            await service.merge("same_id", "same_id")

    @pytest.mark.asyncio
    async def test_merge_source_not_found(self, service: SemanticDedupService) -> None:
        """Merging a non-existent source node raises ValueError."""
        service.kp_repo.get = AsyncMock(side_effect=lambda x: make_kp("2") if x == "2" else None)
        with pytest.raises(ValueError, match="Source node not found"):
            await service.merge("1", "2")

    @pytest.mark.asyncio
    async def test_merge_target_not_found(self, service: SemanticDedupService) -> None:
        """Merging into a non-existent target node raises ValueError."""
        service.kp_repo.get = AsyncMock(side_effect=lambda x: make_kp("1") if x == "1" else None)
        with pytest.raises(ValueError, match="Target node not found"):
            await service.merge("1", "2")
