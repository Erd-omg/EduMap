"""Semantic deduplication engine for knowledge points.

Uses sentence-transformers to find semantically similar knowledge points
and merges duplicates while preserving graph integrity.
"""

import logging

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.kg.models import DedupResult, KnowledgePoint

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.9


class SemanticDedupService:
    """Detect and merge semantically duplicate knowledge points.

    Args:
        kp_repo: KnowledgePointRepository instance.
        edge_repo: EdgeRepository instance.
        vector_index: VectorIndex instance (optional, for embedding storage).
    """

    def __init__(self, kp_repo, edge_repo, vector_index=None, embedding_model: str = "BAAI/bge-small-zh-v1.5"):
        self.kp_repo = kp_repo
        self.edge_repo = edge_repo
        self.vector_index = vector_index
        self._embedding_model = embedding_model
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self._embedding_model)
        return self.model

    async def find_duplicates(
        self, kp_id: str | None = None
    ) -> list[DedupResult]:
        """Scan all nodes (or a specific node) for semantic duplicates.

        Returns a list of DedupResult objects sorted by similarity descending.
        """
        if kp_id:
            target = await self.kp_repo.get(kp_id)
            if not target:
                logger.warning("find_duplicates: node %s not found", kp_id)
                return []
            candidates = await self.kp_repo.list_all()
            pairs = [(target, c) for c in candidates if c.id != kp_id]
        else:
            all_nodes = await self.kp_repo.list_all()
            pairs = self._generate_pairs(all_nodes)

        if not pairs:
            return []

        texts_a = [f"{a.name} {a.description}" for a, b in pairs]
        texts_b = [f"{b.name} {b.description}" for a, b in pairs]

        model = self._get_model()
        embeddings = model.encode(
            texts_a + texts_b, show_progress_bar=False
        )
        n = len(pairs)
        emb_a = embeddings[:n]
        emb_b = embeddings[n:]

        scores = cosine_similarity(emb_a, emb_b).diagonal()

        results: list[DedupResult] = []
        for i, (a, b) in enumerate(pairs):
            sim = float(scores[i])
            if sim >= _SIMILARITY_THRESHOLD:
                results.append(
                    DedupResult(
                        source_id=a.id,
                        target_id=b.id,
                        similarity=round(sim, 4),
                        merged=False,
                    )
                )

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    async def merge(
        self, source_id: str, target_id: str
    ) -> KnowledgePoint:
        """Merge *source* knowledge point into *target*.

        1. Retrieve both nodes.
        2. Append source.id to target.merged_from_ids.
        3. Re-point all PREREQUISITE_OF / RELATED_TO edges from source -> target.
        4. DETACH DELETE source.
        5. Return the updated target.

        Raises:
            ValueError: If source == target, or either node is not found.
        """
        if source_id == target_id:
            raise ValueError("Cannot merge a node into itself")

        source = await self.kp_repo.get(source_id)
        target = await self.kp_repo.get(target_id)

        if not source:
            raise ValueError(f"Source node not found: {source_id}")
        if not target:
            raise ValueError(f"Target node not found: {target_id}")

        # Warn when both non-canonical
        if not source.canonical and not target.canonical:
            logger.warning(
                "Merging two non-canonical nodes (%s -> %s)", source_id, target_id
            )

        # Update target metadata
        merged_from = list(target.merged_from_ids)
        if source_id not in merged_from:
            merged_from.append(source_id)
        merged_from.extend(
            sid for sid in source.merged_from_ids if sid not in merged_from
        )

        # Rewire incoming edges (point source's prerequisites to target)
        await self._rewire_incoming(source_id, target_id)

        # Rewire outgoing edges (point source's dependants to target)
        await self._rewire_outgoing(source_id, target_id)

        # Delete source node
        await self.kp_repo.delete(source_id)

        # Update target properties
        updated = await self.kp_repo.update(
            target_id,
            {
                "merged_from_ids": merged_from,
                "canonical": True,
            },
        )

        if not updated:
            raise RuntimeError(
                f"Failed to update target node after merge: {target_id}"
            )

        if self.vector_index:
            try:
                self.vector_index.delete(source_id)
            except Exception:
                logger.warning(
                    "Could not delete source embedding from vector index: %s",
                    source_id,
                )

        logger.info(
            "Merged %s into %s (merged_from=%s)",
            source_id,
            target_id,
            merged_from,
        )
        return updated

    def _generate_pairs(
        self, nodes: list[KnowledgePoint]
    ) -> list[tuple[KnowledgePoint, KnowledgePoint]]:
        """Generate all unique unordered pairs of nodes for comparison."""
        pairs: list[tuple[KnowledgePoint, KnowledgePoint]] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.append((nodes[i], nodes[j]))
        return pairs

    async def _rewire_incoming(
        self, source_id: str, target_id: str
    ) -> None:
        """Re-point incoming PREREQUISITE_OF edges from source to target.

        For every node X where X -[:PREREQUISITE_OF]-> source,
        create X -[:PREREQUISITE_OF]-> target.
        """
        query = """
        MATCH (x:KnowledgePoint)-[r:PREREQUISITE_OF]->(source:KnowledgePoint {id: $source_id})
        MERGE (x)-[:PREREQUISITE_OF]->(target:KnowledgePoint {id: $target_id})
        """
        await self.edge_repo.conn.execute_write(
            query,
            {"source_id": source_id, "target_id": target_id},
        )

    async def _rewire_outgoing(
        self, source_id: str, target_id: str
    ) -> None:
        """Re-point outgoing PREREQUISITE_OF and RELATED_TO edges from source to target.

        For every node Y where source -[r]-> Y,
        create target -[r]-> Y.
        """
        for rel_type in ("PREREQUISITE_OF", "RELATED_TO"):
            query = f"""
            MATCH (source:KnowledgePoint {{id: $source_id}})
                -[r:{rel_type}]->
                (y:KnowledgePoint)
            MERGE (target:KnowledgePoint {{id: $target_id}})
                -[:{rel_type}]->
                (y)
            """
            await self.edge_repo.conn.execute_write(
                query,
                {"source_id": source_id, "target_id": target_id},
            )
