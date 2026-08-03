"""Load KP embeddings from Neo4j into ChromaDB/VectorIndex.

Usage: python scripts/seed_embeddings.py

Reads all KnowledgePoints from Neo4j, generates embeddings via
sentence-transformers, and upserts them into the VectorIndex (ChromaDB).
"""
from __future__ import annotations

import asyncio
import logging
import time

from src.kg.connection import Neo4jPool
from src.kg.vector_index import VectorIndex
from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)


async def load_embeddings(
    neo4j_uri: str = "bolt://neo4j:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "edumap_dev",
    chroma_host: str = "chromadb",
    chroma_port: int = 8000,
) -> dict:
    """Generate and upsert embeddings for all KPs in Neo4j."""
    # ── Connect to Neo4j ────────────────────────────────────────────
    pool = Neo4jPool(neo4j_uri, neo4j_user, neo4j_password)
    kp_repo = KnowledgePointRepository(pool)

    # ── Load all KPs ────────────────────────────────────────────────
    kps = await kp_repo.list_all()
    if not kps:
        logger.warning("No KPs found in Neo4j — skipping embedding load")
        return {"status": "skipped", "reason": "no KPs", "count": 0}

    logger.info("Loaded %d KPs from Neo4j", len(kps))
    await pool.close()

    # ── Load embedding model ────────────────────────────────────────
    logger.info("Loading sentence-transformers model (BAAI/bge-small-zh-v1.5)...")
    t0 = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        logger.info("Model loaded in %.1fs", time.time() - t0)
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        return {"status": "error", "reason": str(exc), "count": 0}

    # ── Connect to VectorIndex ──────────────────────────────────────
    vi = VectorIndex(host=chroma_host, port=chroma_port)

    # ── Generate and upsert embeddings ──────────────────────────────
    texts = [
        f"{kp.name}: {kp.description} (难度: {kp.difficulty}, 分类: {kp.category})"
        for kp in kps
    ]
    ids = [kp.id for kp in kps]
    names = [kp.name for kp in kps]

    t1 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info("Embeddings generated in %.1fs for %d KPs", time.time() - t1, len(kps))

    upserted = 0
    for i, kp_id in enumerate(ids):
        vi.upsert(
            kp_id=kp_id,
            embedding=embeddings[i].tolist(),
            metadata={
                "name": names[i],
                "description": kps[i].description,
                "difficulty": str(kps[i].difficulty),
                "category": kps[i].category,
                "text": texts[i],
            },
        )
        upserted += 1
        if upserted % 10 == 0:
            logger.info("Upserted %d/%d embeddings", upserted, len(kps))

    size = vi.collection_size()
    logger.info(
        "Done — upserted %d/%d embeddings, collection size=%d",
        upserted, len(kps), size,
    )

    # Log the first few to verify
    if size > 0:
        logger.info("VectorIndex is healthy with %d embeddings", size)
    else:
        logger.warning("VectorIndex reports 0 items — check ChromaDB connection")

    return {
        "status": "completed",
        "kps_loaded": len(kps),
        "embeddings_upserted": upserted,
        "collection_size": size,
    }


async def main():
    logger.info("Starting embedding load...")
    result = await load_embeddings()
    logger.info("Result: %s", result)


if __name__ == "__main__":
    asyncio.run(main())
