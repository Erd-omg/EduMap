#!/usr/bin/env python
"""把 Neo4j 中的知识点（KnowledgePoint）编码后灌入 ChromaDB kp_embeddings 集合。

用途：向量检索依赖 kp_embeddings 集合有数据；本脚本保证评测/演示环境可用。

用法:
    .venv/bin/python scripts/seed_kg_embeddings.py [--dry-run]

注意：脚本会在运行前为本机地址设置 no_proxy（绕过 macOS 系统代理劫持 localhost）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "services" / "backend-core"))


def _bypass_system_proxy_for_local() -> None:
    """macOS 上 httpx/requests 会读取系统代理（如 Clash），劫持 localhost 请求返回 502。"""
    bypass_hosts = {"localhost", "127.0.0.1", "::1"}
    existing = os.environ.get("no_proxy", "")
    merged = ",".join(sorted({*filter(None, existing.split(",")), *bypass_hosts}))
    os.environ["no_proxy"] = merged
    os.environ["NO_PROXY"] = merged


# 强制使用本地缓存的模型，避免联网检查更新时卡死。
# 若需在线下载新模型：HF_HUB_OFFLINE=0 且 HF_ENDPOINT=https://hf-mirror.com（国内镜像，
# huggingface.co 直连被墙；本机 FlClash 未开 HTTP 代理端口时也无法走代理）。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


async def seed(dry_run: bool = False) -> int:
    from src.config import settings
    from src.kg.connection import Neo4jPool
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.vector_index import VectorIndex

    pool = Neo4jPool(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    kp_repo = KnowledgePointRepository(pool)

    try:
        kps = await kp_repo.list_all()
    except Exception as exc:
        print(f"[ERROR] 读取 Neo4j 知识点失败: {exc}")
        await pool.close()
        return 1

    if not kps:
        print("[WARN] Neo4j 中没有知识点，先运行 KG 种子脚本")
        await pool.close()
        return 1

    print(f"Neo4j 共 {len(kps)} 个知识点")

    if dry_run:
        for kp in kps[:10]:
            print(f"  {kp.id}: {kp.name}")
        await pool.close()
        return 0

    from sentence_transformers import SentenceTransformer

    model_name = settings.llm_embedding_model
    print(f"加载 embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)

    vi = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
    texts = [f"{kp.name}: {kp.description}" for kp in kps]
    embeddings = model.encode(texts, show_progress_bar=False)

    for kp, emb in zip(kps, embeddings):
        vi.upsert(
            kp.id,
            emb.tolist(),
            metadata={
                "name": kp.name,
                "description": kp.description,
                "category": kp.category,
                "difficulty": kp.difficulty,
            },
        )

    count = vi.collection_size()
    print(f"完成：kp_embeddings 集合现有 {count} 条向量")
    await pool.close()
    return 0 if count >= len(kps) else 1


if __name__ == "__main__":
    _bypass_system_proxy_for_local()
    parser = argparse.ArgumentParser(description="Seed ChromaDB kp_embeddings from Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="只列出知识点，不写入")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(seed(dry_run=args.dry_run)))
