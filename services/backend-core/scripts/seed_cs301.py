"""Seed a second course (cs301, 操作系统) so fusion can be tested off-corpus.

Why this exists
---------------
The fusion default in ``src/config.py`` (``rrf`` → ``score``) rests on an n=200
ablation run against **cs201 only** — the single course this repo seeded.  A
question raised in review is whether that advantage survives a corpus with
different characteristics, and the answer cannot be produced by re-running the
same benchmark: ``expanded_queries.json`` is cs201 by construction, and its
``relevant_kp_ids`` name knowledge points that only exist for cs201.

To measure anything, the second course must actually be **indexed** —
its KPs present in Neo4j and their embeddings in ChromaDB.  A query set whose
``relevant_kp_ids`` point at non-existent KPs would retrieve nothing and score
every fusion strategy 0.0, which is worse than no number at all because it
looks like a measurement.

This script seeds the graph half; ``seed_embeddings.py`` then indexes it.

Deliberately a different domain (operating systems, not data structures) so
the two corpora differ in vocabulary and in the shape of their queries — the
point is to vary the distribution, not just the wording.

Usage::

    cd services/backend-core
    python scripts/seed_cs301.py            # seed KPs + course + relations
    python scripts/seed_embeddings.py       # index them into ChromaDB

**Caveat on step 2:** ``seed_embeddings.py`` hardcodes compose-only hostnames
(``neo4j:7687``, ``chromadb``) and has no ``--course`` flag — it embeds *every*
KP in the graph.  Running it from the host therefore needs a small shim; the
command that actually worked here was::

    python -c "
    import asyncio
    from scripts.seed_embeddings import load_embeddings
    print(asyncio.run(load_embeddings(
        neo4j_uri='bolt://localhost:7687',
        chroma_host='localhost', chroma_port=8003)))"

Both courses are indexed by one run, which is what we want — but note that
``run_fusion_ablation.py`` reaches Chroma on ``settings.chroma_port`` (8003),
so indexing into a *different* Chroma (e.g. by running this inside compose
against the compose-internal instance) would produce an index the benchmark
cannot see, and every metric would silently read empty.
"""

from __future__ import annotations

import asyncio
import logging
import os

from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COURSE_ID = "cs301"
COURSE_NAME = "操作系统"

# (id, name, description, difficulty, category)
_KPS: list[tuple[str, str, str, int, str]] = [
    ("kp-os-intro", "操作系统概述", "操作系统的定义、功能与在计算机系统中的位置", 1, "foundation"),
    ("kp-os-process", "进程与线程", "进程的概念、状态转换、PCB，以及与线程的区别", 2, "core"),
    ("kp-os-schedule", "进程调度", "调度算法：先来先服务、短作业优先、时间片轮转、多级反馈队列", 3, "core"),
    ("kp-os-sync", "进程同步与互斥", "临界区、信号量、管程、生产者消费者问题", 3, "core"),
    ("kp-os-deadlock", "死锁", "死锁的四个必要条件、预防、避免（银行家算法）与检测", 4, "core"),
    ("kp-os-memory", "内存管理", "连续分配、分页、分段、段页式与地址转换", 3, "memory"),
    ("kp-os-vmemory", "虚拟内存", "请求分页、页面置换算法（FIFO/LRU/Clock）、缺页中断与抖动", 4, "memory"),
    ("kp-os-file", "文件系统", "文件的逻辑与物理结构、目录、索引节点与磁盘空间管理", 3, "storage"),
    ("kp-os-disk", "磁盘调度", "磁盘调度算法：FCFS、SSTF、SCAN、C-SCAN，以及磁盘寻道时间", 3, "storage"),
    ("kp-os-io", "I/O 管理", "I/O 控制方式：程序查询、中断、DMA、通道；缓冲与设备分配", 2, "io"),
]

# (from_id, to_id)
_PREREQS: list[tuple[str, str]] = [
    ("kp-os-intro", "kp-os-process"),
    ("kp-os-process", "kp-os-schedule"),
    ("kp-os-process", "kp-os-sync"),
    ("kp-os-sync", "kp-os-deadlock"),
    ("kp-os-intro", "kp-os-memory"),
    ("kp-os-memory", "kp-os-vmemory"),
    ("kp-os-memory", "kp-os-file"),
    ("kp-os-file", "kp-os-disk"),
    ("kp-os-intro", "kp-os-io"),
]

_RELATED: list[tuple[str, str]] = [
    ("kp-os-schedule", "kp-os-sync"),
    ("kp-os-vmemory", "kp-os-disk"),
    ("kp-os-file", "kp-os-io"),
    ("kp-os-deadlock", "kp-os-sync"),
]


def build_statements() -> list[str]:
    """Cypher for the whole course, idempotent via MERGE."""
    stmts: list[str] = [
        f"MERGE (c:Course {{id: '{COURSE_ID}'}}) SET c.name = '{COURSE_NAME}'",
    ]
    for kp_id, name, desc, diff, cat in _KPS:
        stmts.append(
            f"MERGE (k:KnowledgePoint {{id: '{kp_id}'}}) "
            f"SET k.name = '{name}', k.description = '{desc}', "
            f"k.difficulty = {diff}, k.category = '{cat}'"
        )
        stmts.append(
            f"MATCH (c:Course {{id: '{COURSE_ID}'}}), "
            f"(k:KnowledgePoint {{id: '{kp_id}'}}) MERGE (c)-[:HAS_TOPIC]->(k)"
        )
    for a, b in _PREREQS:
        stmts.append(
            f"MATCH (a:KnowledgePoint {{id: '{a}'}}), (b:KnowledgePoint {{id: '{b}'}}) "
            "MERGE (a)-[:PREREQUISITE_OF]->(b)"
        )
    for a, b in _RELATED:
        stmts.append(
            f"MATCH (a:KnowledgePoint {{id: '{a}'}}), (b:KnowledgePoint {{id: '{b}'}}) "
            "MERGE (a)-[:RELATED_TO]->(b)"
        )
    return stmts


async def main() -> int:
    # The pool host differs between host-run (localhost) and compose (neo4j);
    # an explicit env var keeps this usable from both.
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    driver = AsyncGraphDatabase.driver(
        uri, auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "edumap_dev")),
    )
    success = failed = 0
    try:
        for stmt in build_statements():
            try:
                async with driver.session() as session:
                    await session.run(stmt)
                success += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.warning("FAIL: %s ... (%s)", stmt[:70], exc)
    finally:
        await driver.close()
    logger.info(
        "Seeded %s (%s): %d statements ok, %d failed. "
        "Next: index it — see the 'Caveat on step 2' note in this module's "
        "docstring (seed_embeddings.py defaults to compose-only hostnames).",
        COURSE_ID, COURSE_NAME, success, failed,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
