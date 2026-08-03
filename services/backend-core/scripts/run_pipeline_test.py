"""End-to-End Pipeline Test: Upload → Parse → Index → Retrieve → Answer.

Tests the full resource ingestion pipeline:
1. Create a text file with known content about a specific topic
2. Upload via DocumentParser (simulating user file upload)
3. Verify it's parsed and indexed into ChromaDB
4. Search for related queries and confirm uploaded content is retrieved
5. Verify the MentorAgent can use the uploaded content in answers

This validates the complete "user uploads file → system indexes → RAG retrieves" flow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, "/app")


class PipelineTest:
    """End-to-end pipeline validation test."""

    # Test topics: each has file content + related queries + expected KPs
    TEST_TOPICS = [
        {
            "name": "哈希表冲突解决",
            "filename": "hash_table_collision.txt",
            "content": """哈希表冲突解决详解

哈希表（Hash Table）是一种通过哈希函数将键映射到存储位置的数据结构。
当两个不同的键映射到同一个位置时，就发生了冲突（Collision）。
常见的冲突解决方法有：

1. 链地址法（Separate Chaining）：每个位置维护一个链表，冲突的元素
   链入同一个链表中。查找时需要遍历链表。

2. 开放地址法（Open Addressing）：当冲突发生时，按某种探查序列寻找
   下一个空闲位置。常见的探查方式有线性探查、二次探查和双重哈希。

3. 再哈希法（Rehashing）：准备多个哈希函数，当第一个哈希函数发生
   冲突时，使用第二个哈希函数，依此类推。

4. 公共溢出区法：将冲突的元素统一放到一个公共溢出区域。

链地址法是最常用的冲突解决方法，Java的HashMap和Python的dict都
采用链地址法（或优化的变体）来解决哈希冲突。
""",
            "queries": [
                "哈希表是如何解决冲突的？",
                "链地址法和开放地址法有什么区别？",
                "哈希冲突的常见解决方法有哪些？",
            ],
        },
        {
            "name": "快速排序算法详解",
            "filename": "quick_sort.txt",
            "content": """快速排序（Quick Sort）详解

快速排序是一种高效的排序算法，采用分治（Divide and Conquer）策略。

基本思想：
1. 从数组中选出一个元素作为基准（pivot）
2. 将数组分成两部分：左边都小于基准，右边都大于基准（partition）
3. 递归地对左右两部分进行快速排序

时间复杂度：
- 平均情况：O(n log n)
- 最坏情况：O(n²)（当数组已经有序且每次都选最值作为基准时）
- 最好情况：O(n log n)

空间复杂度：O(log n)（递归调用栈的深度）

优化策略：
- 三数取中法选择基准，避免最坏情况
- 当子数组长度小于某个阈值时，改用插入排序
- 三路快排，处理大量重复元素的情况

快速排序是不稳定的排序算法。
""",
            "queries": [
                "快速排序的时间复杂度是多少？",
                "快速排序是如何实现的？",
                "快速排序的优化策略有哪些？",
            ],
        },
    ]

    def __init__(self, user_id: str = "test-pipeline-user") -> None:
        self._user_id = user_id
        self._rag_service = None
        self._mentor_agent = None
        self._vector_index = None
        self._uploaded_files: list[str] = []

    async def setup(self) -> None:
        """Initialize RAG service + VectorIndex (no upload API needed)."""
        from src.kg.connection import Neo4jPool
        from src.rag.rag_service import RAGRetrievalService
        from src.kg.vector_index import VectorIndex
        from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
        from src.agents.mentor.agent import MentorAgent
        from src.utils.llm_adapter import create_llm
        from src.config import settings

        pool = Neo4jPool(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
        )
        kp_repo = KnowledgePointRepository(pool)
        self._vector_index = VectorIndex(
            host=getattr(settings, "chroma_host", "chromadb"),
            port=getattr(settings, "chroma_port", 8000),
        )
        self._rag_service = RAGRetrievalService(
            vector_index=self._vector_index,
            kp_repo=kp_repo,
        )

        llm = create_llm(settings)
        self._mentor_agent = MentorAgent(
            rag_service=self._rag_service,
            llm_adapter=llm,
        )

        logger.info("Pipeline test services initialized (Neo4j pool kept open)")

    # ── Step 1: Upload & Index ─────────────────────────────────

    async def upload_and_index(self, topic: dict) -> dict:
        """Simulate file upload: parse text, chunk, and index into ChromaDB."""
        from src.resources.parser import DocumentParser
        from sentence_transformers import SentenceTransformer

        # Create temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(topic["content"])
        tmp_path = tmp.name
        tmp.close()

        # Parse the document
        parser = DocumentParser()
        t0 = time.time()
        result = await parser.parse_and_index(
            file_path=Path(tmp_path),
            resource_id=topic["name"],
            resource_name=topic["name"],
            vector_index=self._vector_index,
            kp_id=topic["name"].split(" ")[0],
        )
        parse_time = time.time() - t0
        os.unlink(tmp_path)

        logger.info("  Parsed '%s': %d chunks, %d chars, %.1fs",
                    topic["name"], result.get("chunks", 0),
                    result.get("total_chars", 0), parse_time)

        # Verify in ChromaDB via direct query
        chroma_count = 0
        try:
            model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            q_emb = model.encode([topic["queries"][0]], show_progress_bar=False)[0]
            results = self._vector_index.search(q_emb.tolist(), top_k=10)

            # Check for resource_chunks collection separately
            if self._vector_index._client is not None:
                try:
                    rc = self._vector_index._client.get_collection("resource_chunks")
                    chroma_count = rc.count()
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("  ChromaDB verify warning: %s", exc)

        return {
            "topic": topic["name"],
            "chunks_parsed": result.get("chunks", 0),
            "chars_indexed": result.get("total_chars", 0),
            "chromadb_count": chroma_count,
            "parse_time_s": round(parse_time, 2),
            "indexed": result.get("indexed", 0) > 0,
        }

    # ── Step 2: Retrieve ───────────────────────────────────────

    async def test_retrieval(self, topic: dict) -> dict:
        """Verify that uploaded content is retrievable via RAG search."""
        results = []
        for query in topic["queries"]:
            t0 = time.time()
            search_results = await self._rag_service.search(query, top_k=10)
            latency = (time.time() - t0) * 1000

            # Check if our uploaded content appears
            found_uploaded = any(
                r.source_type == "chroma" and topic["name"] in r.source_name
                for r in search_results
            )

            results.append({
                "query": query,
                "found_uploaded_content": found_uploaded,
                "top_results": [{"name": r.source_name, "type": r.source_type, "score": round(r.score, 3)}
                                for r in search_results[:5]],
                "latency_ms": round(latency, 2),
            })

        n_found = sum(1 for r in results if r["found_uploaded_content"])
        return {
            "topic": topic["name"],
            "queries_tested": len(topic["queries"]),
            "queries_with_uploaded_content": n_found,
            "retrieval_success_rate": round(n_found / max(len(topic["queries"]), 1) * 100, 1),
            "per_query": results,
        }

    # ── Step 3: Generate Answer ────────────────────────────────

    async def test_answer_generation(self, topic: dict) -> dict:
        """Verify the MentorAgent can answer using uploaded content."""
        from src.rag.evaluation.metrics import faithfulness, answer_relevancy

        results = []
        for query in topic["queries"]:
            # Get mentor answer
            answer_parts = []
            async for event in self._mentor_agent.answer_stream(
                query=query, user_id=self._user_id,
            ):
                if event["type"] == "token":
                    answer_parts.append(event["data"].get("content", ""))

            answer = "".join(answer_parts)

            # Evaluate
            f_result = faithfulness(answer, ["哈希表", "冲突解决", "链地址法", "快速排序", "分治"])
            relevancy = answer_relevancy(answer, query)

            results.append({
                "query": query,
                "answer_length": len(answer),
                "faithfulness": f_result["faithfulness"],
                "answer_relevancy": relevancy,
            })

        avg_f = round(sum(r["faithfulness"] for r in results) / len(results), 4)
        avg_r = round(sum(r["answer_relevancy"] for r in results) / len(results), 4)
        return {
            "topic": topic["name"],
            "avg_faithfulness": avg_f,
            "avg_relevancy": avg_r,
            "per_query": results,
        }

    # ── Report ─────────────────────────────────────────────────

    def print_report(
        self, upload_results: list[dict],
        retrieval_results: list[dict],
        generation_results: list[dict],
    ) -> None:
        sep = "=" * 60
        print(f"\n{sep}")
        print("  端到端管线评测报告: 上传 → 解析 → 索引 → 检索 → 回答")
        print(f"{sep}")

        # Upload & Index
        print(f"\n📤 上传 & 解析 & 索引")
        for u in upload_results:
            status = "✅" if u["indexed"] else "❌"
            print(f"  {status} {u['topic']}: {u['chunks_parsed']} chunks, "
                  f"{u['chars_indexed']} chars, {u['parse_time_s']}s")
            if u.get("chromadb_count", 0) > 0:
                print(f"     ChromaDB resource_chunks count: {u['chromadb_count']}")

        # Retrieval
        print(f"\n🔍 检索验证")
        for r in retrieval_results:
            rate = r["retrieval_success_rate"]
            status = "✅" if rate >= 50 else "⚠️" if rate > 0 else "❌"
            print(f"  {status} {r['topic']}: {r['queries_with_uploaded_content']}/"
                  f"{r['queries_tested']} queries found uploaded content ({rate}%)")
            for pq in r["per_query"]:
                icon = "✅" if pq["found_uploaded_content"] else "❌"
                print(f"    {icon} \"{pq['query'][:30]}...\" "
                      f"({pq['latency_ms']}ms)")

        # Generation
        if generation_results:
            print(f"\n🤖 生成质量")
            for g in generation_results:
                print(f"  Faithfulness: {g['avg_faithfulness']}, "
                      f"Relevancy: {g['avg_relevancy']}")

        print(f"\n{'─'*60}")
        all_uploaded = all(u["indexed"] for u in upload_results)
        all_retrieved = all(
            r["queries_with_uploaded_content"] > 0 for r in retrieval_results
        )
        if all_uploaded and all_retrieved:
            print("  ✅ 端到端管线: 全部通过")
        else:
            print("  ⚠️  端到端管线: 部分通过")
        print(f"{sep}\n")


async def main():
    t_start = time.time()
    test = PipelineTest()
    await test.setup()

    try:
        from src.prompts import PromptRegistry
        PromptRegistry.load()

        upload_results = []
        retrieval_results = []
        generation_results = []

        for topic in PipelineTest.TEST_TOPICS:
            logger.info("Testing topic: %s", topic["name"])
            upload_results.append(await test.upload_and_index(topic))
            retrieval_results.append(await test.test_retrieval(topic))
            generation_results.append(await test.test_answer_generation(topic))

        test.print_report(upload_results, retrieval_results, generation_results)
        logger.info("Pipeline test completed in %.1fs", time.time() - t_start)

    finally:
        pass


if __name__ == "__main__":
    asyncio.run(main())
