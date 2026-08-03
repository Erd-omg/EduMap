"""Expand Neo4j seed data with additional KPs for RAG evaluation."""
import asyncio
import logging

from neo4j import AsyncGraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CYPHER_STATEMENTS = [
    # New KPs
    "MERGE (k:KnowledgePoint {id: 'kp-recursion', name: '递归', description: '递归的定义、递归调用栈与分治思想'}) SET k.difficulty=2, k.category='foundation'",
    "MERGE (k:KnowledgePoint {id: 'kp-bst', name: '二叉搜索树', description: '二叉搜索树的定义、查找、插入与删除操作'}) SET k.difficulty=3, k.category='nonlinear'",
    "MERGE (k:KnowledgePoint {id: 'kp-sort-bubble', name: '冒泡排序', description: '相邻元素两两比较的排序算法'}) SET k.difficulty=2, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-sort-selection', name: '选择排序', description: '每次选择最小/最大元素放到正确位置的排序算法'}) SET k.difficulty=2, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-sort-quick', name: '快速排序', description: '基于分治和基准元素的排序算法'}) SET k.difficulty=3, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-sort-merge', name: '归并排序', description: '将序列分成两半分别排序再合并的有序排序算法'}) SET k.difficulty=3, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-hash', name: '哈希表', description: '通过哈希函数实现 O(1) 平均查找时间的数据结构'}) SET k.difficulty=3, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-search-binary', name: '二分查找', description: '在有序数组中通过折半查找目标元素的算法'}) SET k.difficulty=2, k.category='algorithm'",
    "MERGE (k:KnowledgePoint {id: 'kp-string', name: '字符串', description: '字符串的基本操作与模式匹配基础'}) SET k.difficulty=1, k.category='foundation'",
    "MERGE (k:KnowledgePoint {id: 'kp-search-kmp', name: 'KMP 算法', description: '利用部分匹配表避免主串回溯的高效字符串匹配算法'}) SET k.difficulty=5, k.category='algorithm'",

    # Course links
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-recursion'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-bst'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-sort-bubble'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-hash'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-search-binary'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-string'}) MERGE (c)-[:HAS_TOPIC]->(k)",
    "MATCH (c:Course {id: 'cs201'}), (k:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (c)-[:HAS_TOPIC]->(k)",

    # Prerequisite relationships
    "MATCH (a:KnowledgePoint {id: 'kp-array'}), (b:KnowledgePoint {id: 'kp-search-binary'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-search'}), (b:KnowledgePoint {id: 'kp-search-binary'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-array'}), (b:KnowledgePoint {id: 'kp-hash'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-recursion'}), (b:KnowledgePoint {id: 'kp-tree'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-recursion'}), (b:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-recursion'}), (b:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-tree'}), (b:KnowledgePoint {id: 'kp-bst'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort'}), (b:KnowledgePoint {id: 'kp-sort-bubble'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort'}), (b:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort'}), (b:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort'}), (b:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-string'}), (b:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-search'}), (b:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (a)-[:PREREQUISITE_OF]->(b)",

    # Related relationships
    "MATCH (a:KnowledgePoint {id: 'kp-bst'}), (b:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (a)-[:RELATED_TO]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-bst'}), (b:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (a)-[:RELATED_TO]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-hash'}), (b:KnowledgePoint {id: 'kp-search-binary'}) MERGE (a)-[:RELATED_TO]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort-bubble'}), (b:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (a)-[:RELATED_TO]->(b)",
    "MATCH (a:KnowledgePoint {id: 'kp-sort-quick'}), (b:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (a)-[:RELATED_TO]->(b)",
]


async def main():
    driver = AsyncGraphDatabase.driver(
        "bolt://neo4j:7687",
        auth=("neo4j", "edumap_dev"),
    )
    success = 0
    failed = 0
    for stmt in CYPHER_STATEMENTS:
        try:
            async with driver.session() as session:
                await session.run(stmt)
            success += 1
            logger.info("OK: %s ...", stmt[:60])
        except Exception as e:
            failed += 1
            logger.warning("FAIL: %s ... (%s)", stmt[:60], e)
    await driver.close()
    logger.info("Done: %d succeeded, %d failed", success, failed)


if __name__ == "__main__":
    asyncio.run(main())
