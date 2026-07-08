// EduMap Neo4j Seed Data
// Sample course: "数据结构与算法" (Data Structures & Algorithms)
// Run: cypher-shell -u neo4j -p edumap_dev -f scripts/db/seed/neo4j-seed.cypher

// ─── Create Course ─────────────────────────────────────────────
MERGE (c:Course {id: 'cs201', name: '数据结构与算法', description: '计算机科学核心课程'})
SET c.difficulty = 2

// ─── Create Knowledge Points ───────────────────────────────────
MERGE (k1:KnowledgePoint {id: 'kp-intro', name: '算法基础', description: '算法的定义、特性与复杂度分析'})
SET k1.difficulty = 1, k1.category = 'foundation'

MERGE (k2:KnowledgePoint {id: 'kp-array', name: '数组', description: '数组的定义、存储与基本操作'})
SET k2.difficulty = 1, k2.category = 'linear'

MERGE (k3:KnowledgePoint {id: 'kp-linkedlist', name: '链表', description: '单向、双向链表及其操作'})
SET k3.difficulty = 2, k3.category = 'linear'

MERGE (k4:KnowledgePoint {id: 'kp-stack', name: '栈', description: '后进先出数据结构'})
SET k4.difficulty = 2, k4.category = 'linear'

MERGE (k5:KnowledgePoint {id: 'kp-queue', name: '队列', description: '先进先出数据结构'})
SET k5.difficulty = 2, k5.category = 'linear'

MERGE (k6:KnowledgePoint {id: 'kp-tree', name: '树', description: '二叉树、遍历与平衡树'})
SET k6.difficulty = 3, k6.category = 'nonlinear'

MERGE (k7:KnowledgePoint {id: 'kp-heap', name: '堆', description: '最大堆与最小堆'})
SET k7.difficulty = 3, k7.category = 'nonlinear'

MERGE (k8:KnowledgePoint {id: 'kp-graph', name: '图', description: '图的表示、遍历与最短路径'})
SET k8.difficulty = 4, k8.category = 'nonlinear'

MERGE (k9:KnowledgePoint {id: 'kp-sort', name: '排序算法', description: '冒泡、快排、归并等'})
SET k9.difficulty = 3, k9.category = 'algorithm'

MERGE (k10:KnowledgePoint {id: 'kp-search', name: '查找算法', description: '二分查找、哈希查找'})
SET k10.difficulty = 2, k10.category = 'algorithm'

MERGE (k11:KnowledgePoint {id: 'kp-dp', name: '动态规划', description: '最优子结构与状态转移'})
SET k11.difficulty = 5, k11.category = 'algorithm'

MERGE (k12:KnowledgePoint {id: 'kp-complexity', name: '复杂度分析', description: '时间与空间复杂度的大 O 表示法'})
SET k12.difficulty = 2, k12.category = 'foundation'

// ─── Create Relationships (Prerequisite Chain) ─────────────────
MERGE (c)-[:HAS_TOPIC]->(k1)
MERGE (c)-[:HAS_TOPIC]->(k2)
MERGE (c)-[:HAS_TOPIC]->(k3)
MERGE (c)-[:HAS_TOPIC]->(k4)
MERGE (c)-[:HAS_TOPIC]->(k5)
MERGE (c)-[:HAS_TOPIC]->(k6)
MERGE (c)-[:HAS_TOPIC]->(k7)
MERGE (c)-[:HAS_TOPIC]->(k8)
MERGE (c)-[:HAS_TOPIC]->(k9)
MERGE (c)-[:HAS_TOPIC]->(k10)
MERGE (c)-[:HAS_TOPIC]->(k11)
MERGE (c)-[:HAS_TOPIC]->(k12)

// Prerequisite relationships
MERGE (k2)-[:PREREQUISITE_OF]->(k3)
MERGE (k2)-[:PREREQUISITE_OF]->(k4)
MERGE (k2)-[:PREREQUISITE_OF]->(k5)
MERGE (k3)-[:PREREQUISITE_OF]->(k6)
MERGE (k4)-[:PREREQUISITE_OF]->(k6)
MERGE (k5)-[:PREREQUISITE_OF]->(k6)
MERGE (k6)-[:PREREQUISITE_OF]->(k7)
MERGE (k6)-[:PREREQUISITE_OF]->(k8)
MERGE (k6)-[:PREREQUISITE_OF]->(k9)
MERGE (k1)-[:PREREQUISITE_OF]->(k12)
MERGE (k1)-[:PREREQUISITE_OF]->(k10)
MERGE (k10)-[:PREREQUISITE_OF]->(k11)
MERGE (k9)-[:PREREQUISITE_OF]->(k11)

// Related relationships
MERGE (k10)-[:RELATED_TO]->(k12)
MERGE (k7)-[:RELATED_TO]->(k9)
MERGE (k3)-[:RELATED_TO]->(k2)

// ─── Add Indexes ──────────────────────────────────────────────
CREATE INDEX kp_name IF NOT EXISTS FOR (n:KnowledgePoint) ON (n.name)
CREATE INDEX kp_category IF NOT EXISTS FOR (n:KnowledgePoint) ON (n.category)
