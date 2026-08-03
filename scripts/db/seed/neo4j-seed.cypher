// EduMap Neo4j Seed Data
// Sample course: "数据结构与算法" (Data Structures & Algorithms)
// Run: cypher-shell -u neo4j -p edumap_dev -f scripts/db/seed/neo4j-seed.cypher

// ─── Create Course ─────────────────────────────────────────────
MERGE (c:Course {id: 'cs201', name: '数据结构与算法', description: '计算机科学核心课程'})
SET c.difficulty = 2;

// ─── Create Knowledge Points ───────────────────────────────────
MERGE (k1:KnowledgePoint {id: 'kp-intro', name: '算法基础', description: '算法的定义、特性与复杂度分析'})
SET k1.difficulty = 1, k1.category = 'foundation';

MERGE (k2:KnowledgePoint {id: 'kp-array', name: '数组', description: '数组的定义、存储与基本操作'})
SET k2.difficulty = 1, k2.category = 'linear';

MERGE (k3:KnowledgePoint {id: 'kp-linkedlist', name: '链表', description: '单向、双向链表及其操作'})
SET k3.difficulty = 2, k3.category = 'linear';

MERGE (k4:KnowledgePoint {id: 'kp-stack', name: '栈', description: '后进先出数据结构'})
SET k4.difficulty = 2, k4.category = 'linear';

MERGE (k5:KnowledgePoint {id: 'kp-queue', name: '队列', description: '先进先出数据结构'})
SET k5.difficulty = 2, k5.category = 'linear';

MERGE (k6:KnowledgePoint {id: 'kp-tree', name: '树', description: '二叉树、遍历与平衡树'})
SET k6.difficulty = 3, k6.category = 'nonlinear';

MERGE (k7:KnowledgePoint {id: 'kp-heap', name: '堆', description: '最大堆与最小堆'})
SET k7.difficulty = 3, k7.category = 'nonlinear';

MERGE (k8:KnowledgePoint {id: 'kp-graph', name: '图', description: '图的表示、遍历与最短路径'})
SET k8.difficulty = 4, k8.category = 'nonlinear';

MERGE (k9:KnowledgePoint {id: 'kp-sort', name: '排序算法', description: '冒泡、快排、归并等'})
SET k9.difficulty = 3, k9.category = 'algorithm';

MERGE (k10:KnowledgePoint {id: 'kp-search', name: '查找算法', description: '二分查找、哈希查找'})
SET k10.difficulty = 2, k10.category = 'algorithm';

MERGE (k11:KnowledgePoint {id: 'kp-dp', name: '动态规划', description: '最优子结构与状态转移'})
SET k11.difficulty = 5, k11.category = 'algorithm';

MERGE (k12:KnowledgePoint {id: 'kp-complexity', name: '复杂度分析', description: '时间与空间复杂度的大 O 表示法'})
SET k12.difficulty = 2, k12.category = 'foundation';

// ─── Additional KPs for RAG evaluation dataset coverage ────────────

MERGE (k13:KnowledgePoint {id: 'kp-recursion', name: '递归', description: '递归的定义、递归调用栈与分治思想'})
SET k13.difficulty = 2, k13.category = 'foundation';

MERGE (k14:KnowledgePoint {id: 'kp-bst', name: '二叉搜索树', description: '二叉搜索树的定义、查找、插入与删除操作'})
SET k14.difficulty = 3, k14.category = 'nonlinear';

MERGE (k15:KnowledgePoint {id: 'kp-sort-bubble', name: '冒泡排序', description: '相邻元素两两比较的排序算法'})
SET k15.difficulty = 2, k15.category = 'algorithm';

MERGE (k16:KnowledgePoint {id: 'kp-sort-selection', name: '选择排序', description: '每次选择最小/最大元素放到正确位置的排序算法'})
SET k16.difficulty = 2, k16.category = 'algorithm';

MERGE (k17:KnowledgePoint {id: 'kp-sort-quick', name: '快速排序', description: '基于分治和基准元素的排序算法'})
SET k17.difficulty = 3, k17.category = 'algorithm';

MERGE (k18:KnowledgePoint {id: 'kp-sort-merge', name: '归并排序', description: '将序列分成两半分别排序再合并的有序排序算法'})
SET k18.difficulty = 3, k18.category = 'algorithm';

MERGE (k19:KnowledgePoint {id: 'kp-hash', name: '哈希表', description: '通过哈希函数实现 O(1) 平均查找时间的数据结构'})
SET k19.difficulty = 3, k19.category = 'algorithm';

MERGE (k20:KnowledgePoint {id: 'kp-search-binary', name: '二分查找', description: '在有序数组中通过折半查找目标元素的算法'})
SET k20.difficulty = 2, k20.category = 'algorithm';

MERGE (k21:KnowledgePoint {id: 'kp-string', name: '字符串', description: '字符串的基本操作与模式匹配基础'})
SET k21.difficulty = 1, k21.category = 'foundation';

MERGE (k22:KnowledgePoint {id: 'kp-search-kmp', name: 'KMP 算法', description: '利用部分匹配表避免主串回溯的高效字符串匹配算法'})
SET k22.difficulty = 5, k22.category = 'algorithm';

// ─── Create Relationships for New KPs ─────────────────────────────
// Course→KP links
MATCH (c:Course {id: 'cs201'}), (k13:KnowledgePoint {id: 'kp-recursion'}) MERGE (c)-[:HAS_TOPIC]->(k13);
MATCH (c:Course {id: 'cs201'}), (k14:KnowledgePoint {id: 'kp-bst'}) MERGE (c)-[:HAS_TOPIC]->(k14);
MATCH (c:Course {id: 'cs201'}), (k15:KnowledgePoint {id: 'kp-sort-bubble'}) MERGE (c)-[:HAS_TOPIC]->(k15);
MATCH (c:Course {id: 'cs201'}), (k16:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (c)-[:HAS_TOPIC]->(k16);
MATCH (c:Course {id: 'cs201'}), (k17:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (c)-[:HAS_TOPIC]->(k17);
MATCH (c:Course {id: 'cs201'}), (k18:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (c)-[:HAS_TOPIC]->(k18);
MATCH (c:Course {id: 'cs201'}), (k19:KnowledgePoint {id: 'kp-hash'}) MERGE (c)-[:HAS_TOPIC]->(k19);
MATCH (c:Course {id: 'cs201'}), (k20:KnowledgePoint {id: 'kp-search-binary'}) MERGE (c)-[:HAS_TOPIC]->(k20);
MATCH (c:Course {id: 'cs201'}), (k21:KnowledgePoint {id: 'kp-string'}) MERGE (c)-[:HAS_TOPIC]->(k21);
MATCH (c:Course {id: 'cs201'}), (k22:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (c)-[:HAS_TOPIC]->(k22);

// Prerequisite relationships for new KPs
MATCH (k2:KnowledgePoint {id: 'kp-array'}), (k20:KnowledgePoint {id: 'kp-search-binary'}) MERGE (k2)-[:PREREQUISITE_OF]->(k20);
MATCH (k10:KnowledgePoint {id: 'kp-search'}), (k20:KnowledgePoint {id: 'kp-search-binary'}) MERGE (k10)-[:PREREQUISITE_OF]->(k20);
MATCH (k2:KnowledgePoint {id: 'kp-array'}), (k19:KnowledgePoint {id: 'kp-hash'}) MERGE (k2)-[:PREREQUISITE_OF]->(k19);
MATCH (k13:KnowledgePoint {id: 'kp-recursion'}), (k6:KnowledgePoint {id: 'kp-tree'}) MERGE (k13)-[:PREREQUISITE_OF]->(k6);
MATCH (k13:KnowledgePoint {id: 'kp-recursion'}), (k17:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (k13)-[:PREREQUISITE_OF]->(k17);
MATCH (k13:KnowledgePoint {id: 'kp-recursion'}), (k18:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (k13)-[:PREREQUISITE_OF]->(k18);
MATCH (k6:KnowledgePoint {id: 'kp-tree'}), (k14:KnowledgePoint {id: 'kp-bst'}) MERGE (k6)-[:PREREQUISITE_OF]->(k14);
MATCH (k9:KnowledgePoint {id: 'kp-sort'}), (k15:KnowledgePoint {id: 'kp-sort-bubble'}) MERGE (k9)-[:PREREQUISITE_OF]->(k15);
MATCH (k9:KnowledgePoint {id: 'kp-sort'}), (k16:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (k9)-[:PREREQUISITE_OF]->(k16);
MATCH (k9:KnowledgePoint {id: 'kp-sort'}), (k17:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (k9)-[:PREREQUISITE_OF]->(k17);
MATCH (k9:KnowledgePoint {id: 'kp-sort'}), (k18:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (k9)-[:PREREQUISITE_OF]->(k18);
MATCH (k21:KnowledgePoint {id: 'kp-string'}), (k22:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (k21)-[:PREREQUISITE_OF]->(k22);
MATCH (k10:KnowledgePoint {id: 'kp-search'}), (k22:KnowledgePoint {id: 'kp-search-kmp'}) MERGE (k10)-[:PREREQUISITE_OF]->(k22);

// Related relationships
MATCH (k14:KnowledgePoint {id: 'kp-bst'}), (k17:KnowledgePoint {id: 'kp-sort-quick'}) MERGE (k14)-[:RELATED_TO]->(k17);
MATCH (k14:KnowledgePoint {id: 'kp-bst'}), (k18:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (k14)-[:RELATED_TO]->(k18);
MATCH (k19:KnowledgePoint {id: 'kp-hash'}), (k20:KnowledgePoint {id: 'kp-search-binary'}) MERGE (k19)-[:RELATED_TO]->(k20);
MATCH (k15:KnowledgePoint {id: 'kp-sort-bubble'}), (k16:KnowledgePoint {id: 'kp-sort-selection'}) MERGE (k15)-[:RELATED_TO]->(k16);
MATCH (k17:KnowledgePoint {id: 'kp-sort-quick'}), (k18:KnowledgePoint {id: 'kp-sort-merge'}) MERGE (k17)-[:RELATED_TO]->(k18);

// ─── Create Relationships (Prerequisite Chain) ─────────────────
MATCH (c:Course {id: 'cs201'}), (k1:KnowledgePoint {id: 'kp-intro'}) MERGE (c)-[:HAS_TOPIC]->(k1);
MATCH (c:Course {id: 'cs201'}), (k2:KnowledgePoint {id: 'kp-array'}) MERGE (c)-[:HAS_TOPIC]->(k2);
MATCH (c:Course {id: 'cs201'}), (k3:KnowledgePoint {id: 'kp-linkedlist'}) MERGE (c)-[:HAS_TOPIC]->(k3);
MATCH (c:Course {id: 'cs201'}), (k4:KnowledgePoint {id: 'kp-stack'}) MERGE (c)-[:HAS_TOPIC]->(k4);
MATCH (c:Course {id: 'cs201'}), (k5:KnowledgePoint {id: 'kp-queue'}) MERGE (c)-[:HAS_TOPIC]->(k5);
MATCH (c:Course {id: 'cs201'}), (k6:KnowledgePoint {id: 'kp-tree'}) MERGE (c)-[:HAS_TOPIC]->(k6);
MATCH (c:Course {id: 'cs201'}), (k7:KnowledgePoint {id: 'kp-heap'}) MERGE (c)-[:HAS_TOPIC]->(k7);
MATCH (c:Course {id: 'cs201'}), (k8:KnowledgePoint {id: 'kp-graph'}) MERGE (c)-[:HAS_TOPIC]->(k8);
MATCH (c:Course {id: 'cs201'}), (k9:KnowledgePoint {id: 'kp-sort'}) MERGE (c)-[:HAS_TOPIC]->(k9);
MATCH (c:Course {id: 'cs201'}), (k10:KnowledgePoint {id: 'kp-search'}) MERGE (c)-[:HAS_TOPIC]->(k10);
MATCH (c:Course {id: 'cs201'}), (k11:KnowledgePoint {id: 'kp-dp'}) MERGE (c)-[:HAS_TOPIC]->(k11);
MATCH (c:Course {id: 'cs201'}), (k12:KnowledgePoint {id: 'kp-complexity'}) MERGE (c)-[:HAS_TOPIC]->(k12);

// Prerequisite relationships
MATCH (k2:KnowledgePoint {id: 'kp-array'}), (k3:KnowledgePoint {id: 'kp-linkedlist'}) MERGE (k2)-[:PREREQUISITE_OF]->(k3);
MATCH (k2:KnowledgePoint {id: 'kp-array'}), (k4:KnowledgePoint {id: 'kp-stack'}) MERGE (k2)-[:PREREQUISITE_OF]->(k4);
MATCH (k2:KnowledgePoint {id: 'kp-array'}), (k5:KnowledgePoint {id: 'kp-queue'}) MERGE (k2)-[:PREREQUISITE_OF]->(k5);
MATCH (k3:KnowledgePoint {id: 'kp-linkedlist'}), (k6:KnowledgePoint {id: 'kp-tree'}) MERGE (k3)-[:PREREQUISITE_OF]->(k6);
MATCH (k4:KnowledgePoint {id: 'kp-stack'}), (k6:KnowledgePoint {id: 'kp-tree'}) MERGE (k4)-[:PREREQUISITE_OF]->(k6);
MATCH (k5:KnowledgePoint {id: 'kp-queue'}), (k6:KnowledgePoint {id: 'kp-tree'}) MERGE (k5)-[:PREREQUISITE_OF]->(k6);
MATCH (k6:KnowledgePoint {id: 'kp-tree'}), (k7:KnowledgePoint {id: 'kp-heap'}) MERGE (k6)-[:PREREQUISITE_OF]->(k7);
MATCH (k6:KnowledgePoint {id: 'kp-tree'}), (k8:KnowledgePoint {id: 'kp-graph'}) MERGE (k6)-[:PREREQUISITE_OF]->(k8);
MATCH (k6:KnowledgePoint {id: 'kp-tree'}), (k9:KnowledgePoint {id: 'kp-sort'}) MERGE (k6)-[:PREREQUISITE_OF]->(k9);
MATCH (k1:KnowledgePoint {id: 'kp-intro'}), (k12:KnowledgePoint {id: 'kp-complexity'}) MERGE (k1)-[:PREREQUISITE_OF]->(k12);
MATCH (k1:KnowledgePoint {id: 'kp-intro'}), (k10:KnowledgePoint {id: 'kp-search'}) MERGE (k1)-[:PREREQUISITE_OF]->(k10);
MATCH (k10:KnowledgePoint {id: 'kp-search'}), (k11:KnowledgePoint {id: 'kp-dp'}) MERGE (k10)-[:PREREQUISITE_OF]->(k11);
MATCH (k9:KnowledgePoint {id: 'kp-sort'}), (k11:KnowledgePoint {id: 'kp-dp'}) MERGE (k9)-[:PREREQUISITE_OF]->(k11);

// Related relationships
MATCH (k10:KnowledgePoint {id: 'kp-search'}), (k12:KnowledgePoint {id: 'kp-complexity'}) MERGE (k10)-[:RELATED_TO]->(k12);
MATCH (k7:KnowledgePoint {id: 'kp-heap'}), (k9:KnowledgePoint {id: 'kp-sort'}) MERGE (k7)-[:RELATED_TO]->(k9);
MATCH (k3:KnowledgePoint {id: 'kp-linkedlist'}), (k2:KnowledgePoint {id: 'kp-array'}) MERGE (k3)-[:RELATED_TO]->(k2);

// ─── Add Indexes ──────────────────────────────────────────────
CREATE INDEX kp_name IF NOT EXISTS FOR (n:KnowledgePoint) ON (n.name);
CREATE INDEX kp_category IF NOT EXISTS FOR (n:KnowledgePoint) ON (n.category);

// ─── Uniqueness Constraint ───────────────────────────────────
CREATE CONSTRAINT kp_id IF NOT EXISTS FOR (n:KnowledgePoint) REQUIRE n.id IS UNIQUE;
