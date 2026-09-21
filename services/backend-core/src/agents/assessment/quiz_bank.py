"""Pre-built quiz bank for the cs201 course (数据结构与算法).

Provides fallback questions when the LLM-based AssessmentAgent fails to
generate quiz questions.  Each knowledge point has 2-3 multiple-choice
questions with Chinese content.
"""

from __future__ import annotations

from src.agents.models import QuizQuestion

# ── Internal question data ──────────────────────────────────────────────

_QUESTIONS: dict[str, list[dict]] = {
    "kp-intro": [
        {
            "id": "q-intro-1",
            "type": "choice",
            "content": "数据结构主要研究什么内容？",
            "options": ["数据的逻辑结构和存储结构", "数据的加密和解密", "数据的传输和接收", "数据的显示和打印"],
            "correct_answer": "数据的逻辑结构和存储结构",
        },
        {
            "id": "q-intro-2",
            "type": "choice",
            "content": "算法必须具备以下哪个特性？",
            "options": ["有穷性", "无限循环", "随机执行", "无输入"],
            "correct_answer": "有穷性",
        },
        {
            "id": "q-intro-3",
            "type": "choice",
            "content": "下列哪个不是算法的基本控制结构？",
            "options": ["顺序结构", "选择结构", "循环结构", "跳转结构"],
            "correct_answer": "跳转结构",
        },
    ],
    "kp-array": [
        {
            "id": "q-array-1",
            "type": "choice",
            "content": "数组在内存中的存储方式是？",
            "options": ["连续存储", "链式存储", "散列存储", "索引存储"],
            "correct_answer": "连续存储",
        },
        {
            "id": "q-array-2",
            "type": "choice",
            "content": "已知数组 A[0..9]，访问 A[5] 的时间复杂度是？",
            "options": ["O(1)", "O(n)", "O(log n)", "O(n²)"],
            "correct_answer": "O(1)",
        },
    ],
    "kp-linkedlist": [
        {
            "id": "q-list-1",
            "type": "choice",
            "content": "单链表中，每个结点包含哪两部分？",
            "options": ["数据域和指针域", "数据域和索引域", "指针域和索引域", "数据域和散列域"],
            "correct_answer": "数据域和指针域",
        },
        {
            "id": "q-list-2",
            "type": "choice",
            "content": "在单链表中插入一个新结点的时间复杂度是（已知插入位置的前驱结点）？",
            "options": ["O(1)", "O(n)", "O(log n)", "O(n²)"],
            "correct_answer": "O(1)",
        },
        {
            "id": "q-list-3",
            "type": "choice",
            "content": "与数组相比，链表的主要优点是什么？",
            "options": ["插入和删除操作更高效", "随机访问更快", "占用的内存更少", "编程实现更简单"],
            "correct_answer": "插入和删除操作更高效",
        },
    ],
    "kp-stack": [
        {
            "id": "q-stack-1",
            "type": "choice",
            "content": "栈（Stack）的访问原则是？",
            "options": ["后进先出（LIFO）", "先进先出（FIFO）", "随机访问", "优先级访问"],
            "correct_answer": "后进先出（LIFO）",
        },
        {
            "id": "q-stack-2",
            "type": "choice",
            "content": "以下哪个应用场景最适合使用栈？",
            "options": ["函数调用和递归", "打印队列", "CPU 任务调度", "缓存淘汰"],
            "correct_answer": "函数调用和递归",
        },
    ],
    "kp-queue": [
        {
            "id": "q-queue-1",
            "type": "choice",
            "content": "队列（Queue）的访问原则是？",
            "options": ["先进先出（FIFO）", "后进先出（LIFO）", "随机访问", "优先级访问"],
            "correct_answer": "先进先出（FIFO）",
        },
        {
            "id": "q-queue-2",
            "type": "choice",
            "content": "循环队列的主要目的是什么？",
            "options": ["充分利用数组空间，避免假溢出", "提高队列的访问速度", "支持优先级排序", "实现双端操作"],
            "correct_answer": "充分利用数组空间，避免假溢出",
        },
    ],
    "kp-tree": [
        {
            "id": "q-tree-1",
            "type": "choice",
            "content": "二叉树的每个结点最多有几个子结点？",
            "options": ["2 个", "1 个", "3 个", "没有限制"],
            "correct_answer": "2 个",
        },
        {
            "id": "q-tree-2",
            "type": "choice",
            "content": "二叉树的先序遍历顺序是？",
            "options": ["根-左-右", "左-根-右", "左-右-根", "根-右-左"],
            "correct_answer": "根-左-右",
        },
        {
            "id": "q-tree-3",
            "type": "choice",
            "content": "在二叉搜索树中查找一个元素的时间复杂度（平均情况）是？",
            "options": ["O(log n)", "O(n)", "O(1)", "O(n²)"],
            "correct_answer": "O(log n)",
        },
    ],
    "kp-heap": [
        {
            "id": "q-heap-1",
            "type": "choice",
            "content": "大顶堆中，以下哪个关系一定成立？",
            "options": ["父结点的值 ≥ 子结点的值", "父结点的值 ≤ 子结点的值", "所有结点值相等", "叶结点值最小"],
            "correct_answer": "父结点的值 ≥ 子结点的值",
        },
        {
            "id": "q-heap-2",
            "type": "choice",
            "content": "堆排序的最好和最坏时间复杂度都是？",
            "options": ["O(n log n)", "O(n)", "O(n²)", "O(log n)"],
            "correct_answer": "O(n log n)",
        },
    ],
    "kp-graph": [
        {
            "id": "q-graph-1",
            "type": "choice",
            "content": "图的深度优先遍历（DFS）使用的数据结构是？",
            "options": ["栈", "队列", "数组", "链表"],
            "correct_answer": "栈",
        },
        {
            "id": "q-graph-2",
            "type": "choice",
            "content": "有 n 个顶点的无向完全图有多少条边？",
            "options": ["n(n-1)/2", "n(n-1)", "n²", "2n"],
            "correct_answer": "n(n-1)/2",
        },
        {
            "id": "q-graph-3",
            "type": "choice",
            "content": "Dijkstra 算法用于解决什么问题？",
            "options": ["单源最短路径", "最小生成树", "拓扑排序", "关键路径"],
            "correct_answer": "单源最短路径",
        },
    ],
    "kp-sort": [
        {
            "id": "q-sort-1",
            "type": "choice",
            "content": "快速排序的平均时间复杂度是？",
            "options": ["O(n log n)", "O(n)", "O(n²)", "O(log n)"],
            "correct_answer": "O(n log n)",
        },
        {
            "id": "q-sort-2",
            "type": "choice",
            "content": "下列排序算法中，哪一个是稳定的？",
            "options": ["归并排序", "快速排序", "堆排序", "选择排序"],
            "correct_answer": "归并排序",
        },
        {
            "id": "q-sort-3",
            "type": "choice",
            "content": "冒泡排序的最坏时间复杂度是？",
            "options": ["O(n²)", "O(n)", "O(n log n)", "O(log n)"],
            "correct_answer": "O(n²)",
        },
    ],
    "kp-search": [
        {
            "id": "q-search-1",
            "type": "choice",
            "content": "二分查找的前提条件是？",
            "options": ["数据有序排列", "数据无序排列", "数据为整数", "数据为字符串"],
            "correct_answer": "数据有序排列",
        },
        {
            "id": "q-search-2",
            "type": "choice",
            "content": "二分查找的时间复杂度是？",
            "options": ["O(log n)", "O(n)", "O(n log n)", "O(1)"],
            "correct_answer": "O(log n)",
        },
    ],
    "kp-dp": [
        {
            "id": "q-dp-1",
            "type": "choice",
            "content": "动态规划的核心思想是什么？",
            "options": ["将问题分解为重叠子问题并存储子问题的解", "随机搜索最优解", "使用贪心策略一步步选择", "枚举所有可能解"],
            "correct_answer": "将问题分解为重叠子问题并存储子问题的解",
        },
        {
            "id": "q-dp-2",
            "type": "choice",
            "content": "动态规划与分治法的最大区别是什么？",
            "options": ["子问题是否重叠", "问题规模是否相同", "是否需要递归", "是否有最优子结构"],
            "correct_answer": "子问题是否重叠",
        },
        {
            "id": "q-dp-3",
            "type": "choice",
            "content": "斐波那契数列使用动态规划后，时间复杂度从 O(2ⁿ) 降低到？",
            "options": ["O(n)", "O(n²)", "O(log n)", "O(1)"],
            "correct_answer": "O(n)",
        },
    ],
    "kp-complexity": [
        {
            "id": "q-cplx-1",
            "type": "choice",
            "content": "大 O 表示法描述的是什么？",
            "options": ["算法的时间增长率上界", "算法的准确执行时间", "算法的内存占用量", "算法的最优情况"],
            "correct_answer": "算法的时间增长率上界",
        },
        {
            "id": "q-cplx-2",
            "type": "choice",
            "content": "以下复杂度从低到高排列正确的是？",
            "options": [
                "O(1) < O(log n) < O(n) < O(n log n) < O(n²)",
                "O(1) < O(n) < O(log n) < O(n²) < O(n log n)",
                "O(n) < O(1) < O(log n) < O(n log n) < O(n²)",
                "O(log n) < O(1) < O(n) < O(n²) < O(n log n)",
            ],
            "correct_answer": "O(1) < O(log n) < O(n) < O(n log n) < O(n²)",
        },
    ],
}


def get_questions(kp_id: str) -> list[QuizQuestion]:
    """Return pre-built QuizQuestion list for a knowledge point, or empty list."""
    raw_list = _QUESTIONS.get(kp_id, [])
    questions: list[QuizQuestion] = []
    for q_data in raw_list:
        # Ensure correct_answer is in options
        if q_data["correct_answer"] not in q_data.get("options", []):
            others = [
                o for o in q_data.get("options", [])
                if o != q_data["correct_answer"]
            ]
            q_data["options"] = [q_data["correct_answer"]] + others

        questions.append(QuizQuestion(
            id=q_data["id"],
            type=q_data["type"],  # type: ignore[arg-type]
            content=q_data["content"],
            options=list(q_data.get("options", [])),
            correct_answer=q_data["correct_answer"],
            knowledge_point_id=kp_id,
        ))
    return questions


def get_all_kp_ids() -> list[str]:
    """Return all knowledge point IDs that have pre-built questions."""
    return list(_QUESTIONS.keys())
