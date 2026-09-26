# cs301 语料人工复核指南

> 适用文件：`services/backend-core/src/rag/evaluation/datasets/expanded_queries_cs301.json`
> 当前状态：`_meta.review_status == "unreviewed"`
> **复核完成前，任何引用该语料 MRR/召回的数字都不得用于决策。**

## 为什么必须复核

`relevant_kp_ids` 决定 MRR 的**分子**。标注错一条，MRR 就跟着动。
issue #3 的诉求正是"结论要有可复现的独立依据"——而未经复核的标注不是依据，
只是"看起来很权威"的数字，与本仓此前踩过的坑（如记录过 `recall > 1` 的旧
`ablation-fusion.json`）同源。

## 复核目标

对**每一条**查询，确认 `relevant_kp_ids` 是否**恰好**是最佳答案来源：

- **漏标**：某 KP 明显是正确答案的来源，却没列进去 → MRR 被低估
- **多标**：列了不该算相关的 KP → MRR 被高估
- **无关**：列了完全不相干的 KP

「恰好」是关键词。判定标准是"一个提问的学生，答案应该来自哪几个知识点"，
不是"哪些知识点与之沾边"。

## 参照：cs301 的 10 个知识点

| id | 名称 | 描述 |
|---|---|---|
| `kp-os-intro` | 操作系统概述 | 操作系统的定义、功能与在计算机系统中的位置 |
| `kp-os-process` | 进程与线程 | 进程的概念、状态转换、PCB，以及与线程的区别 |
| `kp-os-schedule` | 进程调度 | 先来先服务、短作业优先、时间片轮转、多级反馈队列 |
| `kp-os-sync` | 进程同步与互斥 | 临界区、信号量、管程、生产者消费者问题 |
| `kp-os-deadlock` | 死锁 | 四个必要条件、预防、避免（银行家算法）与检测 |
| `kp-os-memory` | 内存管理 | 连续分配、分页、分段、段页式与地址转换 |
| `kp-os-vmemory` | 虚拟内存 | 请求分页、页面置换算法（FIFO/LRU/Clock）、缺页中断与抖动 |
| `kp-os-file` | 文件系统 | 文件的逻辑与物理结构、目录、索引节点与磁盘空间管理 |
| `kp-os-disk` | 磁盘调度 | FCFS、SSTF、SCAN、C-SCAN，以及磁盘寻道时间 |
| `kp-os-io` | I/O 管理 | 程序控制、中断、DMA、通道；缓冲与设备分配 |

## 怎么做（工具已备好）

```bash
cd services/backend-core
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 生成工作表：40 条查询 × 声明标注 × 三种融合各自的 top-5 实际召回
python scripts/review_cs301_corpus.py > /tmp/cs301_review.json
# 摘要（含结语）走 stderr，JSON 走 stdout
```

输出每条含 `query` / `claimed` / `by_method`（三策略各自 top-5）/
`never_retrieved`（声明了但没被任何方法召回）/ `foreign_hits`（混入的 cs201 知识点）。

**逐条读**，对每条回答：

1. `claimed` 里的每个 KP，是否真的应该算相关？→ 不是则**删掉**
2. 还缺哪些 KP？→ **补上**
3. `by_method` 里出现的高排名 KP，是否其实也该算相关？（漏标最常见的形态）

改完后同步：`_meta.review_status` 改为 `reviewed`，填 `reviewed_by` / `reviewed_at`，
并在 `benchmark_results/README.md` 的表格里去掉 ⚠️ 标记。

## 已自动标出的可疑条目（复核时优先看）

### 1. 声明的 KP 从未被任何方法召回（1 条）

- **「死锁和饥饿有什么区别？」**
  声明 `[kp-os-deadlock, kp-os-sync]`，但 `kp-os-sync` 在三种方法的 top-5 中
  **一次都没出现**。需判断：是 `kp-os-sync` 不该算相关（删除），
  还是它确实相关而检索没找到（保留，说明检索有缺口）。

### 2. ⚠️ 跨课程污染：约 26–27/40 条查询的结果里混入 cs201 知识点

三种方法**都有**（rrf 26、minmax 27、score 26），不是某个策略的缺陷。
例如「什么是临界区？」的 rrf top-5 里出现 `kp-recursion`、`kp-dp`。

原因：ChromaDB 里两个课程的 KP 在**同一个 collection**，检索没有按课程过滤，
所以语义相近的 cs201 知识点（如「递归」之于「调用栈」）会被召回。

**这对复核意味着什么**：不影响 MRR（MRR 只看正确 KP 的排名，多召回不相关的
不影响分子），**但复核时不要被 top-5 里的 cs201 条目误导**——它们只是噪声。
若要让语料真正"课程隔离"，需在检索侧加按课程过滤，那是另一个改动。

## 复核后要做什么

1. 更新语料文件与 `_meta`（`review_status: "reviewed"`）
2. 重跑 `python scripts/run_fusion_ablation.py --dataset expanded --course cs301`
3. 用新数字**重新审视** `src/config.py` 里关于 cs301 的注释——若次序不再反转，
   那段话必须改（它现在说的是"次序会变"）
4. 更新 `benchmark_results/README.md` 的表格与复核状态
