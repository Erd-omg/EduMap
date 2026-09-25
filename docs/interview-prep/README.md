# 面试准备手册 — EduMap 多智能体学习系统

面向 **AI Agent 开发 / 多智能体系统 / LLM 应用开发** 岗位的面试逐字稿。

每个亮点一份独立文档，采用统一结构：

1. **背景阐述**（业务场景、数据量级）
2. **问题剖析**（痛点、不解决会怎样）
3. **方案构思与技术实现**（选型对比 + 核心代码 + 兜底）
4. **优化效果与压测数据**（前后对比）
5. **复盘总结**（核心价值 + 踩坑）
6. **常见面试题与追问点**（含【面试官可能追问】应答策略）

## 文档索引

| # | 亮点 | 文件 | 核心数字 |
|---|------|------|----------|
| 1 | 多 Agent 编排（LangGraph StateGraph） | [01-multi-agent-orchestration.md](01-multi-agent-orchestration.md) | 延迟 -40%，P95 135s→82s，吞吐 39→65/h |
| 2 | 标准化 Agent 执行框架（Harness） | [02-agent-harness.md](02-agent-harness.md) | 样板代码 -60%，失败率 8%→0.3% |
| 3 | RAG 混合检索 + 语义分块 + 重排序 | [03-rag-hybrid-retrieval.md](03-rag-hybrid-retrieval.md) | 双路召回实测 MRR 0.841 / HR@3 0.930（n=200，**重排未启用**）；P50 19.4ms / P95 54.7ms |
| 4 | RAG 评测与优化闭环 | [04-rag-evaluation.md](04-rag-evaluation.md) | n=200：MRR 0.841 / HR@3 0.930 / P@1 0.755；n=20：MRR 0.900；生成侧 Faithfulness 词重叠 0.5407 / 语义 0.813 / Relevancy 0.8539（**已重测修正**）|
| 5 | 三层记忆架构 | [05-memory-system.md](05-memory-system.md) | token 开销 -60%，context 构建 P95 ~18ms |
| 6 | 工具调用 + 结构化输出双模式 | [06-tools-structured-output.md](06-tools-structured-output.md) | 解析成功率 85%→98%+，出题率 88%→97% |
| 7 | 竞品调研与改进路线图（对标分析） | [07-competitive-landscape.md](07-competitive-landscape.md) | 16 项改进按性价比排序；PNAS RCT 验证护栏设计；含来源分级 **[已验证]/[仅营销]/[未核实]** |

> **文档七的使用方式**：它不同于前六篇（讲"我做了什么"），而是讲**"我做了什么、业界做到哪、差距在哪"**。用于回答"你调研过竞品吗""为什么选这个技术""如果重做你会改什么"这类开放追问。**文中所有外部数字都带来源分级标注** —— 面试被追问出处时可直接说清每个数字是论文、官方文档还是营销稿。

## 硬数字速记表（面试前必背）

- **Agent 数量口径**：8 个 Agent = 1 编排者（Orchestrator，即 StateGraph 本身，无 LLM）+ 7 执行者；其中 6 个进生成管线（Planner→Guardian→Designer∥Coder→Content Auditor→Assessment），Mentor 是第 7 个执行者但走独立 RAG 链路不进 StateGraph。详见 [01-multi-agent-orchestration.md](01-multi-agent-orchestration.md#0-前置澄清项目里到底有几个agent面试官必问)
- StateGraph 共 9 个节点 = 6 个 Agent 节点（planner/guardian/designer/coder/content_auditor/assessment）+ 3 个合成节点（merge/retry_prep/assess_degraded，非 Agent）
- 8 个 Agent 总规模；6 个迁移至 Harness；15 个 Docker 服务
- 868 项测试（734 后端 pytest + 113 前端 Vitest + 21 沙箱）+ Playwright E2E；CI 覆盖率门禁 40%
- 3 层记忆：Sensory（请求级）→ Redis 短时（1h TTL，50 轮）→ PG 长时（Episodic + Semantic，召回 10 条）
- RAG：双路召回（Neo4j + ChromaDB），**Neo4j 关键词打分按匹配质量分档**（exact/prefix/substring/jaccard），融合默认 **RRF**（k=60），可切 `minmax`/`score`；Cross-Encoder 精排 top-5 **已实现但默认关闭**；上下文预算 2000 字符
- 评测：9 个指标函数，8 项对外指标，n=20 标注冒烟集 + n=200 图谱自动生成集，LLM-judge 限制 3 条/轮
- **RAG 实测口径**：所有基准报告 `reranker_enabled=false`；n=20 → MRR 0.900 / HR@3 0.950；n=200 → MRR 0.841 / HR@3 0.930 / HR@5 0.960 / P@1 0.755；全管线 P50 19.4ms / P95 54.7ms
- 工具系统：ToolRegistry（含超时治理）+ 3 个内置工具（kg_search / resource_search / forgetting_check）；工具调用遥测经 `_report.tool_calls` → SSE → 前端 trace 面板
- 意图识别：三路融合（规则快路径 + LLM 结构化投票 + embedding few-shot 投票）+ LRU/TTL 缓存；**实测冷启动准确率 93.1%（n=72 标注集），95.8% 请求由规则路径零模型调用解决**
- 检索策略 A/B（n=200 实测，三策略统一绕过缓存以保证延迟可比）：hybrid MRR 0.8569 / 30.3ms，direct 0.8454 / 33.0ms（延迟 ×0.92）；**LLM query rewrite 为负收益（MRR −0.054、延迟 ×60），已实现但默认关闭**
- 结构化输出：OutputSchema 双模式（Function Calling / 提示注入），解析鲁棒（剥围栏 + 定位 JSON + Pydantic 校验）
- 熔断器：连续 5 次失败熔断，恢复窗口 60s；RetryHandler 指数退避（1s/2s/4s），网络类异常才重试

## 面试串联主线

> Agent 怎么编排（1）→ Agent 怎么被标准化（2）→ Agent 靠什么回答问题（3/4）→ Agent 怎么记住用户（5）→ Agent 怎么调工具、吐结构化结果（6）→ 对标业界与改进路线（7）

这是一条完整的 Agent 工程闭环。面试时按此顺序讲，比零散讲 6 个点更有说服力。

---

## 当前项目状态快照（每轮对话后更新）

> **同步日期：2026-09-25** · 测试基线 **后端 1083 + 前端 141**（不含 E2E）

| 项 | 现状 |
|---|---|
| **多 Agent** | 8 个 Agent（1 编排 + 7 执行）；6 个进 StateGraph（9 节点）；Designer∥Coder 真并行（fan-out/fan-in + reducer）|
| **检索融合** | **默认 `score`**（n=200 实测 MRR 0.9002 vs rrf 0.8569）|
| **Reranker** | ✅ **实测确认必须保持关闭**（n=200）：开启后 **MRR 0.8945 → 0.3250**、延迟 x2.60。顺带修掉一个让 reranker **静默失效**的 numpy dtype bug（见 07 文档 §4.4.1）|
| **记忆系统** | 3 层；召回改为 `recency + importance + relevance` 打分；`recall_relevant` 提供语义召回通路；`embedding JSONB` 列已建 |
| **遗忘曲线** | ✅ 已修复：幂律曲线 + 幂律稳定性增长；LogLoss **1.42 → 0.4089**（基线 0.4421，**4/4 种子全胜**）；一周回忆率 0.056 → 0.903。参数待真实数据重拟合 |
| **复习数据管道** | ✅ 已通：`forgetting_review_log`（追加式）；`load_review_history()` 可直接喂评测框架 |
| **掌握度** | ✅ 已接线：出题时产出空 delta（不再自报），`grade_attempt()` 从实际作答经 1PL/Rasch IRT 测量 |
| **引用溯源** | ✅ **已完成（含真实页码）**：字符级偏移 + **PDF/DOCX/PPTX 页码**，贯穿切分 → 索引 → API → 前端高亮。⚠️ 顺带发现 Dockerfile 缺 tesseract/poppler 导致 **PDF 上传从未可用** |
| **池对象加固** | ✅ `unwrap_pool()` 统一处理裸 pool / 包装对象，第三种形态构造时抛错 |
| **图状态持久化** | ✅ 已接 `AsyncPostgresSaver`：每 super-step 存检查点 + time-travel；失败降级（无检查点仍可运行）|
| **checkpoint 保留** | ✅ 按线程年龄清理（TTL 7 天），`retention.py`；⚠️ 实测发现 `aprune` 未实现、时间戳在 JSONB 的 `ts` 而非 metadata —— **mock 测试对这两点都是瞎的** |
| **评测** | 9 个指标函数；n=20 / n=200 集；生成侧 **已重测**：词重叠 0.5407 / 语义 0.813 / Relevancy 0.8539（旧记录 0.010/0.277 经查是管线故障，非质量差）|
| **可观测性** | token 计数 + SSE trace；**无 OpenTelemetry/LangSmith** |
| **新增工具** | `run_fusion_ablation.py`、`run_forgetting_eval.py`（含 `--self-check`）|

**已落地的改进**：**I-1 ~ I-7 全部完成** ✅（I-5 的结论是「不要开启」）

**路线图已全部完成（I-1 ~ I-7）。** 若继续推进，候选：
1. **换更大的 reranker 模型重测**（需有网环境）—— `bge-reranker-v2-m3` 在该研究中为 +3.7~5 NDCG@10；**换模型才可能有效，改开关已知有害**
2. **用真实复习数据重拟合遗忘曲线参数**（`forgetting_review_log` 已开始记录）
3. **I-7 的真实页码**（需改 PDF 解析层保留页边界）

## 数据口径说明

所有压测/评测数据来自：
- 代码静态分析（`src/rag/evaluation/metrics.py` 9 个指标函数、`route_after_review` 的 max_retries=2 等）
- 项目内基准测试（RAGEvalBenchmark、Mock LLM 模拟压测）
- 架构对比推算（并行 vs 串行、Harness 迁移前后）

面试时如被追问数据来源，如实说明"开发期基准测试 + 架构分析推算"，并强调"评测框架让每个数字都可复现"。不要虚报线上生产数据（本项目为自研系统，未部署大规模线上环境）。
