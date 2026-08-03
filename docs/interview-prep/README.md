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
| 3 | RAG 混合检索 + 语义分块 + 重排序 | [03-rag-hybrid-retrieval.md](03-rag-hybrid-retrieval.md) | HitRate@3 0.83→0.930，P95 增量 <150ms |
| 4 | RAG 评测与优化闭环 | [04-rag-evaluation.md](04-rag-evaluation.md) | MRR 0.841，HitRate@3 0.930，P@1 0.755，Faithfulness 0.620 |
| 5 | 三层记忆架构 | [05-memory-system.md](05-memory-system.md) | token 开销 -60%，context 构建 P95 ~18ms |
| 6 | 工具调用 + 结构化输出双模式 | [06-tools-structured-output.md](06-tools-structured-output.md) | 解析成功率 85%→98%+，出题率 88%→97% |

## 硬数字速记表（面试前必背）

- 6 Agent 生成管线：Planner → Guardian → Designer∥Coder → Content Auditor → Assessment
- 8 个 Agent 总规模；6 个迁移至 Harness；15 个 Docker 服务
- 574 项后端 pytest + 前端 Vitest + Playwright E2E；CI 覆盖率门禁 40%
- 3 层记忆：Sensory（请求级）→ Redis 短时（1h TTL，50 轮）→ PG 长时（Episodic + Semantic，召回 10 条）
- RAG：双路召回（Neo4j + ChromaDB），top-20 粗筛 → Cross-Encoder 精排 top-5，上下文预算 2000 字符
- 评测：9 个指标函数，8 项对外指标，20 条标注数据集 + 图谱自动生成集，LLM-judge 限制 3 条/轮
- 工具系统：ToolRegistry + 3 个内置工具（kg_search / resource_search / forgetting_check）
- 结构化输出：OutputSchema 双模式（Function Calling / 提示注入），解析鲁棒（剥围栏 + 定位 JSON + Pydantic 校验）
- 熔断器：连续 5 次失败熔断，恢复窗口 60s；RetryHandler 指数退避（1s/2s/4s），网络类异常才重试

## 面试串联主线

> Agent 怎么编排（1）→ Agent 怎么被标准化（2）→ Agent 靠什么回答问题（3/4）→ Agent 怎么记住用户（5）→ Agent 怎么调工具、吐结构化结果（6）

这是一条完整的 Agent 工程闭环。面试时按此顺序讲，比零散讲 6 个点更有说服力。

## 数据口径说明

所有压测/评测数据来自：
- 代码静态分析（`src/rag/evaluation/metrics.py` 9 个指标函数、`route_after_review` 的 max_retries=2 等）
- 项目内基准测试（RAGEvalBenchmark、Mock LLM 模拟压测）
- 架构对比推算（并行 vs 串行、Harness 迁移前后）

面试时如被追问数据来源，如实说明"开发期基准测试 + 架构分析推算"，并强调"评测框架让每个数字都可复现"。不要虚报线上生产数据（本项目为自研系统，未部署大规模线上环境）。
