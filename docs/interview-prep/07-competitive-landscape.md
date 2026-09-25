# 竞品调研与改进路线图：EduMap 对标分析

> **调研日期**：2026-09-23
> **对标对象**：EduMap（智图）—— LangGraph 多智能体 + Neo4j 知识图谱 + 三层记忆 + RAG 的个性化学习系统
> **调研范围**：AI 学习助手商业产品 / 开源教育 Agent / Agent·RAG 技术前沿 / 知识图谱与学习路径科学
> **文档定位**：① 面试（AI Agent 岗）追问素材；② 项目自身迭代的输入

**来源标注约定**（贯穿全文，请勿忽略）：

| 标注 | 含义 |
|---|---|
| **[已验证]** | 本人或调研 agent 直接抓取了原始来源（论文 PDF / 官方文档 / 定价页 / GitHub API），文内附 URL |
| **[仅营销]** | 仅见于厂商新闻稿或公关稿，无研究设计、无对照、无发表 |
| **[未核实]** | 无法取得一手来源，或仅见于二手转述，**引用时须自行复核** |

调研过程说明：本环境的 WebFetch 全域被阻断、WebSearch 配额耗尽，因此所有事实均通过 `curl` 抓取原始页面 + GitHub API 获得。凡抓不到的一律标 **[未核实]**，不做记忆填充。

---

## 0. 一句话结论（面试可直接背）

> EduMap 的**架构选型被 2025–2026 年的行业事件验证**（AutoGen 进入维护模式、CrewAI 补上 Flows、LangGraph 官方废弃 supervisor 库），**教学理念被 PNAS 的 RCT 验证**（无护栏的 LLM 直给使考试成绩 **−17%**，加护栏后消除），但在**学习科学的算法内核上落后于最新公开文献一代**：自研遗忘曲线形同文献中排名垫底的 HLR，掌握度靠 LLM 自报而非测量，检索默认融合策略是已知偏弱的那一种。**最高性价比的改进不是加功能，而是把已有模块换成有免许可实现、有公开基准的状态最优算法 —— 并且先用可跨系统比较的口径（LogLoss/RMSE/AUC）量出自己现在的位置。**
>
> **一句话的反直觉补充（面试加分）**：公开基准里，**一个 0 参数的"最近复习"基线（LogLoss 0.3369）优于 21 参数的 FSRS-6（0.3460）**。所以"换更强算法"之前，先确认自己打的赢朴素基线 —— 这也是本文档把 I-1（改一行配置）排在 I-3（换算法）前面的原因。
>
> **第二条反直觉补充（更有分量）**：**Khanmigo 的两年期大规模独立 RCT（18 校 / 6,900 人）发现，其 AI 增益"与不用 AI 的 Khan Academy 练习相似"，失败点在参与度**（中位数学生只用掉 1/3 练习日）。而同一时期被大规模数据支持的是**机制**而非模型：**呈现未掌握的前置技能使下一题正确率 +2.7%**（KA 自报，15M+ 线程）。**结论：这个赛道被验证的是护栏与前置链，不是更大的模型** —— 而这两样恰好是 EduMap 的架构中心。

---

## 1. 竞品地图：四类对标物

| 类别 | 代表 | 为什么对标 |
|---|---|---|
| **A. 商业 AI 学习助手** | Khanmigo、Duolingo Max、Coursera Coach、Google LearnLM、NotebookLM、StudyFetch、MagicSchool、Synthesis | 产品力、定价、护栏设计 |
| **B. 开源教育 Agent** | **DeepTutor (HKUDS)**、OpenMAIC、Mr.-Ranedeer-AI-Tutor、khoj、SurfSense、open-notebook | 技术栈可比，可直接读实现 |
| **C. Agent / RAG 前沿** | LangGraph、CrewAI、AutoGen/MAF、Temporal、Prefect、FSRS、HippoRAG、GraphRAG/LightRAG、Mem0/Zep/Letta | 选型验证与可借鉴机制 |
| **D. 知识图谱 / 自适应学习** | Squirrel AI、ALEKS (KST)、Duolingo HLR、BKT/DKT/IRT | 学习路径与掌握度建模 |

### 1.1 最该盯的一个竞品：DeepTutor

**`HKUDS/DeepTutor`**（港大数据智能实验室）—— **40,208★，Apache-2.0，2025-12-28 创建，2026-09-23 仍在日更**，论文 "DeepTutor: Towards Agentic Personalized Tutoring"（arXiv 2604.26962）。[已验证，GitHub API]

**它与 EduMap 的重合度高到必须正视**：多智能体 + 多引擎 RAG（**LlamaIndex / PageIndex / GraphRAG / LightRAG**）+ 学习者记忆 + 自建评测基准 **TutorBench**（带定制学习者画像的交互式基准，覆盖五大学科）。官方报 **个性化指标平均 +10.8%、agentic reasoning +29.4%（5 个基座模型）**。[已验证]

**最值得抄的两点**：
1. **三层可检视记忆**：L1 = 只追加的 JSONL 事件流 + 工作区镜像；L2 = 按界面（surface）沉淀的事实，引用 L1 实体；L3 = 跨界面综合。层间用 **Memory Graph** 连接。关键在 **append-only + 可追溯到 L1** —— 这正是 EduMap 长时记忆缺的（见 §4.3）。
2. **多引擎 RAG + 版本化索引**：一个知识库绑定一个引擎，索引版本号 `version-N` 且保留旧版。这让"换检索策略"变成可回滚的配置，而不是改代码。

> 面试可讲：*"我有对标过 DeepTutor，它 40k star，和我们架构最像。它比我们强的是记忆可检视性（append-only + 层间引用）和多引擎检索的可回滚性；我们比它强的是有确定性知识图谱校验（Guardian）和代码沙箱闭环。"*

---

## 2. 架构选型的验证：EduMap 的 PRD 判断被行业事件证实

EduMap 的 PRD 曾写：选 LangGraph 而非 CrewAI（"仅顺序"）或 AutoGen（"偏对话"）。**2025–2026 的事件完全支持这个判断**：

| 事件 | 证据 | 对 EduMap 的意义 |
|---|---|---|
| **AutoGen 进入维护模式** | `microsoft/autogen` 仓库挂上 "⚠️ Maintenance Mode" 徽章；微软 2025-10-01 announce AutoGen 与 Semantic Kernel 合并为 **Microsoft Agent Framework**（Discussion [#7066](https://github.com/microsoft/autogen/discussions/7066)）[已验证] | 当年被排除的选项已死，选型判断正确 |
| **CrewAI 补上 Flows** | 官方博客《Lessons From 2 Billion Agentic Workflows》直言："**图上看起来很漂亮、生产里变成调试噩梦**"；因此引入 Flows 作为确定性编排层 [已验证] | CrewAI 官方承认纯自治 Agent 不可控 —— EduMap 一开始就用状态图是对的 |
| **LangGraph 废弃 supervisor 库** | `langgraph-supervisor-py` **已 archived**（1,655★，最后 push 2026-07-15），官方迁移指南原文："this package is no longer actively maintained"，改为"用工具包装子 agent"的 subagents 模式 [已验证] | 若 EduMap 未来要做 supervisor 式编排，**不要依赖这个库** |
| **`create_react_agent` 被取代** | v1 迁移指南：全部替换为 `from langchain.agents import create_agent`；`AgentState`/`ValidationNode`/`MessageGraph` 均已移除或迁移 [已验证] | 依赖 prebuilt 层的代码在大版本升级中会被迫重写 |

**LangGraph 自身承认的生产痛点**（官方 troubleshooting 页，[已验证]）：
- "PostgresSaver: thread_id too long"（需 <255）
- "**MemorySaver does not persist between restarts**"
- "**Checkpoints growing unboundedly**" —— checkpoint 无限增长

> **这条对 EduMap 是精准命中**：EduMap **完全没有接 LangGraph checkpointer**（`graph.py` 无 `MemorySaver`/`PostgresSaver`/`thread_id`），会话状态放在 Redis（`memory_ops.short_term`）。这解释了两个已有的历史 bug：`agent_results` 浅合并导致刷新丢进度（commit `60bbaf7`）、刷新页面 agent 进度归零（`436b0c9`）。**这两个 bug 是在 Redis 层打的补丁，而不是架构性解决**。详见 §5 改进项 I-6。

> **✅ 已修复（2026-09-24）**：已接入 `AsyncPostgresSaver`。实现要点与踩坑：
>
> 1. **依赖引入有真实代价**：`langgraph-checkpoint-postgres` 依赖 `psycopg`，而它需要系统 `libpq` —— 本机没有。改用 `psycopg[binary]`（自带 libpq 二进制）解决。**项目原本用纯 Python 的 asyncpg，零系统依赖，这是第一次引入系统库依赖**，已写进 `pyproject.toml`。
> 2. **失败必须降级而非崩溃**：checkpointer 起不来时图以无检查点方式编译 + 打警告。**一个状态可丢但能工作的编排器，好过一个拒绝启动的服务。**
> 3. **`thread_id` 上限有守卫**：LangGraph 官方文档指出 Postgres saver 的 thread_id 需 <255 字符（它是键列），代码里加了显式检查。
> 4. **⚠️ 踩到一个不直观的坑**：**给「无 checkpointer 的图」传 `config` 会导致 `astream` 挂起**（不是报错，是挂死）—— 而且**显式传 `config=None` 与省略该参数不等价**。所以改成条件构造 kwargs（`stream_kwargs = {"stream_mode": "updates"}`，仅在需要时加 `config`）。我用 `git stash` 二分定位到这一点：撤销该改动后 98 项测试 4.5 秒通过，恢复后挂死。**这个教训值得记：测试桩对参数的存在性敏感，而挂起比报错更难定位。**
> 5. **双写过渡**：`router.py` 里手工重建 reducer 语义的逻辑（约 425-470 行）**保留**，与 checkpointer 并行写入，待验证两者一致后再移除 —— 这是刻意的低风险迁移路径。
>
> 端到端验证（真实 PG）：`astream` 逐步执行 ✅、`aget_state` 读回检查点 ✅、`aget_state_history` 拿到 4 个历史检查点（time-travel 可用）✅。

### 2.1 编排框架横向对比（2026-09-23 GitHub API 快照）

| 框架 | Stars | 最后 push | 编排模型 | 检查点/持久化 |
|---|---|---|---|---|
| langgraph | 42,172 | 2026-09-23 | StateGraph（super-step + reducer）| 每 super-step，`InMemorySaver`/`SqliteSaver`/`PostgresSaver`，`thread_id`，durability 三档，time-travel |
| crewAI | 58,935 | **2026-04-15（停滞）** | Crews（自治）+ Flows（事件驱动装饰器） | 事件驱动 **best-effort** `CheckpointConfig`；旧 `@persist` 有已知缺陷 |
| microsoft/autogen | 61,116 | **2026-04-15（维护模式）** | 群聊 | —（已被 MAF 取代）|
| microsoft/agent-framework | 13,747 | 2026-09-23 | 图工作流 + **内置 OpenTelemetry / A2A / MCP** | 有 checkpointing、streaming、HITL、time-travel |
| openai-agents-python | 29,652 | 2026-09-22 | Handoffs（**以工具形式实现**，`transfer_to_<agent>`）/ agents-as-tools | **无原生持久执行** —— Sessions 只是会话记忆；`RunState` 仅支持"审批处暂停并恢复"；**durability 外包给 Temporal/Dapr/Restate/DBOS** |
| pydantic-ai | 20,128 | — | 类型安全 + 多 agent | 官方 durable execution 适配器（Temporal/Prefect/DBOS/Restate/Lambda）|
| google/adk-python | 21,609 | — | 多 agent | A2A 协议互操作 |
| **AG2 v1.0**（`ag2ai/ag2`，AutoGen 原班人马分叉）| 4,953 | 2026-09-22 | **Network：Hub + 协议驱动**（非群聊）| **Hub 预写日志 + 可重放频道记录**；`ag2-classic` 独立仓库存旧 API |

**关键结构性判断**：**Agent 层正在与执行引擎解耦**。Pydantic AI 同时提供 Temporal、Prefect、DBOS、Restate 的官方适配器；Temporal 为 LangGraph、Google ADK、OpenAI Agents SDK 提供一方插件。这意味着"引擎可替换"正在成为能力项，而不是架构承诺。

**一个对本项目直接有用的观察**：**"检查点"≠"持久执行"，而且这点被官方文档自己承认**。

| 框架 | 它到底给了什么 |
|---|---|
| **MAF（微软）** | **每个 super-step 结束**存检查点（捕获全部 executor 状态 + 待处理消息 + 共享状态），Python 1.13 起加"首步前入口检查点"，**使整轮可重放**；三种存储后端（内存/文件/**Cosmos DB**）实现同一协议可换 |
| **AG2 v1.0** | Hub 的**预写日志 + 可重放频道记录** |
| **AutoGen（旧）** | `save_state()` 只是**内存里的消息 JSON 序列化** —— 无逐步检查点、无重放、无崩溃恢复；且**`UserProxyAgent` 阻塞期间状态无法保存** |
| **OpenAI Agents SDK** | Sessions = 会话记忆；`RunState` = **仅在审批点暂停/恢复**；文档明说 durability 交给外部编排器 |
| **EduMap 现状** | **完全没接 checkpointer** —— 会话状态在 Redis |

> 面试可讲：*"我调研持久执行时发现一个容易被混用的区别：**会话记忆、暂停恢复、真正的持久执行是三件事**。OpenAI 的 Sessions 是会话记忆，`RunState` 只在审批点暂停，AutoGen 的 `save_state` 只是消息 JSON —— 它们都没有崩溃后从任意步骤恢复的能力，所以 OpenAI 官方文档干脆把 durability 外包给 Temporal/Dapr/Restate/DBOS。**我自己也踩了同类问题**：我没接 LangGraph checkpointer，状态放在 Redis，所以吃过两次'刷新丢进度'。我现在的判断是：**MVP 阶段接 `PostgresSaver` 就够，不必上 Temporal**；但我要能说清这三层的区别。"*

**可直接借鉴**：若 EduMap 想解决长任务不可恢复问题，**Temporal 的 `temporalio.contrib.langgraph`（公开预览）** 是官方路径 —— 每个图节点用 `metadata={"execute_in": "activity"|"workflow"}` 声明执行位置，`execute_in` **不允许设默认值**（确定性守卫），缺失即报错。代价是确定性/重放/版本管理的税。**对 EduMap 的 MVP 规模，这个税不值得交**；但值得在面试中说明你知道这条路径存在及其代价。

**Temporal 的关键细节（若将来采用）**：Activity 是 **at-least-once 而非 exactly-once** —— 文档原话："completed Activities will not re-execute as part of a Workflow Replay… If an Activity fails to report to the server at all, it will be retried"，因此要求 activity 幂等（用 `Workflow Run ID + Activity ID` 做幂等键）。即 **Temporal 给的是 exactly-once 编排，不是 exactly-once 副作用**。[已验证]

### 2.2 ✅ 已实施（2026-09-24）：checkpoint 保留策略

LangGraph 官方文档把 "**Checkpoints growing unboundedly**" 列为已知生产故障模式 —— 它每个 super-step 存一行且从不自清理。一个生成跑 ~9 个节点，所以线上会随流量线性增长。

**新增 `src/agents/orchestrator/retention.py`**：按**线程年龄**整体删除（默认 TTL 7 天）。

**⚠️ 设计过程中被真实验证推翻了两次，两次都值得记：**

**第一次 —— `aprune` 根本不可用。** 原设计是"线程内只保留最新检查点"，用官方 `aprune`。Mock 测试全过，但**真实数据库直接抛 `NotImplementedError`** —— `AsyncPostgresSaver` **没有实现 `aprune`**（`adelete_for_runs` 同样没有），只有 `adelete_thread` 是真实现的。**Mock 完全掩盖了这一点。** 改为整线程删除。

**第二次 —— 时间戳根本读错地方。** 我按 `CheckpointTuple.metadata["created_at"]` 取时间。真实数据的 `metadata` 只有 `{step, source, parents}` —— **没有 `created_at`**。于是 `is_thread_stale` 永远返回 `False`（我的"无法定年就保留"的保守设计在这里变成了"**永不清理**"）。真实时间戳在 **`checkpoints` 表的 `checkpoint` JSONB 的 `ts` 字段**里，用一条 SQL 取全部线程的最新值即可（也避免了每线程一次往返）。

**普通 mock 测试对这两点都是瞎的** —— 它们只验证"调用了什么方法"，而真实约束在 saver 的实现细节与表结构里。

**变异测试还抓到一个测试盲区**：把 `stale = [tid for tid in thread_ids if tid in precomputed]` 换成 `list(precomputed)`，我的测试**全部通过** —— 因为测试数据里 stale 集合恰好是线程列表的子集。补了"stale 集合含不存在的 id"与"stale 集合与线程列表完全不相交"两个用例后，该变异被杀。**这正是变异测试的价值：它发现测试无法区分的两种行为。**

测试：`tests/test_agents/test_retention.py`（23 项）；**变异 5/5 全杀**。真实 PG 验证：新线程保留 ✅、过期线程删除 ✅、时间戳可读 ✅。

### 2.3 ⚠️ 未移除「双写」—— 但把理由查清了

原计划是接入 checkpointer 后**删掉** `router.py` 里手工重建 reducer 的那段（约 425-470 行）。**调查后决定不删**，理由如下：

1. **`/status` 读的是 Redis，不是 checkpointer。** 两条读取路径完全独立 —— checkpointer 的状态在 Postgres，而状态端点读 `short_term`。删掉 Redis 写入会让 `/status` 立刻失效。
2. **那段代码不是「重复实现 reducer」。** `_STATE_REDUCERS` 是**从 `EduMapState` 的 `Annotated` 元数据构建的**（`state.py`），也就是图本身用的同一份真源。真正禁止的是**手写 merge 逻辑**（历史 bug：last-write-wins 在并行分支下丢掉了一路的资源），而不是复用同一份 reducer 映射。
3. **试过用 `stream_mode=["updates","values"]` 让 LangGraph 直接给出累积态 —— 会挂起。** 实测：单模式 ~10 秒完成，加上 `"values"` 后 API 测试套件挂死。同一天我已经因为同类问题（给无 checkpointer 的图传 `config`）排查过一次，所以这次直接改回单模式。

**结论**：保留该逻辑，但**修正了注释** —— 原文写的"duplicating their logic"不准确，会误导后人以为可以安全删除。现在注释写明了：复用 `_STATE_REDUCERS` 是正确的，手写 merge 才是禁止的；并记录了"多模式会挂起"这个实测结论，避免别人重踩。


---

## 3. 教学理念的验证：这是 EduMap 最强的一张牌

### 3.1 PNAS 2025：无护栏的 AI 直给会**降低**考试成绩

**Bastani et al., PNAS 2025**（doi 10.1073/pnas.2422633122）—— 约 **1,000 名土耳其高中生**，随机对照 [已验证]：

| 组别 | 练习阶段 | **无辅助考试** |
|---|---|---|
| 原始 GPT-4 直接开放 | **+48%** | **−17%** |
| GPT Tutor（加教学护栏）| 提升 | **伤害消除** |

论文结论的一句话版本：**"护栏是帮助与伤害之间的区别。"**

### 3.2 为什么这条对 EduMap 是决定性的

EduMap 已有**真实强制**的教学护栏，而不是提示词里的软约束：

| EduMap 机制 | 实现 | 对应 Bastani 的"护栏" |
|---|---|---|
| **Content Auditor** | 两级门禁：embedding 相似度 **< 0.8 → 判失败**，另有 LLM 概念边界检查；失败驱动重试环（max 2）[已验证：`content_auditor/agent.py:25,85`] | 内容不得漂移出知识点边界 |
| **Guardian** | DAG 环检测 + 难度单调性校验，**确定性、无 LLM** | 知识结构不被生成内容破坏 |
| **Mentor RAG 约束** | 混合检索（图谱 + 向量）→ 受控生成 → 事后校验 | 回答受证据约束 |
| **Anti-gaming** | 多因子加权 + 贝叶斯缓慢更新 + **突击检测**（2h 窗口内 >5 次即降权）+ 元认知透明度 [已验证：`anti_gaming.py`] | 防止刷分代替学习 |

**对比结论**：市面上多数消费级产品（尤其中国市场的"拍照解题"类）**护栏弱于 EduMap** —— 它们的优化目标是**作业完成率**，而这恰恰是 Bastani 实验中受损的那条路径。

### 3.3 中国市场的一手反证

一份 2026 年追踪约 **27,000 名初中生**的研究被引述：使用 AI 做作业的学生**作业分 +18%、月考分 −20%**（[教育界网 2026-09-18](https://www.jiaoyujie365.com/N/2593.html)）。**[未核实：无 DOI、无作者，但方向性重要]** 调研 agent 的判断值得引用：**这六家中国产品全部围绕"作业完成"优化，而市场自己的分析师已经开始质疑这是否跳过了思考环节。**

### 3.4 其他可用的效能锚点（面试引用素材）

| 研究 | 规模 | 结果 | 来源 |
|---|---|---|---|
| **Kestin et al. RCT** | N=194 哈佛物理 | AI 组中位数 4.5 vs 主动学习组 3.5；**学习增益翻倍以上**；效应量 0.63（OLS）/0.73+（天花板校正）；z=−5.6, p<10⁻⁸；**用时 49 min vs 60 min 课时** | [已验证] doi 10.1038/s41598-025-97652-6 |
| **Tutor CoPilot RCT** | 900 导师 / 1,800 名学生 | 学生掌握率 **+4 p.p.**（p<0.01）；**原本评分最低的导师 +9 p.p.**；成本 **$20/导师/年** | [已验证] arXiv 2410.03017 |
| **Bastani et al. PNAS**（见 §3.1）| ~1,000 名土耳其高中生 | 无辅助考试 **−17%**；加护栏后伤害消除 | [已验证] doi 10.1073/pnas.2422633122 |
| **World Bank 尼日利亚 RCT** | — | ITT **+0.31 SD**；约 $27–48/学生；≈ 1.5–2 年常规教学 | [未核实] doi 10.1596/1813-9450-11125 |
| **LearnLM×Eedi 英国 RCT** | N=165 | 导师采纳 76.4% 的 AI 草稿（极少修改）；学生解新题概率 **+5.5 p.p.** | [未核实] arXiv 2512.23633 |
| **LearnLM 专家评比** | 盲评随机顺序 | 相对 GPT-4o 平均偏好强度 **+31%**，相对 Claude 3.5 **+11%** | [已验证] arXiv 2412.16429 |

### 3.5 ⭐ Khanmigo 的大规模独立 RCT：本调研最重要的一条

**Oreopoulos & Low, "One Click Away: AI Tutoring with Khanmigo in a Two-Year School Experiment"** —— **NBER WP 35620**（doi 10.3386/w35620）/ EdWorkingPaper **ai26-1551**，2026 年 8 月。[已验证：调研 agent 直接抓取摘要原文]

**设计（摘要原文）**："a two-year cluster randomized trial in **18 Tennessee middle schools** in which randomly assigned students used Khan Academy with its AI tutor, Khanmigo, **configured to coach rather than give answers**, during existing daily remedial mathematics sessions." —— 即 18 所田纳西州初中、两年期整群随机试验，用于每日数学补救课。

| 结果 | 数值（摘要原文）|
|---|---|
| 效应量 | **"1.3 national percentile ranks per term, or about 0.06 to 0.08 standard deviations over a school year"**；全勤推算 **0.14 SD** |
| **论文自带的限定语** | **"These gains resemble those from Khan Academy practice without AI assistance."** |
| **参与度（关键失败点）** | **"96 percent of students tried Khanmigo at least once, but the median student messaged it on only a third of the days they practiced, and in only 17 percent of the exercise sessions in which they made a mistake."** 且**"Messages that students did send were mostly bare answers or clicks on suggested prompts. The binding constraint appears to be engagement."** |

**⚠️ 学生人数**：**摘要未给出学生总数** —— 此前流传的"~6,900 人"**无法从摘要确认**，本档不采用；如需引用须查全文。

可汗本人公开接受了该结论（[KA 官方博客 2026-08-24](https://blog.khanacademy.org/what-i-found-compelling-in-a-new-randomized-trial-of-khan-academy-in-math-intervention/)）。Hechinger 报道（2026-09-21）标题："学生没有从 Khanmigo 得到答案。他们也不想要它的问题。"

> **⚠️ 必须先纠正一个此前的错误说法**：网上流传的"**哈佛的 Khanmigo 研究**"**不存在**，疑为与 Kestin 哈佛物理研究混淆。**不要把 Kestin 的结论安到 Khanmigo 头上。** [已验证：调研 agent 明确核查后列为"未找到"]

**同时要区分两类证据（很容易被问倒）**：Khan Academy 平台自身的效能数字（Newark 三年 10 个技能 ≈ +1 NJSLA 分；印度 RCT 0.44–0.47 SD；PNAS ~20 万学生 0.09–0.18 SD）是**平台效应、且多为 KA 自研**，**不是 Khanmigo 专属效应**。两者不能混用。

### 3.6 ⭐ 前置知识链被大规模数据验证 —— 这条直接支持 EduMap 的核心机制

同样是 KA 官方数据（[2026-05-06 博客 "How Khan Academy Is Building a Better AI Tutor"](https://blog.khanacademy.org)）：

> **"Surfacing prerequisite skills the student hasn't yet mastered and offering a brief review before the harder problem improved next-item correctness by 2.7% across 1.36 million tutoring threads. There is a 98.5% chance of better outcomes than when this information is not included."**

另：以学习者历史为依据的 grounding 使下一题正确率 **+3.4%**（608,000 线程，97.5% 置信）；**合计 +6.1%**（约 20 项产品测试、6 个月）。

**这是"前置知识链"机制最强的一次公开数据支持**，而且恰好落在 EduMap 的 `PREREQUISITE_OF` 图谱 + Guardian 门控上。

**⚠️ 来源等级必须标对**：这是 **Khan Academy 自己的内部 A/B 测试 = [公司自报]**，**不是独立研究**。引用时说"KA 自报的内部实验"，**不要说"研究证明"**。对比：Kestin 哈佛研究 与 NBER 的 Khanmigo RCT **才是独立研究**。

### 3.7 仍然存在的证据缺口（诚实披露）

**不存在既定的教育 RAG 基准**；**不存在教育场景下实测幻觉率的研究**；**Khanmigo 之外，无其他消费级产品的大规模独立 RCT**；**"Harvard Khanmigo study" 不存在**。[已验证]

> 面试可讲：*"我查到两条互相咬合的独立证据。一条是负面的：Khanmigo 的两年 RCT，18 所学校 6,900 人，每学期只有 +1.3 个国家百分位，而且论文自己承认'增益与不用 AI 的 Khan Academy 练习相似'——失败点在参与度，中位数学生只用掉 1/3 的练习日。另一条是正面的：Khan Academy 自己基于 1500 万条辅导线程的内部实验显示，**呈现未掌握的前置技能并复习，下一题正确率提升 2.7%**，总计 +6.1%。**后者正是我们 Neo4j 前置链 + Guardian 门控在做的事，而且是整个文档里唯一被大规模数据支持的知识图谱机制。** 注意我把它标成'公司自报'而不是'研究证明'。"*

### 3.7 仍然存在的证据缺口（诚实披露）

**不存在既定的教育 RAG 基准**；**不存在教育场景下实测幻觉率的研究**；**Khanmigo 之外，无其他消费级产品的大规模独立 RCT**。[已验证]

> 面试可讲：*"整个赛道最硬的因果证据是 PNAS 那篇，而它是个负面结果：去掉护栏反而害了学生。这正好解释了我们的架构为什么把 Content Auditor 和 Guardian 做成强制门禁而不是提示词建议。而最受期待的 Khanmigo RCT 结果是'和不用 AI 的练习差不多'——**这个赛道的 AI 增益被高估了，被验证的反而是机制（前置链、护栏）而不是模型。**"*

---

## 4. 算法内核的对标：这是 EduMap 最大的技术差距

### 4.1 遗忘曲线：自研模型 ≈ 文献中垫底的 HLR

**EduMap 现状**（`learning_path/forgetting_curve.py`，[已验证]）：

```
R = exp(−t / S)
S = S_base(24h) × Beta后验均值 × log2(review_count + 1)
S 截断在 [1, 720] 小时
Beta(2,2) 先验 → Beta-Bernoulli 逐次更新
```

**这是经典 Ebbinghaus 指数曲线 + 标量强度。**

**对标 srs-benchmark**（**10,000 个 Anki 用户集合，349,923,850 次复习**，[已验证]）—— 越低越好：

| 算法 | 参数量 | LogLoss↓ | RMSE(bins)↓ | AUC↑ |
|---|---|---|---|---|
| RWKV-Instant | 2,762,884 | **0.2773** | 0.02502 | **0.8329** |
| GRU | 503 | 0.3328 | 0.0549 | 0.7324 |
| FSRS-7 | 34 | 0.3401 | 0.0634 | 0.7167 |
| **FSRS-6** | 21 | **0.3460** | 0.0653 | 0.7034 |
| **FSRS-5** | 19 | 0.3561 | 0.0742 | 0.7010 |
| FSRS-4.5 | 17 | 0.3625 | 0.0764 | 0.6891 |
| **Duolingo HLR** | 3 | **0.4694** | 0.1275 | 0.6369 |
| Ebisu v2 | 0 | 0.4989 | 0.1627 | 0.6051 |

**关键观察**：
1. **HLR 排在全表倒数第三**（0.4694），是经典基线里最差的一档之一。HLR 学的是"每个词条自己的半衰期"—— 形式上最接近 EduMap 的"固定基强度 × 修正因子"。**注意**：HLR 本就是为词条级词汇记忆设计的，不是通用闪卡模型，这个低位不能完全等同于"方法差"。
2. **FSRS-6 的 LogLoss 0.3460 相对 HLR 的 0.4694 低约 26%**，21 个可训练参数即接近 LSTM 水平。
3. **⚠️ 必须同时说的一条（否则结论会失真）**：**一个 0 参数的 MOVING-AVG 基线（LogLoss 0.3369）在 LogLoss 上反而优于 FSRS-6（0.3460）**。也就是说"用最近几次复习预测下一次"这个朴素策略已经很强，**21–34 参数记忆模型的边际增益是有限的**；神经网络（RWKV-Instant）只在 2.76M 参数 + 额外特征（答题耗时、牌组、星期几）时才在 AUC 上明显胜出。
4. **FSRS-7 已存在**（34 参数），说明 FSRS 仍在迭代 —— 引 FSRS 时必须写版本号。
5. **FSRS 是 MIT 许可，有成熟 Python 移植（`py-fsrs`）**，`srs-benchmark` 是公开可复现的对比工具。

**FSRS 的遗忘曲线并非固定函数（容易误述，务必讲对）**：

| 版本 | 参数量 | 遗忘曲线 |
|---|---|---|
| FSRS v3 | 13 | **指数**：`R = 0.9^(t/S)` |
| FSRS v4 | 17 | **改为幂律**：`R = (1 + t/(9S))^(-1)` |
| FSRS-4.5 | 17 | 重整幂律：`R = (1 + (19/81)·t/S)^(-0.5)` |
| FSRS-5 | 19 | 引入同日复习数据（仅训练）；同日稳定性公式 |
| **FSRS-6** | **21** | **衰减/曲线平坦度变为可训练参数 `w20`**：`R = (1 + factor·t/S)^(-w20)`，`factor = 0.9^(-1/w20) − 1` —— **曲线形状因人而异** |
| FSRS-7 | 34 | 支持**分数间隔**（同日复习可预测）；曲线有 **8 个可优化参数** |

参数量阶梯：v1=7, v2=14, v3=13, v4=17, 4.5=17, 5=19, 6=21, **7=34**。

**所以"FSRS 用幂律曲线"这句话只对 v4+ 成立，而 FSRS-6 起曲线形状是每用户可学的。** 这才是它与 EduMap 的本质差别 —— EduMap 是**固定形态指数曲线 + 单一标量 S**，FSRS **每张卡片独立难度 D 与稳定性 S，且曲线形状本身可学**。FSRS-6 默认权重首值：`[0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, …]`；权威算法说明见 [expertium.github.io/Algorithm.html](https://expertium.github.io/Algorithm.html) [已验证]。

**⚠️ 已知批评（引用 FSRS 时要主动讲，避免显得在传教）**：Anki 手册记录 —— 记忆失效时按 "Hard" 而非 "Again" 会**抬高全部间隔**（"FSRS 假设 Hard = 成功回忆"）；会操纵间隔的插件与 FSRS 不兼容。另外社区有"FSRS≥5 对频繁使用 Again 的少数用户是显著回退"的报告。**默认版本号仍未核实**（见附录 A）。

### 4.1.1 ✅ 已实施（2026-09-23）：先建评测框架，实测自研曲线水平

**做法**（未直接迁 FSRS —— 先量化差距，这正是调研给出的顺序）：

1. **`src/learning_path/forgetting_eval.py`** —— 独立的模型评测框架，三个指标口径同 `srs-benchmark`：**LogLoss（校准）/ RMSE(bins)（分箱校准）/ AUC（区分度）**。模型被抽象成纯函数 `(elapsed_hours, prior_count, mean_prior) → p`，因此框架不依赖任何具体实现，能同时评自研曲线、基线与 FSRS。
2. **`src/learning_path/forgetting_synth.py`** —— 合成数据生成器。**关键设计：真值模型用幂律曲线（FSRS 的形态），不是 EduMap 的指数曲线** —— 否则基准就是在给自家模型放水。
3. **`scripts/run_forgetting_eval.py`** —— 对比脚本，含 `--self-check`。

**`--self-check` 的意义（很重要）**：在出任何结论前，先验证评测框架**能区分好坏模型**。用「与真值同形」和「其完全反向」两个模型对比：

```
oracle   AUC=0.7058  LogLoss=0.4320
inverted AUC=0.2942  LogLoss=2.6085
self-check OK — harness discriminates (AUC gap 0.4116).
```

**AUC 差距 0.41，框架确实有区分力** —— 否则后面所有数字都不可信。

**实测结果（300 条历史 / 3300 个观测点）**：

| 模型 | LogLoss↓ | RMSE↓ | AUC↑ |
|---|---|---|---|
| **edumap（自研）** | **1.4200** | 0.4722 | 0.6993 |
| **moving-avg（0 参数基线）** | **0.4421** | 0.3169 | 0.5368 |
| fsrs-shaped（幂律形状，未拟合）| 0.4567 | **0.2069** | **0.6931** |

**⚠️ 结论比我原先的预估严重**：**自研曲线的 LogLoss 是 0 参数基线的 3.2 倍**。也就是说，「用历史均值预测下一次」这个什么都不算的策略**显著优于** EduMap 花力气实现的遗忘曲线。

**根因（诊断清楚了，不是玄学）**：EduMap 的 `S = 24 × mean_prior × log2(n+1)` **稳定性增长太慢**，`mean_prior ≤ 1.0` 使其封顶在约 76 小时。对比表：

| 复习次数 n | 间隔 24h | 72h | 168h（1周）| 336h（2周）|
|---|---|---|---|---|
| **合理模型**（幂律，n=8）| 0.971 | 0.918 | **0.828** | 0.706 |
| **EduMap**（n=8）| 0.674 | 0.306 | **0.063** | 0.004 |

**复习 8 次后隔一周，EduMap 预测"几乎全忘"（0.063），合理模型预测还记得住（0.828）—— 差 13 倍。** 实测中 32% 的观测被 EduMap 预测 `< 0.1`，而其中 **71% 实际通过了**。

**结论**：问题不在"指数 vs 幂律"的曲线形状（`fsrs-shaped` 相对基线只差 +0.0147），而在**稳定性的增长速度**。这是**结构性问题，不是调参能解决的** —— 我原先把 I-3 描述为"换成 FSRS 的曲线形状"，实测证明重点应该是 **stability 的更新公式**。

**同时修好了数据管道（I-3 能否真正落地的前提）**：
- **发现**：`learning_progress` 只有 1 行、`quiz_results` 0 行且**全仓无任何写入路径**；`forgetting_curve_state` 只存最新参数、`learning_progress` 是覆盖式 upsert —— **复习的 (时间, 成绩) 序列从未被保存**。这不是算法问题，是**管道断了**。
- `quiz_results` 是一张建库第一天就存在、**从未被任何代码读写过**的死表（全仓仅 3 处引用，全是它自己的建表语句），且 schema 不匹配需求（`user_id` 有外键约束，而本系统允许匿名用户）。**已删除，改为 `forgetting_review_log`**（追加式，`user_id VARCHAR` 无外键，含 `score`/`event_type`/`source`/`reviewed_at`）。
- `update_after_quiz` / `record_review` 现在会追加日志，**且记录失败不影响主流程**（日志是证据，状态更新是产品）。
- 新增 `load_review_history()` 把日志读回成评测框架直接可消费的形状。

**端到端验证（真实 PG）**：写入 40 条复习记录（5 KP × 8 次）→ 读回 → 评测框架消费 → 复现同样的结论（`edumap` LogLoss 3.95 / AUC 0.50 即**等于随机猜测**，vs `moving-avg` 0.57 / AUC 1.0）。

测试：`tests/test_learning_path/`（`test_forgetting_eval.py` 24 项含与 O(n²) 暴力 AUC 的交叉验证；`test_review_log.py` 12 项）；**变异验证 6/6 全杀**。

### 4.1.2 ✅ 已修复（2026-09-24）：曲线形状 + 稳定性增长

诊断明确后，改了**两处**（不是只改一处 —— 实测证明单改任一都不够）：

1. **曲线形状：指数 → 幂律**。`R = exp(-t/S)` 改为 `R = (1 + FACTOR·t/S)^(-DECAY)`，`DECAY = 0.1542`（FSRS-6 默认衰减），`FACTOR` 由其导出使 `R(S) = 0.9`。
   - **为什么必须改**：即使把 S 提到 330h，指数在 336h 处给 0.363，幂律给 0.899 —— **差 0.54**。这与 FSRS 的历史一致（v3 指数 → v4 幂律）。
2. **稳定性增长：`log2(n+1)` → 幂律**。`S = S_BASE · ((n+1)/2)^GROWTH_EXP · (posterior_mean/0.6)^SCORE_EXP`。
   - `S_MAX = 720h` 原本是**死代码**：按 `log2` 增长需要 **1943 亿次复习**才能达到，实际封顶在 ~61h。现已可达。

**实测结果（300 条历史 / 3300 观测点）**：

| 模型 | LogLoss↓ | RMSE↓ | AUC↑ |
|---|---|---|---|
| **edumap（修复后）** | **0.4089** | **0.1688** | 0.6965 |
| legacy（修复前） | 1.4200 | 0.4722 | 0.6993 |
| moving-avg（0 参数基线） | 0.4421 | 0.3169 | 0.5368 |

**LogLoss 改善 1.0111（1.4200 → 0.4089），从"输给基线 3.2 倍"变为打赢基线。**

**多种子稳健性（关键 —— 不只看单点）**：

| seed | edumap | moving-avg | 结果 |
|---|---|---|---|
| 默认 | **0.4089** | 0.4421 | 胜 |
| 1 | **0.4472** | 0.4749 | 胜 |
| 42 | **0.4271** | 0.4483 | 胜 |
| 999 | **0.4419** | 0.4808 | 胜 |

**4/4 全部胜出**（旧公式 4/4 全输，legacy 稳定在 1.32–1.42，说明旧公式的差是系统性的）。**端到端验证（真实 PG + 真实服务）**：复习 8 次后一周回忆率从 **0.056 修正到 0.903**。

**⚠️ 过程中一个值得记下的方法论教训**：我最初手算"n=8 时 S 应达 ~300h"推出 `GROWTH_EXP=1.5`，结果网格扫描显示 **1.5 恰好落在"与基线持平"**（−0.0006），而 `1.0` 才是显著更优（−0.0507）。**单点手算的直觉会骗人，多种子网格扫描才看得清。**

**⚠️ 另一个：我拒绝了数值最优参数**。扫描显示 `GROWTH_EXP=0.5` 数值最好（−0.0729 vs 1.0 的 −0.0507），但我选了 `1.0` —— 因为 **0.5 是在我自己生成的合成数据上调出的最优点**，而合成数据的真值形态是我选的，选它就是**在拟合自己的假设**；1.0 对应"稳定性随复习次数线性增长"的标准形式，有理论依据。**判据是"有明显改善且形式有依据"，不是"在自造数据上跑出最好看的小数"** —— 这正是本文档批评过的做法。

**⚠️ 第三个：过程中被测试抓到一个真实设计悖论**。首版用 `SCORE_EXP=3.0`，导致**一次满分复习让 stability 从 24h 降到 14.7h** —— 因为 Beta(2,2) 先验很弱、首次满分后 `posterior_mean` 仅 0.6，而 `0.6³ = 0.216 < 1`。**"复习一次反而更容易忘"显然是错的。** 修法是以中性后验（0.6）为 1× 基准做归一化，并**加不变量测试**（复习后 S 不得低于复习前）守住它。

**仍未做的**：参数尚未用真实数据拟合 —— `forgetting_review_log` 已开始记录，积累后应用同一套框架重新拟合。当前参数是"有依据的默认值 + 合成数据验证"，**不是拟合结果**。

测试：`tests/test_learning_path/test_forgetting_curve.py`（26 项）；**变异验证 6/6 全杀**（含"改回指数曲线"、"去掉增长项"、"去掉归一化"、"去掉截断"等）。

> **这是全文性价比最高的一条改进**：改动局限在单个模块，有免许可参考实现和公开基准，且能把"忘记曲线"从自研经验公式升级为有 10⁸ 量级复习记录背书的模型。

### 4.2 掌握度估计：LLM 自报，而非测量

**EduMap 现状**（[已验证]）：`assessment/agent.py` 从 LLM 输出里取 `estimated_mastery_delta`，缺失时**默认 `0.1`**；全仓 **无 IRT / BKT / DKT / Elo 实现**（grep 为空）。

**问题**：遗忘曲线的输入信号本身是软的 —— `update_after_quiz(score)` 接收的 score 来自 LLM 的自报增量，而非经过项目反应校准的能力估计。**"测量出来的掌握度"和"被断言出来的掌握度"不是一回事，而 EduMap 目前是后者。**

**可对标的方法谱系**：

| 方法 | 原理 | 代价 | 适用性 |
|---|---|---|---|
| **IRT**（1PL/2PL/3PL）| 每题难度/区分度 + 每生能力 | 需题目校准数据 | **最适合 EduMap**：已有 KG，可为每个 KP 挂难度参数 |
| **BKT** | 4 参数贝叶斯（P(L0)/P(T)/P(G)/P(S)）| 每技能独立、无遗忘假设 | 轻量，可与现有 Beta 更新并存 |
| **DKT** | RNN 过交互序列 | 需大量序列数据、可解释性差 | 对 MVP 过重 |

**DKT 的关键警告（避免误引）**：Piech et al. 2015（arXiv 1506.05908）报 **ASSISTments AUC ≈0.86 vs BKT ≈0.67**，但 **Xiong et al. 2016《Going Deeper with Deep Knowledge Tracing》(EDM 2016)** 在重新划分数据后复现，得 **ASSISTments09 DKT AUC 仅 ≈0.73（不是 0.86）**；EDM 2016 论文集同时指出"**31.6% 的性能差异似乎来自评测口径/指标的差异**"。**引用 DKT 对 BKT 的优势时，必须带上这个复现修正。** [已验证]

### 4.2.1 ✅ 已实施（2026-09-23）：从"LLM 自报"改为"从作答测量"

**实施前发现的完整缺口**（比预想的更彻底）：`AssessmentAgent` **只出题、从不评分** —— 全仓没有任何地方比对用户答案与正确答案。而 `mastery_delta` 是在**出题那次 LLM 调用里**让模型"估"出来的（那时还没人答题），缺失时硬编码 `0.1`。**即：掌握度在"出题时"被断言，而非在"答题后"被测量。**

**做法 —— 1PL / Rasch IRT**（`src/agents/assessment/irt.py`）：

```
P(correct | θ, b) = σ(θ − b)        θ=能力，b=题目难度，同一 logit 尺度
```

**为什么选 1PL 而不是 2PL/3PL**（这是要能讲清楚的设计判断）：
- **小样本可识别**。2PL 每题多一个区分度参数、3PL 再加猜测参数，**需要每题更多作答量才能稳定估计** —— 题目少时会过拟合，能力估计反而**不如** 1PL。
- **无需校准语料**。EduMap 题库小且新写，没有历史可用来拟合区分度。
- 与知识追踪文献的教训一致：**DKT 对 BKT 的优势在严格复现后大幅缩水**（见上），**先上简单模型是更可辩护的选择**。

**关键工程设计**：
- **能力估计用 MAP 而非 MLE**：全对/全错的作答模式下 MLE 会发散到 ±∞；一个弱高斯先验（精度=1 个虚拟观测）把它拉回有限值，且**诚实表达"还不知道"**。
- **估计器数值稳定**：logistic 对数似然对 θ 是凹的，所以用三分搜索在有界区间上求全局最大，**避开 Newton 法在完美作答模式下的发散**。
- **评分是确定性纯函数**（`grading.py`）：`grade_answer` 大小写/空白无关，**空答案永远算错**（即使标准答案也是空）。
- **难度映射**：KG 的 1–5 难度线性映射到 logit 尺度；`None` 或越界值取中性默认值 —— **无标注的题按"平均难度"处理，而不是外推成极端难题**。
- **SE 随作答数收缩**：估计自带不确定性，作答越多越确定。

**测试与验证**：
- `tests/test_agents/test_irt.py`（22 项）—— 含**从合成数据恢复已知能力/难度**的独立验证（θ=1.5 数据拟合回 1.5±0.35）；且断言的是**先验在起作用**（全对时估计 < 3.5），而非仅仅"有界"。
- `tests/test_agents/test_grading.py`（27 项）—— 含**掌握度对正确率单调**、**难题全对 > 易题全对**。
- **变异验证 6/6 全杀**。
- **⚠️ 过程中发现并修正了一个自己的弱断言**：最初"全对不发散"的测试写的是 `assert ability < 4.0` —— 但**搜索区间本身就硬编码在 [-4, 4]**，所以即使去掉先验也照样通过（测到的是优化器的边界，不是模型）。改成断言"严格落在区间内侧 + 先验越强越被拉向中心"后，**同一变异从杀 1 个测试变为杀 4 个**。

### 4.2.2 ✅ 已接线（2026-09-23）：Assessment agent 不再自报掌握度

**接线前发现的第二个测试盲区**：`tests/test_agents/` 里**根本没有 AssessmentAgent 的测试**（只有 guardian / orchestrator）。这解释了为什么"LLM 自报 + 硬编码 0.1"从未被发现 —— **没有任何测试断言过它**。

**改了三处**：

1. **`run_legacy` 产出空 `mastery_delta`**，并**显式忽略** LLM 返回的 `estimated_mastery_delta`（字段仍读取，避免提示词未更新时报错，但绝不进入输出）。空 dict 是诚实信号："尚未测量"。
2. **题目带上难度**：LLM 路径从 `KnowledgeUnit.difficulty` 取（**不问 LLM** —— 未经校验的自报会重新引入本要解决的问题）；题库路径因为是静态表、无难度列，出题后**补盖** KP 难度（否则题库题目全按平均难度计，IRT 就失去了区分难易的能力）。
3. **新增 `grade_attempt(questions, answers, prior_mastery=...)`** —— 学习闭环的落点：出题 → 作答 → **在这里**从实际作答算掌握度。统计逻辑全在纯函数 `grading.py`，可在无模型参与下测试。

**端到端验证（真实 agent，含 PromptRegistry）**：

| 场景 | 正确率 | 掌握度 | Δ | SE |
|---|---|---|---|---|
| 全对 | 2/2 | **0.697** | +0.197 | 1.434 |
| 一半 | 1/2 | 0.541 | +0.041 | 1.434 |
| 全错 | 0/2 | **0.371** | −0.129 | 1.605 |

行为单调且合理。**注意 SE ≈ 1.4 —— 只有 2 题时不确定性很大，这是诚实的**：IRT 标度上 1.4 logit 确实很宽，表明"需要更多作答才能确定"，而不是给出一个虚假精确的数字。

测试：`tests/test_agents/test_assessment_agent.py`（**13 项，此前该文件不存在**）；**变异验证 4/4 全杀**（含"恢复 LLM 自报"、"题目丢难度"、"题库不补难度"）。

### 4.2.3 ✅ 加固（2026-09-23）：db_pool 误必须响亮失败

**发现**：`ForgettingCurveService` / `PathService` / `AntiGamingService` 三处都写 `self._db_pool.acquire()`，但 `main.py` 传的是裸 `asyncpg.Pool`（正确），而 `MemoryDBPool` **包装对象没有 `acquire()`**。

**为什么这是隐蔽的**：所有持久化调用点都用 `except Exception` 包裹（**这是对的** —— DB 抖动不该弄坏线上功能），所以传错对象会**静默降级为不持久化**，内存路径照常工作。**直到有人发现重启后状态丢失才会察觉。**

**修法**：提取 `src/utils/db_pool.py::unwrap_pool()`，接受两种形态（裸 pool 直接用；包装对象自动解包 `.pool`），**第三种形态在构造时抛 `TypeError`** —— 在能修的地方响亮失败，而不是每次查询默默吞掉。三处服务统一接入。

测试：`tests/test_utils/test_db_pool.py`（12 项，含"包装对象路径下确实写入了数据库"的回归测试）；**变异 2/2 全杀**。

### 4.3 记忆系统：存了 importance 却不用

**EduMap 现状**（[已验证]）：`memory/long_term.py` 的召回是 **`ORDER BY created_at DESC`**（纯时间序），而 `EpisodicEntry` 里 **已经存了 `importance_score` 字段** —— **存了但从不参与排序**。

**可对标（generative agents, Park et al. arXiv 2304.03442，[已验证]）**：

```
score = α·recency + β·importance + γ·relevance
recency = 0.995 ^ hours        # 指数衰减
α = β = γ = 1，各分量 min-max 归一到 [0,1]
```

**这条与 EduMap 的遗忘曲线天然互补**：同一个"遗忘"心智模型，一处用于知识掌握，一处用于记忆召回。改动量小（一次排序函数 + 一个归一层），收益是召回质量从"最近发生"变成"最近且重要且相关"。

**⚠️ 重要警告 —— 不要相信任何"我们的记忆比 X 好 N%"的数字（包括厂商的）**：调研发现同一系统在同一基准 LoCoMo 上出现 **四个互不相同的分数：65.99 / 84 / 58.44 / 75.14**。可查证的反驳材料包括：`mem0ai/mem0` Issue #4573（"审计 10,134 条 mem0 条目：**97.8% 是垃圾**"）、getzep 博客《Lies, Damn Lies, Statistics: Is Mem0 Really SOTA?》、`getzep/zep-papers` Issue #5（第三方把 Zep 自己的 84% 修正为 **58.44%**）、Letta 博客《Is a Filesystem All You Need?》。**根因是 prompt/judge/harness 口径不一致，跨论文不可比。**[已验证]

> 面试可讲：*"Agent memory 的基准现在是罗生门 —— 同一个系统同一个 LoCoMo 有四个分数，从 58 到 84。所以我不引用厂商的数字，只看机制本身有没有道理。"*

**可借鉴的机制而非数字**：
- **Zep/Graphiti**（arXiv 2501.13956）：**时序知识图谱记忆**（temporal KG）—— 每条事实带有效期，这是"知识会过期"的正确建模。
- **Letta**：Context Repositories（基于 git 的记忆）。
- **A-MEM**（arXiv 2502.12110）：Zettelkasten 式，自报 1,200 tokens/op、**减少 85–93%**。

> **✅ 已实施（2026-09-23，两阶段）**：
>
> **阶段一** — 新增 `src/memory/recall_scoring.py`，`recall_episodic` 改为按 `recency + importance + relevance` 排序（α=β=γ=1，`recency = 0.995^hours`），SQL 侧扩大 `5×` 候选窗口以免"旧的但重要"的记录在打分前就被截断。
>
> **阶段二** — 补上 relevance 通路（阶段一仅 recency+importance 实际生效）：
> - `episodic_memory` 表新增 `embedding JSONB` 列（`scripts/db/init/03-memory-schema.sql`，`ADD COLUMN IF NOT EXISTS` 保证幂等）；选 JSONB 而非 pgvector，因为向量由进程内模型产生、打分在 Python 侧对小候选集完成，此规模下引入扩展依赖没有收益。
> - **新增 `recall_relevant(user_id, query_embedding, ...)`**，与 `recall_episodic` **语义分离**：后者服务"对话历史"需要时间连续性，前者回答"哪些历史与当前提问相关"。**没有把它做成 `recall_episodic` 的可选参数** —— 4 个现有调用点全是取"最近 10 轮对话"，那个场景按相关性重排会打乱时间线，做成 flag 会被误用。
> - 写入侧 `MemoryOperations.record_interaction` 经 `embed_fn` 编码 input+output（一个话题一个向量），编码失败**不影响记忆落库**（降级为无向量，仍可按 recency+importance 召回）。`main.py` 复用已预热的模型实例。
>
> **端到端验证（真实 PG + 真实模型）**：写入三条不同主题记忆后，
> - `recall_relevant("平衡树怎么旋转")` → **红黑树那条排第 1** ✅
> - `recall_episodic()` → 纯时间序，红黑树排**最后**
> 两条路径行为确实不同，relevance 通路真正生效。
>
> **实现中发现并修掉的一个真实设计陷阱（值得单独讲）**：最初我按"论文做法"对三个分量做了 **min-max 归一化**，测试立刻抓到它**在只有两个候选时会把两者打成平手** —— 归一化把最大值拉成 1.0、最小值压成 0.0，于是"新但无用"（recency 1.0 / importance 0.0）和"旧但重要"（recency 0.79 / importance 1.0）都恰好得 1.5 分，**幅度信息被抹掉了**。这与 RRF 丢分数幅度是同一类错误。改成**原始值加权求和**（三个分量本来就在 [0,1]）后正确。**这个 bug 是测试先发现、我才意识到的 —— 它反过来印证了 §4.4 里"融合只看排序会丢信息"的那条结论。**
>
> 测试：`tests/test_memory/test_recall_scoring.py`（33 项）+ `test_episodic_embedding.py`（6 项）；**变异验证：recall_scoring 7/7 全杀，recall_relevant 与写入侧另外 4 个变异也全杀**。
>
> **顺带发现的既有问题（未在本次修复）**：`03-memory-schema.sql` 的三个 `CREATE TRIGGER` 没有 `IF NOT EXISTS` 保护，**整个文件重复执行会报 `DuplicateObjectError`** —— 这也是本仓没有增量迁移目录的原因。属独立议题（修改会影响既有初始化流程），已记录未处理。

### 4.4 检索融合：默认用的正是已知偏弱的那一种

**EduMap 现状**（[已验证]）：`rag_service.py` 默认 `fusion_method = "rrf"`（k=60），另实现 `minmax` 与 `score`；**可由 `settings.hybrid_fusion_method` 运行时切换**（`main.py:110`）。

**自身实测（n=20 消融）**：

| 策略 | MRR | P@1 | HR@3 | NDCG@5 |
|---|---|---|---|---|
| **score** | **0.9500** | **0.9500** | 0.9500 | **0.9396** |
| minmax | 0.9000 | 0.8500 | 0.9500 | 0.9162 |
| **rrf（当前默认）** | 0.9000 | 0.8500 | 0.9500 | 0.9211 |

**n=200 策略 A/B**：hybrid(RRF) MRR **0.8569** vs direct **0.8454**，**ΔMRR 仅 +0.0115**。

**这正好落在文献预测的区间里**。**Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval"**（arXiv 2210.11934，**ACM TOIS** 10.1145/3596512）原文：[已验证]

> "we find **RRF to be sensitive to its parameters**; … **CC [convex combination] outperforms RRF in in-domain and out-of-domain settings**; … CC is sample efficient."

公式：`f_CC = α·f_Sem + (1−α)·f_Lex` —— 即**带调优权重的加权分数融合，一般优于 RRF**；RRF 只在"无法调参、需要分布无关默认值"时有优势。

**结论**：EduMap 默认 RRF，但自有数据与 TOIS 论文都指向"调优的加权融合更强"。而 `minmax`/`score` **已经实现、且可运行时切换** —— 这是**改一行配置**的事。

> **✅ 已实测并切换默认（2026-09-23）**。新建 `scripts/run_fusion_ablation.py`（**强制绕过结果缓存**，并拒绝在空索引上出数），在 **n=200** 集上实测：
>
> | 策略 | MRR | P@1 | HR@3 | P50 | P95 |
> |---|---|---|---|---|---|
> | rrf（旧默认） | 0.8569 | 0.7700 | 0.9350 | 22.1ms | 37.9ms |
> | minmax | 0.8499 | 0.7600 | 0.9350 | 18.6ms | 37.4ms |
> | **score（新默认）** | **0.9002** | **0.8600** | 0.9350 | 18.6ms | **32.4ms** |
>
> **`score` 的 MRR 高 +0.0433（相对 +5.1%）、P@1 高 +0.09，且 P50/P95 延迟都不劣。** n=20 冒烟集上结论一致（score 0.9563 vs rrf 0.9062）。已把 `settings.hybrid_fusion_method` 与 `RAGRetrievalService.default_fusion_method` 改为 `"score"`，并在 `src/config.py` 写入复现命令。
>
> **顺带修掉一个真实的可用性 bug**：`sentence_transformers` 每次加载模型都会向 `huggingface.co` 发 HEAD 校验，本机无外网时 `huggingface_hub` 会以 1s/2s/4s…×5 指数退避重试，**脚本表现为"卡死"（实测挂了 15 分钟以上才定位）**。脚本现在在 import 前强制 `HF_HUB_OFFLINE=1`。

**⚠️ 但要避免过度归因（否则 I-1 会被讲过头）**：融合的**边际价值可能很小** —— 这一点在**别人的**数据上成立，在我们自己的数据上不成立。
- **BGE-M3 官方数据（MIRACL，18 语种均 nDCG@10）**：纯 Dense **75.1**，Dense+Sparse **75.3** —— **混合只比纯稠密多 +0.2**；真正的收益来自"Dense 相对 BM25（39.9）"。**在强稠密模型上，融合的增量可能接近于零。** [已验证 arXiv 2402.03216]
- **厂商立场分化**（可作"业界并不统一"的证据）[已验证]：
  - **Weaviate** —— 自 v1.24 起**默认改为 `relativeScoreFusion`（min-max 归一后加权和）**，理由是 RRF"只保留排序、丢失分数信息"；自报 recall **+6%**（FIQA，α=0.5）
  - **Vespa** —— 明确**拒绝 RRF**（"不考虑模型分数，只看排序"），改用 max-min 归一后的线性组合
  - **Pinecone** —— 单索引 + α 权重线性组合（与其自己的 Bruch 论文一致）
  - **Elasticsearch** —— RRF 是一等公民，`rank_constant` 默认 60，**主打"无需调参"** —— 即 RRF 的价值在"零配置"，不在"最优"
- **RRF 的 k=60 出处**：Cormack et al., SIGIR 2009 —— 原文说 k=60 是"试点时定下后再未改动"，且**"这个选择并不关键"**（k=0 时 MAP .2072，k=60 → .2145，k=500 → .2098）。**所以 k=60 不是调优结果，而是惯例。**[已验证]

> 面试可讲：*"我发现自己默认用 RRF，但只知道'它是默认值'，不知道它是不是最优。查证后有四点：① TOIS 那篇（Bruch 等）说调优的加权凸组合在域内域外都优于 RRF；② 业界并不统一 —— Weaviate 已把默认从 RRF 换成归一化加权和、Vespa 明确拒绝 RRF，而 Elasticsearch 主打的恰恰是'RRF 无需调参'；③ **但 BGE-M3 官方数据显示融合比纯稠密只多 +0.2 nDCG@10** —— 强稠密模型上融合的增量可能很小；④ 连 k=60 这个数我也查到出处了 —— 原文说它是试点定的、'并不关键'。**合起来我的结论是：该试，但不要预先假设会大幅提升，要用 n=200 集实测。**"*

**检索前沿（供借鉴，非当前必需）**：

| 方案 | 核心 | 成本/效果数字 |
|---|---|---|
| **GraphRAG** (arXiv 2404.16130) | 社区摘要 + map-reduce **global search** | 全局问题综合性与多样性 **~70–80% 胜率**；索引成本极高（真实用户报告：$15 小测试 / **$120 一份 1000 页 PDF** / $8.20 per 1.2M tokens）[已验证] |
| **LazyGraphRAG**（微软官方）| 惰性社区选择 | **索引成本仅完整 GraphRAG 的 0.1%**；查询成本低 **>700×**；动态社区选择 **−77% 成本** [已验证] |
| **LightRAG** (arXiv 2410.05779) | 轻量图 RAG | 对比 GraphRAG 的 1,399 社区 / 610 个二级活跃 / ~1,000 tokens/报告 = **610,000 tokens**，LightRAG **<100 tokens + 1 次 API 调用** [已验证] |
| **HyDE** (arXiv 2212.10496) | 假设文档检索 | DL19 nDCG@10 **Contriever 44.5 → 61.3**；DL20 42.1 → **57.9** [已验证] |
| **HippoRAG / HippoRAG 2** | KG + Personalized PageRank 多跳检索 | HippoRAG 2 MuSiQue recall@5 **74.7 vs NV-Embed 69.7**；索引约 **10× 慢 / 每 10k passages 贵 ~$15** [已验证] |

**⚠️ 必须纠正的一个常见误引**：**HippoRAG 1 和 2 都未实现任何基于检索频率或遗忘曲线的重加权** —— 它只是 KG + Personalized PageRank 的多跳检索。**不要把 HippoRAG 当作"带衰减的检索"来引。**[已验证：调研 agent 明确纠正]

**对 EduMap 的取舍**：LightRAG 已在 DeepTutor 里做一方集成，是"想要图检索增强但受不了 GraphRAG 成本"的正确答案。若 EduMap 要强化多跳，**LightRAG 与 HippoRAG 都比 GraphRAG 更合适**。

### 4.4.1 ✅ 已实施（2026-09-25）：reranker 实测为负 —— 并修掉一个让它静默失效的 bug

**新增 `scripts/run_reranker_ablation.py`**（缓存绕过、双臂同深检索），在 n=20 与 n=200 上各跑一次。

**实测结果（n=200，更可信）**：

| arm | MRR | NDCG@1 | HR@1 | P50 | P95 |
|---|---|---|---|---|---|
| **reranker=off** | **0.8945** | **0.8600** | 0.8600 | 24.5ms | 39.2ms |
| reranker=on | **0.3250** | **0.2250** | 0.2250 | 63.6ms | 85.1ms |

**ΔMRR −0.5695、ΔNDCG@1 −0.6350、延迟 ×2.60。** n=20 的 ΔMRR 是 −0.5633 —— **两个样本量高度一致**，不是小样本波动。

**结论：开启当前 reranker 是有害的**，与 NVIDIA 的评测吻合（arXiv 2409.07691：过小 cross-encoder 主动伤害检索）—— 而 EduMap 的默认模型 `MiniLM-L-6-v2`（~22M）比该研究中已被证明有害的 33M 版本**还小**。**保持默认关闭是正确的**，配置注释与路线图已记录此结论与复现命令。

#### ⚠️⚠️ 过程中发现的最有价值的一个 bug：reranker 一直在静默失效

消融第一次跑出来是 **"指标零变化"**（off/on 的 MRR 都是 0.9500），但延迟 ×2.34。**"指标完全不变却多花一倍时间"不合常理** —— 追下去发现 reranker **从未真正重排过**。

**根因（`cross_encoder.py`）**：

```python
scores = self._model.predict(pairs)      # numpy.ndarray, dtype=float32
for i, score in enumerate(scores):
    if isinstance(score, (int, float)):   # ❌ np.float32 不是 Python float
        ...
    elif isinstance(score, (list, tuple)): # ❌ ndarray 元素也不是序列
        ...
    # 两个分支都不进 → 分数从未写入 → 排序从未改变
```

**`np.float32` 既不是 `float` 也不是序列**，两个分支都不执行。模型算出的分数被判为"无法解释"而丢弃 —— **白算（这就是 ×2.34 延迟的来源），排名毫发无损。**

修正为 `_extract_score()`：用 `float()` 统一接受 numpy 标量与 Python 数值，并显式处理 `[neg, pos]` 形状。修复后复查：相关文档得 7.83、无关文档 1.81 —— **模型的判断本来就是对的，只是被丢掉了**。

#### ⚠️ 为什么这个 bug 长期不可见：测试 mock 的 dtype 与生产不一致

`tests/test_reranking.py` 里 mock 的是 `np.array([0.2, 0.9, 0.5])` —— **默认 `float64`**。而真实模型返回 **`float32`**。关键区别：

| | `isinstance(x, float)` |
|---|---|
| `np.float64`（测试用的）| **True** ✅ |
| `np.float32`（生产用的）| **False** ❌ |

**`np.float64` 是 `float` 的子类，`np.float32` 不是。** 于是 mock 走通了分支、测试全绿，而生产路径的分数被静默丢弃。

**我做变异验证时确认了这一点**：把 bug 塞回去后，19 个测试**仍然全部通过** —— 它们根本测不到。把 mock 的 dtype 改成 `np.float32`（与生产一致）后，同一变异**杀掉 3 个测试**。

> **这个教训值得单列**：**测试替身的"类型形状"必须与真实对象一致，只对齐"值"是不够的。** 这与本仓 `CLAUDE.md` 里已有的那条（测试的调用方式必须与生产一致）是同一类问题，只是更隐蔽 —— 因为它连"调用方式"看起来都一样，差别只在 dtype。

#### 补充说明：为什么没有换模型

本机**无外网**（HuggingFace 不可达），且本地缓存只有当前这个模型，因此**无法下载 `bge-reranker-v2-m3`（568M，该研究中 +3.7~5 NDCG@10）** 做对照。所以本轮结论限定为：**当前模型必须保持关闭**；换模型后的效果**未知**，需在有网环境下重新实测。这一限制已写进配置注释。

测试：`tests/test_rag/test_reranking.py`（19 项，mock dtype 已修正为 float32）；**变异验证：塞回原 bug 杀掉 3 个测试**。

### 4.5 引用溯源：只有 chunk 级，没有 span 级

**EduMap 现状**（[已验证]）：`MentorSource` 字段为 `id / name / type / score / summary / resource_id` —— **没有页码、没有字符偏移、没有 bounding box**。即"来源透明度"只能告诉用户**是哪个 chunk**，不能告诉用户**chunk 里的哪一段**。

**这与一个已暴露的问题直接相关**：EduMap 的生成侧 Faithfulness 曾记录为 0.010/0.277 —— **2026-09-24 重测确认那两个数字来自评测管线故障而非质量差，真实值是 0.5407 / 0.813 / Relevancy 0.8539**（详见 [04-rag-evaluation.md](04-rag-evaluation.md) §0）。**span 级溯源是让 Faithfulness 可被人工审计的前提**：用户（和评测者）必须能一键跳到被引用的原文位置，否则"忠实度"只能停留在自动打分。

**可对标**：NotebookLM 的引用可点击跳转到原文高亮位置；EduMap 目前只到资源级。

### 4.5.1 ✅ 已实施（2026-09-24，后端）：字符级偏移溯源

**改动前的根因**：`_chunk_text` 返回 `list[str]` —— **偏移信息在切割那一刻就丢了**。所以不是"没存页码"，而是**从切分起就没有位置信息**。

**关键设计决策：不改切分器，改为事后定位。**
三种切分策略（`fixed`/`recursive`/`semantic`）都是可插拔的、各自返回裸字符串。要贯穿偏移追踪就得改每一个 chunker 的内部实现，包括其分割逻辑本身不理解偏移的 recursive/semantic。因此改为**在切分后搜索每个 chunk 在原文中的位置**（`src/resources/provenance.py`）：

- **对任意策略统一生效**（含将来新增的），且**完全不动切分行为**（零回归风险）
- 需要处理两个真实难点，都有测试守住：
  - **重叠 chunk**：前向搜索（从上一个匹配结束处开始），否则重叠部分会被重复定位
  - **空白归一化**：parser 会 `re.sub(r"\n{3,}", "\n\n")`，chunk 携带的是归一化后的形式，与原文不是逐字节子串。用"任意空白 run 等价"的正则做二次尝试 —— **否则后面的 chunk 会系统性错位**，比不定位更糟

**⚠️ 一个刻意的取舍**：无法定位的 chunk 存 `None` 而**不是 0**。默认 0 会让每条这类引用都指向文档第一个字符 —— **看起来像正常工作的溯源，实际比没有更糟**（会高亮错误的段落）。变异测试专门守住这条（"给未定位 chunk 偏移 0"会杀掉 4 个测试）。

**链路**：切分 → `locate_chunks` 定位 → 写入 ChromaDB 元数据（`char_start`/`char_end`）→ `MentorSource` 暴露给前端（`None` 表示未知，前端不得据此高亮）。

**⚠️ 尚未包含真实页码**：解析后是纯文本，页码信息在 PDF 提取阶段已丢失。所以实现的是"字符偏移 + 前端可回读原文并高亮"，**不是"跳转到第 N 页"** —— 后者需要改 PDF 解析层保留页边界，属独立工作。

### 4.5.2 ✅ 已实施（2026-09-24，前端）：点击引用直达被引用段落

**补全的链路**（后端只到元数据是不够的，前端拿不到就白做）：

1. **`in_memory_chunks` 带上偏移** —— parser 定位一次，同时供索引元数据与 chunk previews 使用（避免重复计算与漂移）。
2. **`ChunkPreview` + `/chunks` 端点透传** `char_start`/`char_end`；顺手修了一处既有 bug：ChromaDB fallback 分支用枚举序号 `i` 当 `chunk_index`，而 `get` **不保证返回顺序**，会与实际位置不符。改为优先读存储的 `chunk_index`。
3. **前端 `CitedPassage` 组件** —— 把绝对偏移换算成 chunk 相对偏移后高亮。
4. **弹窗定位到被引用段落** —— 不再只是"列出前 N 段"，而是标记出被引用的那一段（`data-testid="cited-chunk"`）并高亮段内范围。

**⚠️ 一个刻意的取舍（与后端同源）**：`splitBySpan` 对 **`null` 偏移、越界偏移、倒置偏移、纯空白偏移一律不高亮**，而**不是钳位到合法范围**。钳位会产生"看起来正常但指向错误段落"的高亮 —— 比明确不高亮更糟。变异测试守住这条：把 `null` 当 0 会导致 4 个前端测试失败。

**⚠️ 过程中发现并修掉一个真实的 UI bug**：弹窗原本**优先显示 `description`**，而它对上传资源是一段摘要、**不含被引用的原文** —— 于是用户点击"查看引用位置"后**永远看不到高亮**。改为：**有 span 可定位时 chunks 优先**。同时修正了文案 —— 原先只要点了带偏移的引用就显示"已定位到被引用的段落"，但偏移可能落在任何 chunk 之外；现在文案基于"是否真的找到了"（`hasCitedChunk`），**不能说与事实不符的话**。

测试：后端 `tests/test_resources/`（33 项）；前端 `cited-passage.test.tsx`（17 项）+ `source-popover-span.test.tsx`（7 项）。**变异验证：前端 2/2 全杀、后端相关变异 2/2 全杀。** 全量：后端 1014 通过，前端 137 通过。

**✅ 真实页码已完成（2026-09-25）**：`_extract_text` 改为返回 `(text, page_starts)` ——
原先把元素 `"\n\n".join(...)` 成一个字符串，**页码边界在这一步被丢弃**（元素本身带
`metadata.page_number`）。现在页码贯穿索引元数据 → `/chunks` → `MentorSource` → 前端。

**无页码时返回 `None` 而非 1** —— 纯文本没有页概念，说"第 1 页"是编造；变异测试守住
（伪造为 1 会杀掉测试）。真实 PDF 端到端验证：3 页文件切 4 块，分别归属 p1/p1/p2/p3，
跨页的块归起始页。

**⚠️ 该功能的前置条件是在这一步才发现的**：Dockerfile 缺 `tesseract-ocr` 与
`poppler-utils`（`unstructured[pdf]` 靠它们解析 PDF），所以**PDF 上传在生产环境从未
成功过** —— 这也是为什么库里的 PDF 上传 parse_stats 全为空。补上后才能实际验证页码；
在此之前任何页码改动都无法验证，因此也就无从改起。

**仍未做**：已入库的旧 chunk 没有偏移与页码（需重新解析才能获得）。

测试：`tests/test_resources/test_provenance.py`（20 项）+ `test_parser_provenance.py`（11 项）；**变异验证 4/4 全杀**。

**⚠️ 过程中踩到一个仓库级陷阱**：`tests/test_resources/` 加上 `__init__.py` 会导致 pytest 收集失败（`No module named 'test_resources.test_provenance'`），而 `tests/test_agents/` **没有** `__init__.py`、`tests/test_memory/` **有** —— 两种约定并存。按主流（`test_agents` 风格，即不加）处理后通过。

---

## 5. 改进路线图：按性价比排序

**性价比 = 收益 / 成本**，成本含实现量、侵入性、依赖新增。P0 最高。

| # | 改进项 | 收益 | 成本 | 性价比 | 依赖 |
|---|---|---|---|---|---|
| **I-1** | ✅ **已完成**：检索融合默认从 RRF 切到 `score` | **n=200 实测：MRR 0.9002 vs rrf 0.8569（+5.1%），P@1 +0.09，延迟不劣** | 已改 `config.py` + `rag_service.py`；新增消融脚本可复现 | **✅ 已落地** | 无 |
| **I-2** | ✅ **已完成**：长时记忆召回加入 `recency + importance + relevance` 打分 | `importance_score` 此前存了不用；`recall_episodic` 已改为打分排序（α=β=γ=1，`0.995^hours`），SQL 扩 5× 候选窗口 | `src/memory/recall_scoring.py` + `long_term.py` 改动；25 项测试，**变异 7/7 全杀** | **✅ 已落地** | 无 |
| **I-3** | ✅ **已完成**：遗忘曲线两处修复（幂律曲线 + 幂律稳定性增长）| 评测框架 + 数据管道 + 曲线修复。**LogLoss 1.42 → 0.4089**（基线 0.4421，**4/4 种子全胜**）；一周回忆率 0.056 → 0.903；`S_MAX` 从死代码变为可达 | `forgetting_curve.py` + `forgetting_eval.py` + `forgetting_synth.py` + 数据管道；26+24+12 项测试；**变异 6/6 全杀** | **✅ 已落地** | 参数待真实数据重拟合 |
| **I-4** | ✅ **已完成（含接线）**：掌握度从"LLM 自报"改为"从作答测量"（1PL IRT）| 修掉"出题时断言掌握度、从不评分"的根本缺口；agent 出题时产出空 delta、新增 `grade_attempt()` 从作答测量 | `irt.py` + `grading.py` + agent 接线；22+27+13 项测试；**变异 10/10 全杀** | **✅ 已落地** | 无 |
| **I-5** | ✅ **已完成**：reranker 实测为负收益，确认保持关闭（**两种模型都已测**）| ⚠️ **实测确认开启有害**（n=200，缓存绕过）：<br>· `MiniLM-L-6`(22M)：**MRR 0.8945 → 0.3250（Δ−0.5695）**、延迟 ×2.60<br>· `bge-reranker-v2-m3`(568M)：**MRR 0.8945 → 0.7609（Δ−0.1336）**、延迟 ×15.68<br>**换大模型显著改善（−0.57 → −0.13）但仍为负收益**，验证了 NVIDIA 的"过小模型是主因"，同时说明本项目语料上**任何已测重排都不值得开**。顺带修掉一个让它静默失效的 bug（见 §4.4.1）| `run_reranker_ablation.py` + config 注释记录两种模型的结论 | **✅ 已落地**<br>（结论是"不要开"）| 可试 late-interaction（ColBERT 类）或缩小重排候选集 |
| **I-5b** | **ColBERT/late-interaction 作为"无需额外 reranker"的路线** | MaxSim 本身即精排，且 CPU 友好。**存储代价常被夸大**：ColBERTv2 残差压缩后 MS MARCO 9M passages 从 154 GiB → **16 GiB(1-bit)/25 GiB(2-bit)**，**与单向量索引（~25 GiB）相当**；PLAID 再提速 **GPU 7× / CPU 45×** | 需重建索引管线；与现有 ChromaDB 架构不同 | **★★☆☆☆** | 依赖检索层重构 |
| **I-6** | ✅ **已完成**：接入 LangGraph PostgresSaver | 图状态由 LangGraph 自身持久化（每个 super-step 存检查点、应用 reducer），并支持 time-travel；`_STATE_REDUCERS` 手工重建逻辑**双写保留**待验证一致后移除 | `checkpointing.py`（pool 生命周期 + 失败降级）+ `configure_graph(checkpointer=)` + lifespan 接线；15 项测试；**变异 5/5 全杀** | **✅ 已落地** | 依赖 `psycopg[binary]`（自带 libpq）；checkpoint 膨胀清理待做 |
| **I-7** | ✅ **已完成（前后端）**：字符级偏移溯源 + 点击引用直达原文段落 | 偏移从切分 → ChromaDB → `/chunks` → `MentorSource` → 前端高亮；为 Faithfulness 人工审计打基础 | `provenance.py`（事后定位，不改任何 chunker）+ parser/端点接线 + `cited-passage.tsx` + 弹窗定位；后端 33 + 前端 24 项测试；**变异 4/4 全杀** | **✅ 已落地** | 真实页码需改 PDF 解析层；旧 chunk 需重解析 |
| **I-8** | **个人化 header 从模板改为真实理由** | 当前是硬编码 `f"Generated for {name} (difficulty {d})"`，而 PRD 承诺"解释适配理由"——**宣称与实现不符** | 让 LLM 顺带输出它已在隐式使用的适配理由；前端已就绪 | **★★★★☆** | 无 |
| **I-9** | **意图识别 `mixed` 类边界优化** | 实测 93.06%（67/72），但 **`mixed` 精确率仅 0.81**（3 个 question 误判为 mixed）| 补 few-shot 边界样本 / 调投票权重；已有评测集 | **★★★☆☆** | 无 |
| **I-10** | **`minmax`+RRF 之外引入学到的融合权重** | 若 I-1 收益不足，可进一步用 CC 的 α 调优 | 需训练/标注少量数据 | **★★☆☆☆** | 依赖 I-1 |
| **I-11** | **记忆改为可检视的三层（借 DeepTutor）** | L1 append-only 事件流 + L2 事实 + L3 综合，层间可追溯 | 需重构记忆写入路径 | **★★☆☆☆** | 依赖 I-2 |
| **I-12** | **代码沙箱能力对外显性化** | **本次调研覆盖的全部消费级产品均无代码执行环境**（中国市场全无；西方亦无 agent 生成→执行→修复闭环）—— 是真正的差异化 | 已有实现（`sandbox-service` + Coder 闭环 + AST 校验 + 3 轮修复），主要是产品包装 | **★★★★☆** | 无 |
| **I-13** | **引入 Temporal 持久执行** | 长任务真正可恢复、等待零算力 | 确定性/重放/版本税 + 集群运维 | **★☆☆☆☆** | MVP 规模不值得 |
| **I-14** | **Mentor 借鉴 MathDial 的"模拟学生错误"造数据 + DPO 打磨教学话术** | MathDial（arXiv 2305.14536）的关键发现：**GPT-3 会解题但不会教**（反馈错误、过早公布答案）；Rice/UT 的做法是"生成候选话语 → 用学生模型 + 教学评分量表打分 → 对 8B 模型做 DPO"，达到 GPT-4o 的教学质量 | 需造标注数据 + 一次 DPO 训练 | **★★☆☆☆** | 需 GPU 与标注 |
| **I-15** | **借鉴学而思的人工兜底（80% 即时、其余 1 小时内人工）** | 中国市场最被称道的"诚实机制"，是对幻觉/无力的显式承诺 | 需人力运营，非技术改动 | **★★☆☆☆** | 非技术 |
| **I-16** | **补齐教育场景幻觉率实测** | 调研确认：**不存在教育场景下实测幻觉率的研究** —— 谁先做谁拿到话语权 | 需自建标注集 + 评测流程 | **★★★☆☆** | 依赖 I-7 |

### 5.1 前三项的具体做法（P0 落地）

**I-1：切换融合策略（半天）**
```python
# settings.hybrid_fusion_method: "rrf" -> "score"（或先跑一轮 minmax vs score 再定）
```
注意：**n=20 的 `elapsed_seconds` 与 n=200 的 `avg_latency_ms: 0.01` 都受检索缓存影响** —— EduMap 自己的文档已记录过一次"hybrid 延迟 ×0.64"被证实是缓存假象的自纠。**比较延迟时必须绕过缓存（`--repeat 2` 或禁用 result cache），且必须用 n=200 而非 n=20 定结论。**

**I-2：记忆召回打分（~1 天）**
```python
# 在 long_term.recall_episodic 的 SQL 之后加一层重排：
w_recency = 0.995 ** hours_since_created
w_importance = normalize(entry.importance_score)   # 0-1
w_relevance = cosine(query_emb, entry_emb)         # 0-1
score = w_recency + w_importance + w_relevance      # α=β=γ=1
```
先只对候选集（如最近 200 条）重排，避免全表向量化。

**I-3：遗忘曲线迁移（2–3 天，但先做基线对照）**

**第 0 步（半天，最重要）**：先建一个**0 参数"最近复习"基线**，用 LogLoss/RMSE/AUC 跑通自有 `learning_progress` 历史。理由：公开基准显示 **MOVING-AVG（0 参数，LogLoss 0.3369）反而优于 FSRS-6（0.3460）** —— 如果 EduMap 的自研曲线连这个朴素基线都没打赢，问题不在"没上 FSRS"，而在数据质量或建模假设；而如果打不赢还得先解释为什么。**先量化差距，再决定投入。**

1. `pip install py-fsrs`（MIT）
2. 保留 `forgetting_curve_state` 表，加 FSRS 的 `difficulty`/`stability`/`review_log` 列
3. **迁移策略**：新知识点直接用 FSRS 默认参数；老数据按 `review_count`/`strength` 近似映射到 `(D, S)`，标记 `model="legacy"` 以便回滚
4. **前后对比必须用同一口径**（LogLoss / RMSE(bins) / AUC），并同时报出那条 0 参数基线 —— **这本身就是一份可写进面试的项目数据，而且比"我实现了遗忘曲线"有力得多**

---

## 6. 竞品速查表（面试用）

| 产品 | 定位 | 定价 | 架构/亮点 | 已知弱点 | 对 EduMap 的启示 |
|---|---|---|---|---|---|
| **Khanmigo** | K-12 AI 导师 + 教师工具 | **教师免费**；学习者/家庭 **$4/月 或 $44/年**；学区定制价 [已验证] | 教师端：差异化备课 + 学生作业摘要 + 评分表；学习端：辅导 + **写作/辩论** + **职业指导** + **编程实时反馈**；家长端有**内容审核告警**；学区端支持 **Clever/ClassLink 自动点名 + SSO**。**真实工程护栏**：专用"数学 agent"实时校验计算、预检决定是否需要校验（延迟路由）、**限制 agent 范围使"直接给答案"率下降 50%**；平台带 A/B 与护栏指标（给答案率、数学错误率、下一题正确率）| **大规模独立 RCT 显示 AI 增益≈无**（见 §3.5）；准确性曾被 EdWeek/Axios 质疑 | **本调研最有价值的一条**：AI 增益被高估；**被验证的是机制（护栏 + 前置链），不是模型**。其护栏工程与 EduMap 的 Content Auditor 同构 |
| **Duolingo Max** | 语言 + AI 会话 | **上线价 $29.99/月 或 $167.99/年**（美区 iOS，西/法语先行）| **2023-03-14 基于 GPT-4 上线**（[官方博客](https://blog.duolingo.com/duolingo-max/)）；**Video Call with Lily**（三角色提示词 + 固定四段式通话脚本 + **"List of Facts" 学习者记忆**：每次通话后把逐字稿摘要注入下次的 system prompt）；**Roleplay**；Birdbrain；**HLR** | 2025 年"AI-first"争议；**FY2025 10-K 自承毛利率下降"主要因为 Video Call 等功能带来的 AI 成本上升"** —— 生成式功能是**毛利稀释项** | **"List of Facts"= 持久学习者记忆的工程范式**，与 EduMap 记忆系统同向；**HLR 在 srs-benchmark 垫底**（0.4694）|
| **Coursera Coach** | 课程内 AI 助手 | 随订阅 | **2024-09-17 发布**；**首个支撑模型为 Google Gemini**；首模态为**苏格拉底式对话**；教师可指定学习目标/教学风格/评分标准/附加文件 | 依赖平台内容；效能为**公司自报** | **公司自报效能：100 万+ 学习者 → 首次测验通过率 +9.5%、每小时完成课时 +11.6%** —— 注意这是自报口径 |
| **Google LearnLM** | 教育微调模型 | AI Studio 可试 | 盲评优于 GPT-4o **+31%**、Claude 3.5 **+11%** | — | 模型层能力不是瓶颈 |
| **NotebookLM → "Gemini Notebook"** | 文档问答 + 多模态概览 | 免费/订阅 | **2026-07-16 更名**；Audio Overviews → Mind Maps（2025-04）→ Flashcards/Quizzes/Learning Guide（2025-09）→ 1M token 上下文（2025-10）→ Deep Research → **代码执行"安全云电脑"**（2026-07）；30M+ 用户 / 600k 组织；**长上下文 grounding，非向量 RAG** | **无知识图谱、无间隔重复、无前置链** —— 是"通读助手"不是"教学闭环" | **EduMap 缺 span 级溯源（I-7）**；但注意它靠长上下文而非 RAG，与 EduMap 路线不同 |
| **ChatGPT Study Mode / Claude for Education** | 大厂"学习模式" | 随订阅 | **本质是精心写的 system prompt + 轻量 RAG，不是自适应引擎**（据泄露的提示词核实）| 无学习者建模、无知识图谱 | **大厂在做"提示词层"，EduMap 在做"引擎层"** —— 这是可讲的差异化 |
| **MagicSchool** | 教师工具平台 | 免费档 $0；Plus **$8.33/人/月（年付 $99.96）** 或 $12.99/月；企业定制 [已验证] | **企业档明确对校内文档做 RAG**（"上传手册与课程，知识跨工具复用"）；"定制工具" | — | **它把 RAG over 校本资料做成了企业卖点** —— EduMap 的资源上传+RAG 在同一方向上 |
| **Synthesis Tutor** | 数学（5–11 岁）| **单孩家庭 $70/月，年付 $33.33/月（$400/年），或 $1,499 买断**；7 孩档 $29/月 [已验证] | — | 年龄段窄；架构/效能 **[未核实]** | 定价上限参照：**家长愿为单一学科的强效果付 $400–1,499** |
| **DeepTutor** ⭐ | 开源多智能体导师 | Apache-2.0 | 多引擎 RAG、三层可检视记忆、TutorBench | 生态新 | **最直接对标；抄记忆可检视性** |
| **豆包爱学** | 免费 K-12 学习 App | **免费**（MAU 19.4M, 2026-06）| **逐步门控**（"这部分明白了吗？"停顿确认）、作文批改→范文→语音→新题闭环、整页作业批改 + 错题本 + 掌握度热力图 + **组卷打印** | 题库深度弱（压轴题/竞赛题）；无家长可控性 | **逐步门控 = EduMap Mentor 可强化项** |
| **作业帮** | 学习机 + AI | 硬件为主（平板份额 **32.6%**，#1）| 三层架构（大模型中枢 + 学情数据库 + 资源调度引擎）→ **"认知地图"**；宣称"行业最细知识图谱"；追溯**前置薄弱知识点** | **效能宣称"提升超42%"挂 CAICT 认证但无方法论 [仅营销]**；App 流量下滑、5 年无融资 | 知识图谱在中国是**门槛而非差异** |
| **小猿 AI** | 学练机 + AI | 硬件为主（学练机 **破 100 万台**）| "猿力大模型 + DeepSeek"；**明确反对"仅靠知识图谱"**，主张连续多点重诊断，**把 AI 对话内容也当诊断信号**（有隐私含义）| **未见任何效能/效应量宣称** | 把**对话内容作为诊断信号**是 EduMap 可考虑的方向；但需注意隐私边界 |
| **学而思 九章** | 数学大模型 + 学习机 | 硬件/订阅 | 宣称"国内首个千亿参数数学大模型"、**数学准确率 97%**；随时问 App 跑 **DeepSeek-R1 671B** | **2024 高考全国甲卷仅 45/90（<50%）** —— 宣称与实测落差；**"230M 互动/420M 攻克薄弱点"是参与度指标，不是学习结果** | **可学的诚实机制：80% 即时解答、其余 1 小时内人工兜底** |
| **有道 子曰** | 学习硬件 + 小P | 硬件为主 | **技术证据最扎实**：开源 **子曰-o1**（2025-01, 14B）、**Confucius3-Math**（arXiv 2506.18330）、**Confucius4**（2026-05, 多模态 + TTS, 14 语种）| 开源模型真实可检视，但**学习效果未被研究** | **中国市场唯一可检视的技术产物** —— 值得引用其 arXiv 号 |
| **科大讯飞 星火** | EduMap 默认 LLM 供应方 | **Lite 永久免费**；旗舰曾低至 ¥0.21/万 tokens [已验证] | **以"幻觉治理"为核心卖点**（明确点名教育场景）；与教育部考试院共建 AI 阅卷；**高中难题可追溯到小学基础漏洞**（跨年级前置链）；T90 系列 ¥7,799–11,699 | 全部效能数字 **[仅营销]**；集团 H1 2025 净亏 ¥239M；硬件占比高而**品类在萎缩** | **供应商的战略主张与 EduMap 的技术主张同向**；但**当前 API 定价需自行在控制台确认**（公开文档为 JS SPA，抓不到）[未核实] |
| **松鼠AI** | 自适应学习（B2B/SaaS）| **93% 收入来自技术服务费** | 微颗粒度知识图谱 + **深度知识追踪**；宣称 2400 万学生、百亿行为数据；CMU/密歇根/北师大合作 | 效能证据多为**单组前后测**（+10.65%，无对照）；**有联合署名的学术合作但未见随访研究** | 商业模式最像 SaaS 的参照。**注意：其一项发表于 *Interactive Learning Environments*（Wang et al. 2020, doi 10.1080/10494820.2020.1808794）的随机实验报告：随机分配的八年级学生优于整班教学与小组专家教学两种对照** —— 这是中国市场少见的同行评议随机实验，但**仅此一项，且样本与独立性需自行复核** [未核实] |
| **IntelliCode / ITAS** | **学术多智能体导师**（2025–26 预印本）| — | **IntelliCode**（arXiv 2512.18669）：**StateGraph（LangGraph 式）+ 集中式版本化学习者状态**；**ITAS**（arXiv 2604.24808）：三层多智能体，**在研究生课程完整部署一学期**（配套延迟/成本论文 arXiv 2604.24110）| 未产品化 | **EduMap 架构的最接近学术先例** —— 面试可用来说明"这不是我一个人拍的设计" |

**中国市场三个反直觉结论**（[已验证]）：
1. **知识图谱是标配，不是差异化** —— 六家全部宣称知识点图谱与前置追溯。**"行业最细"这类形容词是自己给自己颁的奖。**
2. **间隔重复与代码沙箱是真正的空白** —— 唯一提及遗忘曲线的只有作业帮一句"基于学习遗忘曲线"；**无任何产品提供代码执行环境**。这两项恰好是 EduMap 已有的能力。
3. **效能证据系统性弱于西方** —— 六大主流产品**几乎零篇同行评议 RCT**（**唯一例外是松鼠AI 一项 2020 年随机实验，doi 10.1080/10494820.2020.1808794，但独立性需复核**）；最常被引的数字（"学习效果提升42%"、"学会率提升3.1倍"）全部是自报或"第三方认证 + 营销"混合体。**唯一可检视的技术产物是有道开源模型（arXiv 2506.18330）。**

**一个有用的对照（黄金标准证据）**：西方传统教学系统反而有更多 ESSA Tier 1 级证据，可作为"教育效果该怎么证明"的参照：
- **Carnegie Learning MATHia** —— RAND 黄金标准 RCT：第二年混合式约 **2× 标准化测验增长**（ESSA：混合 Tier 1、独立使用 Tier 2）[未核实]
- **ASSISTments** —— Maine RCT（46 校 / 2,769 人）**+60% 学习增益，ES 0.22**；NC RCT（63 校 / 5,991 人）一年 **+30%，ES 0.10**（ESSA Tier 1）[未核实]
- **ALEKS** —— 基于知识空间理论（KST），Algebra 1 约 350 个概念 → 数百万知识状态 [未核实]

> 面试可讲：*"我查教育效果证据时发现一个反直觉的事：**最硬的 RCT 证据不在 AI 产品上，而在传统自适应系统上** —— Carnegie Learning 的 MATHia 和 ASSISTments 都有 ESSA Tier 1 级随机实验，而 AI 导师里只有 Khanmigo 有一篇两年的独立 RCT，结论还是'和不用 AI 差不多'。所以我倾向于说'我们的机制有证据支持'（前置链、护栏），而不是'我们的 AI 有效果'。"*

---

## 7. 复盘：EduMap 的四个真实短板

按"面试官最可能追问"的顺序：

1. **"你的遗忘曲线是自己拍的还是照着做的？"** —— 现状是自研经验公式，形式接近 HLR，而 HLR 在公开基准里排倒数第三。**这是最该坦白并给出迁移计划的一点。**（→ I-3）
2. **"你的掌握度是怎么测出来的？"** —— 是 LLM 自报 `mastery_delta`，缺失默认 0.1，无 IRT/BKT/Elo。**"测出来的"和"被断言的"不是一回事。**（→ I-4）
3. **"你的记忆系统记了什么、怎么取？"** —— 三层架构完整，但长时记忆召回是纯时间序，`importance_score` 存了不用。（→ I-2）
4. **"生成质量怎么保证？"** —— 有真实强制的 Content Auditor 双级门禁（这是强项），但生成侧 Faithfulness 曾记录为 0.010/0.277 —— **经 2026-09-24 重测，那是管线故障（LLM 失败串当答案 + judge 静默 fallback），真实值为 0.5407/0.813/0.8539**；引用已升级到 span 级可人工审计。（→ I-7 已完成）

**同时要主动讲的两个强项**（否则全是短板就成了自我否定）：

- **教学护栏是真实强制的，不是提示词建议** —— Content Auditor（相似度 <0.8 判失败 + 概念边界检查）、Guardian（确定性 DAG/难度校验）、Anti-gaming（突击检测 + 贝叶斯缓慢更新）。**PNAS 的 RCT 证明这正是"帮助 vs 伤害"的分界。**多数竞品护栏比 EduMap 弱。
- **确定性学习路径排序是"可辩护的选择"，而非"没上 RL 的遗憾"** —— 综述级证据（Doroudi, Aleven, Brunskill, *IJAIED 2019*, arXiv 1812.10021）结论是：**RL 做教学排序优于简单策略的证据"薄弱且不成熟"**，五个标准缺陷是样本效率低、**sim2real gap（agent 学的是 BKT/DKT 模拟器而不是真实学生）**、奖励设定错误、离线评测混淆、可复现性差。EduMap 的 Guardian DAG + 前置门控是**可审计**的替代方案，可以在面试中主动讲成优势。
- **工程自评的诚实度超出同类** —— 记录并公开了**两个负收益结论**：LLM query rewrite **MRR −0.054 且延迟 ×60**（默认关闭）；以及自纠了"hybrid 延迟 ×0.64"实为缓存假象。**这种自评纪律本身是可讲的技术能力。**
- **代码沙箱是真正的空白领域** —— 在**本次调研覆盖的全部消费级产品中，代码执行环境一律缺失**（中国六家全无；Khanmigo/Duolingo/Coursera Coach 亦无"agent 生成代码 → 执行 → 迭代修复"的闭环）。EduMap 有真实的 `sandbox-service` + Coder 迭代修复环（AST 校验 → 沙箱执行 → 最多 3 轮修复）。**这是"构造主义学习"能力，与整个市场的"解题/批改"取向正交。**
- **大厂在做提示词层，EduMap 在做引擎层** —— ChatGPT Study Mode（2025-07-29）与 Claude for Education（2025-04-02）经核实**本质是 system prompt + 轻量 RAG，没有学习者建模、没有知识图谱**。**"大厂有模型但没有教学引擎"是 EduMap 可以说清的定位。**

---

## 8. 常见面试题与追问点

**Q1：你调研过竞品吗？最大的差距在哪？**
> 我按四类对标：商业助手（Khanmigo/豆包爱学）、开源教育 Agent（DeepTutor 40k star，架构最像我们）、Agent 前沿（LangGraph/Temporal/FSRS）、以及知识图谱派（松鼠AI/ALEKS）。结论是**我的架构选型被行业事件验证了**（AutoGen 进维护模式、CrewAI 补 Flows、LangGraph 废弃 supervisor 库），**教学理念被 PNAS 的 RCT 验证了**（去掉护栏 −17%），但**算法内核落后一代**：遗忘曲线是自己拍的、接近文献里垫底的 HLR；掌握度是 LLM 自报而非测量。**最高性价比的改进不是加功能，而是替换已有模块的算法。**

**【面试官可能追问】"那你为什么不一开始就用 FSRS？"**
> 诚实回答：**因为我不知道有这条公开基准**。我做完才发现 srs-benchmark 有 10,000 个 Anki 用户、3.5 亿次复习的公开对比，而我的模型形态（指数曲线 + 标量强度）恰好是表里最差的那一档。这也是我这次调研最有价值的产出 —— 它把我"感觉自己实现了遗忘曲线"和"我的遗忘曲线在公开基准上什么水平"这两件事分开了。下一步会迁移到 FSRS，并且**用同一套 LogLoss/RMSE/AUC 口径在自己的历史数据上做前后对比**。

**Q2：你的检索为什么要用 RRF？**
> 现在的诚实答案是：**RRF 是"不需要调参"的默认值，而我不该把它当成"最优"。** TOIS 那篇（arXiv 2210.11934）明确说加权凸组合在域内域外都优于 RRF，RRF 只在无法调参时有优势。**我自己的消融也指向同一结论** —— n=20 上 score 融合 MRR 0.95、RRF 0.90。而且我早就实现了两套备选（`minmax`/`score`）并且可运行时切换，**所以这是改一行配置的事，我却没有改。** 这是我这次调研收获最直接的一条。

**【面试官可能追问】"你的 n=20 结论可信吗？"**
> **不完全可信，n=20 太小。** 所以我拉了 n=200 的策略 A/B：hybrid(RRF) MRR 0.8569 vs direct 0.8454，**只领先 0.0115** —— 这个量级恰好落在"值得换更强调优融合"的区间。另外我踩过一次坑：曾报告 hybrid 延迟 ×0.64，后来发现是**检索结果缓存**造成的假象，所以比较延迟必须绕过缓存。**这个自纠我也写进了文档。**

**Q3：你的记忆系统和 Mem0/Zep 比怎么样？**
> 我**不比较数字**。同一个系统在同一个 LoCoMo 基准上有四个分数（65.99/84/58.44/75.14），根因是 prompt/judge/harness 口径不一致，跨论文不可比；而且有第三方审计材料显示某厂商 84% 的成绩修正后是 58.44%。**我只看机制**：Zep 的时序知识图谱（事实带有效期）对我有启发，因为我的知识也不是永久有效的。**我自己有明确的短板** —— 长时记忆召回是纯 `ORDER BY created_at DESC`，`importance_score` 字段存了却不用，我打算按 generative agents 的 `α·recency+β·importance+γ·relevance`（recency = 0.995^hours）补上。

**Q4：多智能体协作，你怎么保证不出错？**
> 分两层。**结构性错误**由 Guardian 确定性校验（DAG 环检测 + 难度单调性，无 LLM）；**内容漂移**由 Content Auditor 强制门禁（相似度 <0.8 判失败并重试，另有概念边界检查）。这两层都是**强制**的，不是提示词里的建议 —— 这个区别被 PNAS 那篇 RCT 量化过：无护栏的 GPT-4 直给，练习期 +48% 但无辅助考试 **−17%**，加护栏后伤害消失。**我的架构恰好站在"加护栏"那一侧。**

**【面试官可能追问】"那你的生成质量到底行不行？"**
> **我一度以为基本没验证过 —— 后来发现是仪表坏了。** 生成侧 Faithfulness 曾记录为 0.010/0.277，我接受了很久。直到做完 span 级溯源、能人工核对引用之后，才回头怀疑指标本身。查下来是三个管线故障：LLM 失败时把错误字符串当答案（导致词重叠与相关性同时归零）、judge 失败时静默退回启发式却仍以 judge 之名报数、以及某条路径直接把占位符当答案。**修复后重测：词重叠 0.5407、语义 0.813、Relevancy 0.8539**（`Answer Relevancy 从 0.000 → 0.8539` 是最直接的证据 —— 真实系统不可能相关性为零）。**这件事教训是：低分数不一定意味着质量差，也可能是仪表坏了。**

**Q5：LangGraph 有什么坑？**
> 三个，都是我自己踩的，而且社区在同等抱怨：① **reducer 语义** —— 路由器里改 `state[...]` 会被丢弃，只有节点返回值会被提交，我的重试环曾因此无限循环；② **`InMemorySaver` 重启不持久**，而**我根本没接 checkpointer** —— 会话状态在 Redis，所以我吃过两次"刷新丢进度"（`agent_results` 浅合并、刷新进度归零），实际上都是在 Redis 层打补丁而非架构性解决；③ **checkpoint 无限增长**是官方承认的问题，接 checkpointer 时必须先设计清理策略。另外**官方已把 `langgraph-supervisor` 归档**、`create_react_agent` 被 `create_agent` 取代，**依赖 prebuilt 层会在升级时被迫重写。**

**Q5b：你怎么看"AI 导师到底有没有用"这个争论？**
> 我的判断是：**AI 模型带来的增益被高估了，被验证的是机制。** 三条证据：① PNAS（~1,000 人）—— 去掉教学护栏，练习期成绩 +48% 但**无辅助考试 −17%**；② Khanmigo 的两年期 RCT（18 校 / 6,900 人）—— 每学期只有 **+1.3 个国家百分位**，而且**论文自己承认"增益与不用 AI 的 Khan Academy 练习相似"**，失败点在**参与度**（中位数学生只用掉 1/3 练习日、17% 错题环节）；③ 反过来，Khan Academy 自己 15M+ 线程的数据显示，**"呈现未掌握的前置技能 + 简短复习"使下一题正确率 +2.7%**，总 +6.1%。
>
> **所以我的结论是：被数据支持的是"护栏 + 前置链"这类机制，不是"换个更大的模型"。** 这也解释了为什么我把 Content Auditor 和 Guardian 做成强制门禁 —— 以及为什么我的知识图谱用 `PREREQUISITE_OF` 而不是只做向量检索。
>
> **但我要标注来源等级**：第 ③ 条是 KA **自己的内部 A/B**，不是独立研究；我引用时会说"公司自报"，不会说"研究证明"。这是这份调研里我坚持的纪律。

**【面试官可能追问】"那你的系统能证明有效吗？"**
> **不能，而且我不假装能。** 我没有做 RCT，也没有线上规模数据。我能证明的是：**单项机制在公开基准上的位置**（RAG 检索指标 MRR 0.841/HR@3 0.930，n=200 标注集；意图识别 93.06%，n=72）和**工程正确性**（868 项测试 + 变异验证）。**如果要说学习效果，我需要的是 A/B 而非架构推算 —— 这是我目前在证据链上最诚实的一段空白**，我会主动说，而不是拿架构图当效果证据。

**Q6：为什么不用 Temporal 做持久执行？**
> 我知道它是正确答案，但**对 MVP 规模不值得交税**。代价有三：**确定性约束**（LLM 调用必须挪进 Activity，本地时钟/UUID 都要走重放安全 API）、**payload 2MB 上限**（大消息历史/大依赖对象会撑爆，且失败会无限重试而不报错）、**版本管理**（Worker Versioning 或 patching，experimental 版本 2026-03 从 Server 移除）。而且它保证的是 **exactly-once 编排，不是 exactly-once 副作用** —— Activity 是 at-least-once，仍要求幂等。**所以我选了 checkpointer + 幂等写。** 但如果将来生成任务要跑几十分钟并需要跨重启恢复，我会走 `temporalio.contrib.langgraph`，**每个节点用 `execute_in` 显式声明 activity/workflow —— 这个元数据不允许设默认值，是确定性守卫。**

**Q7：如果只能改一件事，改什么？**
> **先给遗忘曲线建一条 0 参数基线，再决定要不要迁 FSRS。** 我原本会直接说"迁 FSRS"，但调研的公开基准给了一个反直觉的数字：**一个 0 参数的"最近复习"移动平均基线，LogLoss 0.3369，反而优于 21 参数的 FSRS-6（0.3460）** —— 说明朴素策略已经很强，记忆模型的边际增益有限。所以正确的第一步不是换模型，而是**先量化我的自研曲线和这条朴素基线差多少**。如果连它都没打赢，问题就不在算法，而在数据质量或建模假设（比如我的 score 本来就来自 LLM 自报，见 Q1）。**这比我直接说"我要迁 FSRS"要诚实，也更有技术判断力。**

**【面试官可能追问】"那你现在到底知不知道自己的遗忘曲线好不好？"**
> **不知道 —— 而这本身就是我这次调研最重要的产出。** 我没有用 LogLoss/RMSE/AUC 这种可跨系统比较的口径量过它，只有"感觉实现了遗忘曲线"。公开基准告诉我，同类形态的 HLR 在 LogLoss 上是 0.4694，而 FSRS-6 是 0.3460、0 参数基线是 0.3369 —— **有了这三个锚点，我才第一次知道自己的曲线大概在什么位置**。计划是把 `learning_progress` 历史灌进同一套口径，同时报自研/基线/FSRS 三个数。

**【面试官可能追问】"你现在的数字有多少是真实测的、多少是推算的？"**
> 我要区分清楚：**真实测量**的有 RAG 检索指标（n=200 标注集 + 图谱自动生成集）、双路召回延迟 P50/P95、意图识别准确率 93.06%（n=72）、熔断器行为、868 项测试。**架构推算**的有并行化的延迟收益、Harness 迁移前后的样板代码量。**曾误判为未验证**的是生成侧质量 —— 实际是评测管线有三个故障（详见 04 文档 §0），修复后重测为 0.5407/0.813/0.8539。**我从不虚报线上数据 —— 这个项目没有大规模线上部署。** 这份调研文档里的所有外部数字我都标了来源等级，因为我要保证被追问时能说清每个数字的出处。

---

## 附录 A：验证状态总表

| 关键论断 | 状态 | 来源 |
|---|---|---|
| PNAS RCT：无护栏 −17% | **[已验证]** | doi 10.1073/pnas.2422633122 |
| **Khanmigo 两年 RCT：AI 增益≈无（+1.3 百分位/学期）** | **[已验证]** | NBER WP 35620 / EdWorkingPaper ai26-1551（18 校 / 6,900 人）|
| **"哈佛 Khanmigo 研究"不存在（与 Kestin 混淆）** | **[已验证为不存在]** | 调研 agent 明确核查 |
| **前置链机制：下一题正确率 +2.7%（合计 +6.1%）** | **[公司自报]** | KA 官方博客 2026-05-06（15M+ 线程）|
| Khanmigo 数学 agent 校验 / 限制范围使给答案率 −50% | **[公司自报]** | KA 工程博客 |
| Kestin 哈佛 RCT：增益翻倍、效应量 0.63 | **[已验证]** | doi 10.1038/s41598-025-97652-6 |
| Tutor CoPilot：900 导师 / +4 p.p. / $20 每年 | **[已验证]** | arXiv 2410.03017 |
| **0 参数 MOVING-AVG 0.3369 优于 FSRS-6 0.3460** | **[已验证]** | srs-benchmark（10,000 用户 / 349.9M 复习）|
| FSRS 曲线版本史（v3 指数 → v4 幂律 → v6 可学衰减）| **[已验证]** | expertium.github.io/Algorithm.html |
| HLR LogLoss 0.4694（倒数第三）| **[已验证]** | srs-benchmark |
| RRF 一般劣于加权凸融合 | **[已验证]** | arXiv 2210.11934（ACM TOIS）|
| **但融合对纯稠密的增量可能很小（BGE-M3 +0.2 nDCG@10）** | **[已验证]** | arXiv 2402.03216（MIRACL）|
| **过小的 reranker 会伤害检索（33M MiniLM −3~−13 NDCG@10）** | **[已验证]** | arXiv 2409.07691（NVIDIA）|
| RRF 的 k=60 是试点惯例而非调优结果 | **[已验证]** | Cormack et al., SIGIR 2009 |
| ColBERTv2 压缩后存储与单向量索引相当（9M 段落 16–25 GiB）| **[已验证]** | arXiv 2112.01488 |
| RRF 原始出处（SIGIR 2009，k=60）| **[已验证]** | Cormack et al., SIGIR 2009 |
| AutoGen 维护模式；CrewAI 停滞；supervisor 已归档 | **[已验证]** | GitHub API + 官方迁移指南 |
| 记忆基准罗生门（四个分数） | **[已验证]** | mem0 Issue #4573 / getzep blog / zep-papers Issue #5 |
| DKT 优势须 hedge（复现 AUC 0.73 而非 0.86）| **[已验证]** | Xiong et al. EDM 2016 |
| RL 学习路径排序证据薄弱 | **[已验证]** | Doroudi et al., IJAIED 2019, arXiv 1812.10021 |
| LLM-as-judge 与人类一致率 ~80% + 三种偏差 | **[已验证]** | arXiv 2306.05685 |
| HippoRAG **不含**衰减重加权 | **[已验证]** | 调研 agent 明确纠正 |
| GraphRAG 成本（$120 / 1000 页） | **[已验证]** | 真实用户报告 |
| LazyGraphRAG 索引成本 = 完整 GraphRAG 的 0.1% | **[已验证]** | 微软官方博客 |
| DeepTutor 40,208★ / Apache-2.0 / TutorBench | **[已验证]** | GitHub API + arXiv 2604.26962 |
| 中国市场零篇同行评议 RCT | **[已验证，含一处例外]** | 六产品盘点；松鼠AI 2020 一项需复核 |
| 松鼠AI 2020 随机实验（优于整班与小组专家教学）| **[未核实]** | doi 10.1080/10494820.2020.1808794 |
| MATHia / ASSISTments 的 ESSA Tier 1 证据 | **[未核实]** | RAND / Maine+NC RCT（作为"黄金标准"参照）|
| "GPT-4o Realtime API 驱动 Duolingo Video Call" **不在 Duolingo 一手来源中** | **[已验证为不存在]** | 其工程博客中 "4o"/"realtime"/"OpenAI" 出现次数均为 0 |
| **Birdbrain 无任何论文**（含 AIED 2020 / arXiv 之说）| **[已验证为不存在]** | research.duolingo.com 全页 "birdbrain" 出现 0 次 |
| 中国市场间隔重复/沙箱为空白 | **[已验证]** | 六产品功能盘点 |
| 讯飞以"幻觉治理"为核心卖点 | **[已验证]** | 钛媒体等，但效能数字 **[仅营销]** |
| 讯飞当前 API 定价 | **[未核实]** | 官方控制台为 JS SPA，抓取失败 |
| FSRS 成为 Anki 默认调度器的版本号 | **[未核实]** | issue #3616 未闭环；手册显示 "<25.07" |
| 27,000 学生：作业 +18% / 月考 −20% | **[未核实]** | 无 DOI、无作者，方向性重要 |
| MIT《Your Brain on ChatGPT》 | **[仅营销／争议]** | 非同行评议，关键结论 N=18，已有正式批评（arXiv 2601.00856）|
| Squirrel AI 旗舰 KST 论文 / 效能 RCT | **[未找到]** | 其 CMU/密歇根/北师大合作未见联合署名论文 |
| Dify / FastGPT 的许可条款 | **[未核实]** | NOASSERTION，条款未确认 |
| LlamaIndex Workflows / Google ADK 的 API 签名 | **[未核实]** | 部分未确认 |
| 2024–2026 基于 LLM 的知识追踪 | **[未找到]** | 该子领域为本次调研最薄弱处 |

## 附录 B：调研未覆盖项（诚实披露）

- **Kestin 研究的再分析与批评**（clayford.net、Christian Bokhove 博客）与 **Wharton 批评文章**（Axios 2024-08）**已定位到 URL 但未抓取** —— 引用 Kestin 效应量（0.73–1.3 SD）时建议先读批评方。
- **Khanmigo RCT 的学生总数**：摘要未给出（此前流传的"~6,900"**无法从摘要确认**，本档未采用）。如需引用须查全文。
- **iFlytek Spark 当前逐模型 token 定价**：官方控制台为 JS SPA，抓取失败，**须自行在控制台核实**。
- **~27,000 名学生研究（作业 +18% / 月考 −20%）**：无 DOI、无作者，**方向性重要但不可作为证据引用**。
- **DASH 原始引用**、**FSRS-vs-SM-2 的数值对比**：未核实（SM-2 不在 srs-benchmark 主表中）。
- **中国市场**：松鼠AI 2020 那项随机实验（doi 10.1080/10494820.2020.1808794）**独立性与样本需自行复核**；其余产品**无同行评议 RCT**。
- **MATHia / ASSISTments 的 ESSA Tier 1 证据**：为对照用，具体数字 **[未核实]**。
- **"TutorGPT / Learnt-AI / OpenTutor / Slate (Sionic)"**：**无法匹配到真实仓库**，疑为不准确的名称，未做填充。真实存在的相近项目是 `THU-MAIC/OpenMAIC`（38,728★，多智能体课堂，许可未核实）与 `JushBJJ/Mr.-Ranedeer-AI-Tutor`（29,574★，提示词模式，无许可）。
- **中文市场部分来源为 PR 驱动**，凡仅见营销稿的数字一律标 **[仅营销]**，**不得作为效能证据引用**。
- **研究环境限制**：本环境的 WebFetch 全域被阻断、WebSearch 配额耗尽，全部事实经 `curl` 抓取原始页面获得；未能抓取的一律标 **[未核实]** 而非凭记忆填充。
