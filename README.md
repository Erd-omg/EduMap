# 智图 EduMap

**多智能体驱动的个性化学习系统**

基于 LangGraph + Neo4j + FastAPI + Next.js 的全栈 AI 学习平台，通过多智能体协作、知识图谱和 RAG 技术，实现从画像分析到资源生成再到自适应学习的完整闭环。

---

## 系统架构

```
                    ┌──────────────────────────────────────────────┐
                    │            Frontend (Next.js :3000)          │
                    │  /chat  /generate  /learn  /profile        │
                    └──────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────────────────────┐
              │            │                            │
              ▼            ▼                            │
     ┌──────────────┐  ┌──────────────────────────┐     │
     │profile-svc   │  │    backend-core (:8000)   │     │
     │(:8001)       │  │  LangGraph + RAG + KG API │     │
     │画像分析 SSE   │  └──────────────────────────┘     │
     └──────┬───────┘            │                      │
            │                    │                      │
              │                                                   │
              │  ┌─────────────────────────────────────────┐      │
              │  │   LangGraph 编排 (6 Agents)             │      │
              │  │  Planner → Guardian ⇉ Designer ∥ Coder  │      │
              │  │  → Content Auditor → Assessment         │      │
              │  └─────────────────────────────────────────┘      │
              │                                                   │
              │  ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌───────────────┐
              │  │ Learning    │  │ Mentor   │  │ Knowledge  │  │ Memory System │
              │  │ Path Service│  │ (RAG QA) │  │ Graph API  │  │ Sensory→Short │
              │  └─────────────┘  └──────────┘  └────────────┘  │ → Long-term   │
              │              ┌───────────────────┐               └───────┬───────┘
              │              │ RAG 增强           │                      │
              │              │ 语义分块+Reranking │                 ┌────▼────┐
              │              │ 评测框架+工具系统   │                 │ Redis   │
              │              └───────────────────┘                 │短期记忆  │
              └────┬───────────┬───────────┬───────────────────────┴─────────┘
                   │           │           │
              ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
              │ Neo4j  │ │Postgres│ │ChromaDB│
              │(图谱)   │ │(画像+   │ │(向量)   │
              │        │ │ 长期记忆)│ │        │
              └────────┘ └────────┘ └────────┘
```

## 5 阶段实现

| Phase | 内容 | 文件数 | 状态 |
|-------|------|--------|------|
| Phase 1 | 项目骨架 + 基础设施 (Monorepo, Docker, CI) | 84 | ✅ |
| Phase 2 | 对话画像 + 知识图谱 (Neo4j, Profile, Radar) | 32 | ✅ |
| Phase 3 | 多智能体资源生成 (LangGraph, 6 Agents) | 39 | ✅ |
| Phase 4 | 个性化学习路径 (PathService, D3 Skill Tree) | 19 | ✅ |
| Phase 5 | Mentor RAG + 导航 + 文档 | ~20 | ✅ |
| **Phase 6** | **记忆系统 + RAG 增强** | **~34** | **✅ 新增** — 三层记忆、工具系统、语义分块、Cross-Encoder Reranking、RAG 评测框架、volatile 存储持久化 |
| **Phase 7** | **Agent Harness 标准化层** | **~14** | **✅ 新增** — BaseAgent ABC、结构化输出(双模式)、统一重试/可观测性、Tool/Memory Mixins、6 个 Agent 迁移 |
| **Phase 8** | **运维加固 + 测试基建 + RAG 评测升级** | **~24** | **✅ 新增** — ChromaDB 锁定、Embedding 预热、种子自动加载、LLM 熔断器、pytest 650+ 项测试（含前端 Vitest + Playwright）、LLM-as-Judge 评测、学生问答数据集、CI 测试回归门禁 |

## 技术栈

| 层 | 技术 |
|---|---|
| **前端** | Next.js 15, React 19, TypeScript, Tailwind CSS v4, D3.js v7 |
| **后端** | Python 3.12, FastAPI, LangChain/LangGraph, httpx, asyncpg |
| **图数据库** | Neo4j 5 (APOC) — 知识图谱、遍历、最短路径 |
| **关系数据库** | PostgreSQL 16 — 用户画像 (JSONB)、长期记忆、评测结果 |
| **向量数据库** | ChromaDB — 语义检索、相似度搜索（具备内存回退模式） |
| **缓存/短期记忆** | Redis 7 — 会话管理、对话轮次存储 |
| **LLM** | 适配器模式 (默认 OpenAI 兼容, 可切换) — 支持工具调用格式 |
| **嵌入/重排序** | sentence-transformers (BAAI/bge-small-zh-v1.5), Cross-Encoder |
| **容器** | Docker Compose — 全部服务一键启动 |

## 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| `web` | 3000 | Next.js 前端 |
| `backend-core` | 8000 | AI/Agent 服务 + API |
| `profile-service` | 8001 | 用户画像服务 |
| `sandbox-service` | 8002 | 代码沙箱执行 |
| `neo4j` | 7474 / 7687 | 浏览器 / Bolt |
| `postgres` | 5432 | 用户数据 |
| `chromadb` | 8003 | 向量索引 |
| `redis` | 6379 | 缓存 |

## API 端点总览

### 知识图谱 (`/api/v1/kg/*`)
```
GET    /nodes/{id}                   获取知识点
POST   /nodes                        创建知识点
PUT    /nodes/{id}                   更新知识点
DELETE /nodes/{id}                   删除知识点
GET    /courses/{id}/graph           课程知识图谱
GET    /traverse/prerequisites/{id}  前置知识 (深度)
GET    /traverse/prerequisites/{id}/path  前置链
GET    /traverse/related/{id}        相关知识点
GET    /path                         最短路径
POST   /edges                        创建边
DELETE /edges                        删除边
POST   /dedup/trigger                语义去重
POST   /dedup/merge                  合并节点
POST   /seed/load                    加载种子数据
GET    /seed/verify                  种子数据统计
GET    /eval/cold-start              冷启动评估
```

### 资源生成 (`/api/v1/orchestrator/*`)
```
POST   /generate                     启动多智能体生成（返回 session_id）
GET    /status/{session_id}          查询生成状态
GET    /stream/{session_id}          SSE 进度流（agent 阶段事件）
```
### 资源管理 (`/api/v1/resources/*`)
```
POST   /upload                       上传资源文件（自动解析 + ChromaDB 索引）
GET    /                             列出资源（支持 user_id/kp_id/type/source 筛选）
GET    /{id}                         获取资源元数据
GET    /{id}/chunks?limit=N          获取上传资源解析的文本段落预览
DELETE /{id}                         删除资源
POST   /sync-generated               同步 AI 生成资源到资源库
```

### 对话画像分析 (`/api/v1/analysis/*`)
```
GET    /stream/{user_id}?message=    统一 SSE 流（画像分析 + Mentor RAG 问答混合路由）
```

### 学习路径 (`/api/v1/learning-path/*`)
```
GET    /{course_id}                  个性化学习路径
GET    /{course_id}/next             下个推荐知识点
POST   /progress                     记录学习进度
GET    /{course_id}/content-style    内容类型建议
```

### Mentor / RAG 增强 (`/api/v1/mentor/*`)
```
GET    /stream/{user_id}             SSE 流式 RAG 问答（支持对话历史上下文）
GET    /search                       知识点搜索 (预览)
GET    /evaluate?sample=true&n=10    自动化评测（Precision/Recall/MRR/NDCG/HitRate）
                                     sample=true 使用手工标注学生问答数据集
POST   /rerank-preview               重排序预览（Cross-Encoder before/after）
```

### Memory 系统（自动集成）
```
- 三层记忆：Sensory（请求级） → Short-term（Redis, 1h TTL） → Long-term（PostgreSQL）
- Episodic 记忆：用户交互记录（mentor 问答自动持久化）
- Semantic 记忆：用户画像、学习偏好、技能摘要
- 工具系统：ToolRegistry 管理 LLM 可用工具（KG 搜索/资源搜索/遗忘检查）
```

## 快速开始

```bash
# 1. 克隆项目
git clone <repo-url>
cd EduMap

# 2. 安装前端依赖
pnpm install

# 3. 启动所有服务 (Docker 必需)
docker compose up -d

# 4. 开发模式
pnpm dev
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | `""` | LLM API 密钥。**留空时系统以演示模式运行** — AI 对话和资源生成会提示未配置 API，但知识图谱、学习路径、资源管理等非 LLM 功能可正常使用 |
| `LLM_API_BASE` | `""` | LLM API 地址（如 `https://api.deepseek.com/v1`） |
| `LLM_MODEL` | `"deepseek-chat"` | 模型名称 |

## 项目结构

```
EduMap/
├── apps/web/                        # Next.js 15 前端
│   ├── src/app/                     # 路由页面
│   │   ├── chat/                    # 统一对话（画像分析+智能辅导）
│   │   ├── generate/                # 资源库（文件上传+管理）
│   │   ├── learn/[courseId]/        # 学习路径
│   │   ├── mentor/                  # 重定向到 /chat
│   │   └── profile/                 # 个人画像
│   ├── src/components/              # 组件
│   │   ├── knowledge-graph/         # 技能树
│   │   ├── layout/                  # 导航栏/侧边栏
│   │   ├── profiling/               # 聊天窗口/雷达图/画像
│   │   ├── mentor/                  # 来源面板 (source-panel)
│   │   └── resources/               # 资源库 + 生成进度
│   ├── src/hooks/                   # React Hooks
│   └── src/stores/                  # Zustand 状态管理
├── packages/shared-types/           # TypeScript 共享类型
├── services/
│   ├── backend-core/src/            # AI/Agent 服务
│   │   ├── agents/                  # 智能体实现 (7个+记忆)
│   │   │   ├── orchestrator/        # LangGraph 编排
│   │   │   ├── planner/             # 知识提取
│   │   │   ├── guardian/            # 结构验证
│   │   │   ├── designer/            # 内容设计
│   │   │   ├── coder/               # 代码生成
│   │   │   ├── content_auditor/     # 质量审核
│   │   │   ├── assessment/          # 测评生成
│   │   │   └── mentor/              # RAG 问答
│   │   ├── kg/                      # 知识图谱层
│   │   ├── learning_path/           # 学习路径
│   │   ├── rag/                     # RAG 检索
│   │   ├── prompts/                 # 提示词管理
│   │   └── utils/                   # LLM 适配器
│   │   ├── memory/                  # 三层记忆系统 (Phase 6)
│   │   ├── tools/                   # 工具系统 (Phase 6)
│   │   └── rag/chunking/            # 语义/递归分块策略
│   │   └── rag/reranking/           # Cross-Encoder 重排序
│   │   └── rag/evaluation/          # 自动化评测框架
│   │       ├── metrics.py           # Precision/Recall/MRR/NDCG/HitRate/Faithfulness
│   │       ├── evaluator.py          # RAGEvaluator 评测管道
│   │       ├── benchmark.py          # 基准测试运行器（支持 KG 生成 + 手工标注双数据源）
│   │       ├── llm_judge.py          # LLM-as-Judge 评测（faithfulness + relevancy）
│   │       └── datasets/             # 评测数据集（20 条手工标注 + 200 条扩展 + 72 条意图标注）
│   │── tests/                         # pytest 自动化测试（734 项，含 API 集成测试）
│   │   ├── conftest.py               # 全局 fixtures
│   │   ├── mocks/                     # MockLLMAdapter
│   │   ├── test_rag/                  # RAG 评测/分块/重排序/策略对比测试（204 项）
│   │   ├── test_harness/              # BaseAgent/Retry/StructuredOutput 测试（64 项）
│   │   ├── test_memory/               # 三层记忆系统测试（83 项）
│   │   ├── test_tools/                # 工具系统 + 工具调用执行测试（63 项）
│   │   ├── test_kg/                   # 知识图谱/向量索引/语义去重测试（80 项）
│   │   ├── test_learning_path/        # 遗忘曲线/路径服务测试（50 项）
│   │   ├── test_agents/               # Guardian/Orchestrator 测试（51 项）
│   │   ├── test_utils/                # LLM 熔断器/TTL 缓存测试（43 项）
│   │   ├── test_analysis/             # 三路融合意图识别测试（15 项）
│   │   └── test_api/                  # FastAPI 集成测试（78 项）
│   └── sandbox-service/src/         # 代码沙箱服务
├── scripts/db/                      # 数据库初始化脚本
└── docker-compose.yml               # 开发环境编排
```

## 智能体系统

| 智能体 | 角色 | 技术 |
|--------|------|------|
| **Orchestrator** | LangGraph 状态图路由 | StateGraph, 条件边, 并行 fan-out/fan-in |
| **Planner** | 知识点提取 | LLM + KG 查重 |
| **Guardian** | DAG 校验 + 难度单调性 | DFS 拓扑排序 |
| **Designer** | 多模态内容生成 | LLM 提示词工程 |
| **Coder** | 代码生成 + 沙箱验证 | AST 分析 + Docker |
| **Assessment** | 微测验生成 | LLM 结构化输出 |
| **Content Auditor** | 生成质量审核 | ChromaDB 相似度 |
| **Mentor** | RAG 问答（支持对话历史） | ChromaDB + Neo4j 混合检索 + Cross-Encoder Reranking |

**并行生成分支**：Guardian 通过后 **fan-out** 到 Designer 与 Coder 两个分支，二者在同一个 superstep 内并发执行，再由 merge 节点 **fan-in** 汇合（`route_after_generation` 是两条分支共用的扇入路由器）。代价是两个分支并发写共享状态，因此 `EduMapState` 中 `generated_resources`（追加/重置）、`agent_results`（浅合并）、`current_phase` 三个字段必须声明 reducer —— 缺任一个 LangGraph 都会抛 `InvalidUpdateError` 使整次生成失败。`retry_prep` 的重入同样会重新触发双分支。

### 增强系统（Phase 6-7）

| 系统 | 角色 | 技术 |
|------|------|------|
| **记忆系统** | 三层记忆：Sensory → Short-term (Redis) → Long-term (PostgreSQL) | Redis + asyncpg + Pydantic |
| **工具系统** | 为 agent 提供可调用工具（KG搜索/资源搜索/遗忘检查） | ToolRegistry（含超时治理）+ `!tool:name(k=v)` 文本协议；工具调用遥测经 SSE 推送到前端 trace 面板 |
| **分块引擎** | 多策略文档分块（语义/递归/固定） | sentence-transformers + 分离器层级 |
| **RAG 评测** | 检索质量/生成质量自动化评测管道 | Precision@K / Recall@K / MRR / NDCG / HitRate / Faithfulness (LLM-as-Judge + 词重叠双模式) / Citation Accuracy / Context Coverage |
| **Agent Harness** | 标准化 Agent 执行层：BaseAgent ABC、结构化输出(双模式)、统一重试、可观测性、Tool/Memory 注入 | `src/harness/` 8 个文件，6 个 Agent 迁移，全部 5 个 graph 节点统一 |

## RAG 评测结果

对 RAG 系统在真实数据库（Neo4j + ChromaDB）上进行了多维度评测，覆盖检索质量、响应延迟、端到端管线验证和反馈闭环。

### 检索质量（200 条学生问答数据集，覆盖全部 22 个知识点）

整体指标（n=200，含中文 jieba 分词支持的 faithfulnes/relevancy 评测）：

| 指标 | @1 | @3 | @5 |
|------|---|---|---|
| **Precision** | 0.755 | 0.393 | 0.258 |
| **Recall** | 0.597 | 0.855 | 0.916 |
| **Hit Rate** | 0.755 | 0.930 | 0.960 |
| **MRR** | — | **0.841** | — |

按难度分解：

| 难度 | n | P@1 | MRR | HR@3 |
|------|---|-----|-----|------|
| 基础 | 77 | 0.818 | 0.896 | 0.974 |
| 中等 | 83 | 0.771 | 0.847 | 0.940 |
| 高级 | 40 | 0.600 | 0.721 | 0.825 |

### 端到端管线验证（上传→解析→索引→检索→回答）
- ✅ 文件上传+ChromaDB 索引: 0.6s/文件，全部成功
- ✅ 检索召回: 5/6 条查询找到上传内容（83.3%）
- ✅ 生成回答: LLM 调用正常

### 响应延迟（100 次调用，全管道）
| P50 | P95 | P99 | 平均 |
|-----|-----|-----|------|
| 19.42ms | 54.66ms | 96.85ms | 25.75ms |

### 反馈闭环验证
- ✅ 记忆系统存储→召回正常
- ✅ 纠错内容传递到 LLM 对话历史，改进回答质量
- ⚠️ 当前架构限制：RAG 检索无状态，反馈不影响检索结果
- 详情见 `benchmark_results/latest_expanded.json` 和 `benchmark_results/latest_sample.json`

### 检索策略 A/B 对比（direct / hybrid / rewrite）

```bash
cd services/backend-core
python3 scripts/run_strategy_comparison.py --dataset expanded \
    --strategies direct,hybrid,rewrite --output benchmark_results
```

n=200 扩展集实测（DeepSeek 真实 LLM，重排关闭，**三条策略均绕过检索结果缓存以保证延迟可比**）：

| 策略 | R@5 | P@5 | MRR | NDCG@5 | HR@5 | 平均延迟 |
|------|-----|-----|-----|--------|------|----------|
| direct（纯向量） | 0.9000 | 0.254 | 0.8454 | 0.8634 | 0.955 | 33.0 ms |
| hybrid（向量+KG+RRF） | **0.9054** | **0.255** | **0.8569** | **0.8726** | 0.955 | **30.3 ms** |
| rewrite（LLM 改写后 hybrid） | 0.9054 | 0.254 | 0.8029 | 0.8275 | 0.95 | 1822.9 ms |

**结论（含一条负面结果）**：
- hybrid 相对 direct **MRR +0.012、R@5 +0.005，延迟 ×0.92**（略快且更准），因此是生产默认。
- **LLM query rewrite 是负收益**：200/200 改写全部成功，但 MRR 比 hybrid **低 0.054**、延迟是 hybrid 的 **×60**。原因是评测集查询本身就是教科书式问法，改写反而丢掉了原句语义结构。因此 `rag_rewrite_enabled` **默认关闭**——这是一个如实记录的负结果，未做 prompt 调优以美化数字。
- **口径修订（重要）**：早期版本曾报告 hybrid 延迟为 direct 的 **×0.64**。复核发现那是因为 `direct` 直接调 `_chroma_search`、而 `hybrid` 走 `search()` 并命中了检索结果缓存——**该延迟优势部分是缓存假象**。现在三条策略统一 `use_cache=False`，公平口径为 **×0.92**。

### 意图识别准确率评测

```bash
python3 scripts/run_intent_eval.py --no-cache --output benchmark_results
```

自建 72 条人工标注集（`datasets/intent_labeled.json`，含刻意难例），冷启动实测：

| 版本 | 准确率 | 规则路径占比 | 融合路径准确率 |
|------|--------|--------------|----------------|
| 修复前 | 0.7500 (54/72) | 94.4% | 1.0000 |
| **修复后** | **0.9306 (67/72)** | **95.8%** | 1.0000 |

分类为 profile / question / mixed 三类的每类 F1：0.971 / 0.943 / 0.872。
**95.8% 的消息由规则快路径以零模型调用解决**，仅 4.2% 升级到 LLM + embedding 融合投票。

## 许可证

MIT
