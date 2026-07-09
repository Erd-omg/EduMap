# 智图 EduMap

**多智能体驱动的个性化学习系统**

基于 LangGraph + Neo4j + FastAPI + Next.js 的全栈 AI 学习平台，通过多智能体协作、知识图谱和 RAG 技术，实现从画像分析到资源生成再到自适应学习的完整闭环。

---

## 系统架构

```
                    ┌──────────────────────────────────────────────┐
                    │            Frontend (Next.js :3000)          │
                    │  /chat  /generate  /learn  /mentor  /profile │
                    └──────┬──────────────┬───────────────────────-┘
                           │              │
              REST/SSE (HTTP)      EventSource (SSE)
                           │              │
              ┌────────────▼──────────────▼──────────────────────┐
              │              backend-core (:8000)                 │
              │                                                   │
              │  ┌─────────────────────────────────────────┐      │
              │  │   LangGraph 编排 (6 Agents)             │      │
              │  │  Planner → Guardian → Designer+Coder    │      │
              │  │  → Content Auditor → Assessment         │      │
              │  └─────────────────────────────────────────┘      │
              │                                                   │
              │  ┌─────────────┐  ┌──────────┐  ┌─────────────┐  │
              │  │ Learning    │  │ Mentor   │  │ Knowledge   │  │
              │  │ Path Service│  │ (RAG QA) │  │ Graph API   │  │
              │  └─────────────┘  └──────────┘  └─────────────┘  │
              │                                                   │
              └────┬───────────┬───────────┬──────────────────────┘
                   │           │           │
              ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
              │ Neo4j  │ │Postgres│ │ChromaDB│
              │(图谱)   │ │(画像)   │ │(向量)   │
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

## 技术栈

| 层 | 技术 |
|---|---|
| **前端** | Next.js 15, React 19, TypeScript, Tailwind CSS v4, D3.js v7 |
| **后端** | Python 3.12, FastAPI, LangChain/LangGraph, httpx |
| **图数据库** | Neo4j 5 (APOC) — 知识图谱、遍历、最短路径 |
| **关系数据库** | PostgreSQL 16 — 用户画像 (JSONB) |
| **向量数据库** | ChromaDB — 语义检索、相似度搜索 |
| **缓存** | Redis 7 |
| **LLM** | 适配器模式 (默认 OpenAI 兼容, 可切换) |
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
POST   /generate                     启动多智能体生成
GET    /status/{session_id}          查询生成状态
GET    /stream/{session_id}          SSE 进度流
```

### 学习路径 (`/api/v1/learning-path/*`)
```
GET    /{course_id}                  个性化学习路径
GET    /{course_id}/next             下个推荐知识点
POST   /progress                     记录学习进度
GET    /{course_id}/content-style    内容类型建议
```

### Mentor (`/api/v1/mentor/*`)
```
GET    /stream/{user_id}             SSE 流式 RAG 问答
GET    /search                       知识点搜索 (预览)
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
| `LLM_API_KEY` | `""` | LLM API 密钥 (空=mock) |
| `LLM_API_BASE` | `""` | LLM API 地址 |
| `LLM_MODEL` | `"spark"` | 模型名称 |

## 项目结构

```
EduMap/
├── apps/web/                        # Next.js 15 前端
│   ├── src/app/                     # 路由页面
│   │   ├── chat/                    # 对话画像
│   │   ├── generate/                # 资源生成
│   │   ├── learn/[courseId]/        # 学习路径
│   │   ├── mentor/                  # Mentor 问答
│   │   └── profile/                 # 个人画像
│   ├── src/components/              # 组件
│   │   ├── generation/              # 生成工作流
│   │   ├── knowledge-graph/         # 技能树
│   │   ├── layout/                  # 导航栏
│   │   ├── mentor/                  # Mentor 聊天
│   │   ├── profiling/               # 雷达图/画像
│   │   └── resources/               # 资源展示
│   ├── src/hooks/                   # React Hooks
│   └── src/stores/                  # Zustand 状态管理
├── packages/shared-types/           # TypeScript 共享类型
├── services/
│   ├── backend-core/src/            # AI/Agent 服务
│   │   ├── agents/                  # 智能体实现 (7个)
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
│   ├── profile-service/src/         # 用户画像服务
│   └── sandbox-service/src/         # 代码沙箱服务
├── scripts/db/                      # 数据库初始化脚本
└── docker-compose.yml               # 开发环境编排
```

## 智能体系统

| 智能体 | 角色 | 技术 |
|--------|------|------|
| **Orchestrator** | LangGraph 状态图路由 | StateGraph, 条件边 |
| **Planner** | 知识点提取 | LLM + KG 查重 |
| **Guardian** | DAG 校验 + 难度单调性 | DFS 拓扑排序 |
| **Designer** | 多模态内容生成 | LLM 提示词工程 |
| **Coder** | 代码生成 + 沙箱验证 | AST 分析 + Docker |
| **Assessment** | 微测验生成 | LLM 结构化输出 |
| **Content Auditor** | 生成质量审核 | ChromaDB 相似度 |
| **Mentor** | RAG 问答 | ChromaDB + Neo4j 混合检索 |

## 许可证

MIT
