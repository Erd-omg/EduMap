# 智图 EduMap

多智能体个性化学习系统 —— 基于 LangGraph + Neo4j + FastAPI + Next.js

## 项目结构

```
EduMap/
├── apps/web/                        # Next.js 15 前端
├── packages/shared-types/           # TypeScript 共享类型
├── services/
│   ├── backend-core/                # AI/Agent 服务 (FastAPI)
│   ├── profile-service/             # 用户画像服务 (数据隔离)
│   └── sandbox-service/             # 代码沙箱服务 (安全边界)
├── scripts/db/                      # 数据库脚本
└── docker-compose.yml               # 开发环境编排
```

## 快速开始

```bash
# 安装依赖
pnpm install

# 启动所有服务 (需要 Docker)
docker compose up -d

# 开发模式
pnpm dev
```

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Next.js 15, React 19, Tailwind CSS v4, D3.js |
| 后端 | FastAPI, LangChain/LangGraph, Celery |
| 数据库 | Neo4j 5 (图谱), PostgreSQL 16 (画像), ChromaDB (向量) |
| 缓存 | Redis 7 |
| LLM | 适配器模式 (默认讯飞星火, 可切换) |

## 8 智能体系统

| Agent | 角色 |
|---|---|
| Orchestrator | LangGraph 状态图路由 |
| Planner | 文档 → 知识点提取 |
| Guardian | DAG 校验 + 语义去重 |
| Designer | 多模态内容生成 |
| Coder | 代码生成 + 沙箱验证 |
| Assessment | Micro-Quiz 教学反馈 |
| Content Auditor | 生成质量审核 |
| Mentor | RAG 问答 (Phase 5) |

## 许可

MIT
