# 产品需求文档 (PRD)：智图 (EduMap)

**项目代号**：智图 (EduMap) —— 高等教育多智能体个性化学习系统
**版本**：v2.0
**状态**：设计定稿（基于 V1 + 架构评审反馈）

---

## 1. 项目概述

### 1.1 背景与目标

在高等教育数字化转型背景下，针对学生面临的学习资源繁杂、缺乏精准指导等痛点，**智图 (EduMap)** 旨在构建一个基于多智能体协同（Multi-Agent）与知识图谱技术的**独立 SaaS 式**智能化学习系统。通过深度解析学生画像，实现从资源自动生成到路径动态规划的"因材施教"。

本系统的核心定位是**学生个人使用的 AI 学习伙伴**，无需教师介入。系统直接面向大学生个体，通过自然语言交互理解学生的专业背景和学习需求，自主生成个性化的多模态学习资源。

### 1.2 核心 Slogan

> **"懂知识，更懂你：因材施教的数字化实践，从构建专属你的学习图谱开始。"**

---

## 2. 系统架构设计 (Multi-Agent System)

### 2.1 总体架构

采用"编排者 (Orchestrator) + 执行者 (Executor)"架构，**MVP 阶段将核心 AI 服务合并为单一容器**以减少网络开销：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js 15)                        │
│  skill-tree-canvas | chat-window | resource-viewer | profile-card   │
│  progress-slider | personalization-banner | source-transparency     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ API Routes (BFF)
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      backend-core (FastAPI)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │Orchestr. │ │ Planner  │ │ Guardian │ │ Designer │ │  Coder   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌────────────────┐ ┌───────────────┐  │
│  │Assessment│ │  Mentor  │ │Content Auditor │ │ Prompt       │  │
│  │  Agent   │ │(Phase 5) │ │  (审核闸门)     │ │ Registry     │  │
│  └──────────┘ └──────────┘ └────────────────┘ └───────────────┘  │
│                                                                      │
│  ┌─────────────────────┐ ┌──────────────────────┐                   │
│  │  Semantic Dedup     │ │  Cold Start Eval     │                   │
│  │  (知识图谱防漂移)    │ │  (语料置信度评估)    │                   │
│  └─────────────────────┘ └──────────────────────┘                   │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌──────────┐    ┌──────────────┐    ┌──────────────────┐
│ PostgreSQL│   │  Neo4j 5 CE   │    │ ChromaDB + Redis  │
│ 画像/用户  │   │ 知识图谱   │    │ 向量库 + 缓存/队列 │
└──────────┘    └──────────────┘    └──────────────────┘
```

### 2.2 智能体角色定义（8 Agent）

| # | Agent | 角色 | LLM | 确定性 | 引入阶段 |
|---|---|---|---|---|---|
| 1 | **Orchestrator** (编排者) | LangGraph 状态图任务路由、Agent 选择、质量门调度、错误处理 | ✗ | ✓ | Phase 3 |
| 2 | **Planner** (课程规划师) | PDF/MD/DOCX/PPT 语义解析 → 知识点提取（含前置关系、难度分级）→ 图谱构建建议 | ✓ | ✗ | Phase 3 |
| 3 | **Guardian** (知识库维护官) | DAG 环检测（拓扑排序）、难度单调性校验、元数据完整性、触发语义去重 | ✗ | ✓ | Phase 3 |
| 4 | **Designer** (多媒体内容官) | 生成个性化讲解文档、Mermaid 思维导图、交互式 HTML 图表、分层习题、拓展阅读 | ✓ | ✗ | Phase 3 |
| 5 | **Coder** (代码专家) | 实操代码生成 → AST 静态分析 → Docker 沙箱执行 → 最多 3 轮迭代修复 | ✓ | ✓ | Phase 3 |
| 6 | **Assessment** (评估智能体) | Micro-Quiz 生成（1-3 选择/判断/填空）→ 自动评分 → mastery 和遗忘曲线参数更新 | ✓ | ✓ | Phase 3 |
| 7 | **Content Auditor** (内容审核员) | 生成内容与知识点元数据向量相似度审核（阈值 0.8），低于则丢弃重做；检查知识边界漂移 | ✓ | ✓ | Phase 3 |
| 8 | **Mentor** (导师) | RAG 约束问答，混合检索（图谱 + 向量）→ 受控生成 → 事后校验（Phase 5） | ✓ | ✓ | Phase 5 |

### 2.3 核心技术栈

| 层 | 技术选择 | 决策理由 |
|---|---|---|
| 前端框架 | Next.js 15 App Router | SSR 流式渲染（画像对话/导师问答）、API Routes 即 BFF、D3.js 图谱可视化生态成熟 |
| UI 组件 | shadcn/ui + Tailwind CSS v4 | 无障碍、可组合、零设计系统锁定 |
| 后端语言 | Python 3.12+ | AI/ML 生态最成熟，LangChain/LangGraph 原生支持 |
| Web 框架 | FastAPI | 异步原生、自动 OpenAPI、高性能 |
| Agent 框架 | LangChain / LangGraph | 状态图模型天然适配教育场景的分支/循环/条件路由，优于 CrewAI（仅顺序）和 AutoGen（偏对话） |
| 知识图谱 | Neo4j 5 CE + APOC | 多跳遍历（前置链查询）是核心场景，Cypher 表达力远超 PostgreSQL 递归 CTE |
| 向量数据库 | ChromaDB | 零基础设施（pip install），MVP 规模足够 |
| 关系数据库 | PostgreSQL 16 | 用户/画像/路径持久化 |
| 缓存/队列 | Redis 7 + Celery | 缓存 + Session + 异步长任务（>10s 资源生成） |
| LLM 接口 | 适配器模式（LangChain BaseLLM） | 默认讯飞星火，可切换 GPT/Claude/DeepSeek 无代码改动 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 本地运行，中文支持好，零 API 依赖 |
| 文档解析 | python-unstructured + PaddleOCR | PDF/DOCX/PPT/MD 统一处理，中文 PDF 增强 |
| 代码沙箱 | Docker SDK (256MB / 5s / 无网络) | 安全隔离，临时容器 |
| 部署 | Docker Compose | 开源一键部署，可选托管版 |

### 2.4 分层记忆系统

- **短期对话**：Redis 缓存（当前会话上下文，TTL 30 分钟）
- **中期行为**：PostgreSQL（学习行为日志、Quiz 历史、交互轨迹）
- **长期图谱**：Neo4j（知识图谱 + 学生进度 + 遗忘曲线参数）

---

## 3. 核心功能需求

### 3.1 对话式动态学生画像

| 维度 | 说明 | 来源 |
|---|---|---|
| 知识基础 (knowledge_base) | 各学科知识水平评分 (0-1)，含学科细分 | 对话抽取 + Assessment 验证 |
| 学习能力 (learning_ability) | 理解速度/解题/批判思维/记忆保持/分析能力 | 对话抽取 + 行为模式分析 |
| 学习动机 (learning_motivation) | 内在兴趣/目标导向/外部动机权重 | 对话抽取 |
| 知识覆盖率 (knowledge_coverage) | 已掌握/学习中/未开始统计，缺口识别 | 图谱推导 + Quiz 验证 |
| 交互风格偏好 (interaction_style) | 视觉/文本/交互/听觉评分 | 对话抽取 + 行为追踪 |
| 专注力特征 (focus_characteristics) | 平均专注时长、分心频率、推荐会话长度 | 行为数据分析 |

**构建方式**：通过自然语言对话自动抽取，摒弃传统填表，随学随新。

### 3.2 多智能体协同的资源生成

支持输入格式：PDF，MD，DOCX，PPT

**生成资源类型**（≥5 种）：
1. 课程讲解文档（知识点分步讲解，适配学生认知风格）
2. Mermaid 思维导图（可视化知识结构）
3. 分层练习题（选择/判断/代码填空，按难度分级）
4. 拓展阅读材料（精选外部参考，附带摘要和关联知识点标注）
5. 交互式可视化图表（HTML+JS 渲染，网页内可直接交互）
6. 代码实操案例（含自动沙箱验证，支持 Python/JavaScript）

### 3.3 个性化学习路径规划与资源推送

- **路径算法**：基于前置关系 DAG + 难度单调性约束 + 画像适配 + Assessment 结果
- **动态调整**：每完成一个 Learning Step + Assessment 后，路径自动重排
- **遗忘曲线**：Ebbinghaus 模型 + 贝叶斯个性化参数，技能树颜色编码（红 < 0.3 / 黄 0.3-0.6 / 绿 > 0.6）
- **预加载缓存**：N+1/N+2 步资源异步预生成，消除等待感

### 3.4 Assessment Agent（教学反馈闭环）

**核心作用**：解决"如何证明学懂了"的问题，为遗忘曲线参数收敛提供真实数据。

- **触发时机**：每个 Learning Step 完成后自动触发
- **Quiz 形式**：1-3 道选择/判断/填空题，2 分钟内完成
- **反馈路径**：答题结果 → 更新 mastery_level → 重新拟合遗忘曲线参数 r → 影响路径重排
- **反游戏化设计**：①结合行为隐信号（停留时间、交互深度）多因子加权 ②贝叶斯缓慢更新（抑制单次刷分）③间隔突击复习（随机抽查历史知识点）④元认知透明度（展示评估依据和权重）

### 3.5 Content Auditor（生成质量审核闸门）

- **审核时机**：Designer/Coder 生成完成后，Orchestrator 的 REVIEW 阶段
- **审核方法**：生成内容与 Context_ID（知识点元数据）向量相似度检查
- **阈值**：cosine_similarity < 0.8 → 丢弃重做（最多 3 次），> 0.8 通过
- **额外维度**：Coder 产物检查 API/语法版本兼容性；Designer 产物检查概念是否超出知识点难度等级
- **连续 3 次失败**：降级返回最佳版本 + 低置信度标志

### 3.6 冷启动策略（语料不足时的优雅降级）

| 置信度区间 | 行为 | 前端表现 |
|---|---|---|
| ≥ 0.7 | 正常生成 | 可选 Banner 显示置信度 |
| 0.4 ~ 0.7 | 生成 + 提示上传 | Banner："当前主题语料稀缺，上传 1-2 份参考资料可提升质量" |
| < 0.4 | 主动阻断 | "暂无足够语料，请先上传参考文档" → 增量处理后重评估 |

- **置信度公式**：`confidence = w1 * corpus_density + w2 * kg_coverage + w3 * cross_validate_count`
- **增量增强**：用户上传补充材料后仅处理新增部分，无需重新全量生成

### 3.7 个性化透明度

每个生成的资源文件开头嵌入结构化 Header，展示个人化适配依据：

```
personalization:
  rationale: "根据您的[认知风格: 视觉型]和[易错点: 数组下标越界]，
              本材料特别强化了逻辑示意图和代码边界检查"
  target_gaps:
    - knowledge_point: "数组"
      gap: "下标越界处理"
  confidence: 0.94
```

前端渲染为可折叠 Banner，默认展开。

### 3.8 智能辅导（可选加分项）

Mentor Agent（Phase 5）：
- 即时多模态答疑
- 混合检索（Neo4j 精确事实 + ChromaDB 语义片段）→ RAG 约束生成
- 事后校验确保回答有据可依

### 3.9 学习效果评估（可选加分项）

通过实时跟踪学习行为、Quiz 情况、资源使用反馈，依托多维度数据分析实现精准评估，并根据评估结果动态调整资源推送策略和学习计划。

---

## 4. 关键架构机制

### 4.1 Prompt Registry（提示词注册中心）

所有 Agent 的 System Prompt 以 Markdown 文件版本化管理，禁止代码中写死字符串：

```
backend-core/src/prompts/
├── registry.json              # 索引（prompt_id → version, path, agent_id）
├── planner/v1-system.md
├── designer/v1-explanation.md
├── content-auditor/v1-audit.md
└── assessment/v1-micro-quiz.md
```

代码中通过 `PromptRegistry.get("designer/v1-explanation").render(kwargs)` 引用。

### 4.2 语义去重/融合层（防图谱漂移）

Planner 提取新知识点节点时，强制进行向量相似度召回：

```
新节点 proposal → 向量嵌入 → ChromaDB 语义搜索
  → cosine_similarity > 0.9 → 返回已有节点 ID → Merge（别名/描述/难度加权平均）
  → ≤ 0.9 → 创建新节点
```

Neo4j 节点增加 `merged_from_ids` 和 `canonical` 字段追踪。

### 4.3 三层幻觉防护

| 层 | 时机 | 机制 |
|---|---|---|
| 输入约束 | 生成前 | 每个 Agent 的 System Prompt 注入"核心知识元数据说明书" |
| RAG 约束 | 生成中 | 上下文仅含 KG 精确定义 + 向量库相关片段 |
| Content Auditor | 生成后 | 向量相似度审核（阈值 0.8）+ 知识边界检查 |

### 4.4 反游戏化机制

1. **行为隐信号多因子加权**：Quiz 结果 + 资源停留时间 + 交互深度 + 求助行为 + 复习触发 → 综合评估 mastery
2. **贝叶斯缓慢更新**：`posterior = prior * alpha + quiz_evidence * beta`，其中 `alpha >> beta`
3. **间隔突击复习**：当前 Quiz 中随机混入 1 道历史知识点题目，若答错则下调对应 mastery
4. **元认知透明度**：Quiz 后展示评估依据和权重，降低单一维度操控动机

### 4.5 数据合规（PIPL）

- 学生画像（6 维 + 对话历史 + Quiz 记录）属于敏感个人信息
- 需要明确告知用户数据用途、存储期限、删除权利
- 若使用第三方 LLM API，需确认数据处理协议合规
- 支持数据导出和账号注销时数据删除
- 毕业数据默认匿名化保留（仅用于模型改进，不可反推个人身份）

---

## 5. 演示功能（验收辅助）

### 5.1 Progress Slider（技能树进度模拟）

- **位置**：skill-tree-canvas 底部
- **类型**：Range slider（0% ~ 100%）
- **数据源**：`POST /backend-core/debug/simulate-growth` 接口
- **效果**：拖动时技能树节点颜色实时变化、遗忘曲线重绘、路径重排动画
- **用途**：评委直接拖动滑块，展示"从零到完全掌握某章节"的全过程

### 5.2 来源透明度悬浮提示

- 所有资源文档显示"来源透明"标志
- 鼠标悬停弹窗显示基于哪些知识图谱节点生成、AI 置信度、幻觉校验结果

### 5.3 debug/simulate-growth 接口

- 一键模拟学生"在一周内从 0 到 100 掌握某章节"的全链路数据
- 返回画像更新、技能树快照、路径重排事件、遗忘曲线数据的时间序列

---

## 6. 实现阶段

### Phase 1：项目骨架 + 基础设施（约 2 周）
- Monorepo 结构 + pnpm workspace + Docker Compose
- shared-types 包 + Next.js 脚手架 (Tailwind + shadcn/ui)
- 所有 Python 服务 FastAPI + /health
- Neo4j seed 数据 + ChromaDB + Redis + PostgreSQL
- CI pipeline + `.env.example`

### Phase 2：对话画像 + 知识图谱（约 3 周）
- Neo4j 数据访问层 + 语义去重/融合层
- Knowledge Graph API（CRUD + 遍历 + 路径查询 + 去重）
- Profile Service + LLM 对话分析器 + 6 维雷达图组件
- 画像聊天界面（SSE 流式）
- 冷启动评估模块原型

### Phase 3：多智能体资源生成 + 评估闭环（约 7 周）
- Prompt Registry + LangGraph 状态图 + LLM 抽象层
- Planner → Guardian → Designer → Coder 四级管线
- Assessment Agent（Micro-Quiz → 评分 → 画像反馈）
- Content Auditor（相似度 < 0.8 丢弃重做，最多 3 次）
- Personalization Header 注入 + 前端 Banner 渲染
- RAG 模块 + Celery 异步管道
- 冷启动策略完整实现
- 反游戏化机制（多因子加权 + 贝叶斯更新）

### Phase 4：学习路径 + 推送 + 演示（约 3 周）
- 学习路径算法 + 遗忘曲线模型 + 技能树颜色叠加
- 预加载缓存 + Progress Slider + debug/simulate-growth
- 间隔突击复习 + 元认知透明度展示

### Phase 5：Mentor + 优化 + 文档（约 2 周）
- Mentor Agent + 性能优化（< 2s / < 10s）
- 安全审计 + 文档 + 负载测试 + 幻觉率测量（> 95%）

---

## 7. 非功能性需求

1. **安全性**：代码执行采用 Docker 临时容器，禁用外网访问，内存 < 256MB，时间 < 5s
2. **交互规范**：流式输出，内容卡片化展示，支持 Markdown 渲染
3. **响应效率**：核心链路 < 2s，资源生成 < 10s（异步 + 预加载）
4. **幻觉率**：防幻觉校验准确率 > 95%
5. **可维护性**：所有 Agent 提示词通过 Prompt Registry 版本化管理

---

## 8. 量化关键指标

| 指标 | 目标 |
|---|---|
| 核心考点资源覆盖率 | 100% |
| 资源生成平均响应时间 | < 10s（异步骨架预加载） |
| 防幻觉校验准确率 | > 95% |
| Assessment 覆盖知识点的比率 | > 90%（每知识点生成后必有 Quiz） |
| Content Auditor 拦截率 | 可测量，无固定目标（前期预期较高中期降低） |
| 画像维度持续更新频率 | 每次交互后更新至少 1 个维度 |

---

## 9. 后续演进方向

- **社交学习**：基于知识图谱相似度推荐学习伙伴
- **全平台部署**：Web → 移动端（小程序/App）
- **教师仪表盘**：班级画像分布、学习进度概览（非核心定位，按需）
- **LMS 集成**：与学校现有学习管理系统对接
- **评估效度对标**：与课程期末考试成绩做相关性分析，验证 Assessment 准确性