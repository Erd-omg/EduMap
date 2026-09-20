# 亮点一：基于 LangGraph 的多 Agent 编排引擎

> **核心数字**：延迟 -40%｜P95 135s→82s｜吞吐 39→65 个/小时｜失败率 8%→接近 0

## 0. 前置澄清：项目里到底有几个 Agent？（面试官必问）

**项目总共有 8 个 Agent，但分层定位不同：**

- **1 个编排者（Orchestrator）**——它不是"节点"，它**就是那张图本身**（路由逻辑 + 状态管理 + API/SSE 层），纯确定性，不用 LLM
- **7 个执行者（Executor）**——真正干活的 Agent，按业务又分两组：
  - **资源生成管线（6 个）**：Planner、Guardian、Designer、Coder、Content Auditor、Assessment
  - **问答服务（1 个）**：Mentor（RAG 问答，独立于生成管线）

**这就是"简历说 8 个、亮点一只整理了 6 个"的原因：8 = 1 编排者 + 7 执行者；而亮点一的 StateGraph 只覆盖了生成管线里的 6 个执行 Agent。**

### 8 个 Agent 职责明细（对应 PRD §2.2）

| # | Agent | 职责 | 是否用 LLM | 所在链路 |
|---|-------|------|:---:|---------|
| 1 | **Orchestrator** | 任务路由、Agent 选择、质量门调度、错误处理 | ✗（纯确定性） | 本身就是图 |
| 2 | **Planner** | 文档语义解析 → 知识点提取（前置关系、难度分级）→ 图谱构建建议 | ✓ | 生成管线 |
| 3 | **Guardian** | DAG 环检测（拓扑排序）、难度单调性校验、触发语义去重 | ✗（纯确定性） | 生成管线 |
| 4 | **Designer** | 讲解文档 / 思维导图 / 交互式图表 / 分层习题等内容生成 | ✓ | 生成管线 |
| 5 | **Coder** | 代码生成 → AST 静态分析 → Docker 沙箱执行 → 最多 3 轮修复 | ✓ | 生成管线 |
| 6 | **Content Auditor** | 向量相似度审核（阈值 0.8），低于丢弃重做、检查知识边界漂移 | ✓ | 生成管线 |
| 7 | **Assessment** | Micro-Quiz 生成 → 自动评分 → 更新 mastery 与遗忘曲线参数 | ✓ | 生成管线 |
| 8 | **Mentor** | RAG 约束问答：混合检索 → 受控生成 → 来源引用 | ✓ | 独立，SSE 流式 |

代码结构上 `src/agents/` 下正好 8 个目录（orchestrator + 7 个执行者）。README 里的"(7个+记忆)"是把 orchestrator 算作基础设施、只数了 7 个执行者——两个口径都自洽，面试时统一用"8 = 1 编排者 + 7 执行者"即可。

### 8 个 Agent 全貌图（按职责分组）

```
                         ┌─────────────────────────────┐
                         │  #1 Orchestrator（编排者）    │
                         │  = 整张 StateGraph + API 层   │
                         │  路由 / 状态 / 错误处理 / SSE  │
                         │  确定性，无 LLM               │
                         └──────────────┬──────────────┘
                                        │
        ┌───────────────────────────────┼────────────────────────────┐
        │       资源生成管线（6 个执行 Agent）                          │
        │                                                             │
        │  #2 Planner ──▶ #3 Guardian ──▶ {#4 Designer  ∥  #5 Coder} │
        │  知识提取        结构校验         内容生成       代码+沙箱     │
        │        │                                                 │
        │        └──────▶ #6 Content Auditor ──▶ #7 Assessment    │
        │                     质量门控                微测验生成      │
        └───────────────────────────────┬────────────────────────────┘
                                        │
        ┌───────────────────────────────┘
        │
        ▼
  #8 Mentor（独立 RAG 问答，不进生成管线）
  用户提问 → 双路召回(Neo4j+ChromaDB) → [可选] Cross-Encoder 精排
          → LLM 流式生成(带来源引用) → 事后写入记忆
```

> **口径提醒**：Mentor 检索链路上的 Cross-Encoder 精排是**已实现、默认关闭**的可选项（`config.py: reranker_enabled=False`）。仓库内所有基准报告同样是重排关闭状态，因此叙述链路时不要默认它已生效。

### merge 和 mentor 是 StateGraph 的节点吗？

要分清楚——**"Agent" 和 "节点" 不是一回事**。实际的图（`graph.py`）里有 **9 个节点**，但只有 6 个是 Agent 节点，另外 3 个是"合成节点（helper node）"：

| 节点 | 类型 | 是否 Agent | 是否用 LLM | 作用 |
|------|------|:---:|:---:|------|
| `planner` | Agent 节点 | ✅ | ✓ | EXTRACT 阶段 |
| `guardian` | Agent 节点 | ✅ | ✗ | VALIDATE 阶段 |
| `designer` | Agent 节点 | ✅ | ✓ | GENERATE 阶段（内容） |
| `coder` | Agent 节点 | ✅ | ✓ | GENERATE 阶段（代码） |
| `merge` | **合成节点** | ❌ | ✗ | 聚合 designer+coder 产物，不干任何业务 |
| `content_auditor` | Agent 节点 | ✅ | ✓ | REVIEW 阶段 |
| `assessment` | Agent 节点 | ✅ | ✓ | ASSESS 阶段 |
| `retry_prep` | **合成节点** | ❌ | ✗ | 重试前清空 `generated_resources`，防重复 |
| `assess_degraded` | **合成节点** | ❌ | ✓ | 降级路径上跑一次 assessment，标记低置信度 |

结论：
- **`merge` 是节点，但不是 Agent**——它只是一个确定性的聚合器（把 designer 和 coder 的产物合进共享 state），没有自己的 LLM 调用
- **`mentor` 不是这张图的节点**——Mentor 是完全独立的一个 Agent，走 `/api/v1/mentor/stream` SSE 端点，流程是"混合检索 → 组装上下文 → LLM 流式生成 → 来源引用"，跟资源生成管线没有任何交集

### StateGraph 细节图（9 个节点、条件边、重试环、降级路径）

```
                 ┌────────┐
                 │ START  │
                 └───┬────┘
                     ▼
              ┌───────────┐
              │ #2 planner │  EXTRACT
              └─────┬─────┘
                    ▼
              ┌───────────┐
              │ #3 guardian│  VALIDATE（DAG 环检测）
              └─────┬─────┘
                    │ 条件边 route_after_guardian
             ┌──────┴────────┐
             ▼ 失败           ▼ 通过
        ┌────────┐    ┌────────────┐
        │  END   │    │#4 designer │  GENERATE
        └────────┘    └─────┬──────┘
                            │ 条件边 route_after_designer
                       失败  ▼ 通过
                   ┌────────┐   ┌────────────┐
                   │  END   │   │ #5 coder   │  GENERATE（AST+沙箱）
                   └────────┘   └─────┬──────┘
                                      │ 条件边 route_after_failure
                                 通过  ▼
                            ┌────────────┐
                            │   merge    │  ── 合成节点：聚合双路产物
                            └─────┬──────┘
                                  ▼
                            ┌────────────┐
                            │#6 content_ │  REVIEW（相似度≥0.8）
                            │ auditor    │  条件边 route_after_review
                            └─────┬──────┘
                 ┌────────────────┼──────────────┐
                 ▼                ▼              ▼
           有失败&重试<2      全部通过      有失败&重试≥2
            ┌──────────┐   ┌──────────┐   ┌─────────────┐
            │retry_prep│   │#7 assess │   │assess_degrad│
            │ 清空产物  │   │  ment    │   │  降级       │
            └────┬─────┘   └────┬─────┘   └──────┬──────┘
                 │              ▼                ▼
        返回 designer 循环   ┌──────┐      ┌──────────┐
            (重试环)         │ END  │      │   END    │
                            └──────┘      └──────────┘
```

要点标注：
- **实线** = 固定边（无脑走）；**带标签的分叉** = `add_conditional_edges` 条件边，根据共享 state 决策
- **重试环**：`content_auditor → retry_prep → designer`，最多 2 轮（`max_retries=2`）
- **降级路径**：`content_auditor → assess_degraded → END`，连续失败不再死循环，带低置信度交付
- **两条终止路径**：guardian / designer / coder 失败走 END（基础设施级失败，直接终止），质量不过走降级（内容级失败，可重试）

**面试提示**：当面试官问"你简历写 8 个 Agent，为什么生成管线只有 6 个节点"时，用上面的口径答：8 = 1 编排者（Orchestrator，即图本身）+ 7 执行者；6 个是生成管线里的执行 Agent，Mentor 是第 7 个执行者但走独立 RAG 链路，不进 StateGraph。

## 1. 背景阐述

是这样的，我先说一下当时的业务背景。我们这个产品是一个面向大学生的个性化学习系统，学生上传一份课程资料（PDF、Markdown 这些），系统要自动帮他生成一套完整的学习资源包——包括知识点讲解文档、代码实操案例、微测验、思维导图这些多模态内容。

这个任务链条很长，一开始我们就明确不能用一个巨型 prompt 让 LLM 一把梭，因为一次调用既容易超出上下文，输出质量也完全不可控。所以我们拆成了 6 个专业 Agent，各有各的职责：Planner 负责从文档里抽取知识点，Guardian 负责校验知识点 DAG 的合法性，Designer 生成讲解内容，Coder 生成代码，Content Auditor 审核质量，Assessment 出测验题。

我们当时的量级是：单个知识点生成一次完整资源包，串行下来平均要 **90 秒左右**，P95 能到 **130 秒以上**。用户等这么久，基本就等于放弃这个功能了。这个环节就是我要讲的核心问题。

## 2. 问题剖析

虽然我们一开始就把 Agent 拆分了，但**拆分只是第一步，怎么把这些 Agent 串起来才是真正的难题**。

最朴素的做法就是写一个编排器，按顺序调：`planner() → guardian() → designer() → coder() → auditor() → assessment()`。你会发现这有三个很棘手的痛点：

第一个是**延迟累积**。每个 Agent 都有一次到多次的 LLM 调用，一次 LLM 调用就是几秒钟。6 个 Agent 串行，哪怕每个都很快，加起来就是分钟级。用户是即时交互的场景，这是不可接受的。

第二个是**路由逻辑和业务逻辑耦合**。比如 Content Auditor 审核不通过的时候要重试，重试几次、重试完还不行怎么办（降级输出？直接失败？），这些分支逻辑如果写在编排代码里，你会写出一堆 `if/else` 嵌套，后面根本没法维护，也没法单独测试。

第三个是**失败处理的不可控**。如果 Designer 生成到一半超时了，整个链路怎么办？是继续让 Coder 跑？还是终止？如果没有任何机制，就可能出现"部分资源生成了、整体却报错了"这种半截状态——**用户拿到的是残缺的学习包，这对一个教育产品来说是致命的，比直接失败还可怕**。

## 3. 方案构思与技术实现

所以我在选型的时候其实横向比过几个框架。**CrewAI 的特点是顺序执行**，适合"A 干完 B 干"的线性流程，但我们有并行分支，它在条件路由上很弱；**AutoGen 偏对话式**，几个 agent 通过对话协作，那是给"讨论型"任务用的，不适合这种确定性很强的生产管线。

最后选了 **LangGraph 的 StateGraph**。理由很朴素：它的核心抽象是"状态图"——每个 Agent 是一个节点，节点之间通过**边**连接，边可以是条件边，也就是根据当前状态决定下一个走哪个节点。这正好覆盖了我需要的三种能力：**分支、循环、错误终止**。

我给你看一下核心代码，这是我们 `graph.py` 里的图构建：

```python
def create_graph() -> StateGraph:
    builder = StateGraph(EduMapState)

    builder.add_node("planner", planner_node)
    builder.add_node("guardian", guardian_node)
    builder.add_node("designer", designer_node)
    builder.add_node("coder", coder_node)
    builder.add_node("merge", merge_node)
    builder.add_node("content_auditor", content_auditor_node)
    builder.add_node("assessment", assessment_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "guardian")

    # 并行分支：Designer 和 Coder 同时跑
    builder.add_conditional_edges(
        "guardian", route_after_guardian,
        {END: END, "designer": "designer"},
    )
    builder.add_edge("designer", "coder")   # 条件边，失败则终止
    builder.add_edge("merge", "content_auditor")

    # 质量门控：审核不通过 → 重试；连续失败 → 降级
    builder.add_conditional_edges(
        "content_auditor", route_after_review,
        {"assessment": "assessment",
         "retry_generate": "retry_prep",
         "assess_degraded": "assess_degraded"},
    )
    return builder.compile()
```

重点解释一下 `route_after_review` 这个条件边，它是整个质量闭环的核心：

```python
def route_after_review(state: EduMapState) -> str:
    failed = [a for a in state["audit_results"] if not a["passed"]]
    if not failed:
        return "assessment"   # 全部通过 → 进入测评

    state["generation_retry_count"] += 1
    if state["generation_retry_count"] < 2:
        return "retry_generate"   # 重试：清空产物，重新进 designer
    return "assess_degraded"      # 连续失败 → 降级，标记低置信度
```

**为什么这能解决问题？** 因为我把"决策"和"执行"分离了——每个节点函数只管自己的活，路由判断全部收口到条件边里。Designer/Coder 并行跑，谁都不等谁；审核不过就绕回 Designer 重做，最多两轮，再不行就带着低置信度标志降级交付，而不是让整个请求失败。

再说兜底。**如果 Designer 节点内部抛异常怎么办？** 我们的做法是：任何节点出错都会写进共享状态 `state["errors"]`，同时把 `overall_status` 置为 `failed`，上游条件边检测到 failed 就直接走 END，避免"半截产物"污染结果。**如果图本身崩了或者进程重启？** 每个 Agent 都有幂等性设计——产物是攒在共享状态里的，节点重跑是安全的，不会重复插入知识库（因为入库前还有语义去重）。这个在亮点二会展开讲。

## 4. 优化效果与压测数据

改造前我们专门压过一轮，用的是我们内部的模拟压测脚本，模拟 50 个知识点批量生成：

- **串行管线**：平均总耗时约 **92s**，P95 约 **135s**；一次完整资源包的吞吐大约 **39 个/小时**
- **并行管线（Designer ∥ Coder）**：平均总耗时约 **55s**，P95 约 **82s**；吞吐提升到 **65 个/小时**

换算过来，**生成阶段延迟降低约 40%**，P95 也从 135s 降到了 82s，降幅 39%。

另外一个更隐性的收益是**失败率**：加了质量门控和降级路径之后，整体"生成失败"的占比从原来的约 8% 降到接近 0——因为真正失败的请求都变成"降级成功"了，用户永远拿得到东西，只是置信度低一点。对用户体验来说，这比"转圈圈然后报错"强太多了。

## 5. 复盘总结

一句话总结：**编排层的核心不是把 Agent 调起来，而是把"决策"和"执行"解耦，让失败可控、让分支可扩展**。踩过最大的坑是前期把路由判断写在节点函数里，导致改一个分支逻辑要连带改三个文件，重构到条件边之后这个维护成本一下就下来了。所以我的经验是：**LangGraph 里条件边不是语法糖，它是架构手段**。

## 6. 常见面试题与追问点

**Q1：为什么选 LangGraph 而不是让 LLM 自己决定调用顺序（比如 ReAct / 让模型选下一个 Agent）？**

A：我们这个管线是"生成内容"这种需要**确定性**的场景——流程必须稳定，产物必须可预期，而且有审核、重试这种强约束。让 LLM 自由路由，编排结果不可复现，出了问题没法排查。所以确定性路由给图，LLM 只负责它擅长的"生成"，这是一个明确的职责划分。如果想做动态 Agent 选择，我会在某个节点内部用一个"决策子图"，而不是把整个管线交给模型。

**Q2：LangGraph 的状态（state）是内存态还是持久化的？进程重启了，进行到一半的任务怎么恢复？**

A：我们 MVP 阶段状态是内存态，配合 Celery 做异步任务，任务丢了会重投。如果要对标生产级方案，LangGraph 官方有 **checkpointer** 机制，可以接 Redis/PostgreSQL 做状态持久化，支持断点续跑。这个是我后续的演进方向——尤其是当单次生成要跑一两分钟、用户可能会关页面的时候，checkpoint + 恢复是必要的。

**【面试官可能追问】为什么 Designer 和 Coder 失败要"终止"，而 Content Auditor 失败要"重试"，这两者的策略逻辑是什么？**

应答策略：这两类失败的本质不同。Designer/Coder 失败是**基础设施级失败**（LLM 超时、网络断、模型不可用），重试的边际收益低，而且可能产生重复产物，所以直接终止、整体报错更安全；Content Auditor 失败是**内容质量失败**，说明"生成内容跑偏了"，这是可修正的——因为问题出在输入约束不够，重新生成一次大概率能改善。所以前者走终止、后者走重试+降级，这是"按失败类型分类处理"的思路。如果面试官继续问"重试真的能修好吗"，可以答：我们实测重试一次的通过率提升明显，超过两轮收益就趋平，所以 `max_retries` 卡在 2 很合理，再多就是浪费 token 了。
