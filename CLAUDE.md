# EduMap — Claude Code 工作约定

本文件是本仓的强制约定。与默认行为冲突时，以本文件为准。

## 测试

### 铁律：每个新测试都必须做变异验证

**写完一个测试后，把被测的 bug 人为塞回去，确认该测试会失败。不失败就说明它没在测它声称测的东西。**

这条规则不是洁癖，而是本仓踩出来的。以下四个测试全都"通过"，但都没测到它们声称覆盖的东西：

| 测试 | 声称覆盖 | 实际上 |
|---|---|---|
| `test_retry_count_incremented`（旧版） | 重试计数会递增 | 直接把 dict 传给路由器，**原地修改在单测里生效、在 LangGraph 里被丢弃**。真实后果是重试环无限循环 |
| `TestParallelFanOutIsLegal` | 并行扇出合法 | 自己搭了个"形状像生产"的图，没验证真实图接线。`designer_node` 改回返回全量列表它照样通过 |
| 注解元数据断言（旧版） | reducer 已声明 | `get_type_hints()` **会剥离 `Annotated` 元数据**，断言恒为真 |
| `test_status_is_not_the_initial_running_state_after_completion`（我写的第一版） | 完成态不再是 running | 测试 app 从不调 `configure_graph`，图在 planner 处以 "Agent not configured" 失败，**根本走不到 assessment_node**；塞回 bug 它也照样通过 |

共同点：**测试的调用方式与生产调用方式不一致**。

#### 怎么做变异验证

小改动不需要装工具，手工即可：

```bash
cp src/path/to/module.py /tmp/module.bak
# 手工把 bug 塞回去（删掉 await、调换分支、去掉校验）
pytest tests/path/to/test_file.py -q      # 必须失败
cp /tmp/module.bak src/path/to/module.py  # 恢复
git diff --stat src/                      # 确认恢复干净
```

**恢复后务必 `git diff --stat src/` 确认**，否则会把故意引入的 bug 留在树里。

对核心模块（见下）或大批量新增测试，用仓库自带的脚本自动做：

```bash
cd services/backend-core
python scripts/mutation_check.py           # 跑全部配置目标
python scripts/mutation_check.py --list    # 只看有哪些变异，不执行
python scripts/mutation_check.py -v        # 幸存者附测试输出
```

**为什么不用 mutmut**：mutmut 3.8 无法用于本项目。它的 trampoline 硬断言被变异模块名**不得以 `src.` 开头**（`mutmut/stats.py:151`），而本仓的 import 根**就是** `src`（`src/__init__.py` 存在，测试写 `from src.agents... import`），于是每个变异体都触发断言、stats 阶段直接中止。该限制硬编码且不可配置，因此不去改第三方库，而是用脚本自己实现同样的技术（配置见 `scripts/mutation_check.py` 的 `MUTATIONS`）。

`mutation_check.py` 会**在 finally 中无条件恢复**每个被改动的文件——若中途崩溃也可能留下变异体，所以跑完请 `git diff --stat src/` 确认。**注意：脚本运行期间不要编辑 `src/`**，否则会和变异体混在一起。

### 让测试走真实入口

优先级从高到低：

1. **HTTP 端点**（`TestClient` 驱动真实 app）——最高保真
2. **真实编排入口**（`_run_generation`、`BaseAgent.execute()`、`create_graph()` 编译后的图）
3. **公开服务方法**（`svc.search()`、`ops.build_context()`）
4. **内部函数**——仅在纯函数（无框架语义）时可接受

**框架语义只在完整链路里才成立**，典型如：LangGraph 的节点/路由器/状态通道/reducer、装饰器与依赖注入、中间件、Pydantic 边界校验、async 上下文。直接调内部函数会绕过这些。

已知必须走真实入口的场景：

- **LangGraph 状态通道**：路由器里改 `state[...]` 会被丢弃（只有节点返回值会被提交）。要测重试环、reducer、扇出扇入，必须 `create_graph()` + `ainvoke`。
- **工具层**：`_handle_tool_calls` 只是解析，真实链路是 `_gather_tool_context → _last_tool_calls → ExecutionReport.tool_calls → JSON`。这条链断过两次，都是集成级，单测测不到。
- **经 HTTP 回读会话的端点**：`GET /orchestrator/status`、`POST /cancel` 依赖 `memory_ops.short_term` 真实往返。conftest 的桩已改为内存回滚存取；若你要"会话不存在"的用例，显式覆盖。

### 不要写弱断言

```python
# 禁止：接受任何结果，等于没断言
assert resp.status_code in (200, 422, 500)

# 允许：断言具体契约
assert resp.status_code == 200
GenerateResponse.model_validate(resp.json())   # schema 是唯一真源
```

字段存在性断言（`assert "x" in body`）在字段改名时不会失败。能用 Pydantic 模型就优先用——前后端共享 `packages/shared-types`，目前**没有任何测试验证后端响应符合共享类型**。

### 不要复制生产逻辑到测试里

测试里重新实现一遍被测逻辑（复制合并循环、复制 prompt 字符串）只会验证副本。**导入并调用真实实现**。

已有反例：`TestParallelFanOutIsLegal` 自建图；`ParallelBranchGraph` 手写 `astream` 的 step 形状。前者应改为 `create_graph()`（同文件的 `TestRetryLoopTerminates` 是正确示范）。

### 用例必须自带说明它测的是哪条真实路径

在测试类的 docstring 里写明生产入口，例如：

```python
class TestOrchestratorStatus:
    """GET /api/v1/orchestrator/status/{session_id} — 真实 HTTP。"""

class TestRetryLoopTerminates:
    """真实编译图（create_graph + _graph_ctx 注入假 Agent）—— 重试环必须终止。"""
```

目标是让 reviewer 一眼看出"这条路径在生产里怎么被触发的"。

### 断言不了的东西要如实说明

若某测试受环境限制无法覆盖它名义上的目标，**在 docstring 里写明**并指出真正覆盖它的是哪个测试。不要留下一段看起来在守护某 bug、实际永远为真的断言。

## 变异测试范围

配置在 `services/backend-core/scripts/mutation_check.py` 的 `MUTATIONS`（**不用** `pyproject.toml` 的 `[tool.mutmut]`，原因见上）。当前 scope：

```
src/agents/orchestrator/graph.py    (编排图：扇出/扇入、条件路由、重试环)
src/agents/orchestrator/state.py    (状态 reducer)
src/harness/retry.py                (重试 + 熔断桥接)
src/harness/base.py                 (Agent 生命周期)
src/tools/registry.py               (工具执行 + 超时)
src/utils/llm_adapter.py            (熔断状态与归因)
src/main.py                         (readiness 契约)
```

变异体是**手写**的而非通用 AST 变异：目的是探测本仓真正依赖的那些不变量（reducer 语义、失败归因、重试边界），而不是产出上百个等价变异体。

**不纳入 CI 门禁**，只作为 triage 清单——幸存者里常有等价变异（无论如何都测不出来）。**只对核心模块跑**，不要扩到全仓。

新增测试后，建议把该测试要守的那条不变量加成一条 `Mutation`——这样以后别人改坏它会被自动发现，而不依赖记性。

## 已知的未修问题

写在这里避免被重新发现或误以为已修：

- **熔断器的恢复仍依赖"到期后有一次成功调用"**：`_record_success` 是唯一重置路径，而熔断期间的降级响应不会调用它。所以上游持续故障时会呈现"到期→试一次→再开 60s"的循环。**归因与可观测性已补**（`circuit_breaker_state()` + `/health/ready` 的 `circuit_breaker` 段，`_circuit_open_count` 让这种 flap 循环可见），但**恢复策略本身未改**——若要让熔断期内的降级响应也算作探活，需要单独决策。
- **`router.py` 的实时进度合并**必须应用与图相同的 reducer（`_STATE_REDUCERS`，由注解派生）。若新增 reducer 却绕过它，`/status` 快照会与图真实状态漂移。
- **并行分支的失败语义**：一条分支失败不会中止另一条（LangGraph 会把两条都跑完），失败分支的 LLM 调用已经花掉。
- **`readiness()` 闭包捕获的是 `main.py` 的模块级 `app`**，因此 `_create_test_app()` 的测试 app 看不到它（conftest 里仍有一个只返回 `{"status":"ok"}` 的桩）。真实契约由 `tests/test_api/test_readiness.py` 直接调 `main.readiness()` 覆盖。
- **三个用例会挂起（既有问题，与本次改动无关；均已用 `git worktree` 检出 d0b470e 复现）**——同一个根因：**测试路径上加载了真实的 `SentenceTransformer` 嵌入模型**，进程内加载耗时长且不稳定（有时秒过、有时挂死 >20 分钟）。

  | 用例 | 触发路径 |
  |---|---|
  | `test_api/test_resources.py::TestResourcesUpload::test_upload_text_file` | 上传接口 → `DocumentParser.parse_and_index` → `parser.py:289-291` 回退加载真实模型（conftest 未设 `app.state._embedding_model`） |
  | `test_rag/test_chunking.py::TestSemanticChunkerChunk::test_single_sentence` | `chunk()` → `_load_model()`（`semantic_chunker.py:66`） |
  | `test_rag/test_chunking.py::TestSemanticChunkerChunk::test_fallback_when_model_none` | 该用例**本意**是测 `_model=None` 的降级路径，但 `chunk()` 会调 `_load_model()` **把模型重新加载回来**，于是降级路径根本没被走到，反而去加载模型 |

  模型已缓存（`~/.cache/huggingface`），所以不是下载问题，是进程内加载 + 可能的 HF 联网检查。**复现方式是"每次必挂"，但全量跑时是否挂取决于执行顺序**——这让它长期未被发现。

  **修复方向**：① 测试里注入 mock 的 `_embedding_model` / 打桩 `SentenceTransformer`；② `test_fallback_when_model_none` 应改为直接测 `_find_boundaries`-级别的降级分支，或 mock 掉 `_load_model`，否则它连自己的断言目标都没覆盖到。**注意后两点是真实的测试缺陷，不只是"慢"。**

  **临时绕过**：`pytest tests/ --deselect tests/test_api/test_resources.py::TestResourcesUpload::test_upload_text_file --deselect tests/test_rag/test_chunking.py::TestSemanticChunkerChunk::test_single_sentence --deselect tests/test_rag/test_chunking.py::TestSemanticChunkerChunk::test_fallback_when_model_none`

### 排障提醒：慢 ≠ 挂，且不要靠百分比定位

排查上述挂起时我反复误判，浪费了大量时间。教训：

- 全量测试正常约 **4–5 分钟**（其中 `test_rag` 约 55s、`test_kg` 约 37s、`test_harness` 约 13s 是真实耗时，不是卡住）。
- **不要用百分比估算卡在哪个用例**——我按 dot-stream 百分比猜了三次，三次都错。正确做法：`pytest -v`，然后找**最后一条没有 PASSED/FAILED verdict 的 `::` 行**；但注意 pytest 会缓冲输出，这一行可能是"刚启动"而非"卡住"，需结合耗时判断。
- **最有效的定位手段是按目录分别跑**（`for d in tests/*/; do pytest $d; done`），每个目录单独跑通常都能跑完，从而快速排除大部分范围。
- **`--timeout` 不可用**：本仓没装 `pytest-timeout`，会直接报参数错误。
- 怀疑卡死时先 `pkill -f pytest` 清残留——被杀的后台运行会留下孤儿进程，干扰后续判断（我自己制造过 6 个）。
