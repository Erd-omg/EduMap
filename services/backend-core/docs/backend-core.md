# backend-core — 运行方式与单 worker 约定

> 这份文件由 `tests/test_agents/test_worker_mode.py::TestDeploymentFiles`
> 守护：它必须保留下面的 `edumap:single-worker` 标记，且不得出现带多个 worker 的启动参数。

## edumap:single-worker

backend-core 必须以**单 uvicorn worker** 运行。原因不在性能，而在状态归属：

| 状态 | 位置 | 多 worker 下的后果 |
|---|---|---|
| `router._sse_queues` | 进程内 | SSE 重连落到另一个 worker → 收不到任何事件 |
| `router._tasks` | 进程内 | `/orchestrator/cancel` 落到另一个 worker → 409，取消无效 |
| `graph._graph_ctx` | 进程内（每 worker 各一份） | 本身没问题；配错时由启动检查拒绝 |
| `learning_path.QuizStore` | **Redis**（跨 worker 共享） | 已解决（2026-11-19） |

**纵向扩容**（更多 CPU / 内存）照常。**横向扩容**需要先把表头三项进程内状态迁到 Redis，
而那是架构改动，不是配置改动——详见 `src/agents/orchestrator/worker_mode.py`。

## 能做什么、不能做什么

- `docker run` / `docker compose` 使用镜像默认命令：受限（单 worker），正确。
- 启动脚本里用 `guard_run`：

  ```python
  from uvicorn import run
  from src.agents.orchestrator.worker_mode import guard_run
  guard_run(run, "src.main:app", host="0.0.0.0", port=8000)
  ```

  `guard_run(..., workers=4)` 会在**父进程 fork 子进程之前**抛 `RuntimeError`。

  也可以直接用 CLI 跑单 worker：
  `uvicorn src.main:app --host 0.0.0.0 --port 8000`

- **不能**依赖运行期检查发现命令行里给的 worker 数（例如 `workers=4`）：它用 `spawn` 启动子进程，
  标志不进入子进程环境，uvicorn 0.51 也没有可供 ASGI 应用查找的 server 注册表。
  该参数会**静默生效**——这正是本文件存在的原因。

## 旁注：quiz 答案密钥也必须跨进程

`QuizStore` 存 `/quiz/generate` 产出的题目与答案（`quiz:` 前缀，TTL 15 分钟，
`GETDEL` 原子消费），之前是进程内的——出题与评分必须落在同一个 worker。换成 Redis 后
两个请求可以落到任意进程。**Redis 不可用时两个 quiz 端点返回 503（fail closed）**，
不要加进程内回退：那会让多 worker 部署「有时正确」，比明确失败更难排查。

## 旁注：持久化边界

`learning_progress`（学习进度）不是进程内状态：`PathService` 读写 PostgreSQL，
写入键为 `<course_id>::<kp_id>`，跨重启可见（见 PR #9）。
