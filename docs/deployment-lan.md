# EduMap 部署手册（内网 / 局域网先行）

> 适用范围：把系统部署到一台局域网服务器，真实用户用浏览器访问试用。
> **当前不包含账号体系**——每个人在浏览器里的身份是一个本地生成的 UUID，
> 见下文「试用边界」。公开到互联网之前请先读最后一节。

## 0. 试用前必须知道的三件事

| 事实 | 影响 |
|---|---|
| **身份 = 浏览器 localStorage 里的 UUID**（`apps/web/src/lib/user-id.ts`） | 换浏览器 / 清缓存 = 新身份，旧数据仍留在库里但不显示；**同一台共享电脑上的不同人无法互相隔离** |
| **后端信任请求里的 `user_id`**（编排、进度、画像都是 body/query 参数） | 任何人改请求即可读写他人的数据；局域网试用可接受，公网不可 |
| 无登录、无权限 | 只发给熟人，URL 不外传 |

## 1. 环境变量（`.env`，放在仓库根目录，与 `docker-compose.yml` 同级）

```dotenv
# ── LLM（真实调用）──────────────────────────────────────────────
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-...
LLM_API_BASE=https://api.deepseek.com
# 可选：单次调用超时（秒）与重试次数，默认 120 / 3
# LLM_TIMEOUT=120
# LLM_MAX_RETRIES=3

# ── 访问控制 ────────────────────────────────────────────────────
# 留空 = 完全不鉴权（局域网试用 OK）。设置后所有请求须带 Authorization: Bearer
# 或 ?api_key=。注意：浏览器前端目前**不会**自动附带它——设了之后页面会 401。
API_KEY=

# ── CORS ────────────────────────────────────────────────────────
# 默认已允许：localhost / 127.0.0.1 / 10.x / 192.168.x / 172.16-31.x / *.local
# 只有使用自定义域名（如 https://edu.example.com）时才需要补充：
# CORS_ORIGIN_REGEX=^https://edu\.example\.com$

# ── 前端定位后端 ─────────────────────────────────────────────────
# 默认：浏览器用“打开页面时的主机名 + 8000 端口”访问后端。
# 后端不在同一台机器时才需要设置（构建期生效）：
# NEXT_PUBLIC_API_BASE_URL=http://192.168.1.10:8000
# 后端端口若不是 8000：
# BACKEND_PORT=8000

# ── 数据库口令（dev 默认值是 edumap_dev，内网也应改掉）────────────
POSTGRES_PASSWORD=请改成强口令
```

> `docker-compose.yml` 里的 Postgres / Neo4j 口令目前硬编码为 `edumap_dev`
> （第 18-20 行、第 36-37 行）。**内网服务器上也应修改**，且要同步改三个
> 消费者的连接串（backend-core、profile-service、neo4j 客户端）。

## 2. 启动与验证

```bash
docker compose up -d --build
docker compose ps                      # 期望全部 healthy
curl -s http://localhost:8000/health/ready | python -m json.tool   # 8 个 agent ready
```

验证浏览器访问（**在局域网另一台机器上**）：

1. 打开 `http://<服务器IP>:3000`（不要用容器内部的 localhost 判断）
2. DevTools → Network → 确认请求发往 `http://<服务器IP>:8000`，而不是 `localhost:8000`
3. 依次点一遍：`/chat`（画像 + Mentor）→ `/generate`（编排 SSE）→ `/learn/cs201`（路径 + 出题评分）
4. 出题后评分一次，刷新页面，确认学习进度仍在（进度存 PostgreSQL）

## 3. 已知的部署相关限制

| 限制 | 原因 | 处理 |
|---|---|---|
| **backend-core 只能单 worker** | SSE 事件队列与取消句柄在进程内 | 启动检查会拒绝 >1 worker；纵向扩容（更多 CPU/内存）可用 |
| `docs/backend-core.md` 里说明了裸 CLI `--workers N` 防不住 | spawn 出的子进程看不到该参数 | 必须走 Dockerfile 或 `guard_run` |
| **profile-service 地址在浏览器侧** | 画像请求直发 `:8001` | 需开放 8001 端口；若只想暴露 3000，需要下一步做反向代理 |
| 无 HTTPS | 内网先行 | 需要域名/证书时再加 nginx/caddy + `CORS_ORIGIN_REGEX` |
| 无备份流程 | — | 见下文 |
| `web` 容器以 `pnpm dev`（开发模式）运行 | compose 里写死 | 内网试用可用；公网前改成 `prod` 阶段 `node server.js` |

## 4. 备份（每天一次即可，数据在四个命名卷里）

```bash
# 建议放到服务器 crontab，03:17 执行
docker exec edumap-postgres pg_dump -U edumap edumap | gzip > /backup/pg-$(date +%F).sql.gz
docker exec edumap-neo4j neo4j-admin database dump neo4j --to-path=/backups   # 输出在容器卷内，需再拷出
docker run --rm -v edumap_redis-data:/data -v /backup:/backup alpine \
  tar czf /backup/redis-$(date +%F).tgz -C /data .
docker run --rm -v edumap_chroma-data:/data -v /backup:/backup alpine \
  tar czf /backup/chroma-$(date +%F).tgz -C /data .
```

Chroma 与 Neo4j 的索引可以重建（`scripts/db/seed*`、`scripts/seed_embeddings.py`），
**Postgres 是唯一不能再生的数据**（进度、画像、记忆、quiz 结果）。

## 5. LLM 配额与成本

每次“生成”会触发多个 agent（Planner → Guardian → Designer ∥ Coder → Content Auditor
→ Assessment），一次完整生成通常是 **10+ 次 LLM 调用**。上线前至少确认：

- 供应商的并发/每分钟限额与你的预期同时在线人数
- 熔断器在三态间切换时，前端能否看到“稍后重试”（`/health/ready` 暴露
  `circuit_breaker.state`）
- 单用户单位时间内的生成次数上限（目前**没有**任何限流）

**没有 API key 时不会报错**：`create_llm` 总会返回一个适配器，缺 key 的调用会以请求失败的形式出现，再由熔断器累计。上线第一天先跑一次完整生成，确认 `/health/ready` 的 `llm.reachable` 为 `true`。

## 6. 公开到互联网之前必须完成

1. **账号体系**：替换 localStorage UUID；后端从 token 推导 `user_id`，
   不接受客户端传入
2. **反向代理 + HTTPS**（只暴露 3000，`/api` 转发到 8000，画像转发到 8001）
3. 设置 `API_KEY`，并让前端在请求里带上它（或改用会话 cookie）
4. 关闭调试模拟端点（`main.py` 第 482-484 行无条件挂载 `POST /simulate-growth`，
   注释写着 "non-production only" 但代码里没有环境判断）
5. 给生成请求加限流；LLM 费用上限告警
6. `web` 容器切到生产模式
