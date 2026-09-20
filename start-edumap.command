#!/usr/bin/env bash
# EduMap 一键启动：在 EduMap 根目录执行 docker compose up -d，
# 然后用 Docker Desktop 内的 compose 项目视图查看运行状态。
set -euo pipefail

cd "$(dirname "$0")"

echo "EduMap 一键启动中..."
docker compose up -d

echo
echo "当前运行中的服务："
docker compose ps

echo
echo "提示：在 Docker Desktop 左栏「edumap」项目下查看 8 个容器状态。"
echo "启动后端开发模式：cd services/backend-core && .venv/bin/python -m src.main"
echo "启动前端开发模式：pnpm --filter web dev"

# 主动弹出 Docker Desktop（如果正在跑）以便用户立即看到
open -a Docker 2>/dev/null || true