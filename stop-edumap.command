#!/usr/bin/env bash
# EduMap 一键停止：停掉所有 edumap compose 容器（保留数据卷）。
set -euo pipefail

cd "$(dirname "$0")"
echo "EduMap 一键停止中..."
docker compose stop
echo "已停止。再次启动请双击 start-edumap.command。"