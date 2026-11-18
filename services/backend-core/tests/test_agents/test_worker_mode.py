"""Worker-mode guard — ``src/agents/orchestrator/worker_mode.py``, real entry points.

``guard_run`` 包在真实的 ``uvicorn.run`` 外面（用假 run 记录是否被调用）；
启动检查 ``verify_single_worker`` 按 lifespan 的调用方式（``await``）执行；
部署文件用真实读取解析。
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.orchestrator import graph as graph_module
from src.agents.orchestrator import worker_mode

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class TestGuardRun:
    """``guard_run``：多 worker 在任何子进程启动前即被拒绝。"""

    def test_four_workers_are_refused_before_uvicorn_is_called(self):
        calls: list[tuple] = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))

        with pytest.raises(RuntimeError, match="multiple uvicorn workers"):
            worker_mode.guard_run(fake_run, "src.main:app", workers=4)
        assert calls == [], "uvicorn.run must not run when it was refused"

    @pytest.mark.parametrize("workers", [1, None])
    def test_one_worker_passes_through_unchanged(self, workers):
        calls: list[dict] = []

        def fake_run(*args, **kwargs):
            calls.append(kwargs)
            return "served"

        assert worker_mode.guard_run(fake_run, "src.main:app", workers=workers) == "served"
        assert calls == [{"workers": workers}]

    def test_omitted_workers_passes_through(self):
        """没传 workers 就是 uvicorn 的默认单进程行为，不拒绝、不改参数。"""
        assert worker_mode.guard_run(lambda *a, **kw: kw, "src.main:app") == {}


class TestStartupCheck:
    """``verify_single_worker`` —— 在 lifespan 末尾调用（``await``）。"""

    def test_unconfigured_graph_refuses_to_serve(self, monkeypatch):
        """没接 agent 的进程每个生成请求都会失败——启动时就拒绝。

        走的是真实 lifespan 的调用形态（``await verify_single_worker()``），
        lifespan 抛出的异常会让 uvicorn 启动失败，而不是带上坏图开始服务。
        """
        monkeypatch.setattr(graph_module, "is_configured", lambda: False)
        with pytest.raises(RuntimeError, match="not configured"):
            asyncio.run(worker_mode.verify_single_worker())

    def test_configured_graph_passes(self, monkeypatch):
        monkeypatch.setattr(graph_module, "is_configured", lambda: True)
        asyncio.run(worker_mode.verify_single_worker())

    def test_multi_worker_config_is_refused(self, monkeypatch):
        monkeypatch.setattr(graph_module, "is_configured", lambda: True)
        server = SimpleNamespace(config=SimpleNamespace(workers=3))
        with pytest.raises(RuntimeError, match="multiple uvicorn workers"):
            asyncio.run(worker_mode.verify_single_worker(server))

    def test_declared_workers_treats_unknown_as_single(self):
        assert worker_mode.declared_workers(SimpleNamespace(config=SimpleNamespace(workers=3))) == 3
        assert worker_mode.declared_workers(SimpleNamespace(config=SimpleNamespace(workers=None))) is None
        assert worker_mode.declared_workers(None) is None


class TestDeploymentFiles:
    """Dockerfile / docs 的单 worker 约定与 runtime 规则一致。"""

    _MULTI = re.compile(r"--workers\s+([2-9]|\d{2,})")

    def _files(self):
        return [BACKEND_ROOT / "Dockerfile", BACKEND_ROOT / "docs" / "backend-core.md"]

    def test_both_files_carry_the_sentinel(self):
        for path in self._files():
            assert "edumap:single-worker" in path.read_text(encoding="utf-8"), path

    def test_neither_file_launches_multiple_workers(self):
        for path in self._files():
            # Prose is not a launch instruction: lines that merely *point at*
            # an example's label cannot execute anything.
            text = "\n".join(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if not line.strip().startswith((">", "#"))
            )
            assert not self._MULTI.search(text), f"{path} asks for >1 worker"

    def test_dockerfile_cmd_is_the_single_worker_form(self):
        text = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")
        cmd = next(line for line in text.splitlines() if line.startswith("CMD "))
        assert '"uvicorn"' in cmd and '"src.main:app"' in cmd
        assert "--workers" not in cmd
