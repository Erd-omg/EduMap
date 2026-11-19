"""CORS 默认允许局域网浏览器来源 —— 读**真实 app 的中间件配置**。

局域网部署时浏览器打开的是 ``http://192.168.x.y:3000``，Origin 就是它。
原来的正则只认 localhost/127.0.0.1，浏览器会因 CORS 拒掉所有请求，
表现为“后端挂了”。这里不另建 middleware，而是从 ``main.app`` 已装配的
CORS 中间件读 ``allow_origin_regex`` 并逐个匹配——与生产用的是同一个值。
"""

from __future__ import annotations

import re

import pytest

from src import main as main_module


def _origin_regex() -> str:
    for m in main_module.app.user_middleware:
        kwargs = getattr(m, "kwargs", None) or {}
        if "allow_origin_regex" in kwargs:
            return kwargs["allow_origin_regex"]
    raise AssertionError("CORS middleware with allow_origin_regex not found")


@pytest.mark.parametrize("origin", [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.20:3000",
    "http://10.0.0.5:3000",
    "http://172.16.4.9:3000",
    "http://edumap.local:3000",
])
def test_lan_origins_are_allowed(origin: str) -> None:
    assert re.match(_origin_regex(), origin), origin


@pytest.mark.parametrize("origin", [
    "https://evil.example.com",
    "http://8.8.8.8:3000",
    "http://172.32.0.1:3000",   # just outside the private 172.16/12 block
])
def test_public_origins_are_not_allowed_by_default(origin: str) -> None:
    assert not re.match(_origin_regex(), origin), origin


def test_cors_origin_regex_setting_is_appended(monkeypatch) -> None:
    """``CORS_ORIGIN_REGEX`` extends (never replaces) the defaults.

    Re-runs ``main``'s middleware construction through a fresh ``Settings``;
    the module-level app keeps its startup-time value in the test process.
    """
    import importlib

    from src import config as config_module

    monkeypatch.setenv("CORS_ORIGIN_REGEX", r"^https://edu\.example\.com$")
    importlib.reload(config_module)
    try:
        fresh = importlib.reload(main_module)
        regex = next(
            m.kwargs["allow_origin_regex"]
            for m in fresh.app.user_middleware
            if "allow_origin_regex" in getattr(m, "kwargs", {})
        )
        assert re.match(regex, "https://edu.example.com")
        assert re.match(regex, "http://192.168.1.20:3000"), "defaults must survive"
    finally:
        monkeypatch.delenv("CORS_ORIGIN_REGEX")
        importlib.reload(config_module)
        importlib.reload(main_module)
