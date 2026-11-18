"""``main.lifespan`` 的 quiz-store 装配 —— 读真实源码，按生产导入方式导入。

单独成文件，原因有二：

1. 它约束的是 ``main.lifespan`` 的装配形状，不是 ``QuizStore`` 的实现；
2. 最初它写在 ``test_quiz_store.py`` 里，用的是 ``import main``——本机后端目录恰好在
   ``sys.path`` 上所以通过，CI 只把仓库根加进 ``sys.path``，于是
   ``ModuleNotFoundError: No module named 'main'``。这正是 CLAUDE.md 说的
   「测试的调用方式与生产调用方式不一致」。本仓其它测试统一用 ``from src import main``。
"""

from __future__ import annotations

import inspect

from src import main as main_module


class TestQuizStoreWiring:
    """Redis 是 quiz 答案密钥的唯一后端；lifespan 必须在 ping 成功后才建 store。"""

    def test_store_is_created_only_from_a_live_redis_client(self) -> None:
        src = inspect.getsource(main_module.lifespan)
        assert 'redis_client = getattr(app.state, "redis_client", None)' in src, (
            "lifespan must derive the store's backend from the Redis client the memory "
            "layer already pinged successfully"
        )
        assert "QuizStore(redis_client)" in src
        assert "QuizStore()" not in src, (
            "an argument-less QuizStore would be process-local again — the store exists "
            "precisely so generate and grade can land on different workers"
        )
