"""Unit tests for the profile-service PUT /{user_id} route.

These tests validate the "unified double-Profile" merge logic without a live
database: the repository is mocked so we exercise only the router's contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routers.profiles import router as profile_router


@pytest.fixture
def app() -> FastAPI:
    """Build a minimal FastAPI app with a stubbed repository."""
    app = FastAPI()
    app.include_router(profile_router)
    app.state.profile_repo = MagicMock()
    return app


@pytest.fixture
def client(app: FastAPI):
    return TestClient(app)


def _patch_repo(app: FastAPI, *, stored=None, expected_profile=None) -> AsyncMock:
    repo = AsyncMock()
    repo.get_profile = AsyncMock(return_value=stored)
    repo.upsert_profile = AsyncMock(
        return_value={
            "user_id": "u1",
            "profile": expected_profile or {},
            "version": 1,
            "updated_at": "2026-09-20T10:00:00+00:00",
        }
    )
    app.state.profile_repo = repo
    return repo


class TestPutProfileUnifiedMerge:
    """PUT /{user_id} should merge top-level optional fields into profile_data."""

    def test_put_merges_confidence_scores_into_profile_data(self, client, app):
        repo = _patch_repo(app)
        body = {
            "profile_data": {
                "learning_ability": {"comprehension": 0.8},
                "learning_motivation": {"intrinsic": 0.7},
            },
            "confidence_scores": {
                "learning_ability": 0.92,
                "learning_motivation": 0.5,
            },
        }

        resp = client.put("/api/v1/profiles/u1", json=body)
        assert resp.status_code == 200
        repo.upsert_profile.assert_awaited_once()

        args = repo.upsert_profile.await_args
        user_id, merged = args.args
        assert user_id == "u1"
        # 6 维画像原样保留
        assert merged["learning_ability"] == {"comprehension": 0.8}
        assert merged["learning_motivation"] == {"intrinsic": 0.7}
        # confidence_scores 被合入同一个 dict
        assert merged["confidence_scores"] == {
            "learning_ability": 0.92,
            "learning_motivation": 0.5,
        }

    def test_put_merges_notifications_enabled(self, client, app):
        repo = _patch_repo(app)
        body = {
            "profile_data": {"display_name": "张三"},
            "notifications_enabled": False,
        }

        resp = client.put("/api/v1/profiles/u1", json=body)
        assert resp.status_code == 200

        _, merged = repo.upsert_profile.await_args.args
        assert merged["display_name"] == "张三"
        assert merged["notifications_enabled"] is False

    def test_put_omits_optional_fields_keeps_profile_data_intact(
        self, client, app
    ):
        """未传 confidence_scores / notifications_enabled 时，原 profile_data 不变。"""
        repo = _patch_repo(app)
        body = {
            "profile_data": {
                "learning_ability": {"comprehension": 0.5},
                "notifications_enabled": True,
                "confidence_scores": {"learning_ability": 0.3},
            }
        }

        resp = client.put("/api/v1/profiles/u1", json=body)
        assert resp.status_code == 200

        _, merged = repo.upsert_profile.await_args.args
        assert merged["learning_ability"] == {"comprehension": 0.5}
        assert merged["notifications_enabled"] is True
        assert merged["confidence_scores"] == {"learning_ability": 0.3}

    def test_put_top_level_overrides_profile_data_value(self, client, app):
        """若 client 同时传入 profile_data.confidence_scores 和顶层 confidence_scores，
        顶层字段以"最新写入"语义覆盖——保证 SSE 推过来的总是新的。"""
        repo = _patch_repo(app)
        body = {
            "profile_data": {
                "confidence_scores": {"learning_ability": 0.1},  # 旧值
            },
            "confidence_scores": {"learning_ability": 0.99},  # 新值
        }

        resp = client.put("/api/v1/profiles/u1", json=body)
        assert resp.status_code == 200

        _, merged = repo.upsert_profile.await_args.args
        assert merged["confidence_scores"] == {"learning_ability": 0.99}

    def test_put_does_not_mutate_caller_profile_data(self, client, app):
        """防御性：不能因为合并而污染调用方传入的 dict（避免 zustand 状态被反向写入）。"""
        repo = _patch_repo(app)
        original = {"learning_ability": {"comprehension": 0.5}}
        snapshot = {k: v for k, v in original.items()}

        client.put(
            "/api/v1/profiles/u1",
            json={
                "profile_data": original,
                "confidence_scores": {"learning_ability": 0.8},
            },
        )

        assert original == snapshot  # 未被合入 confidence_scores / notifications_enabled

    def test_put_validation_error_on_missing_profile_data(self, client):
        """profile_data 必填；缺字段时返回 422。"""
        resp = client.put(
            "/api/v1/profiles/u1",
            json={"confidence_scores": {"learning_ability": 0.5}},
        )
        assert resp.status_code == 422

    def test_put_partial_update_preserves_stored_fields(self, client, app):
        """局部 PUT 只覆盖出现的顶层字段，已存画像旧字段必须保留（顶层合并）。

        profile-service 是唯一真理源：多端/局部写入方不能把其他来源写入的
        字段抹掉。
        """
        repo = _patch_repo(
            app,
            stored={
                "user_id": "u1",
                "profile": {
                    "knowledge_level": "beginner",
                    "learning_style": {"preferred": "visual"},
                    "notifications_enabled": True,
                },
                "version": 3,
                "updated_at": "2026-09-19T10:00:00+00:00",
            },
        )

        resp = client.put(
            "/api/v1/profiles/u1",
            json={"profile_data": {"learning_style": {"pace": "slow"}}},
        )
        assert resp.status_code == 200

        _, merged = repo.upsert_profile.await_args.args
        assert merged["knowledge_level"] == "beginner"  # 旧字段保留
        assert merged["notifications_enabled"] is True  # 旧字段保留
        # 新值覆盖；嵌套 dict 整体替换（顶层 key 级"最新写入"语义）
        assert merged["learning_style"] == {"pace": "slow"}