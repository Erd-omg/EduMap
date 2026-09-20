"""Fixtures for API integration tests.

Creates a minimal FastAPI TestClient with mocked dependencies,
avoiding the heavy model-loading in the production lifespan.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.prompts import PromptRegistry


@pytest.fixture(scope="session", autouse=True)
def _load_prompts():
    """Load PromptRegistry once per test session (required by all agents)."""
    PromptRegistry.load()


@pytest.fixture
def mock_llm():
    """Mock LLM adapter."""
    from tests.mocks.llm_adapter import MockLLMAdapter
    return MockLLMAdapter(response="模拟回答。来源[1]显示数组插入复杂度为O(n)。")


@pytest.fixture
def mock_neo4j_pool():
    """Mock Neo4jPool."""
    pool = AsyncMock()
    pool.execute_read = AsyncMock(return_value=[])
    pool.execute_write = AsyncMock(return_value=[])
    return pool


@pytest.fixture
def mock_rag_service():
    """Mock RAGRetrievalService."""
    from src.rag.models import RAGContext

    svc = AsyncMock()
    svc.search = AsyncMock(return_value=[])
    svc.assemble_context = AsyncMock(return_value=RAGContext(
        text="模拟上下文内容。数组插入复杂度为O(n)。",
        sources=[],
        total_tokens=20,
    ))
    return svc


def _make_memory_ops_mock():
    """Create a mock MemoryOperations that simulates realistic behavior.

    - short_term.get_session() returns None by default (no existing session)
    - Other methods return empty AsyncMock results
    """
    from unittest.mock import AsyncMock

    short_term = AsyncMock()
    short_term.get_session = AsyncMock(return_value=None)
    short_term.create_session = AsyncMock()
    short_term.update_metadata = AsyncMock()

    memory_ops = AsyncMock()
    memory_ops.short_term = short_term
    return memory_ops


def _make_resource_repo_mock():
    """Create a mock ResourceRepository with in-memory storage.

    Supports create(), get(), update_parse_status(), list(), delete(),
    and sync_generated() — all backed by a local dict so the upload
    endpoint can persist and retrieve metadata through its normal flow.
    """
    from unittest.mock import AsyncMock
    from src.resources.models import ResourceMetadata

    _store: dict[str, ResourceMetadata] = {}

    async def _create(metadata: ResourceMetadata) -> ResourceMetadata:
        _store[metadata.id] = metadata
        return metadata

    async def _get(resource_id: str) -> ResourceMetadata | None:
        return _store.get(resource_id)

    async def _update_parse_status(
        resource_id: str,
        status: str,
        stats: dict | None = None,
    ) -> None:
        meta = _store.get(resource_id)
        if meta:
            _store[resource_id] = ResourceMetadata(
                **{**meta.model_dump(), "parse_status": status, "parse_stats": stats}
            )

    async def _list(**kwargs):
        return list(_store.values()), len(_store)

    async def _delete(resource_id: str) -> ResourceMetadata | None:
        return _store.pop(resource_id, None)

    async def _sync_batch(resources: list[ResourceMetadata]) -> int:
        count = 0
        for r in resources:
            _store[r.id] = r
            count += 1
        return count

    repo = AsyncMock()
    repo.create = AsyncMock(side_effect=_create)
    repo.get = AsyncMock(side_effect=_get)
    repo.update_parse_status = AsyncMock(side_effect=_update_parse_status)
    repo.list = AsyncMock(side_effect=_list)
    repo.delete = AsyncMock(side_effect=_delete)
    repo.sync_batch = AsyncMock(side_effect=_sync_batch)
    repo.__bool__ = lambda self: True
    return repo


def _create_test_app(**overrides) -> FastAPI:
    """Create a minimal FastAPI app for testing with mocked state.

    Registers all routers but skips the heavyweight lifespan.
    """
    app = FastAPI(title="EduMap Test")

    # Register routers (same as production main.py)
    from src.kg.router import router as kg_router
    from src.agents.orchestrator.router import router as orchestrator_router
    from src.learning_path.router import router as learning_path_router
    from src.rag.router import router as mentor_router
    from src.resources.router import router as resource_router
    from src.analysis.router import router as analysis_router
    from src.debug.simulate_growth import router as debug_router
    from src.privacy.router import router as privacy_router

    app.include_router(kg_router)
    app.include_router(orchestrator_router)
    app.include_router(learning_path_router)
    app.include_router(mentor_router)
    app.include_router(resource_router)
    app.include_router(analysis_router)
    app.include_router(debug_router)
    app.include_router(privacy_router)

    # Health endpoints (same as main.py)
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "backend-core", "version": "0.1.0"}

    @app.get("/health/ready")
    async def ready():
        return {"status": "ok", "embedding_model": "mocked"}

    # Set default mock state
    from tests.mocks.llm_adapter import MockLLMAdapter
    # Proper return values for services that need them
    from src.learning_path.models import PersonalizedPath
    from datetime import datetime, timezone

    from src.learning_path.forgetting_curve import ForgettingState

    now_iso = datetime.now(timezone.utc).isoformat()

    path_service = AsyncMock()
    path_service.get_personalized_path = AsyncMock(return_value=PersonalizedPath(
        course_id="cs101", user_id="test-user",
        nodes=[], total_count=0, mastered_count=0, completed_count=0,
        progress_percent=0.0, created_at=now_iso,
    ))
    path_service.record_progress = AsyncMock(return_value=PersonalizedPath(
        course_id="cs101", user_id="test-user",
        nodes=[], total_count=0, mastered_count=0, completed_count=0,
        progress_percent=0.0, created_at=now_iso,
    ))
    path_service.get_next_recommendation = AsyncMock(return_value=None)
    path_service.get_content_type_suggestion = MagicMock(
        return_value=MagicMock(kp_id="kp-test", content_types=["explanation"])
    )
    path_service.save_recommendation = AsyncMock()
    path_service.get_recommendation_history = AsyncMock(return_value=[])

    forgetting_service = AsyncMock()
    forgetting_service.predict_recall = AsyncMock(return_value=0.5)
    forgetting_service.get_state = AsyncMock(return_value=ForgettingState(
        kp_id="kp-test", user_id="test-user",
    ))
    forgetting_service.get_alerts = AsyncMock(return_value=[])
    forgetting_service.get_all_states = AsyncMock(return_value=[])
    forgetting_service.get_recall_map = AsyncMock(return_value={})
    forgetting_service.update_after_quiz = AsyncMock(return_value=ForgettingState(
        kp_id="kp-test", user_id="test-user",
    ))

    anti_gaming_service = AsyncMock()
    # 必须与 AntiGamingService.calculate_weighted_score 的真实返回结构一致，
    # 否则 progress 端点会在取 weighted_score 时 KeyError 并静默降级。
    anti_gaming_service.calculate_weighted_score = AsyncMock(return_value={
        "raw_score": 0.9,
        "weighted_score": 0.75,
        "factors": {
            "mastery_factor": 0.95,
            "cram_factor": 0.8,
            "difficulty_factor": 1.0,
        },
        "is_cramming": False,
        "details": {},
    })
    anti_gaming_service.get_state = AsyncMock(return_value={
        "mastery": 0.7, "cramming_score": 0.0,
    })

    # Mock for ResourceRepository (in-memory storage)
    resource_repo = _make_resource_repo_mock()

    defaults = {
        "neo4j_pool": AsyncMock(),
        "llm_adapter": MockLLMAdapter(response="测试回复"),
        "memory_ops": _make_memory_ops_mock(),
        "rag_service": AsyncMock(),
        "vector_index": AsyncMock(),
        "path_service": path_service,
        "forgetting_service": forgetting_service,
        "anti_gaming_service": anti_gaming_service,
        "kp_repo": AsyncMock(),
        "edge_repo": AsyncMock(),
        "mentor_agent": AsyncMock(),
        "tool_registry": AsyncMock(),
        "reranker": AsyncMock(),
        "resource_repo": resource_repo,
    }
    defaults.update(overrides)

    for key, val in defaults.items():
        setattr(app.state, key, val)

    return app


@pytest.fixture
def client():
    """FastAPI TestClient with mocked dependencies."""
    app = _create_test_app()
    with TestClient(app) as c:
        yield c
