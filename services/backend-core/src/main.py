from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.auth import verify_api_key
from src.config import settings
from src.kg.connection import Neo4jPool
from src.kg.router import router as kg_router
from src.privacy.router import router as privacy_router
from src.logging_middleware import SanitizeQueryFilter
from src.prompts import PromptRegistry
from src.utils.llm_adapter import create_llm

# Register sanitizing filter on uvicorn access logs to prevent user
# messages (sent as GET query params) from appearing in plain text.
logging.getLogger("uvicorn.access").addFilter(SanitizeQueryFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Prompt Registry (must load before any agent uses prompts) ────────
    PromptRegistry.load()

    # ── Infrastructure ──────────────────────────────────────────────────
    pool = Neo4jPool(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    app.state.neo4j_pool = pool

    # ── Auto-seed Neo4j with demo data if empty ────────────────────────
    await _auto_seed_neo4j(pool)

    # ── Auto-load ChromaDB embeddings
    await _auto_seed_embeddings(pool)

    # ── LLM adapter ─────────────────────────────────────────────────────
    llm = create_llm(settings)
    app.state.llm_adapter = llm

    # ── Embedding model pre-warming ─────────────────────────────────────
    # Pre-load sentence-transformers at startup so the first RAG request
    # doesn't block on a 300 MB model download.
    _preload_embedding_model(app)

    # ── Memory system ───────────────────────────────────────────────────
    memory_ops = await _init_memory(app)
    app.state.memory_ops = memory_ops

    # (Tool registry is initialized further down, once kp_repo / rag_service /
    #  forgetting_service exist — the tools need real dependencies.)

    # ── Extract DB pool from memory system (used by multiple services) ────
    memory_db_pool_inner = getattr(memory_ops.long_term, '_db', None)
    forgetting_db_pool = (
        memory_db_pool_inner.pool
        if memory_db_pool_inner and hasattr(memory_db_pool_inner, "pool")
        else None
    )

    # ── Learning Path service ───────────────────────────────────────────
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.repositories.edge_repo import EdgeRepository
    from src.learning_path.path_service import PathService

    kp_repo = KnowledgePointRepository(pool)
    edge_repo = EdgeRepository(pool)
    path_service = PathService(kp_repo=kp_repo, edge_repo=edge_repo, db_pool=forgetting_db_pool)
    app.state.path_service = path_service

    # ── Forgetting Curve service ──────────────────────────────────────────
    from src.learning_path.forgetting_curve import ForgettingCurveService
    forgetting_service = ForgettingCurveService(db_pool=forgetting_db_pool)
    app.state.forgetting_service = forgetting_service

    # ── Anti-Gaming service ───────────────────────────────────────────────
    from src.agents.assessment.anti_gaming import AntiGamingService
    anti_gaming_service = AntiGamingService(db_pool=forgetting_db_pool)
    app.state.anti_gaming_service = anti_gaming_service

    # ── Quiz store (server-side answer keys) ──────────────────────────────
    # Holds generated quiz questions so /quiz/grade can key off a server-side
    # record instead of trusting questions echoed back by the client.  Purely
    # in-process (single uvicorn worker), so there is nothing to tear down.
    from src.learning_path.quiz_store import QuizStore
    app.state.quiz_store = QuizStore()

    # ── Resource Repository (PostgreSQL) ──────────────────────────────────
    from src.resources.repository import ResourceRepository
    if forgetting_db_pool:
        resource_repo = ResourceRepository(db_pool=forgetting_db_pool)
        app.state.resource_repo = resource_repo
        logging.info("ResourceRepository initialized with PostgreSQL pool")
    else:
        logging.warning("No DB pool available — ResourceRepository not initialized")

    # ── RAG service + Mentor agent ──────────────────────────────────────
    from src.kg.vector_index import VectorIndex
    from src.rag.rag_service import RAGRetrievalService
    from src.agents.mentor.agent import MentorAgent

    vector_index: VectorIndex | None = None
    try:
        vector_index = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
    except Exception as exc:
        logging.warning("VectorIndex unavailable — RAG/mentor will be degraded: %s", exc)

    rag_service = RAGRetrievalService(
        vector_index=vector_index,
        kp_repo=kp_repo,
        embedding_model=settings.llm_embedding_model,
        preloaded_model=getattr(app.state, "_embedding_model", None),
    )
    rag_service.fusion_method = settings.hybrid_fusion_method
    rag_service.fusion_k = settings.hybrid_rrf_k
    app.state.rag_service = rag_service
    app.state.vector_index = vector_index

    # ── Cross-Encoder Reranker (lazy-loaded on first use) ─────────────
    from src.rag.reranking.cross_encoder import CrossEncoderReranker
    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )
    if settings.reranker_enabled:
        try:
            reranker.load()
        except Exception as exc:
            logging.warning("Reranker model failed to load: %s", exc)
    app.state.reranker = reranker

    # Wire reranker into RAG service if enabled
    if settings.reranker_enabled and reranker.is_loaded:
        rag_service._reranker = reranker
        logging.info("Reranker enabled for RAG retrieval")

    # ── LLM query rewrite (default off — negative result on the eval set) ──
    if settings.rag_rewrite_enabled:
        from src.rag.query_rewrite import QueryRewriter

        rag_service._rewriter = QueryRewriter(
            llm, timeout_seconds=settings.rag_rewrite_timeout
        )
        rag_service._rewrite_enabled = True
        logging.info("LLM query rewrite enabled for RAG retrieval")

    # ── Retrieval result cache ─────────────────────────────────────────────
    if settings.rag_cache_enabled:
        rag_service.enable_result_cache(
            maxsize=settings.rag_cache_size,
            ttl_seconds=settings.rag_cache_ttl_seconds,
        )
        logging.info(
            "RAG result cache enabled (size=%d, ttl=%ds)",
            settings.rag_cache_size,
            settings.rag_cache_ttl_seconds,
        )

    mentor_agent = MentorAgent(llm_adapter=llm, rag_service=rag_service)
    app.state.mentor_agent = mentor_agent

    # ── Tool registry ─────────────────────────────────────────────────────
    # Built here (not at the top of the lifespan) because the tools need the
    # real KP repo / RAG service / forgetting service as dependencies.
    tool_registry = _init_tools(
        kp_repo=kp_repo,
        rag_service=rag_service,
        forgetting_service=forgetting_service,
    )
    app.state.tool_registry = tool_registry

    # ── Durable graph state (I-6) ───────────────────────────────────────
    # Started here rather than inside _configure_agents because pool creation
    # is async while that function is not. A failure is non-fatal: the graph
    # then compiles without a checkpointer and progress reporting continues via
    # the Redis snapshot.
    from src.agents.orchestrator.checkpointing import CheckpointerHolder

    checkpointer_holder = CheckpointerHolder(settings.database_url)
    await checkpointer_holder.start()
    app.state.checkpointer_holder = checkpointer_holder

    # ── Configure agents + orchestrator graph ───────────────────────────
    _configure_agents(app, pool, llm, checkpointer=checkpointer_holder.saver)

    yield
    # Close the checkpointer before the pools it depends on: its psycopg pool
    # is separate from the asyncpg ones, but tearing it down first keeps the
    # ordering obvious and avoids closing a database that is still in use.
    await checkpointer_holder.stop()
    await pool.close()
    if memory_ops is not None:
        from src.memory.db import MemoryDBPool
        db_pool = getattr(memory_ops.long_term, '_db', None)
        if db_pool and isinstance(db_pool, MemoryDBPool):
            await db_pool.close()


def _configure_agents(app: FastAPI, pool: Neo4jPool, llm, checkpointer=None) -> None:
    """Wire agent dependencies and configure the LangGraph."""
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.vector_index import VectorIndex
    from src.agents.orchestrator.graph import configure_graph
    from src.agents.planner.agent import PlannerAgent
    from src.agents.guardian.agent import GuardianAgent
    from src.agents.designer.agent import DesignerAgent
    from src.agents.coder.agent import CoderAgent
    from src.agents.content_auditor.agent import ContentAuditorAgent
    from src.agents.assessment.agent import AssessmentAgent

    kp_repo = KnowledgePointRepository(pool)

    # Inject harness services (tool_registry + memory_ops) into agents
    tool_registry = getattr(app.state, "tool_registry", None)
    memory_ops = getattr(app.state, "memory_ops", None)
    harness_kwargs = {"tool_registry": tool_registry, "memory_ops": memory_ops}

    # Use existing vector_index from lifespan (don't recreate and overwrite)
    existing_vi = getattr(app.state, "vector_index", None)
    content_auditor_vi = existing_vi
    if existing_vi is None:
        try:
            content_auditor_vi = VectorIndex(host=settings.chroma_host, port=settings.chroma_port)
        except Exception as exc:
            logging.warning("VectorIndex (alt) unavailable — Content Auditor degrades gracefully: %s", exc)
    else:
        logging.info("VectorIndex reused from lifespan")

    sandbox_url = settings.sandbox_url

    configure_graph(
        planner=PlannerAgent(llm_adapter=llm, kp_repo=kp_repo, **harness_kwargs),
        guardian=GuardianAgent(),
        designer=DesignerAgent(llm_adapter=llm, **harness_kwargs),
        coder=CoderAgent(llm_adapter=llm, sandbox_url=sandbox_url, **harness_kwargs),
        content_auditor=ContentAuditorAgent(
            vector_index=content_auditor_vi,
            llm_adapter=llm,
            embedding_model_name=settings.llm_embedding_model,
            **harness_kwargs,
        ),
        assessment=AssessmentAgent(llm_adapter=llm, **harness_kwargs),
        checkpointer=checkpointer,
    )

    # Also update the MentorAgent on app.state with harness services
    mentor_agent = getattr(app.state, "mentor_agent", None)
    if mentor_agent and tool_registry:
        mentor_agent._tool_registry = tool_registry
        logging.info("ToolRegistry injected into MentorAgent")
    if mentor_agent and memory_ops:
        mentor_agent._memory_ops = memory_ops
        logging.info("MemoryOps injected into MentorAgent")

    # Do NOT overwrite app.state.vector_index — it was set in lifespan


async def _init_memory(app: FastAPI):
    """Initialize the three-tier memory system and attach to app state."""
    from src.memory.db import MemoryDBPool
    from src.memory.short_term import ShortTermMemory
    from src.memory.long_term import LongTermMemory
    from src.memory.operations import MemoryOperations

    # PostgreSQL long-term memory
    db_pool = MemoryDBPool(dsn=settings.database_url)
    await db_pool.create()
    long_term = LongTermMemory(db_pool)

    # Redis short-term memory
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        await redis_client.ping()
        app.state.redis_client = redis_client
        short_term = ShortTermMemory(redis_client)
        logging.info("Redis connected for short-term memory at %s", settings.redis_url)
    except Exception as exc:
        logging.warning("Redis unavailable — short-term memory will use in-memory fallback: %s", exc)
        short_term = ShortTermMemory._in_memory_fallback()

    # Encoder for episodic-memory embeddings. Uses the model preloaded into
    # app.state by _preload_embedding_model, so no second model instance is
    # held in memory. If preloading failed the memory system still works —
    # recall just falls back to recency + importance (see MemoryOperations._embed).
    def _encode_for_memory(text: str) -> list[float]:
        model = getattr(app.state, "_embedding_model", None)
        if model is None:
            return []
        return model.encode(text, normalize_embeddings=True).tolist()

    memory_ops = MemoryOperations(
        short_term=short_term,
        long_term=long_term,
        embed_fn=_encode_for_memory,
    )
    app.state.memory_ops = memory_ops
    logging.info("Memory system initialized")
    return memory_ops


def _init_tools(kp_repo=None, rag_service=None, forgetting_service=None):
    """Initialize and register built-in tools.

    Tools are constructed with their real dependencies — constructing them
    bare (``KGTool()``) left every tool returning "dependency unavailable",
    so a tool call could never produce a useful result.
    """
    from src.tools.registry import ToolRegistry
    registry = ToolRegistry()

    # Register built-in tools
    try:
        from src.tools.builtin.forgetting_check import ForgettingCheckTool
        from src.tools.builtin.kg_search import KGTool
        from src.tools.builtin.resource_search import ResourceSearchTool

        registry.register(KGTool(kp_repo=kp_repo))
        registry.register(ResourceSearchTool(rag_service=rag_service))
        registry.register(ForgettingCheckTool(forgetting_service=forgetting_service))
        logging.info(
            "Tool registry initialized with %d tools: %s",
            len(registry.list_tools()),
            registry.list_tools(),
        )
    except Exception as exc:
        logging.warning("Failed to register built-in tools: %s", exc)

    return registry


async def _auto_seed_neo4j(pool) -> None:
    """Auto-load demo course seed data if the Neo4j graph is empty.

    Reads ``scripts/db/seed/neo4j-seed.cypher`` (copied into the Docker image)
    and executes it against Neo4j if no ``Course`` nodes exist yet.
    """
    try:
        from src.kg.seed_loader import SeedLoader

        loader = SeedLoader(pool)
        verify = await loader.verify_seed()
        if verify.get("course_count", 0) > 0:
            logging.info(
                "Neo4j already seeded (%d courses, %d nodes) — skipping",
                verify.get("course_count", 0),
                verify.get("node_count", 0),
            )
            return

        logging.info("Neo4j graph is empty — loading demo course seed data...")
        result = await loader.load_seed_data()
        logging.info(
            "Auto-seed complete: %d nodes, %d edges",
            result.nodes_created,
            result.edges_created,
        )
    except FileNotFoundError:
        logging.warning("Seed file not found — skipping auto-seed (dev mode without scripts/)")
    except Exception as exc:
        logging.warning("Auto-seed failed (graph may be intentionally empty): %s", exc)


async def _auto_seed_embeddings(pool) -> None:
    """Auto-load ChromaDB embeddings for seeded KPs if empty."""
    try:
        from src.kg.seed_loader import SeedLoader

        loader = SeedLoader(pool)
        result = await loader.load_embeddings()
        if result.get("status") == "completed":
            logging.info(
                "Auto-embeddings: %d KPs upserted to ChromaDB (collection size=%d)",
                result.get("count", 0),
                result.get("collection_size", 0),
            )
        elif result.get("status") == "skipped":
            logging.debug("Auto-embeddings skipped: %s", result.get("reason"))
    except Exception as exc:
        logging.warning("Auto-embeddings failed (non-fatal): %s", exc)


def _preload_embedding_model(app: FastAPI) -> None:
    """Pre-download and cache the sentence-transformers embedding model at startup.

    Without this step the first RAG request or document upload blocks on a
    300 MB model download, which can take 30+ seconds and may fail silently.
    """
    model_name = settings.llm_embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        app.state._embedding_model = model
        logging.info("Embedding model pre-loaded: %s (%d MB)", model_name, 300)
    except Exception as exc:
        logging.warning(
            "Embedding model pre-load failed — will lazy-load at first use: %s",
            exc,
        )
        app.state._embedding_model = None


app = FastAPI(
    title="EduMap Backend Core",
    version="0.1.0",
    description="Multi-agent personalized learning system - AI/Agent service",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)] if settings.api_key else [],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Set via CORS_ORIGINS env var; empty means follow allow_origin_regex
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kg_router)

# Register the orchestrator router (lazy-import to avoid circular deps)
from src.agents.orchestrator.router import (  # noqa: E402
    router as orchestrator_router,
)
app.include_router(orchestrator_router)

# Register the learning path router
from src.learning_path.router import router as learning_path_router  # noqa: E402  (deferred: avoids circular imports)
app.include_router(learning_path_router)

# Register the mentor / RAG router
from src.rag.router import router as mentor_router  # noqa: E402  (deferred: avoids circular imports)
app.include_router(mentor_router)

# Register the resource management router
from src.resources.router import router as resource_router  # noqa: E402  (deferred: avoids circular imports)
app.include_router(resource_router)

# Register the unified analysis / profile SSE router
from src.analysis.router import router as analysis_router  # noqa: E402  (deferred: avoids circular imports)
app.include_router(analysis_router)

# Register the debug / simulation router (non-production only)
from src.debug.simulate_growth import router as debug_router  # noqa: E402  (deferred: avoids circular imports)
app.include_router(debug_router)

# Register the privacy (PIPL compliance) router
app.include_router(privacy_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend-core", "version": "0.1.0"}


@app.get("/health/ready")
async def readiness():
    embedding_model = getattr(app.state, "_embedding_model", None)
    llm = getattr(app.state, "llm_adapter", None)
    llm_health = None
    if llm and hasattr(llm, "health_check"):
        try:
            llm_health = await llm.health_check()
        except Exception as exc:
            llm_health = {"reachable": False, "detail": str(exc)}

    # Circuit-breaker state.  The breaker is shared by every agent + Mentor, so
    # without this the only signal that LLM traffic is suspended is a log line —
    # an operator cannot tell "degraded because the breaker is open" from
    # "degraded because the upstream is slow", nor which agent tripped it.
    circuit_breaker = None
    if llm and hasattr(llm, "circuit_breaker_state"):
        try:
            circuit_breaker = llm.circuit_breaker_state()
        except Exception as exc:
            circuit_breaker = {"error": str(exc)}

    return {
        "status": "ok",
        "service": "backend-core",
        "version": "0.1.0",
        "components": {
            "orchestrator": "ready",
            "planner": "ready",
            "guardian": "ready",
            "designer": "ready",
            "coder": "ready",
            "assessment": "ready",
            "content_auditor": "ready",
            "mentor": "ready",
        },
        "embedding_model": {
            "loaded": embedding_model is not None,
            "name": settings.llm_embedding_model,
        },
        "llm": llm_health,
        "circuit_breaker": circuit_breaker,
    }
