from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.kg.connection import Neo4jPool
from src.kg.router import router as kg_router
from src.utils.llm_adapter import create_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Infrastructure ──────────────────────────────────────────────────
    pool = Neo4jPool(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    app.state.neo4j_pool = pool

    # ── LLM adapter ─────────────────────────────────────────────────────
    llm = create_llm(settings)
    app.state.llm_adapter = llm

    # ── Learning Path service ───────────────────────────────────────────
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.repositories.edge_repo import EdgeRepository
    from src.learning_path.path_service import PathService

    kp_repo = KnowledgePointRepository(pool)
    edge_repo = EdgeRepository(pool)
    path_service = PathService(kp_repo=kp_repo, edge_repo=edge_repo)
    app.state.path_service = path_service

    # ── Configure agents + orchestrator graph ───────────────────────────
    _configure_agents(app, pool, llm)

    yield
    await pool.close()


def _configure_agents(app: FastAPI, pool: Neo4jPool, llm) -> None:
    """Wire agent dependencies and configure the LangGraph."""
    from src.kg.repositories.knowledge_point_repo import KnowledgePointRepository
    from src.kg.repositories.edge_repo import EdgeRepository
    from src.kg.vector_index import VectorIndex
    from src.agents.orchestrator.graph import configure_graph
    from src.agents.planner.agent import PlannerAgent
    from src.agents.guardian.agent import GuardianAgent
    from src.agents.designer.agent import DesignerAgent
    from src.agents.coder.agent import CoderAgent
    from src.agents.content_auditor.agent import ContentAuditorAgent
    from src.agents.assessment.agent import AssessmentAgent

    kp_repo = KnowledgePointRepository(pool)
    edge_repo = EdgeRepository(pool)

    vector_index: VectorIndex | None = None
    try:
        vector_index = VectorIndex(host=settings.chroma_host, port=8000)
    except Exception:
        pass  # ChromaDB may not be available — Content Auditor degrades gracefully

    sandbox_url = "http://sandbox-service:8002"

    configure_graph(
        planner=PlannerAgent(llm_adapter=llm, kp_repo=kp_repo),
        guardian=GuardianAgent(),
        designer=DesignerAgent(llm_adapter=llm),
        coder=CoderAgent(llm_adapter=llm, sandbox_url=sandbox_url),
        content_auditor=ContentAuditorAgent(vector_index=vector_index, llm_adapter=llm),
        assessment=AssessmentAgent(llm_adapter=llm),
    )


app = FastAPI(
    title="EduMap Backend Core",
    version="0.1.0",
    description="Multi-agent personalized learning system - AI/Agent service",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kg_router)

# Register the orchestrator router (lazy-import to avoid circular deps)
from src.agents.orchestrator.router import router as orchestrator_router
app.include_router(orchestrator_router)

# Register the learning path router
from src.learning_path.router import router as learning_path_router
app.include_router(learning_path_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "backend-core", "version": "0.1.0"}


@app.get("/health/ready")
async def readiness():
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
            "mentor": "not_implemented",
        },
    }
