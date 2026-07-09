from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.kg.connection import Neo4jPool
from src.kg.router import router as kg_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = Neo4jPool(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
    )
    app.state.neo4j_pool = pool
    yield
    await pool.close()


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
            "orchestrator": "not_implemented",
            "planner": "not_implemented",
            "guardian": "not_implemented",
            "designer": "not_implemented",
            "coder": "not_implemented",
            "assessment": "not_implemented",
            "content_auditor": "not_implemented",
            "mentor": "not_implemented",
        },
    }
