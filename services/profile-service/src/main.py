from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.auth import verify_api_key
from src.config import settings
from src.db.database import DatabasePool
from src.db.repository import ProfileRepository
from src.routers.analysis import router as analysis_router
from src.routers.profiles import router as profile_router
from src.services.llm_analyzer import ProfileAnalyzer


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = DatabasePool(settings.database_url)
    await db.connect()
    app.state.db_pool = db
    app.state.profile_repo = ProfileRepository(db)
    app.state.profile_analyzer = ProfileAnalyzer(
        api_key=settings.llm_api_key,
        api_base=settings.llm_api_base,
        model=settings.llm_model,
    )
    yield
    await db.close()


app = FastAPI(
    lifespan=lifespan,
    title="EduMap Profile Service",
    version="0.1.0",
    description="User profilling service - data isolation layer",
    dependencies=[Depends(verify_api_key)] if settings.api_key else [],
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(profile_router)
app.include_router(analysis_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "profile-service", "version": "0.1.0"}


@app.get("/health/ready")
async def readiness(request: Request):
    db_ok = "connected" if request.app.state.db_pool._pool else "not_connected"
    return {
        "status": "ok",
        "service": "profile-service",
        "version": "0.1.0",
        "components": {"database": db_ok},
    }
