from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="EduMap Backend Core",
    version="0.1.0",
    description="Multi-agent personalized learning system - AI/Agent service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
