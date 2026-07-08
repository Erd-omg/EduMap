from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="EduMap Sandbox Service",
    version="0.1.0",
    description="Secure code execution sandbox with Docker isolation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout_seconds: int = 5
    memory_limit_mb: int = 256


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sandbox-service", "version": "0.1.0"}


@app.post("/execute")
async def execute_code(request: CodeExecutionRequest):
    return {
        "status": "not_implemented",
        "message": "Sandbox execution pending Phase 3 implementation",
    }
