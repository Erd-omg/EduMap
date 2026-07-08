from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="EduMap Profile Service",
    version="0.1.0",
    description="User profile management with data privacy isolation",
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
    return {"status": "ok", "service": "profile-service", "version": "0.1.0"}


@app.get("/health/ready")
async def readiness():
    return {
        "status": "ok",
        "service": "profile-service",
        "version": "0.1.0",
        "components": {
            "database": "not_implemented",
        },
    }
