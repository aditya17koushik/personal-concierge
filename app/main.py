from fastapi import FastAPI

from app.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }