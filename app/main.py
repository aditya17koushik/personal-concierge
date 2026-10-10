import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.agent import router as agent_router
from app.api.approvals import router as approvals_router
from app.api.auth import router as auth_router
from app.config import get_settings
from app.llm.base import LLMError

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
)

app.include_router(agent_router)
app.include_router(auth_router)
app.include_router(approvals_router)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    logger.error("LLM error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "The language model is currently unavailable."},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }