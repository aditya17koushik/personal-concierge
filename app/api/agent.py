from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.approvals.service import ApprovalService
from app.config import get_settings
from app.database.session import get_db
from app.decisions.jev import build_jev
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.factory import get_llm_provider
from app.schemas.agent import ChatRequest, ChatResponse
from app.tools.registry import build_default_registry

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(db: Session = Depends(get_db)) -> AgentService:
    settings = get_settings()
    registry = build_default_registry()
    google = GoogleOAuthService(db, settings)
    return AgentService(
        llm=get_llm_provider(),
        db=db,
        jev=build_jev(settings),
        registry=registry,
        default_currency=settings.default_currency,
        google_oauth=google,
        approvals=ApprovalService(
            db,
            registry,
            ttl_minutes=settings.approval_ttl_minutes,
            google_oauth=google,
            default_currency=settings.default_currency,
        ),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    return await service.chat(user_id=request.user_id, message=request.message)