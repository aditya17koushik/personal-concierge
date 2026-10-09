from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.config import get_settings
from app.database.session import get_db
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.factory import get_llm_provider
from app.schemas.agent import ChatRequest, ChatResponse

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(db: Session = Depends(get_db)) -> AgentService:
    settings = get_settings()
    return AgentService(
        llm=get_llm_provider(),
        db=db,
        default_currency=settings.default_currency,
        google_oauth=GoogleOAuthService(db, settings),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    return await service.chat(user_id=request.user_id, message=request.message)