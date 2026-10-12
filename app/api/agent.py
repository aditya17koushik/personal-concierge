from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.agent.factory import build_agent_service
from app.agent.service import AgentService
from app.config import get_settings
from app.database.session import get_db
from app.schemas.agent import ChatRequest, ChatResponse

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(db: Session = Depends(get_db)) -> AgentService:
    return build_agent_service(db, get_settings())


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    return await service.chat(user_id=request.user_id, message=request.message)