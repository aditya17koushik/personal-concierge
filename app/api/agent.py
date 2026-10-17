from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.agent.factory import build_agent_service
from app.agent.service import AgentService
from app.api.deps import CurrentUser, get_current_user
from app.config import get_settings
from app.conversation.service import ConversationService
from app.database.session import get_db
from app.schemas.agent import ChatRequest, ChatResponse
from app.usage.service import RateLimitExceeded

router = APIRouter(prefix="/agent", tags=["agent"])


def get_agent_service(db: Session = Depends(get_db)) -> AgentService:
    return build_agent_service(db, get_settings())


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current: CurrentUser = Depends(get_current_user),
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    try:
        return await service.chat(user_id=current.external_id, message=request.message)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {exc.limit} messages reached.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    settings = get_settings()
    return ConversationService(
        db,
        ttl_minutes=settings.history_ttl_minutes,
        max_messages=settings.history_max_messages,
        max_chars=settings.history_max_chars,
        retention_days=settings.history_retention_days,
    )


@router.delete("/history")
def clear_history(
    current: CurrentUser = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
) -> dict[str, int]:
    """Forget this conversation: deletes your stored messages."""
    return {"deleted": service.clear(current.user_id)}