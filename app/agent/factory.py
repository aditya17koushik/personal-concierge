"""Builds the agent and approval services. Used by the HTTP API and the
Telegram bot, so both run exactly the same Jev + tools + approvals pipeline."""

from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.approvals.service import ApprovalService
from app.config import Settings
from app.decisions.jev import build_jev
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.factory import get_llm_provider
from app.tools.registry import ToolRegistry, build_default_registry


def build_approval_service(
    db: Session,
    settings: Settings,
    registry: ToolRegistry | None = None,
    google_oauth: GoogleOAuthService | None = None,
) -> ApprovalService:
    return ApprovalService(
        db,
        registry or build_default_registry(),
        ttl_minutes=settings.approval_ttl_minutes,
        google_oauth=google_oauth or GoogleOAuthService(db, settings),
        default_currency=settings.default_currency,
        default_timezone=settings.default_timezone,
    )


def build_agent_service(db: Session, settings: Settings) -> AgentService:
    registry = build_default_registry()
    google = GoogleOAuthService(db, settings)
    return AgentService(
        llm=get_llm_provider(),
        db=db,
        jev=build_jev(settings),
        registry=registry,
        default_currency=settings.default_currency,
        default_timezone=settings.default_timezone,
        daily_message_limit=settings.daily_message_limit,
        history_ttl_minutes=settings.history_ttl_minutes,
        history_max_messages=settings.history_max_messages,
        history_max_chars=settings.history_max_chars,
        history_retention_days=settings.history_retention_days,
        google_oauth=google,
        approvals=build_approval_service(db, settings, registry, google),
    )