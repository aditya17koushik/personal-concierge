from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.approvals.service import ApprovalError, ApprovalService
from app.config import get_settings
from app.database.session import get_db
from app.integrations.google.oauth import GoogleOAuthService
from app.schemas.approvals import ApprovalOut
from app.tools.registry import build_default_registry

router = APIRouter(prefix="/approvals", tags=["approvals"])

UserIdQuery = Query(min_length=1, max_length=255)


def get_approval_service(db: Session = Depends(get_db)) -> ApprovalService:
    settings = get_settings()
    return ApprovalService(
        db,
        build_default_registry(),
        ttl_minutes=settings.approval_ttl_minutes,
        google_oauth=GoogleOAuthService(db, settings),
        default_currency=settings.default_currency,
    )


@router.get("", response_model=list[ApprovalOut])
async def list_approvals(
    user_id: str = UserIdQuery,
    status: str | None = Query(
        default="pending",
        description="pending, executed, failed, rejected, expired. Empty for all.",
    ),
    service: ApprovalService = Depends(get_approval_service),
) -> list[ApprovalOut]:
    return await service.list_approvals(user_id, status or None)


@router.post("/{approval_id}/approve", response_model=ApprovalOut)
async def approve(
    approval_id: str,
    user_id: str = UserIdQuery,
    service: ApprovalService = Depends(get_approval_service),
) -> ApprovalOut:
    try:
        return await service.approve(user_id, approval_id)
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/{approval_id}/reject", response_model=ApprovalOut)
async def reject(
    approval_id: str,
    user_id: str = UserIdQuery,
    service: ApprovalService = Depends(get_approval_service),
) -> ApprovalOut:
    try:
        return await service.reject(user_id, approval_id)
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)