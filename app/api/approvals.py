from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agent.factory import build_approval_service
from app.api.deps import CurrentUser, get_current_user
from app.approvals.service import ApprovalError, ApprovalService
from app.config import get_settings
from app.database.session import get_db
from app.schemas.approvals import ApprovalOut

router = APIRouter(prefix="/approvals", tags=["approvals"])

def get_approval_service(db: Session = Depends(get_db)) -> ApprovalService:
    return build_approval_service(db, get_settings())


@router.get("", response_model=list[ApprovalOut])
async def list_approvals(
    current: CurrentUser = Depends(get_current_user),
    status: str | None = Query(
        default="pending",
        description="pending, executed, failed, rejected, expired. Empty for all.",
    ),
    service: ApprovalService = Depends(get_approval_service),
) -> list[ApprovalOut]:
    return await service.list_approvals(current.external_id, status or None)


@router.post("/{approval_id}/approve", response_model=ApprovalOut)
async def approve(
    approval_id: str,
    current: CurrentUser = Depends(get_current_user),
    service: ApprovalService = Depends(get_approval_service),
) -> ApprovalOut:
    try:
        return await service.approve(current.external_id, approval_id)
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


@router.post("/{approval_id}/reject", response_model=ApprovalOut)
async def reject(
    approval_id: str,
    current: CurrentUser = Depends(get_current_user),
    service: ApprovalService = Depends(get_approval_service),
) -> ApprovalOut:
    try:
        return await service.reject(current.external_id, approval_id)
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)