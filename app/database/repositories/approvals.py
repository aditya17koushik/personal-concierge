import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.database.models import PendingAction


class ApprovalRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        user_id: int,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        request_text: str | None,
        expires_at: datetime,
    ) -> PendingAction:
        action = PendingAction(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tool_name=tool_name,
            arguments=arguments,
            summary=summary,
            request_text=request_text,
            status="pending",
            expires_at=expires_at,
        )
        self._db.add(action)
        self._db.commit()
        self._db.refresh(action)
        return action

    def get(self, user_id: int, approval_id: str) -> PendingAction | None:
        """Scoped to the user: another user's id behaves like 'not found'."""
        return self._db.scalar(
            select(PendingAction).where(
                PendingAction.id == approval_id, PendingAction.user_id == user_id
            )
        )

    def find(
        self, user_id: int, status: str | None = None, limit: int = 50
    ) -> list[PendingAction]:
        stmt = select(PendingAction).where(PendingAction.user_id == user_id)
        if status:
            stmt = stmt.where(PendingAction.status == status)
        stmt = stmt.order_by(PendingAction.created_at.desc()).limit(limit)
        return list(self._db.scalars(stmt))

    def claim(self, approval_id: str) -> bool:
        """Atomically move pending -> executing. Exactly one caller can win,
        so an action can never run twice, even on double clicks or retries."""
        result = self._db.execute(
            update(PendingAction)
            .where(PendingAction.id == approval_id, PendingAction.status == "pending")
            .values(status="executing")
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    def finish(
        self,
        action: PendingAction,
        status: str,
        decided_at: datetime,
        result: dict[str, Any] | None = None,
    ) -> PendingAction:
        action.status = status
        action.decided_at = decided_at
        action.result = result
        self._db.commit()
        self._db.refresh(action)
        return action