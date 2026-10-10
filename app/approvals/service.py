import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.database.models import PendingAction
from app.database.repositories.approvals import ApprovalRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.oauth import GoogleOAuthService
from app.schemas.approvals import ApprovalOut, PendingApprovalOut
from app.tools.base import ToolContext
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ApprovalError(Exception):
    """Maps to an HTTP status in the API layer. Messages are safe to show."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; Postgres returns aware ones.
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _to_out(action: PendingAction) -> ApprovalOut:
    return ApprovalOut(
        id=action.id,
        tool=action.tool_name,
        summary=action.summary,
        status=action.status,
        created_at=action.created_at,
        expires_at=action.expires_at,
        decided_at=action.decided_at,
        result=action.result,
    )


@dataclass(frozen=True)
class _Snapshot:
    """Plain values, so no ORM object is touched on the event loop after a
    commit expires it (that would trigger a blocking lazy load)."""

    id: str
    user_id: int
    tool_name: str
    arguments: dict[str, Any]


class ApprovalService:
    """Stores proposed actions and runs them only after explicit approval.

    All database work runs in worker threads (see the event-loop tests).
    """

    def __init__(
        self,
        db: Session,
        registry: ToolRegistry,
        ttl_minutes: int = 30,
        google_oauth: GoogleOAuthService | None = None,
        default_currency: str = "INR",
        today_fn: Callable[[], date] = date.today,
        now_fn: Callable[[], datetime] = _now,
    ) -> None:
        self._db = db
        self._registry = registry
        self._repo = ApprovalRepository(db)
        self._users = UserRepository(db)
        self._ttl = timedelta(minutes=ttl_minutes)
        self._google_oauth = google_oauth
        self._default_currency = default_currency
        self._today_fn = today_fn
        self._now = now_fn

    # --------------------------------------------------------------- propose

    async def propose(
        self,
        ctx: ToolContext,
        tool_name: str,
        arguments: dict[str, Any],
        request_text: str,
    ) -> tuple[dict[str, Any], PendingApprovalOut | None]:
        """Validate + summarise a proposed call and store it. Executes nothing.

        Returns (result to show the model, pending approval or None on error).
        """
        prepared = await self._registry.prepare(tool_name, arguments, ctx)
        if "error" in prepared:
            return prepared, None

        pending = await asyncio.to_thread(
            self._create,
            ctx.user_id,
            tool_name,
            prepared["arguments"],
            prepared["summary"],
            request_text[:500],
        )
        logger.info("approval proposed: tool=%s id=%s", tool_name, pending.id)
        return (
            {
                "status": "pending_approval",
                "approval_id": pending.id,
                "summary": pending.summary,
                "note": (
                    "Prepared but NOT done. It happens only if the user "
                    "approves it. Tell the user what you prepared."
                ),
            },
            pending,
        )

    def _create(
        self,
        user_id: int,
        tool_name: str,
        arguments: dict[str, Any],
        summary: str,
        request_text: str,
    ) -> PendingApprovalOut:
        action = self._repo.create(
            user_id=user_id,
            tool_name=tool_name,
            arguments=arguments,
            summary=summary,
            request_text=request_text,
            expires_at=self._now() + self._ttl,
        )
        return PendingApprovalOut(
            id=action.id,
            tool=action.tool_name,
            summary=action.summary,
            expires_at=action.expires_at,
        )

    # ------------------------------------------------------------------ list

    async def list_approvals(
        self, external_user_id: str, status: str | None
    ) -> list[ApprovalOut]:
        return await asyncio.to_thread(self._list, external_user_id, status)

    def _list(self, external_user_id: str, status: str | None) -> list[ApprovalOut]:
        user = self._users.get(external_user_id)
        if user is None:
            return []
        for action in self._repo.find(user.id, status="pending"):
            if _as_utc(action.expires_at) <= self._now():
                self._repo.finish(action, "expired", self._now())
        return [_to_out(a) for a in self._repo.find(user.id, status=status)]

    # ---------------------------------------------------------------- approve

    async def approve(self, external_user_id: str, approval_id: str) -> ApprovalOut:
        snap = await asyncio.to_thread(self._load_and_claim, external_user_id, approval_id)

        ctx = ToolContext(
            db=self._db,
            user_id=snap.user_id,
            today=self._today_fn(),
            default_currency=self._default_currency,
            external_user_id=external_user_id,
            google_oauth=self._google_oauth,
        )
        # Arguments are re-validated inside execute(); state may have changed
        # since the proposal (e.g. the expense was already deleted).
        result = await self._registry.execute(snap.tool_name, snap.arguments, ctx)
        status = "failed" if "error" in result else "executed"
        logger.info("approval %s -> %s", snap.id, status)

        return await asyncio.to_thread(self._finish, snap.id, status, result)

    def _load_and_claim(self, external_user_id: str, approval_id: str) -> _Snapshot:
        action = self._load(external_user_id, approval_id)
        snap = _Snapshot(action.id, action.user_id, action.tool_name, dict(action.arguments))
        if not self._repo.claim(action.id):
            raise ApprovalError(409, "This action was already handled.")
        return snap

    def _finish(self, approval_id: str, status: str, result: dict[str, Any]) -> ApprovalOut:
        action = self._db.get(PendingAction, approval_id)
        assert action is not None
        return _to_out(self._repo.finish(action, status, self._now(), result))

    # ----------------------------------------------------------------- reject

    async def reject(self, external_user_id: str, approval_id: str) -> ApprovalOut:
        return await asyncio.to_thread(self._reject, external_user_id, approval_id)

    def _reject(self, external_user_id: str, approval_id: str) -> ApprovalOut:
        action = self._load(external_user_id, approval_id)
        if not self._repo.claim(action.id):  # same atomic gate as approve
            raise ApprovalError(409, "This action was already handled.")
        action = self._db.get(PendingAction, action.id)
        assert action is not None
        return _to_out(self._repo.finish(action, "rejected", self._now()))

    # ----------------------------------------------------------------- shared

    def _load(self, external_user_id: str, approval_id: str) -> PendingAction:
        """Fetch a pending action owned by this user, or raise ApprovalError."""
        user = self._users.get(external_user_id)
        action = self._repo.get(user.id, approval_id) if user else None
        if action is None:
            raise ApprovalError(404, "Approval not found.")

        if action.status != "pending":
            raise ApprovalError(409, f"This action is already {action.status}.")

        if _as_utc(action.expires_at) <= self._now():
            self._repo.finish(action, "expired", self._now())
            raise ApprovalError(410, "This approval has expired. Ask again to redo it.")

        return action