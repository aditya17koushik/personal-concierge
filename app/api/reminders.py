import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user
from app.config import get_settings
from app.database.session import get_db
from app.reminders.service import ReminderService, aware, describe

router = APIRouter(prefix="/reminders", tags=["reminders"])


class ReminderOut(BaseModel):
    id: int
    text: str
    when: str
    due_at: datetime
    timezone: str
    repeat: str | None


def _service(db: Session) -> ReminderService:
    settings = get_settings()
    return ReminderService(
        db, max_pending=settings.reminder_max_pending, max_days=settings.reminder_max_days_ahead
    )


@router.get("", response_model=list[ReminderOut])
async def list_reminders(
    current: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ReminderOut]:
    """Your pending reminders. Create them by chatting: "remind me to call Ravi at 6pm"."""

    def load() -> list[ReminderOut]:
        out: list[ReminderOut] = []
        for r in _service(db).pending(current.user_id):
            info = describe(r)
            out.append(
                ReminderOut(
                    id=r.id,
                    text=r.text,
                    when=str(info["when"]),
                    due_at=aware(r.due_at),
                    timezone=r.timezone,
                    repeat=r.repeat,
                )
            )
        return out

    return await asyncio.to_thread(load)


@router.delete("/{reminder_id}")
async def cancel_reminder(
    reminder_id: int,
    current: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    done = await asyncio.to_thread(_service(db).cancel, current.user_id, reminder_id)
    if not done:
        raise HTTPException(status_code=404, detail="Reminder not found.")
    return {"cancelled": reminder_id}