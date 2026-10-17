from datetime import datetime, timedelta

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import EventAlert, Reminder

STALE_SENDING_MINUTES = 5


class ReminderRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------ user side

    def add(
        self,
        user_id: int,
        text: str,
        due_at: datetime,
        timezone: str,
        repeat: str | None,
        now: datetime,
    ) -> Reminder:
        reminder = Reminder(
            user_id=user_id,
            text=text,
            due_at=due_at,
            timezone=timezone,
            repeat=repeat,
            status="pending",
            attempts=0,
            created_at=now,
            updated_at=now,
        )
        self._db.add(reminder)
        self._db.commit()
        self._db.refresh(reminder)
        return reminder

    def count_pending(self, user_id: int) -> int:
        return int(
            self._db.scalar(
                select(func.count())
                .select_from(Reminder)
                .where(Reminder.user_id == user_id, Reminder.status.in_(("pending", "sending")))
            )
            or 0
        )

    def pending(self, user_id: int, limit: int = 50) -> list[Reminder]:
        return list(
            self._db.scalars(
                select(Reminder)
                .where(Reminder.user_id == user_id, Reminder.status.in_(("pending", "sending")))
                .order_by(Reminder.due_at, Reminder.id)
                .limit(limit)
            )
        )

    def get(self, user_id: int, reminder_id: int) -> Reminder | None:
        return self._db.scalar(
            select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == user_id)
        )

    def cancel(self, user_id: int, reminder_id: int, now: datetime) -> bool:
        result = self._db.execute(
            update(Reminder)
            .where(
                Reminder.id == reminder_id,
                Reminder.user_id == user_id,
                Reminder.status.in_(("pending", "sending")),
            )
            .values(status="cancelled", updated_at=now)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    # ------------------------------------------------------- scheduler side

    def release_stale(self, now: datetime) -> int:
        """A reminder stuck in 'sending' (the process died mid-send) is tried again."""
        result = self._db.execute(
            update(Reminder)
            .where(
                Reminder.status == "sending",
                Reminder.updated_at < now - timedelta(minutes=STALE_SENDING_MINUTES),
            )
            .values(status="pending", updated_at=now)
        )
        self._db.commit()
        return int(result.rowcount or 0)  # type: ignore[attr-defined]

    def due(self, now: datetime, limit: int = 50) -> list[Reminder]:
        return list(
            self._db.scalars(
                select(Reminder)
                .where(Reminder.status == "pending", Reminder.due_at <= now)
                .order_by(Reminder.due_at, Reminder.id)
                .limit(limit)
            )
        )

    def claim(self, reminder_id: int, now: datetime) -> bool:
        """pending -> sending in ONE conditional UPDATE: only one worker can win."""
        result = self._db.execute(
            update(Reminder)
            .where(Reminder.id == reminder_id, Reminder.status == "pending")
            .values(status="sending", attempts=Reminder.attempts + 1, updated_at=now)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    def mark_sent(self, reminder_id: int, now: datetime, next_due: datetime | None) -> None:
        """Delivered. A repeating reminder goes back to pending with its next time."""
        values: dict[str, object] = {"updated_at": now, "sent_at": now, "last_error": None}
        if next_due is None:
            values["status"] = "sent"
        else:
            values.update(status="pending", due_at=next_due, attempts=0)
        self._db.execute(
            update(Reminder)
            .where(Reminder.id == reminder_id, Reminder.status == "sending")
            .values(**values)
        )
        self._db.commit()

    def mark_failed_attempt(
        self, reminder_id: int, now: datetime, error: str, retry_at: datetime | None
    ) -> None:
        """Delivery failed: retry later, or give up (retry_at=None)."""
        values: dict[str, object] = {"updated_at": now, "last_error": error[:200]}
        if retry_at is None:
            values["status"] = "failed"
        else:
            values.update(status="pending", due_at=retry_at)
        self._db.execute(
            update(Reminder)
            .where(Reminder.id == reminder_id, Reminder.status == "sending")
            .values(**values)
        )
        self._db.commit()

    def mark_expired(self, reminder_id: int, now: datetime, next_due: datetime | None) -> None:
        """Missed by too long: dropped. A repeating one skips ahead to its next time."""
        values: dict[str, object] = {"updated_at": now}
        if next_due is None:
            values["status"] = "expired"
        else:
            values.update(status="pending", due_at=next_due, attempts=0)
        self._db.execute(
            update(Reminder)
            .where(Reminder.id == reminder_id, Reminder.status == "sending")
            .values(**values)
        )
        self._db.commit()

    # ------------------------------------------------------- meeting alerts

    def record_alert(self, user_id: int, event_id: str, starts_at: str) -> bool:
        """True the FIRST time this event start is recorded for the user."""
        self._db.add(EventAlert(user_id=user_id, event_id=event_id, starts_at=starts_at))
        try:
            self._db.commit()
            return True
        except IntegrityError:
            self._db.rollback()
            return False

    def forget_alert(self, user_id: int, event_id: str, starts_at: str) -> None:
        """Used when sending failed, so the next poll can try again."""
        self._db.execute(
            delete(EventAlert).where(
                EventAlert.user_id == user_id,
                EventAlert.event_id == event_id,
                EventAlert.starts_at == starts_at,
            )
        )
        self._db.commit()

    def prune_alerts(self, before: datetime) -> int:
        result = self._db.execute(delete(EventAlert).where(EventAlert.created_at < before))
        self._db.commit()
        return int(result.rowcount or 0)  # type: ignore[attr-defined]