"""Reminder rules: turning what the user said into a UTC time, and repeating reminders.

All "local" times are the user's wall-clock time in their profile timezone. They are
converted to UTC for storage, and a repeating reminder keeps its LOCAL time (a daily 08:00
reminder stays at 08:00 across daylight-saving changes).
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.database.models import Reminder
from app.database.repositories.reminders import ReminderRepository

REPEATS = ("daily", "weekdays", "weekly")
MAX_TEXT_CHARS = 300
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class ReminderError(Exception):
    """A safe-to-show reason a reminder could not be made."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def clean_text(text: str) -> str:
    value = _CONTROL.sub(" ", text).replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ReminderError("A reminder needs some text.")
    return value[:MAX_TEXT_CHARS]


def aware(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; they are always UTC."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def local_to_utc(local: datetime, tz_name: str) -> datetime:
    return local.replace(tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)


def resolve_due(
    *,
    at: datetime | None,
    in_minutes: int | None,
    tz_name: str,
    now: datetime,
    max_days: int,
) -> datetime:
    """UTC time for 'at <local time>' or 'in N minutes'. Exactly one must be given."""
    if (at is None) == (in_minutes is None):
        raise ReminderError("Give either a time (at) or a delay (in_minutes), not both.")
    if in_minutes is not None:
        due = now + timedelta(minutes=in_minutes)
    else:
        assert at is not None
        if at.tzinfo is not None:
            raise ReminderError("Give local time without a UTC offset, e.g. 2026-10-20T18:00.")
        due = local_to_utc(at, tz_name)
    if due <= now:
        raise ReminderError("That time has already passed. Ask for a time in the future.")
    if due > now + timedelta(days=max_days):
        raise ReminderError(f"Reminders can be at most {max_days} days ahead.")
    return due


def next_occurrence(due_at: datetime, repeat: str, tz_name: str, now: datetime) -> datetime:
    """The first repeat AFTER `now`, at the same local wall-clock time as `due_at`."""
    tz = ZoneInfo(tz_name)
    local = aware(due_at).astimezone(tz)
    day = local.date()
    wall = local.time()
    now_local = now.astimezone(tz)

    for _ in range(400):
        if repeat == "weekly":
            day += timedelta(days=7)
        else:
            day += timedelta(days=1)
            if repeat == "weekdays":
                while day.weekday() >= 5:
                    day += timedelta(days=1)
        candidate = datetime.combine(day, wall, tzinfo=tz)
        if candidate > now_local:
            return candidate.astimezone(timezone.utc)
    raise ReminderError("Could not work out the next repeat.")


def format_local(value: datetime, tz_name: str) -> str:
    return aware(value).astimezone(ZoneInfo(tz_name)).strftime("%a %d %b, %H:%M")


def describe(reminder: Reminder) -> dict[str, object]:
    return {
        "id": reminder.id,
        "text": reminder.text,
        "when": format_local(reminder.due_at, reminder.timezone),
        "timezone": reminder.timezone,
        "repeat": reminder.repeat,
    }


class ReminderService:
    """Create, list and cancel a user's reminders (delivery lives in app.scheduler)."""

    def __init__(
        self,
        db: Session,
        *,
        max_pending: int = 50,
        max_days: int = 366,
        now_fn: Callable[[], datetime] = utcnow,
    ) -> None:
        self._repo = ReminderRepository(db)
        self._max_pending = max_pending
        self._max_days = max_days
        self._now = now_fn

    def create(
        self,
        user_id: int,
        text: str,
        tz_name: str,
        *,
        at: datetime | None = None,
        in_minutes: int | None = None,
        repeat: str | None = None,
        now: datetime | None = None,
    ) -> Reminder:
        if repeat is not None and repeat not in REPEATS:
            raise ReminderError(f"repeat must be one of: {', '.join(REPEATS)}.")
        clock = now or self._now()
        due = resolve_due(
            at=at, in_minutes=in_minutes, tz_name=tz_name, now=clock, max_days=self._max_days
        )
        text = clean_text(text)
        if self._repo.count_pending(user_id) >= self._max_pending:
            raise ReminderError(
                f"You already have {self._max_pending} pending reminders. Cancel some first."
            )
        return self._repo.add(user_id, text, due, tz_name, repeat, clock)

    def pending(self, user_id: int) -> list[Reminder]:
        return self._repo.pending(user_id)

    def cancel(self, user_id: int, reminder_id: int) -> bool:
        return self._repo.cancel(user_id, reminder_id, self._now())