"""Plain-text messages the scheduler sends. Built by CODE, never by the model, so a calendar
invite can't smuggle instructions into them. Event titles are third-party text: trimmed
and stripped of control characters, then shown as plain text only."""

import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.database.models import Reminder
from app.reminders.service import aware

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_TITLE = 80


def safe(text: str | None, limit: int = MAX_TITLE) -> str:
    value = _CONTROL.sub(" ", text or "").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _clock(value: str, tz: ZoneInfo) -> str:
    return datetime.fromisoformat(value).astimezone(tz).strftime("%H:%M")


def event_line(event: dict[str, Any], tz: ZoneInfo) -> str:
    if event.get("all_day"):
        when = "All day"
    else:
        when = f"{_clock(event['start'], tz)}-{_clock(event['end'], tz)}"
    line = f"- {when}  {safe(event.get('title'))}"
    if event.get("location"):
        line += f" ({safe(event['location'], 40)})"
    return line


def build_agenda(
    day: date,
    tz_name: str,
    events: list[dict[str, Any]] | None,
    reminders: list[Reminder],
) -> str:
    """`events=None` means the calendar could not be read (not connected or an error)."""
    tz = ZoneInfo(tz_name)
    lines = [f"Good morning. Here is {day.strftime('%A %d %B')}:", ""]

    if events is None:
        lines.append("Calendar: not available right now (use /google to connect it).")
    elif not events:
        lines.append("Calendar: nothing scheduled today.")
    else:
        lines.append("Calendar:")
        lines += [event_line(e, tz) for e in events]

    todays = [
        r
        for r in reminders
        if aware(r.due_at).astimezone(tz).date() == day
    ]
    if todays:
        lines += ["", "Reminders today:"]
        for r in todays:
            when = aware(r.due_at).astimezone(tz).strftime("%H:%M")
            lines.append(f"- {when}  {safe(r.text, 120)}")
    return "\n".join(lines)


def build_event_alert(event: dict[str, Any], now: datetime, tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    minutes = max(1, round((event["starts"] - now).total_seconds() / 60))
    unit = "minute" if minutes == 1 else "minutes"
    text = (
        f"Starting in {minutes} {unit}: {safe(event['title'])} "
        f"at {event['starts'].astimezone(tz).strftime('%H:%M')}"
    )
    if event.get("location"):
        text += f"\nWhere: {safe(event['location'], 80)}"
    if event.get("meeting_link"):
        text += f"\nJoin: {event['meeting_link']}"
    return text