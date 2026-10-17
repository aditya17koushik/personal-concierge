import logging
import re
from datetime import date as date_type
from datetime import datetime, time, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

from app.integrations.google.calendar import CalendarClient, CalendarError
from app.tools.base import Tool, ToolContext, ToolError
from app.tools.google_common import UNTRUSTED_NOTE, with_google

logger = logging.getLogger(__name__)

DEFAULT_RANGE_DAYS = 7
MAX_RANGE_DAYS = 366


class ListEventsArgs(BaseModel):
    start_date: date_type | None = Field(
        default=None, description="First day, YYYY-MM-DD. Omit for today."
    )
    end_date: date_type | None = Field(
        default=None,
        description="Last day, inclusive, YYYY-MM-DD. Omit for 7 days after start_date.",
    )
    query: str | None = Field(
        default=None,
        max_length=200,
        description="Free-text search over event titles, descriptions and locations.",
    )
    max_results: int = Field(default=25, ge=1, le=50)


async def list_events(ctx: ToolContext, args: ListEventsArgs) -> dict[str, Any]:
    start = args.start_date or ctx.today
    end = args.end_date or (start + timedelta(days=DEFAULT_RANGE_DAYS))

    if end < start:
        return {"error": "end_date must not be before start_date"}
    if (end - start).days > MAX_RANGE_DAYS:
        return {"error": f"Date range too large (max {MAX_RANGE_DAYS} days)."}

    async def call(
        token: str, transport: httpx.AsyncBaseTransport | None
    ) -> dict[str, Any]:
        result = await CalendarClient(token, transport).list_events(
            start, end, args.query, args.max_results
        )
        logger.info("calendar.list returned %d events", len(result["events"]))
        return {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "timezone": result["timezone"],
            "count": len(result["events"]),
            "more_results_exist": result["truncated"],
            "events": result["events"],
            "note": UNTRUSTED_NOTE,
        }

    return await with_google(ctx, call, (CalendarError,))


class GetEventArgs(BaseModel):
    event_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="The id of an event, taken from list_events results.",
    )


async def get_event(ctx: ToolContext, args: GetEventArgs) -> dict[str, Any]:
    async def call(
        token: str, transport: httpx.AsyncBaseTransport | None
    ) -> dict[str, Any]:
        event = await CalendarClient(token, transport).get_event(args.event_id)
        return {"event": event, "note": UNTRUSTED_NOTE}

    return await with_google(ctx, call, (CalendarError,))


# ------------------------------------------------------------------ helpers

EVENT_ID = Field(
    min_length=1,
    max_length=128,
    pattern=r"^[A-Za-z0-9_-]+$",
    description="The id of an event, taken from list_events results.",
)
EMAIL_RE = re.compile(r"^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$")
MAX_GUESTS = 10
MAX_EVENT_MINUTES = 24 * 60
LOCAL_TIME_HELP = (
    "Local time in the user's calendar time zone, as YYYY-MM-DDTHH:MM, "
    "with no UTC offset."
)


def _naive(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is not None:
        raise ValueError("Give local time without a UTC offset, e.g. 2026-10-20T15:00")
    return value


def _when(dt: datetime | date_type) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%a %d %b %Y, %H:%M")
    return dt.strftime("%a %d %b %Y") + " (all day)"


def _span(start: datetime | date_type, end: datetime | date_type) -> str:
    if isinstance(start, datetime) and isinstance(end, datetime):
        if start.date() == end.date():
            return f"{start.strftime('%a %d %b %Y, %H:%M')}-{end.strftime('%H:%M')}"
        return f"{_when(start)} to {_when(end)}"
    if start == end:
        return _when(start)
    return f"{start.strftime('%a %d %b %Y')} to {_when(end)}"


def _clean_guests(emails: list[str]) -> list[str]:
    out: list[str] = []
    for raw in emails:
        email = raw.strip().lower()
        if not EMAIL_RE.match(email):
            raise ValueError(f"'{raw}' is not a valid email address")
        if email not in out:
            out.append(email)
    if len(out) > MAX_GUESTS:
        raise ValueError(f"At most {MAX_GUESTS} guests")
    return out


async def _client_for(ctx: ToolContext) -> CalendarClient:
    """A CalendarClient for the user's token. Raises ToolError with a safe
    message (used by summarize, where there is no tool-result channel)."""
    from app.integrations.google.oauth import GoogleAuthError, GoogleNotConnectedError

    if ctx.google_oauth is None:
        raise ToolError("Google integration is not configured.")
    try:
        token = await ctx.google_oauth.get_access_token(ctx.user_id)
    except GoogleNotConnectedError as exc:
        raise ToolError("The user has not connected their Google account.") from exc
    except GoogleAuthError as exc:
        raise ToolError(str(exc)) from exc
    return CalendarClient(token, ctx.google_oauth.http_transport)


async def _existing(ctx: ToolContext, event_id: str) -> dict[str, Any]:
    client = await _client_for(ctx)
    try:
        return await client.get_event(event_id)
    except CalendarError as exc:
        raise ToolError(str(exc)) from exc


def _parse(value: str, all_day: bool) -> datetime | date_type:
    if all_day:
        return date_type.fromisoformat(value)
    return datetime.fromisoformat(value).replace(tzinfo=None)  # as the calendar shows it


def _describe(event: dict[str, Any]) -> str:
    try:
        all_day = bool(event.get("all_day"))
        when = _span(_parse(event["start"], all_day), _parse(event["end"], all_day))
    except ValueError:
        when = f'{event["start"]} to {event["end"]}'
    return f'"{event["title"]}" ({when})'


def _guest_note(event: dict[str, Any], verb: str) -> str:
    others = [a for a in event.get("attendees") or [] if not a.get("is_me")]
    if not others:
        return ""
    return f" {len(others)} guest(s) will be emailed about the {verb}."


# ------------------------------------------------------------- create_event


class CreateEventArgs(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    start: datetime | None = Field(default=None, description=LOCAL_TIME_HELP)
    end: datetime | None = Field(
        default=None,
        description=LOCAL_TIME_HELP + " Omit to use duration_minutes (default 60).",
    )
    duration_minutes: int | None = Field(default=None, ge=5, le=MAX_EVENT_MINUTES)
    all_day_date: date_type | None = Field(
        default=None, description="Set this INSTEAD of start/end for an all-day event."
    )
    location: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=2000)
    guests: list[str] = Field(
        default_factory=list,
        description=(
            "Email addresses to invite. They are emailed an invitation on approval. "
            "Only include people the user explicitly asked to invite."
        ),
    )

    _start = field_validator("start", "end")(_naive)

    @field_validator("guests")
    @classmethod
    def _guests(cls, value: list[str]) -> list[str]:
        return _clean_guests(value)

    @model_validator(mode="after")
    def _shape(self) -> "CreateEventArgs":
        if self.all_day_date is not None:
            if self.start or self.end or self.duration_minutes:
                raise ValueError("Use all_day_date alone, or start with end/duration_minutes")
            return self
        if self.start is None:
            raise ValueError("start is required (or all_day_date for an all-day event)")
        if self.end is None:
            self.end = self.start + timedelta(minutes=self.duration_minutes or 60)
        elif self.duration_minutes is not None:
            raise ValueError("Give end or duration_minutes, not both")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if self.end - self.start > timedelta(minutes=MAX_EVENT_MINUTES):
            raise ValueError("Events can be at most 24 hours; use all_day_date for longer")
        self.duration_minutes = None
        return self

    def span(self) -> tuple[datetime | date_type, datetime | date_type]:
        if self.all_day_date is not None:
            return self.all_day_date, self.all_day_date
        assert self.start is not None and self.end is not None
        return self.start, self.end


async def summarize_create_event(ctx: ToolContext, args: CreateEventArgs) -> str:
    client = await _client_for(ctx)
    try:
        tz = await client.timezone_name()
    except CalendarError as exc:
        raise ToolError(str(exc)) from exc
    start, end = args.span()
    text = f'Create event "{args.title}" on {_span(start, end)} ({tz})'
    if args.location:
        text += f", at {args.location}"
    if args.guests:
        text += f". Invite {', '.join(args.guests)}: they will be emailed an invitation"
    return text


async def create_event(ctx: ToolContext, args: CreateEventArgs) -> dict[str, Any]:
    async def call(token: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
        start, end = args.span()
        event = await CalendarClient(token, transport).create_event(
            title=args.title,
            start=start,
            end=end,
            location=args.location,
            description=args.description,
            attendees=args.guests,
        )
        return {"created": event}

    return await with_google(ctx, call, (CalendarError,))


# ------------------------------------------------------------- update_event


class UpdateEventArgs(BaseModel):
    event_id: str = EVENT_ID
    title: str | None = Field(default=None, min_length=1, max_length=200)
    start: datetime | None = Field(default=None, description=LOCAL_TIME_HELP)
    end: datetime | None = Field(
        default=None, description=LOCAL_TIME_HELP + " Omit to keep the event's length."
    )
    all_day_date: date_type | None = Field(
        default=None, description="Move an event to this day, as an all-day event."
    )
    location: str | None = Field(default=None, max_length=300)
    description: str | None = Field(default=None, max_length=2000)

    _start = field_validator("start", "end")(_naive)

    @model_validator(mode="after")
    def _shape(self) -> "UpdateEventArgs":
        if self.all_day_date is not None and (self.start or self.end):
            raise ValueError("Use all_day_date alone, or start/end")
        if self.end is not None and self.start is None:
            raise ValueError("Give start when giving end")
        if self.start and self.end and self.end <= self.start:
            raise ValueError("end must be after start")
        if not any(
            v is not None
            for v in (self.title, self.start, self.all_day_date, self.location, self.description)
        ):
            raise ValueError("Nothing to change")
        return self


def _new_span(
    args: UpdateEventArgs, current: dict[str, Any]
) -> tuple[datetime | date_type, datetime | date_type] | None:
    if args.all_day_date is not None:
        return args.all_day_date, args.all_day_date
    if args.start is None:
        return None
    end = args.end
    if end is None:
        if current.get("all_day"):
            raise ToolError("This is an all-day event; give both start and end, or all_day_date.")
        old_start = datetime.fromisoformat(current["start"]).replace(tzinfo=None)
        old_end = datetime.fromisoformat(current["end"]).replace(tzinfo=None)
        end = args.start + (old_end - old_start)
    return args.start, end


async def summarize_update_event(ctx: ToolContext, args: UpdateEventArgs) -> str:
    current = await _existing(ctx, args.event_id)
    changes: list[str] = []
    span = _new_span(args, current)
    if span:
        changes.append(f"move to {_span(*span)}")
    if args.title is not None:
        changes.append(f'rename to "{args.title}"')
    if args.location is not None:
        changes.append(f"set location to {args.location or '(none)'}")
    if args.description is not None:
        changes.append("change the description")
    return f"Change {_describe(current)}: {'; '.join(changes)}." + _guest_note(current, "change")


async def update_event(ctx: ToolContext, args: UpdateEventArgs) -> dict[str, Any]:
    async def call(token: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
        client = CalendarClient(token, transport)
        current = await client.get_event(args.event_id)
        try:
            span = _new_span(args, current)
        except ToolError as exc:
            return {"error": str(exc)}
        event = await client.update_event(
            args.event_id,
            title=args.title,
            start=span[0] if span else None,
            end=span[1] if span else None,
            location=args.location,
            description=args.description,
        )
        return {"updated": event}

    return await with_google(ctx, call, (CalendarError,))


# ------------------------------------------------------------- delete_event


class DeleteEventArgs(BaseModel):
    event_id: str = EVENT_ID


async def summarize_delete_event(ctx: ToolContext, args: DeleteEventArgs) -> str:
    current = await _existing(ctx, args.event_id)
    text = f"Delete {_describe(current)} from your calendar."
    if current.get("recurring"):
        text += " Only this occurrence is deleted."
    return text + _guest_note(current, "cancellation")


async def delete_event(ctx: ToolContext, args: DeleteEventArgs) -> dict[str, Any]:
    async def call(token: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
        event = await CalendarClient(token, transport).delete_event(args.event_id)
        return {"deleted": event}

    return await with_google(ctx, call, (CalendarError,))


# -------------------------------------------------------------- rsvp_event


class RsvpEventArgs(BaseModel):
    event_id: str = EVENT_ID
    response: Literal["accepted", "declined", "tentative"]


_RSVP_WORDS = {"accepted": "Accept", "declined": "Decline", "tentative": "Reply maybe to"}


async def summarize_rsvp_event(ctx: ToolContext, args: RsvpEventArgs) -> str:
    current = await _existing(ctx, args.event_id)
    if not any(a.get("is_me") for a in current.get("attendees") or []):
        raise ToolError("You are not a guest of this event, so there is nothing to answer.")
    return (
        f"{_RSVP_WORDS[args.response]} the invitation {_describe(current)}. "
        f"The organizer ({current.get('organizer')}) will be notified."
    )


async def rsvp_event(ctx: ToolContext, args: RsvpEventArgs) -> dict[str, Any]:
    async def call(token: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
        event = await CalendarClient(token, transport).respond(args.event_id, args.response)
        return {"responded": args.response, "event": event}

    return await with_google(ctx, call, (CalendarError,))


# ---------------------------------------------------------- find_free_time


class FindFreeTimeArgs(BaseModel):
    duration_minutes: int = Field(default=30, ge=5, le=480)
    start_date: date_type | None = Field(default=None, description="Omit for today.")
    end_date: date_type | None = Field(
        default=None, description="Last day, inclusive. Omit for 7 days after start_date."
    )
    day_start: time = Field(default=time(9, 0), description="Earliest time of day, HH:MM.")
    day_end: time = Field(default=time(18, 0), description="Latest time of day, HH:MM.")
    include_weekends: bool = False
    max_slots: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def _check(self) -> "FindFreeTimeArgs":
        if self.day_end <= self.day_start:
            raise ValueError("day_end must be after day_start")
        return self


MAX_FREE_RANGE_DAYS = 31


async def find_free_time(ctx: ToolContext, args: FindFreeTimeArgs) -> dict[str, Any]:
    start = args.start_date or ctx.today
    end = args.end_date or (start + timedelta(days=DEFAULT_RANGE_DAYS))
    if end < start:
        return {"error": "end_date must not be before start_date"}
    if (end - start).days > MAX_FREE_RANGE_DAYS:
        return {"error": f"Date range too large (max {MAX_FREE_RANGE_DAYS} days)."}

    async def call(token: str, transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
        info = await CalendarClient(token, transport).busy_periods(start, end)
        tz = info["tz"]
        need = timedelta(minutes=args.duration_minutes)
        now = datetime.now(tz)
        slots: list[dict[str, str]] = []

        day = start
        while day <= end and len(slots) < args.max_slots:
            if args.include_weekends or day.weekday() < 5:
                cursor = datetime.combine(day, args.day_start, tzinfo=tz)
                limit = datetime.combine(day, args.day_end, tzinfo=tz)
                cursor = max(cursor, now)
                for b_start, b_end in sorted(info["busy"]):
                    if b_end <= cursor or b_start >= limit:
                        continue
                    if b_start - cursor >= need:
                        slots.append({"start": cursor.isoformat(), "end": b_start.isoformat()})
                    cursor = max(cursor, b_end)
                if limit - cursor >= need:
                    slots.append({"start": cursor.isoformat(), "end": limit.isoformat()})
            day += timedelta(days=1)

        return {
            "timezone": info["timezone"],
            "duration_minutes": args.duration_minutes,
            "free_windows": slots[: args.max_slots],
            "note_all_day_events": [e["title"] for e in info["all_day"]],
            "may_be_incomplete": info["truncated"],
            "tip": "Each window is free for at least the requested duration; "
            "propose a start inside it with create_event.",
        }

    return await with_google(ctx, call, (CalendarError,))


CALENDAR_TOOLS: list[Tool] = [
    Tool(
        name="list_events",
        description=(
            "List events on the user's primary Google Calendar between two dates "
            "(read-only), in chronological order. All-day events have all_day=true "
            "and date-only start/end (end is the last day, inclusive)."
        ),
        args_model=ListEventsArgs,
        handler=list_events,
        domain="calendar",
    ),
    Tool(
        name="get_event",
        description=(
            "Get full details of one calendar event: description, organizer and "
            "attendees with their responses (read-only)."
        ),
        args_model=GetEventArgs,
        handler=get_event,
        domain="calendar",
    ),
    Tool(
        name="find_free_time",
        description=(
            "Find open time slots (gaps) when the user asks when they are FREE or wants "
            "a slot for something. NOT for showing what is scheduled: use list_events "
            "for that. Read-only; skips declined and 'free' events and, by default, "
            "weekends."
        ),
        args_model=FindFreeTimeArgs,
        handler=find_free_time,
        domain="calendar",
    ),
    Tool(
        name="create_event",
        description=(
            "Prepare a new calendar event. Only PREPARES it; it is created only "
            "after the user approves. Times are local to the user's calendar."
        ),
        args_model=CreateEventArgs,
        handler=create_event,
        domain="calendar",
        risk="high",
        side_effects=True,
        summarize=summarize_create_event,
    ),
    Tool(
        name="update_event",
        description=(
            "Prepare a change to an existing event (move it, rename it, change "
            "location or description). Look the event up first to get its id. Only "
            "PREPARES the change; it happens after the user approves."
        ),
        args_model=UpdateEventArgs,
        handler=update_event,
        domain="calendar",
        risk="high",
        side_effects=True,
        summarize=summarize_update_event,
    ),
    Tool(
        name="delete_event",
        description=(
            "Prepare deleting one calendar event by id (from list_events). Only "
            "PREPARES it; it happens after the user approves."
        ),
        args_model=DeleteEventArgs,
        handler=delete_event,
        domain="calendar",
        risk="high",
        side_effects=True,
        summarize=summarize_delete_event,
    ),
    Tool(
        name="rsvp_event",
        description=(
            "Prepare the user's reply (accepted, declined or tentative) to a "
            "calendar invitation. Only PREPARES it; it is sent after approval."
        ),
        args_model=RsvpEventArgs,
        handler=rsvp_event,
        domain="calendar",
        risk="high",
        side_effects=True,
        summarize=summarize_rsvp_event,
    ),
]