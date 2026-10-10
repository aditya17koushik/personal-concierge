import logging
from datetime import date as date_type
from datetime import timedelta
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.integrations.google.calendar import CalendarClient, CalendarError
from app.tools.base import Tool, ToolContext
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
]