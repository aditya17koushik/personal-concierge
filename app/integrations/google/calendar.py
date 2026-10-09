import logging
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.integrations.google.gmail import html_to_text

logger = logging.getLogger(__name__)

BASE_URL = "https://www.googleapis.com/calendar/v3"
MAX_DESCRIPTION_CHARS = 3000
MAX_ATTENDEES = 50


class CalendarError(Exception):
    """Safe-to-show error about a Calendar request."""


def day_bounds(start: date, end: date, tz: ZoneInfo) -> tuple[str, str]:
    """RFC3339 [start 00:00, end+1 00:00) in the calendar's own time zone."""
    time_min = datetime.combine(start, time.min, tzinfo=tz).isoformat()
    time_max = datetime.combine(end + timedelta(days=1), time.min, tzinfo=tz).isoformat()
    return time_min, time_max


def _when(part: dict[str, Any]) -> tuple[str, bool]:
    if part.get("dateTime"):
        return part["dateTime"], False
    return part.get("date", ""), True


def _inclusive_end(end_date: str) -> str:
    # Google's all-day end date is exclusive; humans expect the last day.
    try:
        return (date.fromisoformat(end_date) - timedelta(days=1)).isoformat()
    except ValueError:
        return end_date


def _meeting_link(event: dict[str, Any]) -> str | None:
    if event.get("hangoutLink"):
        return event["hangoutLink"]
    for entry in (event.get("conferenceData") or {}).get("entryPoints") or []:
        if entry.get("entryPointType") == "video" and entry.get("uri"):
            return entry["uri"]
    return None


def summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    start, all_day = _when(event.get("start") or {})
    end, _ = _when(event.get("end") or {})
    if all_day and end:
        end = _inclusive_end(end)

    return {
        "id": event.get("id"),
        "title": event.get("summary") or "(no title)",
        "start": start,
        "end": end,
        "all_day": all_day,
        "location": event.get("location"),
        "meeting_link": _meeting_link(event),
        "status": event.get("status"),
        "recurring": bool(event.get("recurringEventId")),
    }


def detail_event(event: dict[str, Any]) -> dict[str, Any]:
    description = html_to_text(event.get("description") or "").strip()
    truncated = len(description) > MAX_DESCRIPTION_CHARS
    attendees = event.get("attendees") or []

    return {
        **summarize_event(event),
        "description": description[:MAX_DESCRIPTION_CHARS],
        "description_truncated": truncated,
        "organizer": (event.get("organizer") or {}).get("email"),
        "attendee_count": len(attendees),
        "attendees": [
            {
                "email": a.get("email"),
                "name": a.get("displayName"),
                "response": a.get("responseStatus"),
                "optional": bool(a.get("optional")),
                "is_me": bool(a.get("self")),
            }
            for a in attendees[:MAX_ATTENDEES]
        ],
    }


class CalendarClient:
    def __init__(
        self,
        access_token: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = access_token
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self._token}"},
            transport=self._transport,
            timeout=20.0,
        )

    async def _get(
        self, client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            response = await client.get(path, params=params)
        except httpx.HTTPError as exc:
            logger.error("Could not reach Google Calendar: %s", exc)
            raise CalendarError("Could not reach Google Calendar. Please try again.") from exc

        if response.status_code == 200:
            return response.json()

        logger.warning("Calendar API returned %s for %s", response.status_code, path)
        if response.status_code == 401:
            raise CalendarError("Google rejected the access token. Please reconnect Google.")
        if response.status_code == 403:
            raise CalendarError(
                "Calendar access was denied. Check that the Google Calendar API "
                "is enabled and that all permissions were granted."
            )
        if response.status_code == 404:
            raise CalendarError("Event not found.")
        if response.status_code == 429:
            raise CalendarError("Calendar rate limit reached. Try again in a moment.")
        raise CalendarError("Google Calendar returned an unexpected error.")

    async def list_events(
        self, start: date, end: date, query: str | None, max_results: int
    ) -> dict[str, Any]:
        async with self._client() as client:
            calendar = await self._get(client, "/calendars/primary")
            tz_name = calendar.get("timeZone") or "UTC"
            try:
                tz = ZoneInfo(tz_name)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise CalendarError(
                    "Could not determine the calendar's time zone."
                ) from exc

            time_min, time_max = day_bounds(start, end, tz)
            params: dict[str, Any] = {
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",  # expand recurring events
                "orderBy": "startTime",
                "maxResults": max_results,
                "timeZone": tz_name,
            }
            if query:
                params["q"] = query

            data = await self._get(client, "/calendars/primary/events", params)

        return {
            "timezone": tz_name,
            "events": [summarize_event(e) for e in data.get("items") or []],
            "truncated": bool(data.get("nextPageToken")),
        }

    async def get_event(self, event_id: str) -> dict[str, Any]:
        async with self._client() as client:
            event = await self._get(client, f"/calendars/primary/events/{event_id}")
        return detail_event(event)