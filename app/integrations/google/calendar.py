import logging
from datetime import date, datetime, time, timedelta, timezone
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

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            logger.error("Could not reach Google Calendar: %s", exc)
            raise CalendarError("Could not reach Google Calendar. Please try again.") from exc

        if response.status_code in (200, 204):
            return response.json() if response.content else {}

        logger.warning(
            "Calendar API returned %s for %s %s", response.status_code, method, path
        )
        if response.status_code == 401:
            raise CalendarError("Google rejected the access token. Please reconnect Google.")
        if response.status_code == 403:
            if method != "GET":
                raise CalendarError(
                    "Google did not allow changing the calendar. Your Google login "
                    "predates calendar editing: use /google to log in again and "
                    "allow all permissions."
                )
            raise CalendarError(
                "Calendar access was denied. Check that the Google Calendar API "
                "is enabled and that all permissions were granted."
            )
        if response.status_code in (404, 410):
            raise CalendarError("Event not found.")
        if response.status_code == 400:
            raise CalendarError("Google rejected those event details.")
        if response.status_code == 429:
            raise CalendarError("Calendar rate limit reached. Try again in a moment.")
        raise CalendarError("Google Calendar returned an unexpected error.")

    async def _get(
        self, client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return await self._request(client, "GET", path, params)

    async def _tz(self, client: httpx.AsyncClient) -> tuple[str, ZoneInfo]:
        calendar = await self._get(client, "/calendars/primary")
        tz_name = calendar.get("timeZone") or "UTC"
        try:
            return tz_name, ZoneInfo(tz_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise CalendarError("Could not determine the calendar's time zone.") from exc

    async def timezone_name(self) -> str:
        async with self._client() as client:
            return (await self._tz(client))[0]

    async def list_events(
        self, start: date, end: date, query: str | None, max_results: int
    ) -> dict[str, Any]:
        async with self._client() as client:
            tz_name, tz = await self._tz(client)

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

    # ------------------------------------------------------------ busy times

    async def busy_periods(self, start: date, end: date) -> dict[str, Any]:
        """Timed events that block time between two days, as aware datetimes."""
        async with self._client() as client:
            tz_name, tz = await self._tz(client)
            time_min, time_max = day_bounds(start, end, tz)
            data = await self._get(
                client,
                "/calendars/primary/events",
                {
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "maxResults": 250,
                    "timeZone": tz_name,
                },
            )

        busy: list[tuple[datetime, datetime]] = []
        all_day: list[dict[str, Any]] = []
        for event in data.get("items") or []:
            if event.get("status") == "cancelled" or event.get("transparency") == "transparent":
                continue
            me = next((a for a in event.get("attendees") or [] if a.get("self")), None)
            if me and me.get("responseStatus") == "declined":
                continue
            begin = event.get("start") or {}
            finish = event.get("end") or {}
            if begin.get("dateTime") and finish.get("dateTime"):
                busy.append(
                    (
                        datetime.fromisoformat(begin["dateTime"]).astimezone(tz),
                        datetime.fromisoformat(finish["dateTime"]).astimezone(tz),
                    )
                )
            else:
                all_day.append(summarize_event(event))
        return {
            "timezone": tz_name,
            "tz": tz,
            "busy": busy,
            "all_day": all_day,
            "truncated": bool(data.get("nextPageToken")),
        }

    # ---------------------------------------------------------------- writes

    @staticmethod
    def _timed(local: datetime, tz_name: str) -> dict[str, str]:
        return {"dateTime": local.replace(tzinfo=None).isoformat(), "timeZone": tz_name}

    async def create_event(
        self,
        *,
        title: str,
        start: datetime | date,
        end: datetime | date,
        location: str | None,
        description: str | None,
        attendees: list[str],
    ) -> dict[str, Any]:
        async with self._client() as client:
            tz_name = (await self._tz(client))[0]
            body: dict[str, Any] = {"summary": title}
            if isinstance(start, datetime) and isinstance(end, datetime):
                body["start"] = self._timed(start, tz_name)
                body["end"] = self._timed(end, tz_name)
            else:
                # all-day: Google's end date is exclusive
                body["start"] = {"date": start.isoformat()}
                body["end"] = {"date": (end + timedelta(days=1)).isoformat()}
            if location:
                body["location"] = location
            if description:
                body["description"] = description
            if attendees:
                body["attendees"] = [{"email": e} for e in attendees]
            event = await self._request(
                client,
                "POST",
                "/calendars/primary/events",
                {"sendUpdates": "all" if attendees else "none"},
                body,
            )
        return summarize_event(event)

    async def update_event(
        self,
        event_id: str,
        *,
        title: str | None,
        start: datetime | date | None,
        end: datetime | date | None,
        location: str | None,
        description: str | None,
    ) -> dict[str, Any]:
        async with self._client() as client:
            current = await self._get(client, f"/calendars/primary/events/{event_id}")
            tz_name = (await self._tz(client))[0]
            body: dict[str, Any] = {}
            if title is not None:
                body["summary"] = title
            if location is not None:
                body["location"] = location
            if description is not None:
                body["description"] = description
            if start is not None and end is not None:
                if isinstance(start, datetime) and isinstance(end, datetime):
                    body["start"] = self._timed(start, tz_name)
                    body["end"] = self._timed(end, tz_name)
                else:
                    body["start"] = {"date": start.isoformat()}
                    body["end"] = {"date": (end + timedelta(days=1)).isoformat()}
            notify = bool(current.get("attendees"))
            event = await self._request(
                client,
                "PATCH",
                f"/calendars/primary/events/{event_id}",
                {"sendUpdates": "all" if notify else "none"},
                body,
            )
        return summarize_event(event)

    async def delete_event(self, event_id: str) -> dict[str, Any]:
        async with self._client() as client:
            current = await self._get(client, f"/calendars/primary/events/{event_id}")
            notify = bool(current.get("attendees"))
            await self._request(
                client,
                "DELETE",
                f"/calendars/primary/events/{event_id}",
                {"sendUpdates": "all" if notify else "none"},
            )
        return summarize_event(current)

    async def respond(self, event_id: str, response: str) -> dict[str, Any]:
        async with self._client() as client:
            current = await self._get(client, f"/calendars/primary/events/{event_id}")
            attendees = current.get("attendees") or []
            if not any(a.get("self") for a in attendees):
                raise CalendarError("You are not a guest of this event, so there is nothing to answer.")
            updated = [
                {**a, "responseStatus": response} if a.get("self") else a for a in attendees
            ]
            event = await self._request(
                client,
                "PATCH",
                f"/calendars/primary/events/{event_id}",
                {"sendUpdates": "all"},
                {"attendees": updated},
            )
        return summarize_event(event)