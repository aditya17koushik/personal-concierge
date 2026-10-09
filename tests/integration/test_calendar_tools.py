from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import httpx
import pytest
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.config import get_settings
from app.database.models import User
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.base import LLMProvider
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.security.crypto import encrypt
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 9)
Handler = Callable[[httpx.Request], httpx.Response]


def connect_user(db: Session, external_id: str = "me") -> User:
    user = UserRepository(db).get_or_create(external_id)
    GoogleCredentialRepository(db).upsert(
        user_id=user.id,
        google_email="me@example.com",
        access_token_enc=encrypt("tok"),
        refresh_token_enc=encrypt("ref"),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        scopes="x",
    )
    return user


def make_service(db: Session, handler: Handler) -> GoogleOAuthService:
    return GoogleOAuthService(db, get_settings(), transport=httpx.MockTransport(handler))


def make_ctx(db: Session, user: User, handler: Handler) -> ToolContext:
    return ToolContext(
        db=db,
        user_id=user.id,
        today=TODAY,
        default_currency="INR",
        external_user_id=user.external_id,
        google_oauth=make_service(db, handler),
    )


def calendar_handler(
    seen: list[httpx.Request],
    events: list[dict[str, Any]] | None = None,
    single: dict[str, Any] | None = None,
    next_page: bool = False,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == "Bearer tok"
        path = request.url.path
        if path == "/calendar/v3/calendars/primary":
            return httpx.Response(200, json={"timeZone": "Asia/Kolkata"})
        if path == "/calendar/v3/calendars/primary/events":
            body: dict[str, Any] = {"items": events or []}
            if next_page:
                body["nextPageToken"] = "more"
            return httpx.Response(200, json=body)
        if path.startswith("/calendar/v3/calendars/primary/events/"):
            if single is None:
                return httpx.Response(404, json={})
            return httpx.Response(200, json=single)
        return httpx.Response(500)

    return handler


TIMED = {
    "id": "e1",
    "summary": "Standup",
    "status": "confirmed",
    "start": {"dateTime": "2026-10-10T09:30:00+05:30"},
    "end": {"dateTime": "2026-10-10T10:00:00+05:30"},
    "hangoutLink": "https://meet.google.com/abc-defg-hij",
    "recurringEventId": "series1",
    "location": "Room 4",
}
ALL_DAY = {
    "id": "e2",
    "summary": "Offsite",
    "start": {"date": "2026-10-12"},
    "end": {"date": "2026-10-14"},  # exclusive in Google's API
}


@pytest.mark.asyncio
async def test_list_events_default_range_uses_calendar_timezone(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, calendar_handler(seen, events=[TIMED]))

    result = await build_default_registry().execute("list_events", {}, ctx)

    assert result["start_date"] == "2026-10-09"
    assert result["end_date"] == "2026-10-16"  # +7 days
    assert result["timezone"] == "Asia/Kolkata"
    assert result["count"] == 1

    params = seen[-1].url.params
    assert params["timeMin"] == "2026-10-09T00:00:00+05:30"
    assert params["timeMax"] == "2026-10-17T00:00:00+05:30"  # end day inclusive
    assert params["singleEvents"] == "true"
    assert params["orderBy"] == "startTime"
    assert params["timeZone"] == "Asia/Kolkata"
    assert "q" not in params


@pytest.mark.asyncio
async def test_event_summaries(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, calendar_handler([], events=[TIMED, ALL_DAY]))

    result = await build_default_registry().execute("list_events", {}, ctx)
    timed, all_day = result["events"]

    assert timed["title"] == "Standup"
    assert timed["all_day"] is False
    assert timed["meeting_link"] == "https://meet.google.com/abc-defg-hij"
    assert timed["recurring"] is True
    assert timed["location"] == "Room 4"

    assert all_day["all_day"] is True
    assert all_day["start"] == "2026-10-12"
    assert all_day["end"] == "2026-10-13"  # converted to inclusive
    assert "untrusted" in result["note"]


@pytest.mark.asyncio
async def test_explicit_range_query_and_truncation(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, calendar_handler(seen, events=[TIMED], next_page=True))

    result = await build_default_registry().execute(
        "list_events",
        {"start_date": "2026-11-01", "end_date": "2026-11-01", "query": "standup", "max_results": 5},
        ctx,
    )

    params = seen[-1].url.params
    assert params["timeMin"] == "2026-11-01T00:00:00+05:30"
    assert params["timeMax"] == "2026-11-02T00:00:00+05:30"
    assert params["q"] == "standup"
    assert params["maxResults"] == "5"
    assert result["more_results_exist"] is True


@pytest.mark.asyncio
async def test_empty_calendar(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, calendar_handler([], events=[]))

    result = await build_default_registry().execute("list_events", {}, ctx)

    assert result["count"] == 0 and result["events"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"start_date": "2026-10-10", "end_date": "2026-10-01"},
        {"start_date": "2026-01-01", "end_date": "2028-01-01"},
        {"end_date": "2026-10-01"},  # before the default start (today)
    ],
)
async def test_invalid_ranges_return_errors_without_calling_google(
    db: Session, args: dict[str, str]
):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, calendar_handler(seen))

    result = await build_default_registry().execute("list_events", args, ctx)

    assert "error" in result
    assert seen == []


@pytest.mark.asyncio
async def test_get_event_details(db: Session):
    event = {
        **TIMED,
        "description": "<p>Agenda:</p><ul><li>Roadmap</li></ul><script>x()</script>",
        "organizer": {"email": "boss@example.com"},
        "attendees": [
            {"email": "me@example.com", "self": True, "responseStatus": "accepted"},
            {"email": "a@example.com", "displayName": "Asha", "responseStatus": "needsAction", "optional": True},
        ],
    }
    user = connect_user(db)
    ctx = make_ctx(db, user, calendar_handler([], single=event))

    result = await build_default_registry().execute("get_event", {"event_id": "e1"}, ctx)

    ev = result["event"]
    assert "Agenda:" in ev["description"] and "Roadmap" in ev["description"]
    assert "x()" not in ev["description"]
    assert ev["organizer"] == "boss@example.com"
    assert ev["attendee_count"] == 2
    assert ev["attendees"][0]["is_me"] is True
    assert ev["attendees"][1]["name"] == "Asha"
    assert ev["attendees"][1]["optional"] is True
    assert "untrusted" in result["note"]


@pytest.mark.asyncio
async def test_get_event_not_found(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, calendar_handler([], single=None))

    result = await build_default_registry().execute("get_event", {"event_id": "nope"}, ctx)

    assert result == {"error": "Event not found."}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../x", "a b", "e1?alt=media", ""])
async def test_event_id_is_validated(db: Session, bad_id: str):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, calendar_handler(seen, single=TIMED))

    result = await build_default_registry().execute("get_event", {"event_id": bad_id}, ctx)

    assert result["error"] == "Invalid arguments"
    assert seen == []


@pytest.mark.asyncio
async def test_not_connected_user(db: Session):
    user = UserRepository(db).get_or_create("fresh")
    ctx = make_ctx(db, user, calendar_handler([]))

    result = await build_default_registry().execute("list_events", {}, ctx)

    assert "not connected" in result["error"]
    assert "/auth/google/login?user_id=fresh" in result["action"]


@pytest.mark.asyncio
async def test_forbidden_returns_safe_error(db: Session):
    user = connect_user(db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "secret internals"}})

    ctx = make_ctx(db, user, handler)
    result = await build_default_registry().execute("list_events", {}, ctx)

    assert "Calendar API is enabled" in result["error"]
    assert "secret internals" not in str(result)


class ScriptedProvider(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


@pytest.mark.asyncio
async def test_agent_lists_events_and_prompt_mentions_calendar(db: Session):
    connect_user(db)
    service = make_service(db, calendar_handler([], events=[TIMED]))
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c1", name="list_events", arguments={})],
                model="scripted-model",
            ),
            LLMResponse(content="You have a standup tomorrow.", model="scripted-model"),
        ]
    )
    agent = AgentService(provider, db, today_fn=lambda: TODAY, google_oauth=service)

    result = await agent.chat("me", "what's on my calendar?")

    assert result.tools_used == ["list_events"]
    system_prompt = provider.calls[0][0].content or ""
    assert "Google Calendar" in system_prompt
    assert "calendar events" in system_prompt  # in the untrusted-content rule
    assert "Standup" in (provider.calls[1][-1].content or "")