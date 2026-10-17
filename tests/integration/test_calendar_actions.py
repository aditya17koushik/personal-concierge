"""Phase 14: calendar actions are approval-gated, and find_free_time is read-only."""

import json
from datetime import date
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.tools.registry import build_default_registry
from tests.integration.test_calendar_tools import connect_user, make_ctx

GATED = ["create_event", "update_event", "delete_event", "rsvp_event"]
MONDAY = date(2035, 3, 5)  # far in the future, so "now" never clamps the day

MEETING = {
    "id": "e1",
    "summary": "Design review",
    "status": "confirmed",
    "start": {"dateTime": "2035-03-06T15:00:00+05:30"},
    "end": {"dateTime": "2035-03-06T16:00:00+05:30"},
    "organizer": {"email": "boss@example.com"},
    "attendees": [
        {"email": "me@example.com", "self": True, "responseStatus": "needsAction"},
        {"email": "boss@example.com", "responseStatus": "accepted"},
    ],
}
SOLO = {**MEETING, "id": "e2", "summary": "Gym", "attendees": []}


def api(
    seen: list[httpx.Request],
    event: dict[str, Any] | None = MEETING,
    items: list[dict[str, Any]] | None = None,
    write_status: int = 200,
):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/calendar/v3/calendars/primary":
            return httpx.Response(200, json={"timeZone": "Asia/Kolkata"})
        if path == "/calendar/v3/calendars/primary/events":
            if request.method == "POST":
                if write_status != 200:
                    return httpx.Response(write_status, json={})
                body = json.loads(request.content)
                return httpx.Response(200, json={"id": "new1", **body})
            return httpx.Response(200, json={"items": items or []})
        if path.startswith("/calendar/v3/calendars/primary/events/"):
            if request.method == "GET":
                return httpx.Response(200, json=event) if event else httpx.Response(404, json={})
            if write_status != 200:
                return httpx.Response(write_status, json={})
            if request.method == "DELETE":
                return httpx.Response(204)
            body = json.loads(request.content)
            return httpx.Response(200, json={**(event or {}), **body})
        return httpx.Response(500)

    return handler


def writes(seen: list[httpx.Request]) -> list[httpx.Request]:
    return [r for r in seen if r.method != "GET"]


# ---------------------------------------------------------------- gating


def test_every_calendar_write_requires_approval_and_a_summary():
    registry = build_default_registry()
    for name in GATED:
        tool = registry.get(name)
        assert tool is not None and tool.requires_approval and tool.summarize
    free = registry.get("find_free_time")
    assert free is not None and not free.requires_approval and not free.side_effects


@pytest.mark.asyncio
async def test_preparing_never_writes(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    for name, args in [
        ("create_event", {"title": "Lunch", "start": "2035-03-07T13:00"}),
        ("update_event", {"event_id": "e1", "start": "2035-03-07T10:00"}),
        ("delete_event", {"event_id": "e1"}),
        ("rsvp_event", {"event_id": "e1", "response": "declined"}),
    ]:
        out = await registry.prepare(name, args, ctx)
        assert "summary" in out, (name, out)

    assert writes(seen) == []


# ------------------------------------------------------------ validation


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"title": "x"},  # no start
        {"title": "x", "start": "2035-03-07T13:00:00+05:30"},  # offset
        {"title": "x", "start": "2035-03-07T13:00", "end": "2035-03-07T12:00"},
        {"title": "x", "start": "2035-03-07T13:00", "end": "2035-03-09T13:00"},  # > 24h
        {"title": "x", "start": "2035-03-07T13:00", "guests": ["not-an-email"]},
        {"title": "x", "start": "2035-03-07T13:00", "guests": [f"a{i}@b.co" for i in range(11)]},
        {"title": "x", "all_day_date": "2035-03-07", "start": "2035-03-07T13:00"},
        {"title": "", "start": "2035-03-07T13:00"},
    ],
)
async def test_bad_create_arguments_are_rejected(db: Session, args: dict[str, Any]):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([]))
    out = await build_default_registry().prepare("create_event", args, ctx)
    assert out["error"] == "Invalid arguments"


@pytest.mark.asyncio
async def test_update_needs_a_change(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([]))
    out = await build_default_registry().prepare("update_event", {"event_id": "e1"}, ctx)
    assert out["error"] == "Invalid arguments"


@pytest.mark.asyncio
async def test_event_id_is_validated_for_writes(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    out = await build_default_registry().prepare("delete_event", {"event_id": "../x"}, ctx)
    assert out["error"] == "Invalid arguments"
    assert seen == []


# ---------------------------------------------------------------- create


@pytest.mark.asyncio
async def test_create_summary_and_request(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    prepared = await registry.prepare(
        "create_event",
        {
            "title": "Dinner",
            "start": "2035-03-07T19:30",
            "duration_minutes": 90,
            "location": "Home",
            "guests": ["Ravi@Example.com", "ravi@example.com"],
        },
        ctx,
    )
    assert "Dinner" in prepared["summary"]
    assert "19:30-21:00" in prepared["summary"]
    assert "Asia/Kolkata" in prepared["summary"]
    assert "ravi@example.com" in prepared["summary"] and "emailed" in prepared["summary"]
    assert prepared["arguments"]["guests"] == ["ravi@example.com"]  # lowercased, de-duplicated

    result = await registry.execute("create_event", prepared["arguments"], ctx)
    assert result["created"]["id"] == "new1"

    (post,) = writes(seen)
    assert post.method == "POST"
    assert post.url.params["sendUpdates"] == "all"
    body = json.loads(post.content)
    assert body["start"] == {"dateTime": "2035-03-07T19:30:00", "timeZone": "Asia/Kolkata"}
    assert body["end"] == {"dateTime": "2035-03-07T21:00:00", "timeZone": "Asia/Kolkata"}
    assert body["attendees"] == [{"email": "ravi@example.com"}]


@pytest.mark.asyncio
async def test_create_without_guests_sends_nobody_anything(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    result = await registry.execute("create_event", {"title": "Gym", "start": "2035-03-07T07:00"}, ctx)

    assert "created" in result
    (post,) = writes(seen)
    assert post.url.params["sendUpdates"] == "none"
    body = json.loads(post.content)
    assert "attendees" not in body
    assert body["end"]["dateTime"] == "2035-03-07T08:00:00"  # default 60 minutes


@pytest.mark.asyncio
async def test_create_all_day_event_uses_exclusive_end_date(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))

    await build_default_registry().execute(
        "create_event", {"title": "Holiday", "all_day_date": "2035-03-10"}, ctx
    )

    body = json.loads(writes(seen)[0].content)
    assert body["start"] == {"date": "2035-03-10"}
    assert body["end"] == {"date": "2035-03-11"}


@pytest.mark.asyncio
async def test_create_needs_the_new_google_permission(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([], write_status=403))

    result = await build_default_registry().execute(
        "create_event", {"title": "x", "start": "2035-03-07T07:00"}, ctx
    )

    assert "log in again" in result["error"]


# ---------------------------------------------------------------- update


@pytest.mark.asyncio
async def test_move_keeps_the_event_length_and_warns_about_guests(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    prepared = await registry.prepare(
        "update_event", {"event_id": "e1", "start": "2035-03-08T11:00"}, ctx
    )
    assert "Design review" in prepared["summary"]
    assert "Tue 06 Mar 2035, 15:00-16:00" in prepared["summary"]  # current time, readable
    assert "08 Mar 2035, 11:00-12:00" in prepared["summary"]  # new time
    assert "1 guest(s) will be emailed" in prepared["summary"]

    result = await registry.execute("update_event", prepared["arguments"], ctx)
    assert "updated" in result

    (patch,) = writes(seen)
    assert patch.method == "PATCH"
    assert patch.url.params["sendUpdates"] == "all"
    body = json.loads(patch.content)
    assert body["start"]["dateTime"] == "2035-03-08T11:00:00"
    assert body["end"]["dateTime"] == "2035-03-08T12:00:00"
    assert set(body) == {"start", "end"}  # nothing else is touched


@pytest.mark.asyncio
async def test_rename_a_personal_event_notifies_nobody(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen, event=SOLO))

    await build_default_registry().execute(
        "update_event", {"event_id": "e2", "title": "Leg day"}, ctx
    )

    (patch,) = writes(seen)
    assert patch.url.params["sendUpdates"] == "none"
    assert json.loads(patch.content) == {"summary": "Leg day"}


@pytest.mark.asyncio
async def test_update_unknown_event_fails_at_proposal_time(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([], event=None))
    out = await build_default_registry().prepare(
        "update_event", {"event_id": "nope", "title": "x"}, ctx
    )
    assert out == {"error": "Event not found."}


# ---------------------------------------------------------------- delete


@pytest.mark.asyncio
async def test_delete_summary_and_request(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    prepared = await registry.prepare("delete_event", {"event_id": "e1"}, ctx)
    assert "Delete" in prepared["summary"] and "Design review" in prepared["summary"]
    assert "cancellation" in prepared["summary"]
    assert writes(seen) == []

    result = await registry.execute("delete_event", prepared["arguments"], ctx)
    assert result["deleted"]["title"] == "Design review"
    (delete,) = writes(seen)
    assert delete.method == "DELETE" and delete.url.params["sendUpdates"] == "all"


@pytest.mark.asyncio
async def test_delete_already_gone_is_reported(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([], event=None))
    result = await build_default_registry().execute("delete_event", {"event_id": "e1"}, ctx)
    assert result == {"error": "Event not found."}


# ------------------------------------------------------------------ rsvp


@pytest.mark.asyncio
async def test_rsvp_changes_only_my_answer(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, api(seen))
    registry = build_default_registry()

    prepared = await registry.prepare(
        "rsvp_event", {"event_id": "e1", "response": "declined"}, ctx
    )
    assert prepared["summary"].startswith("Decline")
    assert "boss@example.com" in prepared["summary"]

    result = await registry.execute("rsvp_event", prepared["arguments"], ctx)
    assert result["responded"] == "declined"

    (patch,) = writes(seen)
    sent = json.loads(patch.content)["attendees"]
    assert sent[0]["responseStatus"] == "declined" and sent[0]["email"] == "me@example.com"
    assert sent[1]["responseStatus"] == "accepted"  # other guests untouched


@pytest.mark.asyncio
async def test_rsvp_to_an_event_i_am_not_invited_to(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([], event=SOLO))
    out = await build_default_registry().prepare(
        "rsvp_event", {"event_id": "e2", "response": "accepted"}, ctx
    )
    assert "not a guest" in out["error"]


@pytest.mark.asyncio
async def test_rsvp_rejects_unknown_answers(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([]))
    out = await build_default_registry().prepare(
        "rsvp_event", {"event_id": "e1", "response": "maybe-later"}, ctx
    )
    assert out["error"] == "Invalid arguments"


# ------------------------------------------------------------- free time


def timed(start: str, end: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": "x",
        "summary": "busy",
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        **extra,
    }


@pytest.mark.asyncio
async def test_free_time_finds_gaps_inside_working_hours(db: Session):
    user = connect_user(db)
    items = [
        timed("2035-03-05T10:00:00+05:30", "2035-03-05T11:00:00+05:30"),
        timed("2035-03-05T10:30:00+05:30", "2035-03-05T12:00:00+05:30"),  # overlaps
        timed("2035-03-05T14:00:00+05:30", "2035-03-05T17:30:00+05:30"),
        timed("2035-03-05T15:00:00+05:30", "2035-03-05T16:00:00+05:30", transparency="transparent"),
        timed(
            "2035-03-05T12:30:00+05:30",
            "2035-03-05T13:30:00+05:30",
            attendees=[{"self": True, "responseStatus": "declined"}],
        ),
    ]
    ctx = make_ctx(db, user, api([], items=items))

    result = await build_default_registry().execute(
        "find_free_time",
        {
            "duration_minutes": 60,
            "start_date": MONDAY.isoformat(),
            "end_date": MONDAY.isoformat(),
        },
        ctx,
    )

    windows = [(w["start"][11:16], w["end"][11:16]) for w in result["free_windows"]]
    assert windows == [("09:00", "10:00"), ("12:00", "14:00")]  # 17:30-18:00 is too short


@pytest.mark.asyncio
async def test_free_time_skips_weekends_unless_asked(db: Session):
    user = connect_user(db)
    saturday = date(2035, 3, 10)
    ctx = make_ctx(db, user, api([]))
    registry = build_default_registry()
    base = {"start_date": saturday.isoformat(), "end_date": saturday.isoformat()}

    assert (await registry.execute("find_free_time", base, ctx))["free_windows"] == []
    weekend = await registry.execute("find_free_time", {**base, "include_weekends": True}, ctx)
    assert len(weekend["free_windows"]) == 1


@pytest.mark.asyncio
async def test_free_time_validates_its_arguments(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, api([]))
    registry = build_default_registry()

    bad = await registry.execute("find_free_time", {"day_start": "18:00", "day_end": "09:00"}, ctx)
    assert bad["error"] == "Invalid arguments"
    huge = await registry.execute(
        "find_free_time", {"start_date": "2035-01-01", "end_date": "2035-12-31"}, ctx
    )
    assert "too large" in huge["error"]


# ------------------------------------------- proposals during ordinary tool use


class FixedJev:
    """Jev scored the message as LOW risk (use_tools), as can happen for 'reschedule'
    or for a bare 'yes'. Uses the real policy to build the decision."""

    def __init__(self, intents: list[str]) -> None:
        self.intents = intents

    async def decide(self, message: str, registry: Any, context: Any = None) -> Any:
        from app.decisions.policy import apply_policy
        from app.decisions.schemas import JevOutput

        return apply_policy(
            JevOutput.model_validate(
                {"intents": self.intents, "action": "use_tools", "risk": "low"}
            ),
            registry,
        )


class Script:
    """Scripted LLM: records which tools it was offered, replays canned replies."""

    name = "scripted"
    model = "scripted-model"

    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.offered: list[set[str]] = []

    async def chat(self, messages: Any, tools: Any = None, **_: Any) -> Any:
        self.offered.append({t["function"]["name"] for t in tools or []})
        return self.replies[min(len(self.offered) - 1, len(self.replies) - 1)]


def make_agent(db: Session, seen: list[httpx.Request], llm: Script, jev: Any):
    from app.agent.service import AgentService
    from app.approvals.service import ApprovalService
    from tests.integration.test_calendar_tools import TODAY, make_service

    service = make_service(db, api(seen))
    registry = build_default_registry()
    return AgentService(
        llm,  # type: ignore[arg-type]
        db,
        jev=jev,
        registry=registry,
        approvals=ApprovalService(db, registry, google_oauth=service),
        google_oauth=service,
        today_fn=lambda: TODAY,
    )


@pytest.mark.asyncio
async def test_a_low_risk_reschedule_still_produces_an_approval_not_a_dead_end(db: Session):
    from app.schemas.llm import LLMResponse, ToolCall

    connect_user(db)
    seen: list[httpx.Request] = []
    llm = Script(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="update_event",
                        arguments={"event_id": "e1", "start": "2035-03-06T23:00"},
                    )
                ],
                model="scripted-model",
            ),
            LLMResponse(content="Prepared. Please approve.", model="scripted-model"),
        ]
    )
    agent = make_agent(db, seen, llm, FixedJev(["calendar"]))

    result = await agent.chat("me", "Yep")

    assert {"update_event", "create_event", "delete_event", "rsvp_event"} <= llm.offered[0]
    assert "list_events" in llm.offered[0]
    assert [p.tool for p in result.pending_approvals] == ["update_event"]
    assert result.tools_used == []  # nothing ran
    assert writes(seen) == []  # Google was never changed


@pytest.mark.asyncio
async def test_expense_deletion_is_not_offered_during_ordinary_tool_use(db: Session):
    from app.schemas.llm import LLMResponse

    connect_user(db)
    llm = Script([LLMResponse(content="ok", model="scripted-model")])
    agent = make_agent(db, [], llm, FixedJev(["expense"]))

    await agent.chat("me", "I spent 250 on lunch")

    assert "delete_expense" not in llm.offered[0]
    assert not {"update_event", "create_event"} & llm.offered[0]