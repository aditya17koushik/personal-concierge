"""Phase 15: reminders (rules, tools), the scheduler that delivers them, the morning agenda,
meeting alerts, and the Telegram commands that control them."""

import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database.models import EventAlert, Reminder
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.profiles import ProfileRepository
from app.database.repositories.reminders import ReminderRepository
from app.database.repositories.telegram import TelegramRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.oauth import GoogleOAuthService
from app.integrations.telegram.client import TelegramError
from app.profiles.service import ProfileError, ProfileService
from app.reminders.service import (
    ReminderError,
    ReminderService,
    aware,
    clean_text,
    next_occurrence,
    resolve_due,
)
from app.scheduler.agenda import build_agenda, safe
from app.scheduler.runner import SchedulerRunner
from app.security.crypto import encrypt
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

IST = "Asia/Kolkata"
NOW = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)  # 15:30 in Kolkata
TODAY = date(2026, 10, 17)


# ------------------------------------------------------------------ rules


def test_in_minutes_is_relative_to_now():
    due = resolve_due(at=None, in_minutes=30, tz_name=IST, now=NOW, max_days=366)
    assert due == NOW + timedelta(minutes=30)


def test_local_time_is_converted_to_utc():
    due = resolve_due(
        at=datetime(2026, 10, 17, 18, 0), in_minutes=None, tz_name=IST, now=NOW, max_days=366
    )
    assert due == datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc)  # 18:00 IST


@pytest.mark.parametrize(
    "kwargs",
    [
        {"at": datetime(2026, 10, 17, 15, 0), "in_minutes": None},  # 15:00 IST is already past
        {"at": None, "in_minutes": None},
        {"at": datetime(2026, 10, 20, 9, 0), "in_minutes": 5},
        {"at": datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc), "in_minutes": None},
        {"at": datetime(2030, 1, 1, 9, 0), "in_minutes": None},  # too far ahead
    ],
)
def test_bad_times_are_rejected(kwargs: dict[str, Any]):
    with pytest.raises(ReminderError):
        resolve_due(tz_name=IST, now=NOW, max_days=366, **kwargs)


def test_text_is_cleaned():
    assert clean_text("  call\nRavi \t now \x07 ") == "call Ravi now"
    assert len(clean_text("x" * 1000)) == 300
    with pytest.raises(ReminderError):
        clean_text("  \n ")


def test_daily_repeat_keeps_local_time():
    due = datetime(2026, 10, 17, 2, 30, tzinfo=timezone.utc)  # 08:00 IST
    nxt = next_occurrence(due, "daily", IST, NOW)
    assert nxt == datetime(2026, 10, 18, 2, 30, tzinfo=timezone.utc)


def test_repeat_skips_missed_occurrences():
    due = datetime(2026, 10, 10, 2, 30, tzinfo=timezone.utc)  # a week ago
    nxt = next_occurrence(due, "daily", IST, NOW)
    assert nxt > NOW and nxt - NOW <= timedelta(days=1)


def test_weekdays_skip_the_weekend():
    friday = datetime(2026, 10, 16, 3, 30, tzinfo=timezone.utc)  # Fri 09:00 IST
    nxt = next_occurrence(friday, "weekdays", IST, datetime(2026, 10, 16, 4, 0, tzinfo=timezone.utc))
    assert nxt.astimezone(timezone.utc).weekday() == 0  # Monday


def test_weekly_repeat():
    due = datetime(2026, 10, 17, 2, 30, tzinfo=timezone.utc)
    assert next_occurrence(due, "weekly", IST, NOW) == due + timedelta(days=7)


def test_daily_reminder_stays_at_the_same_local_hour_across_daylight_saving():
    # New York clocks go forward on 2027-03-14: 08:00 local is 13:00 UTC before, 12:00 after.
    before = datetime(2027, 3, 13, 13, 0, tzinfo=timezone.utc)
    nxt = next_occurrence(before, "daily", "America/New_York", before + timedelta(minutes=1))
    assert nxt == datetime(2027, 3, 14, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ tools


def tool_ctx(db: Session, user_id: int, tz: str = IST) -> ToolContext:
    return ToolContext(
        db=db, user_id=user_id, today=TODAY, default_currency="INR", timezone=tz, now=NOW
    )


def make_user(db: Session, external_id: str = "me") -> int:
    return UserRepository(db).get_or_create(external_id).id


def test_reminder_tools_need_no_approval_and_have_their_own_domain():
    registry = build_default_registry()
    for name in ("create_reminder", "list_reminders", "cancel_reminder"):
        tool = registry.get(name)
        assert tool is not None and tool.domain == "reminder" and not tool.requires_approval


@pytest.mark.asyncio
async def test_create_list_and_cancel_through_the_registry(db: Session):
    uid = make_user(db)
    ctx = tool_ctx(db, uid)
    registry = build_default_registry()

    made = await registry.execute(
        "create_reminder", {"text": "Call Ravi", "at": "2026-10-17T18:00"}, ctx
    )
    assert made["created"]["when"] == "Sat 17 Oct, 18:00"
    assert made["created"]["timezone"] == IST
    stored = db.query(Reminder).one()
    assert aware(stored.due_at) == datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc)
    assert stored.status == "pending"

    listed = await registry.execute("list_reminders", {}, ctx)
    assert listed["count"] == 1 and listed["reminders"][0]["text"] == "Call Ravi"

    cancelled = await registry.execute("cancel_reminder", {"reminder_id": stored.id}, ctx)
    assert cancelled == {"cancelled": stored.id}
    assert (await registry.execute("list_reminders", {}, ctx))["count"] == 0
    again = await registry.execute("cancel_reminder", {"reminder_id": stored.id}, ctx)
    assert "not found" in again["error"].lower()


@pytest.mark.asyncio
async def test_in_minutes_and_repeat(db: Session):
    uid = make_user(db)
    result = await build_default_registry().execute(
        "create_reminder",
        {"text": "Stretch", "in_minutes": 90, "repeat": "weekdays"},
        tool_ctx(db, uid),
    )
    assert result["created"]["repeat"] == "weekdays"
    assert aware(db.query(Reminder).one().due_at) == NOW + timedelta(minutes=90)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"text": "x"},  # no time
        {"text": "x", "at": "2026-10-20T09:00", "in_minutes": 5},
        {"text": "x", "at": "2026-10-20T09:00:00+05:30"},
        {"text": "x", "in_minutes": 0},
        {"text": "x", "in_minutes": 5, "repeat": "hourly"},
        {"text": "", "in_minutes": 5},
    ],
)
async def test_bad_reminder_arguments(db: Session, args: dict[str, Any]):
    uid = make_user(db)
    result = await build_default_registry().execute("create_reminder", args, tool_ctx(db, uid))
    assert result["error"] == "Invalid arguments"
    assert db.query(Reminder).count() == 0


@pytest.mark.asyncio
async def test_a_reminder_in_the_past_is_refused_with_a_reason(db: Session):
    uid = make_user(db)
    result = await build_default_registry().execute(
        "create_reminder", {"text": "x", "at": "2026-10-17T09:00"}, tool_ctx(db, uid)
    )
    assert "already passed" in result["error"]


@pytest.mark.asyncio
async def test_pending_reminders_are_capped(db: Session, monkeypatch: pytest.MonkeyPatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "reminder_max_pending", 2)
    uid = make_user(db)
    ctx = tool_ctx(db, uid)
    registry = build_default_registry()
    for n in (1, 2):
        assert "created" in await registry.execute(
            "create_reminder", {"text": f"r{n}", "in_minutes": 10 * n}, ctx
        )
    third = await registry.execute("create_reminder", {"text": "r3", "in_minutes": 30}, ctx)
    assert "pending reminders" in third["error"]


@pytest.mark.asyncio
async def test_users_cannot_cancel_or_see_each_others_reminders(db: Session):
    alice, bob = make_user(db, "alice"), make_user(db, "bob")
    registry = build_default_registry()
    await registry.execute("create_reminder", {"text": "secret", "in_minutes": 10}, tool_ctx(db, alice))
    rid = db.query(Reminder).one().id

    assert (await registry.execute("list_reminders", {}, tool_ctx(db, bob)))["count"] == 0
    stolen = await registry.execute("cancel_reminder", {"reminder_id": rid}, tool_ctx(db, bob))
    assert "not found" in stolen["error"].lower()
    assert db.query(Reminder).one().status == "pending"


# -------------------------------------------------------------- scheduler


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.errors: list[TelegramError] = []  # raised one per send, in order

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    @property
    def texts(self) -> list[str]:
        return [t for _, t in self.sent]


class Clock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "sqlite://",
        "app_secret_key": "test-secret-key-not-for-production",
        "scheduler_tick_seconds": 1,
        "calendar_poll_seconds": 0,
    }
    values.update(overrides)
    return Settings(**values)


def link_telegram(factory: sessionmaker[Session], user_id: int, telegram_id: int = 555) -> None:
    with factory() as db:
        TelegramRepository(db).create_link(telegram_id, user_id, "Test")


def add_reminder(
    factory: sessionmaker[Session],
    user_id: int,
    due: datetime,
    text: str = "Call Ravi",
    repeat: str | None = None,
) -> int:
    with factory() as db:
        return ReminderRepository(db).add(user_id, text, due, IST, repeat, NOW).id


def reminder_row(factory: sessionmaker[Session], rid: int) -> Reminder:
    with factory() as db:
        row = db.get(Reminder, rid)
        assert row is not None
        db.expunge(row)
        return row


def runner_for(
    factory: sessionmaker[Session],
    tg: FakeTelegram,
    clock: Clock,
    handler: Any = None,
    **overrides: Any,
) -> SchedulerRunner:
    google = None
    if handler is not None:
        google = lambda db: GoogleOAuthService(  # noqa: E731
            db, settings(), transport=httpx.MockTransport(handler)
        )
    return SchedulerRunner(
        tg,  # type: ignore[arg-type]
        factory,
        settings(**overrides),
        google_factory=google,
        now_fn=clock,
    )


@pytest.fixture
def user(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as db:
        uid = make_user(db)
    link_telegram(session_factory, uid)
    return uid


@pytest.mark.asyncio
async def test_a_due_reminder_is_sent_once(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(seconds=30))
    tg = FakeTelegram()
    runner = runner_for(session_factory, tg, Clock())

    assert await runner.deliver_due_reminders() == 1
    assert await runner.deliver_due_reminders() == 0

    assert tg.sent == [(555, "Reminder: Call Ravi")]
    row = reminder_row(session_factory, rid)
    assert row.status == "sent" and row.sent_at is not None


@pytest.mark.asyncio
async def test_a_future_or_cancelled_reminder_is_not_sent(session_factory, user):
    add_reminder(session_factory, user, NOW + timedelta(minutes=10))
    cancelled = add_reminder(session_factory, user, NOW - timedelta(minutes=1), "nope")
    with session_factory() as db:
        ReminderRepository(db).cancel(user, cancelled, NOW)
    tg = FakeTelegram()

    assert await runner_for(session_factory, tg, Clock()).deliver_due_reminders() == 0
    assert tg.sent == []


@pytest.mark.asyncio
async def test_two_workers_never_double_send(session_factory, user):
    """The claim is one atomic UPDATE, so only one worker can win a reminder.

    (Tested sequentially: the in-memory SQLite test database shares one connection, so truly
    parallel threads would interleave on it. Postgres gives each worker its own connection.)"""
    rid = add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    with session_factory() as first, session_factory() as second:
        assert ReminderRepository(first).claim(rid, NOW) is True
        assert ReminderRepository(second).claim(rid, NOW) is False

    tg = FakeTelegram()
    # the winner is mid-delivery ("sending"), so another worker finds nothing to send
    assert await runner_for(session_factory, tg, Clock()).deliver_due_reminders() == 0
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_late_reminder_says_when_it_was_due(session_factory, user):
    add_reminder(session_factory, user, NOW - timedelta(hours=2))
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()
    assert "It was due Sat 17 Oct, 13:30" in tg.texts[0]


@pytest.mark.asyncio
async def test_a_very_late_reminder_is_dropped_not_sent(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(hours=30))
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()
    assert tg.sent == [] and reminder_row(session_factory, rid).status == "expired"


@pytest.mark.asyncio
async def test_a_repeating_reminder_is_rescheduled(session_factory, user):
    due = datetime(2026, 10, 17, 9, 55, tzinfo=timezone.utc)
    rid = add_reminder(session_factory, user, due, "Stretch", repeat="daily")
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()

    row = reminder_row(session_factory, rid)
    assert tg.texts == ["Reminder: Stretch"]
    assert row.status == "pending" and aware(row.due_at) == due + timedelta(days=1)


@pytest.mark.asyncio
async def test_a_missed_repeat_skips_ahead_without_a_flood(session_factory, user):
    due = NOW - timedelta(days=3)  # the bot was off for days
    rid = add_reminder(session_factory, user, due, "Stretch", repeat="daily")
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()

    assert tg.sent == []  # too late to be useful
    row = reminder_row(session_factory, rid)
    assert row.status == "pending" and aware(row.due_at) > NOW


@pytest.mark.asyncio
async def test_a_temporary_telegram_failure_is_retried_then_given_up(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    tg = FakeTelegram()
    clock = Clock()
    runner = runner_for(session_factory, tg, clock)

    for attempt, expected in enumerate(("pending", "pending", "failed"), start=1):
        tg.errors = [TelegramError("Telegram is down", status=502)]
        await runner.deliver_due_reminders()
        row = reminder_row(session_factory, rid)
        assert row.status == expected, f"after attempt {attempt}"
        clock.now = aware(row.due_at) + timedelta(seconds=1)  # jump to the retry time

    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_retry_that_succeeds_is_sent(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    tg = FakeTelegram()
    clock = Clock()
    runner = runner_for(session_factory, tg, clock)

    tg.errors = [TelegramError("flaky", status=502)]
    await runner.deliver_due_reminders()
    clock.now = aware(reminder_row(session_factory, rid).due_at) + timedelta(seconds=1)
    await runner.deliver_due_reminders()

    assert len(tg.sent) == 1 and reminder_row(session_factory, rid).status == "sent"


@pytest.mark.asyncio
async def test_a_blocked_bot_is_not_retried(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    tg = FakeTelegram()
    tg.errors = [TelegramError("Forbidden: bot was blocked by the user", status=403)]
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()

    row = reminder_row(session_factory, rid)
    assert row.status == "failed" and "blocked" in (row.last_error or "")


@pytest.mark.asyncio
async def test_a_user_without_telegram_fails_cleanly(session_factory):
    with session_factory() as db:
        uid = make_user(db, "web-only")
    rid = add_reminder(session_factory, uid, NOW - timedelta(seconds=5))
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()

    assert tg.sent == []
    assert reminder_row(session_factory, rid).status == "failed"


@pytest.mark.asyncio
async def test_a_reminder_stuck_in_sending_is_recovered(session_factory, user):
    rid = add_reminder(session_factory, user, NOW - timedelta(minutes=10))
    with session_factory() as db:
        repo = ReminderRepository(db)
        assert repo.claim(rid, NOW - timedelta(minutes=9))  # then the process "died"
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()
    assert len(tg.sent) == 1


@pytest.mark.asyncio
async def test_reminder_text_from_the_user_is_sent_as_plain_text(session_factory, user):
    add_reminder(session_factory, user, NOW - timedelta(seconds=5), "buy milk\nand eggs")
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock()).deliver_due_reminders()
    assert "\n" not in tg.texts[0].split("\n(It was")[0]


@pytest.mark.asyncio
async def test_the_loop_runs_ticks_and_stops_cleanly(session_factory, user):
    add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    tg = FakeTelegram()
    runner = runner_for(session_factory, tg, Clock(), scheduler_tick_seconds=0.05)
    stop = asyncio.Event()

    task = asyncio.create_task(runner.run(stop))
    await asyncio.sleep(0.3)
    stop.set()
    await asyncio.wait_for(task, timeout=2)

    assert len(tg.sent) == 1


# ------------------------------------------------- agenda and meeting alerts

MEETING = {
    "id": "evt1",
    "summary": "Design review",
    "status": "confirmed",
    "location": "Room 4",
    "start": {"dateTime": "2026-10-17T11:00:00+05:30"},  # today 11:00 IST
    "end": {"dateTime": "2026-10-17T12:00:00+05:30"},
}


def calendar_api(events: list[dict[str, Any]], seen: list[httpx.Request] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/calendar/v3/calendars/primary":
            return httpx.Response(200, json={"timeZone": IST})
        if request.url.path == "/calendar/v3/calendars/primary/events":
            return httpx.Response(200, json={"items": events})
        return httpx.Response(500)

    return handler


def connect_google(factory: sessionmaker[Session], user_id: int) -> None:
    with factory() as db:
        GoogleCredentialRepository(db).upsert(
            user_id=user_id,
            google_email="me@example.com",
            access_token_enc=encrypt("tok"),
            refresh_token_enc=encrypt("ref"),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scopes="x",
        )


def set_nudges(factory: sessionmaker[Session], user_id: int, **kwargs: Any) -> None:
    with factory() as db:
        service = ProfileService(db, "INR", IST)
        service.update(user_id, timezone=IST)
        if "agenda" in kwargs:
            service.set_agenda(user_id, True, kwargs["agenda"])
        if "alerts" in kwargs:
            service.set_event_alerts(user_id, kwargs["alerts"])


AGENDA_TIME = datetime(2026, 10, 17, 2, 35, tzinfo=timezone.utc)  # 08:05 IST


@pytest.mark.asyncio
async def test_morning_agenda_lists_events_and_reminders_once_a_day(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, agenda="08:00")
    add_reminder(session_factory, user, datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc))
    tg = FakeTelegram()
    runner = runner_for(session_factory, tg, Clock(AGENDA_TIME), calendar_api([MEETING]))

    await runner.calendar_pass()
    await runner.calendar_pass()  # a second poll must not repeat it

    assert len(tg.sent) == 1
    text = tg.texts[0]
    assert "Saturday 17 October" in text
    assert "11:00-12:00  Design review (Room 4)" in text
    assert "18:00  Call Ravi" in text


@pytest.mark.asyncio
async def test_agenda_is_not_sent_before_its_time_or_long_after(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, agenda="08:00")
    tg = FakeTelegram()

    early = Clock(datetime(2026, 10, 17, 2, 0, tzinfo=timezone.utc))  # 07:30 IST
    await runner_for(session_factory, tg, early, calendar_api([])).calendar_pass()
    late = Clock(datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc))  # 11:30 IST, past the window
    await runner_for(session_factory, tg, late, calendar_api([])).calendar_pass()

    assert tg.sent == []


@pytest.mark.asyncio
async def test_agenda_without_google_still_lists_reminders(session_factory, user):
    set_nudges(session_factory, user, agenda="08:00")
    add_reminder(session_factory, user, datetime(2026, 10, 17, 12, 30, tzinfo=timezone.utc))
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock(AGENDA_TIME), calendar_api([])).calendar_pass()

    assert "not available" in tg.texts[0] and "Call Ravi" in tg.texts[0]


@pytest.mark.asyncio
async def test_a_failed_agenda_send_is_tried_again(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, agenda="08:00")
    tg = FakeTelegram()
    runner = runner_for(session_factory, tg, Clock(AGENDA_TIME), calendar_api([]))

    tg.errors = [TelegramError("down", status=502)]
    await runner.calendar_pass()
    assert tg.sent == []
    await runner.calendar_pass()
    assert len(tg.sent) == 1


@pytest.mark.asyncio
async def test_agenda_is_off_unless_the_user_turns_it_on(session_factory, user):
    connect_google(session_factory, user)
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock(AGENDA_TIME), calendar_api([MEETING])).calendar_pass()
    assert tg.sent == []


def soon(minutes: int, **extra: Any) -> dict[str, Any]:
    start = (NOW + timedelta(minutes=minutes)).astimezone(timezone(timedelta(hours=5, minutes=30)))
    return {
        "id": f"soon{minutes}",
        "summary": "Standup",
        "status": "confirmed",
        "start": {"dateTime": start.isoformat()},
        "end": {"dateTime": (start + timedelta(minutes=30)).isoformat()},
        **extra,
    }


@pytest.mark.asyncio
async def test_meeting_alert_is_sent_once(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, alerts=15)
    tg = FakeTelegram()
    runner = runner_for(
        session_factory,
        tg,
        Clock(),
        calendar_api([soon(10, hangoutLink="https://meet.example/abc", location="Room 2")]),
    )

    await runner.calendar_pass()
    await runner.calendar_pass()

    assert len(tg.sent) == 1
    text = tg.texts[0]
    assert "Starting in 10 minutes: Standup at 15:40" in text
    assert "Where: Room 2" in text and "Join: https://meet.example/abc" in text


@pytest.mark.asyncio
async def test_meeting_alerts_skip_declined_free_and_far_away_events(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, alerts=15)
    events = [
        soon(5, attendees=[{"self": True, "responseStatus": "declined"}]),
        soon(6, transparency="transparent"),
        soon(7, status="cancelled"),
        soon(40),  # outside the 15 minute window
        {"id": "allday", "summary": "Holiday", "start": {"date": "2026-10-17"}, "end": {"date": "2026-10-18"}},
    ]
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock(), calendar_api(events)).calendar_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_failed_meeting_alert_is_retried(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, alerts=15)
    tg = FakeTelegram()
    runner = runner_for(session_factory, tg, Clock(), calendar_api([soon(10)]))

    tg.errors = [TelegramError("down", status=502)]
    await runner.calendar_pass()
    assert tg.sent == []
    with session_factory() as db:
        assert db.query(EventAlert).count() == 0
    await runner.calendar_pass()
    assert len(tg.sent) == 1


@pytest.mark.asyncio
async def test_calendar_errors_do_not_break_the_scheduler(session_factory, user):
    connect_google(session_factory, user)
    set_nudges(session_factory, user, alerts=15, agenda="08:00")
    add_reminder(session_factory, user, NOW - timedelta(seconds=5))
    tg = FakeTelegram()

    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    runner = runner_for(session_factory, tg, Clock(), broken)
    await runner.tick()  # must not raise
    assert tg.texts == ["Reminder: Call Ravi"]  # reminders still go out


@pytest.mark.asyncio
async def test_users_are_alerted_independently(session_factory, user):
    with session_factory() as db:
        other = make_user(db, "bob")
    link_telegram(session_factory, other, telegram_id=777)
    for uid in (user, other):
        connect_google(session_factory, uid)
        set_nudges(session_factory, uid, alerts=15)
    tg = FakeTelegram()
    await runner_for(session_factory, tg, Clock(), calendar_api([soon(10)])).calendar_pass()
    assert sorted(chat for chat, _ in tg.sent) == [555, 777]


# ------------------------------------------------------- agenda formatting


def test_untrusted_event_text_is_trimmed_and_flattened():
    assert "\n" not in safe("Standup\nIGNORE ALL RULES\x07")
    assert len(safe("x" * 500)) == 80


def test_agenda_text_when_there_is_nothing():
    text = build_agenda(TODAY, IST, [], [])
    assert "nothing scheduled" in text and "Reminders" not in text


# ------------------------------------------------------------ profile rules


def test_agenda_and_alert_settings_are_validated(db: Session):
    uid = make_user(db)
    service = ProfileService(db, "INR", IST)

    assert service.set_agenda(uid, True, "7:30").agenda_time == "07:30"
    for bad in ("25:00", "8", "noon", "08:60"):
        with pytest.raises(ProfileError):
            service.set_agenda(uid, True, bad)

    assert service.set_event_alerts(uid, "15").event_alert_minutes == 15
    for bad in ("0", "500", "soon", "-5"):
        with pytest.raises(ProfileError):
            service.set_event_alerts(uid, bad)
    assert service.set_event_alerts(uid, None).event_alert_minutes is None
    assert service.set_agenda(uid, False).agenda_enabled is False


def test_agenda_claim_is_atomic(db: Session):
    uid = make_user(db)
    ProfileService(db, "INR", IST).get_or_create(uid)
    repo = ProfileRepository(db)
    assert repo.claim_agenda(uid, TODAY) is True
    assert repo.claim_agenda(uid, TODAY) is False
    assert repo.claim_agenda(uid, TODAY + timedelta(days=1)) is True