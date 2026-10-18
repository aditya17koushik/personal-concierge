"""Delivers reminders, the morning agenda and meeting alerts through Telegram.

Runs inside the bot process (`python -m app.telegram`), next to the polling loop. Everything
is at-least-once with an atomic claim, so a second worker or a restart can never double-send
a reminder:

  pending --claim--> sending --sent--> sent            (or back to pending with the next time
                          \\--failed--> pending/failed   for a repeating reminder)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypeVar
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.observability import get_tracer
from app.config import Settings
from app.coach.service import CoachService
from app.database.repositories.coach import CoachSettingsRepository
from app.database.repositories.profiles import ProfileRepository
from app.database.repositories.reminders import ReminderRepository
from app.database.repositories.tracking import TrackingRepository
from app.database.repositories.telegram import TelegramRepository
from app.integrations.google.calendar import CalendarClient, CalendarError
from app.integrations.google.oauth import (
    GoogleAuthError,
    GoogleNotConnectedError,
    GoogleOAuthService,
)
from app.integrations.telegram.client import TelegramAPI, TelegramError
from app.reminders.service import aware, format_local, next_occurrence
from app.tracking.service import TrackingService
from app.scheduler.agenda import build_agenda, build_event_alert, safe

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_ATTEMPTS = 3
RETRY_MINUTES = (1, 5)  # waits before attempt 2 and 3
LATE_NOTE_AFTER = timedelta(minutes=5)
ALERT_RETENTION = timedelta(days=2)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DueReminder:
    id: int
    user_id: int
    text: str
    due_at: datetime
    timezone: str
    repeat: str | None
    attempts: int


@dataclass(frozen=True)
class NudgeUser:
    user_id: int
    chat_id: int | None
    timezone: str
    agenda_enabled: bool
    agenda_time: str
    last_agenda_on: date | None
    event_alert_minutes: int | None


class SchedulerRunner:
    def __init__(
        self,
        client: TelegramAPI,
        session_factory: Callable[[], Session],
        settings: Settings,
        *,
        google_factory: Callable[[Session], GoogleOAuthService] | None = None,
        now_fn: Callable[[], datetime] = _utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._session_factory = session_factory
        self._settings = settings
        self._google_factory = google_factory or (lambda db: GoogleOAuthService(db, settings))
        self._now = now_fn
        self._sleep = sleep
        self._last_calendar_poll: datetime | None = None
        self._last_coach_poll: datetime | None = None

    # ------------------------------------------------------------------ loop

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        logger.info("Scheduler started")
        while not stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler tick failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._settings.scheduler_tick_seconds)
            except asyncio.TimeoutError:
                pass
        logger.info("Scheduler stopped")

    async def tick(self) -> None:
        """One pass: due reminders every tick, calendar work every `calendar_poll_seconds`."""
        await self.deliver_due_reminders()

        now = self._now()
        poll = timedelta(seconds=self._settings.calendar_poll_seconds)
        if self._last_calendar_poll is None or now - self._last_calendar_poll >= poll:
            self._last_calendar_poll = now
            await self.calendar_pass()

        coach_poll = timedelta(seconds=self._settings.coach_poll_seconds)
        if self._last_coach_poll is None or now - self._last_coach_poll >= coach_poll:
            self._last_coach_poll = now
            await self.coach_pass()
            await self.tracking_pass()

    async def _db(self, fn: Callable[[Session], T]) -> T:
        def run() -> T:
            with self._session_factory() as db:
                return fn(db)

        return await asyncio.to_thread(run)

    # ------------------------------------------------------------- reminders

    async def deliver_due_reminders(self) -> int:
        now = self._now()

        def claim_all(db: Session) -> list[DueReminder]:
            repo = ReminderRepository(db)
            repo.release_stale(now)
            claimed: list[DueReminder] = []
            for r in repo.due(now):
                # Read everything BEFORE claim(): its commit expires the row, and a reload
                # afterwards would already include the attempt the claim just counted.
                snapshot = DueReminder(
                    r.id, r.user_id, r.text, aware(r.due_at), r.timezone, r.repeat,
                    r.attempts + 1,
                )
                if repo.claim(r.id, now):
                    claimed.append(snapshot)
            return claimed

        claimed = await self._db(claim_all)
        for item in claimed:
            await self._deliver(item, now)
        return len(claimed)

    async def _deliver(self, item: DueReminder, now: datetime) -> None:
        late = now - item.due_at
        next_due = (
            next_occurrence(item.due_at, item.repeat, item.timezone, now) if item.repeat else None
        )

        if late > timedelta(hours=self._settings.reminder_late_hours):
            logger.info("reminder %s expired (late by %s)", item.id, late)
            await self._db(lambda db: ReminderRepository(db).mark_expired(item.id, now, next_due))
            return

        chat_id = await self._db(
            lambda db: (
                link.telegram_user_id
                if (link := TelegramRepository(db).get_link_for_user(item.user_id))
                else None
            )
        )
        if chat_id is None:
            await self._fail(item, now, "user has no Telegram link", retry=False)
            return

        text = f"Reminder: {safe(item.text, 300)}"
        if late > LATE_NOTE_AFTER:
            text += f"\n(It was due {format_local(item.due_at, item.timezone)}.)"

        try:
            await self._client.send_message(chat_id, text)
            get_tracer().event("scheduler.reminder_sent", metadata={"user": f"u{item.user_id}", "late_seconds": int(late.total_seconds())})
        except TelegramError as exc:
            # 400/403: the chat is gone or the bot was blocked. Retrying will not help.
            permanent = exc.status in (400, 403)
            await self._fail(item, now, str(exc), retry=not permanent)
            return

        await self._db(lambda db: ReminderRepository(db).mark_sent(item.id, now, next_due))
        logger.info("reminder %s sent", item.id)

    async def _fail(self, item: DueReminder, now: datetime, error: str, retry: bool) -> None:
        retry_at: datetime | None = None
        if retry and item.attempts < MAX_ATTEMPTS:
            wait = RETRY_MINUTES[min(item.attempts - 1, len(RETRY_MINUTES) - 1)]
            retry_at = now + timedelta(minutes=wait)
        logger.warning(
            "reminder %s not delivered (%s); %s",
            item.id,
            error,
            f"retry at {retry_at}" if retry_at else "giving up",
        )
        await self._db(
            lambda db: ReminderRepository(db).mark_failed_attempt(item.id, now, error, retry_at)
        )

    # ----------------------------------------------------- agenda and alerts

    async def calendar_pass(self) -> None:
        now = self._now()

        def load(db: Session) -> list[NudgeUser]:
            telegram = TelegramRepository(db)
            users: list[NudgeUser] = []
            for p in ProfileRepository(db).with_nudges():
                link = telegram.get_link_for_user(p.user_id)
                users.append(
                    NudgeUser(
                        p.user_id,
                        link.telegram_user_id if link else None,
                        p.timezone,
                        bool(p.agenda_enabled),
                        p.agenda_time or "08:00",
                        p.last_agenda_on,
                        p.event_alert_minutes,
                    )
                )
            ReminderRepository(db).prune_alerts(now - ALERT_RETENTION)
            return users

        for user in await self._db(load):
            if user.chat_id is None:
                continue
            try:
                if user.agenda_enabled:
                    await self._maybe_send_agenda(user, now)
                if user.event_alert_minutes:
                    await self._send_event_alerts(user, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Calendar pass failed for user_id=%s", user.user_id)

    async def _google_calendar(self, user: NudgeUser, db: Session) -> CalendarClient | None:
        service = self._google_factory(db)
        try:
            token = await service.get_access_token(user.user_id)
        except (GoogleNotConnectedError, GoogleAuthError):
            return None
        return CalendarClient(token, service.http_transport)

    async def _maybe_send_agenda(self, user: NudgeUser, now: datetime) -> None:
        tz = ZoneInfo(user.timezone)
        local = now.astimezone(tz)
        today = local.date()
        hour, minute = (int(x) for x in user.agenda_time.split(":"))
        scheduled = datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(
            hour=hour, minute=minute
        )
        window = timedelta(hours=self._settings.agenda_window_hours)
        if not (scheduled <= local < scheduled + window):
            return
        if user.last_agenda_on is not None and user.last_agenda_on >= today:
            return

        claimed = await self._db(lambda db: ProfileRepository(db).claim_agenda(user.user_id, today))
        if not claimed:
            return  # already sent (or being sent) by someone else

        try:
            text = await self._compose_agenda(user, today)
            assert user.chat_id is not None
            await self._client.send_message(user.chat_id, text)
            get_tracer().event("scheduler.agenda_sent", metadata={"user": f"u{user.user_id}"})
        except Exception:
            logger.exception("Agenda failed for user_id=%s; will retry", user.user_id)
            await self._db(
                lambda db: ProfileRepository(db).release_agenda(user.user_id, user.last_agenda_on)
            )

    async def _compose_agenda(self, user: NudgeUser, today: date) -> str:
        events: list[dict[str, Any]] | None = None
        session = self._session_factory()
        try:
            client = await self._google_calendar(user, session)
            if client is not None:
                try:
                    events = (await client.list_events(today, today, None, 25))["events"]
                except CalendarError as exc:
                    logger.warning("agenda: calendar unavailable for user_id=%s: %s", user.user_id, exc)
        finally:
            await asyncio.to_thread(session.close)

        reminders = await self._db(lambda db: ReminderRepository(db).pending(user.user_id))
        return build_agenda(today, user.timezone, events, reminders)

    async def _send_event_alerts(self, user: NudgeUser, now: datetime) -> None:
        assert user.event_alert_minutes is not None and user.chat_id is not None
        session = self._session_factory()
        try:
            client = await self._google_calendar(user, session)
            if client is None:
                return
            try:
                events = await client.upcoming_events(now, user.event_alert_minutes)
            except CalendarError as exc:
                logger.warning("alerts: calendar unavailable for user_id=%s: %s", user.user_id, exc)
                return
        finally:
            await asyncio.to_thread(session.close)

        for event in events:
            first = await self._db(
                lambda db, e=event: ReminderRepository(db).record_alert(
                    user.user_id, e["id"], e["start_raw"]
                )
            )
            if not first:
                continue
            try:
                await self._client.send_message(
                    user.chat_id, build_event_alert(event, now, user.timezone)
                )
                get_tracer().event("scheduler.meeting_alert", metadata={"user": f"u{user.user_id}"})
            except TelegramError:
                await self._db(
                    lambda db, e=event: ReminderRepository(db).forget_alert(
                        user.user_id, e["id"], e["start_raw"]
                    )
                )


    # ------------------------------------------------------------ coach

    async def coach_pass(self) -> None:
        """Evening check-in, or the weekly review on the user's chosen weekday (one message a day)."""
        now = self._now()

        def load(db: Session) -> list[CoachUser]:
            telegram = TelegramRepository(db)
            profiles = ProfileRepository(db)
            users: list[CoachUser] = []
            for c in CoachSettingsRepository(db).enabled():
                link = telegram.get_link_for_user(c.user_id)
                profile = profiles.get(c.user_id)
                if link is None or profile is None:
                    continue
                users.append(
                    CoachUser(
                        c.user_id,
                        link.telegram_user_id,
                        profile.timezone,
                        c.checkin_time or "20:00",
                        c.weekly_day,
                        c.last_sent_on,
                    )
                )
            return users

        for user in await self._db(load):
            try:
                await self._maybe_send_coach(user, now)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Coach pass failed for user_id=%s", user.user_id)

    async def _maybe_send_coach(self, user: "CoachUser", now: datetime) -> None:
        tz = ZoneInfo(user.timezone)
        local = now.astimezone(tz)
        today = local.date()
        hour, minute = (int(x) for x in user.checkin_time.split(":"))
        scheduled = datetime.combine(today, datetime.min.time(), tzinfo=tz).replace(
            hour=hour, minute=minute
        )
        window = timedelta(hours=self._settings.coach_window_hours)
        if not (scheduled <= local < scheduled + window):
            return
        if user.last_sent_on is not None and user.last_sent_on >= today:
            return

        claimed = await self._db(
            lambda db: CoachSettingsRepository(db).claim_send(user.user_id, today)
        )
        if not claimed:
            return

        def compose(db: Session) -> str:
            service = CoachService(db)
            if local.weekday() == user.weekly_day:
                return service.weekly_text(user.user_id, today)
            return service.checkin_text(user.user_id, today, local.hour)

        try:
            text = await self._db(compose)
            await self._client.send_message(user.chat_id, text)
            get_tracer().event("scheduler.coach_sent", metadata={"user": f"u{user.user_id}"})
        except Exception:
            logger.exception("Coach message failed for user_id=%s; will retry", user.user_id)
            await self._db(
                lambda db: CoachSettingsRepository(db).release_send(user.user_id, user.last_sent_on)
            )


    # --------------------------------------------------------- tracking nudges

    async def tracking_pass(self) -> None:
        """Nudge users whose planned meal or workout is not logged. At most one claim per slot/day."""
        now = self._now()
        window = timedelta(hours=self._settings.tracking_window_hours)

        def load(db: Session) -> list[tuple[int, int, str]]:
            telegram = TelegramRepository(db)
            profiles = ProfileRepository(db)
            users: list[tuple[int, int, str]] = []
            for t in TrackingRepository(db).enabled():
                link = telegram.get_link_for_user(t.user_id)
                profile = profiles.get(t.user_id)
                if link is not None and profile is not None:
                    users.append((t.user_id, link.telegram_user_id, profile.timezone))
            return users

        for user_id, chat_id, tz in await self._db(load):
            try:
                local = now.astimezone(ZoneInfo(tz))
                nudges = await self._db(lambda db: TrackingService(db).due(user_id, local, window))
                for nudge in nudges:
                    claimed = await self._db(
                        lambda db: TrackingRepository(db).claim_nudge(user_id, local.date(), nudge.slot)
                    )
                    if not claimed:
                        continue
                    try:
                        await self._client.send_message(chat_id, nudge.text)
                        get_tracer().event(
                            "scheduler.tracking_nudge", metadata={"user": f"u{user_id}", "slot": nudge.slot}
                        )
                    except Exception:
                        logger.exception("Tracking nudge failed for user_id=%s; will retry", user_id)
                        await self._db(
                            lambda db: TrackingRepository(db).release_nudge(
                                user_id, local.date(), nudge.slot
                            )
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Tracking pass failed for user_id=%s", user_id)


@dataclass(frozen=True)
class CoachUser:
    user_id: int
    chat_id: int
    timezone: str
    checkin_time: str
    weekly_day: int
    last_sent_on: date | None