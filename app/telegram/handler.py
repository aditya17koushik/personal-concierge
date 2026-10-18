import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar

from sqlalchemy.orm import Session

from app.observability import get_tracer
from app.agent.service import AgentService
from app.approvals.service import ApprovalError, ApprovalService
from app.config import Settings
from app.conversation.service import ConversationService
from app.integrations.google.links import LoginLinkService
from app.integrations.telegram.client import TelegramAPI, TelegramError
from app.llm.base import LLMError
from app.profiles.service import (
    ProfileError,
    ProfileService,
    local_today,
    seconds_until_local_midnight,
)
from app.schemas.approvals import PendingApprovalOut
from app.telegram.users import InviteError, LinkedUser, TelegramUsers
from app.coach.service import WEEKDAYS, CoachService
from app.fitness.photos import FoodDraftService, PhotoAnalyzer, PhotoError
from app.fitness.plan_service import PlanService, format_plan_today, format_plan_week
from app.fitness.service import FitnessError, default_meal_type
from app.fitness.service import (
    FitnessService,
    format_day,
    format_goal,
    format_progress,
    goal_out,
)
from app.tracking.service import TrackingService
from app.reminders.service import ReminderService, describe
from app.usage.service import QuotaService, RateLimitExceeded

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_TELEGRAM_CHARS = 4000  # hard limit is 4096
MAX_INPUT_CHARS = 4000
CALLBACK_RE = re.compile(r"^([ar]):([0-9a-f-]{36})$")
PHOTO_CB_RE = re.compile(r"^p([ld]):([0-9a-f-]{36})$")  # p l = log it, p d = discard
REDEEM_WINDOW_SECONDS = 600
REDEEM_MAX_FAILURES = 5

UNAUTHORIZED = (
    "This is a private assistant. If you have an invite code, send /start <code>."
)
HELP_TEXT = (
    "I can track your expenses, read your Gmail, manage your Calendar (changes need your "
    "approval) and send you reminders.\n\n"
    "Just write to me normally, for example:\n"
    "- I spent 250 on lunch\n"
    "- Any unread emails from my bank?\n"
    "- What's on my calendar tomorrow?\n"
    "- Remind me to call Ravi at 6pm\n"
    "- Had 2 rotis and dal for lunch / I weigh 78.5 / I did a 5 km run\n"
    "- Send a photo of your meal and I will estimate it (you tap Log it)\n\n"
    "Commands:\n"
    "/new - start a fresh conversation (I forget the recent messages)\n"
    "/approvals - changes waiting for your approval\n"
    "/today - what you ate, trained and weigh today (/today yesterday)\n"
    "/progress - weight, food and training trend (/progress 7, /progress 90)\n"
    "/goal - your current goal and targets\n"
    "/plan - today's planned workout and meals (/plan week for all 7 days)\n"
    "/tracking - reminders when a meal or workout is not logged (/tracking on, /tracking off, "
    "/tracking lunch 13:30)\n"
    "/review - your coach's review of the last 7 days\n"
    "/coach - evening check-ins and a weekly review (/coach on 20:00, /coach off, "
    "/coach weekly sunday)\n"
    "/reminders - your pending reminders (/reminders cancel <id>)\n"
    "/agenda - daily morning summary (/agenda on 08:00, /agenda off)\n"
    "/meetings - alert before meetings (/meetings 15, /meetings off)\n"
    "/google - connect your Google account\n"
    "/profile - your timezone, currency and units\n"
    "/timezone <name> - e.g. /timezone Asia/Kolkata\n"
    "/currency <code> - e.g. /currency INR\n"
    "/units <metric|imperial>\n"
    "/help - this message"
)
WELCOME_TIP = (
    "\n\nFirst, set your timezone so dates and reminders are right, "
    "for example: /timezone Asia/Kolkata"
)
PROFILE_COMMANDS = {
    "/timezone": ("timezone", "Timezone", "/timezone Asia/Kolkata"),
    "/currency": ("currency", "Currency", "/currency INR"),
    "/units": ("unit_system", "Units", "/units metric"),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_wait(seconds: int) -> str:
    hours, rest = divmod(max(seconds, 0), 3600)
    minutes = rest // 60
    return f"{hours}h {minutes}m" if hours else f"{max(minutes, 1)}m"
ADMIN_HELP = (
    "\n\nAdmin:\n"
    "/invite - create a single-use invite code\n"
    "/users - list who can use this bot\n"
    "/remove <telegram_id> - remove someone"
)


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def plain_text(text: str) -> str:
    """Strip the Markdown models like to emit. Messages are sent as plain text
    (so stray characters can't break them), where '**bold**' would show as-is."""
    text = _BOLD.sub(r"\1", text)
    text = _HEADING.sub("", text)
    return _INLINE_CODE.sub(r"\1", text)


def split_message(text: str, limit: int = MAX_TELEGRAM_CHARS) -> list[str]:
    """Split long text on line breaks (then spaces) so each part fits Telegram."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def _photo_of(msg: dict[str, Any]) -> tuple[str, str, int | None] | None:
    """(file_id, mime, size) of the photo in a message, or None. Telegram sends photos as a list
    of sizes; take the largest. A document is accepted only when it is an image."""
    sizes = msg.get("photo")
    if isinstance(sizes, list) and sizes:
        best = max(
            (s for s in sizes if isinstance(s, dict) and s.get("file_id")),
            key=lambda s: int(s.get("file_size") or 0),
            default=None,
        )
        if best is not None:
            return str(best["file_id"]), "image/jpeg", best.get("file_size")
    doc = msg.get("document")
    if isinstance(doc, dict) and str(doc.get("mime_type", "")).startswith("image/") and doc.get("file_id"):
        return str(doc["file_id"]), str(doc["mime_type"]), doc.get("file_size")
    return None


def approval_keyboard(approval_id: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [
            [
                {"text": "\u2705 Approve", "callback_data": f"a:{approval_id}"},
                {"text": "\u274c Reject", "callback_data": f"r:{approval_id}"},
            ]
        ]
    }


def approval_text(summary: str, expires_at: Any) -> str:
    return f"Approval needed:\n{summary}\n\nExpires {expires_at.strftime('%H:%M')} UTC"


def _display_name(sender: dict[str, Any]) -> str | None:
    name = sender.get("username") or sender.get("first_name")
    return str(name)[:100] if name else None


class TelegramHandler:
    """Turns Telegram updates into agent calls and approval decisions.

    Approve and Reject exist ONLY as buttons a human presses. They are never
    tools the model can call, so injected text can't approve its own proposals.
    """

    def __init__(
        self,
        client: TelegramAPI,
        session_factory: Callable[[], Session],
        settings: Settings,
        agent_factory: Callable[[Session], AgentService],
        approvals_factory: Callable[[Session], ApprovalService],
        bot_username: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], datetime] = _utcnow,
        vision: PhotoAnalyzer | None = None,
    ) -> None:
        self._vision = vision
        self._client = client
        self._session_factory = session_factory
        self._settings = settings
        self._agent_factory = agent_factory
        self._approvals_factory = approvals_factory
        self._bot_username = bot_username
        self._clock = clock
        self._now_fn = now_fn
        self._failed_redeems: dict[int, list[float]] = {}

    # ----------------------------------------------------------------- entry

    async def handle(self, update: dict[str, Any]) -> None:
        if "callback_query" in update:
            await self._on_callback(update["callback_query"])
        elif "message" in update:
            await self._on_message(update["message"])

    # --------------------------------------------------------------- helpers

    async def _db(self, fn: Callable[[Session], T]) -> T:
        """Run synchronous database work in a worker thread with its own session."""

        def run() -> T:
            with self._session_factory() as db:
                return fn(db)

        return await asyncio.to_thread(run)

    async def _resolve(self, telegram_id: int, name: str | None) -> LinkedUser | None:
        return await self._db(
            lambda db: TelegramUsers(db, self._settings).resolve(telegram_id, name)
        )

    async def _send(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        for index, chunk in enumerate(split_message(plain_text(text)) or ["(no response)"]):
            await self._client.send_message(
                chat_id, chunk, reply_markup if index == 0 and reply_markup else None
            )

    # --------------------------------------------------------------- messages

    async def _on_message(self, msg: dict[str, Any]) -> None:
        chat = msg.get("chat") or {}
        sender = msg.get("from") or {}
        # Private chats only: ignore groups, channels and other bots.
        if chat.get("type") != "private" or sender.get("is_bot") or "id" not in sender:
            return

        chat_id: int = chat["id"]
        telegram_id: int = sender["id"]
        name = _display_name(sender)
        text = (msg.get("text") or "").strip()

        command: str | None = None
        arg = ""
        if text.startswith("/"):
            head, _, arg = text.partition(" ")
            command = head.split("@")[0].lower()
            arg = arg.strip()

        link = await self._resolve(telegram_id, name)

        if link is None:
            if command == "/start" and arg:
                await self._redeem(chat_id, telegram_id, name, arg)
            else:
                await self._send(chat_id, UNAUTHORIZED)
            return

        if command:
            await self._command(link, chat_id, command, arg)
            return

        photo = _photo_of(msg)
        if photo is not None:
            await self._photo(link, chat_id, photo, str(msg.get("caption") or ""))
            return

        if not text:
            await self._send(chat_id, "I can read text messages and photos of food.")
            return

        await self._chat(link, chat_id, text[:MAX_INPUT_CHARS])

    # ------------------------------------------------------------------ photos

    async def _photo(
        self, link: LinkedUser, chat_id: int, photo: tuple[str, str, int | None], caption: str
    ) -> None:
        """A food photo: estimate it, then WAIT for [Log it]. Nothing is logged by itself."""
        if self._vision is None:
            await self._send(chat_id, "Photo analysis is not set up on this server.")
            return
        file_id, mime, size = photo
        limit = self._settings.photo_max_bytes
        if size is not None and size > limit:
            await self._send(chat_id, f"That photo is too large (max {limit // 1_000_000} MB).")
            return

        now = self._now_fn()

        def prepare(db: Session) -> tuple[str, str]:
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            QuotaService(db, self._settings.daily_message_limit).consume(
                link.user_id, local_today(profile.timezone, now),
                seconds_until_local_midnight(profile.timezone, now),
            )  # fmt: skip
            return profile.timezone, default_meal_type(now, profile.timezone)

        try:
            tz_name, meal_type = await self._db(prepare)
        except RateLimitExceeded as exc:
            await self._send(
                chat_id,
                f"You've reached today's limit of {exc.limit} messages. It resets at midnight "
                f"in your timezone, in about {format_wait(exc.retry_after_seconds)}.",
            )
            return
        del tz_name

        try:
            await self._client.send_chat_action(chat_id, "typing")
        except TelegramError:
            pass

        try:
            with get_tracer().trace(
                "photo.analyze", user_id=f"u{link.user_id}", input=caption or None,
                metadata={"mime": mime, "size_bytes": size},
            ) as trace:  # the image itself is never sent to Langfuse
                estimate = await asyncio.wait_for(
                    self._estimate(file_id, mime, caption, limit),
                    timeout=self._settings.photo_timeout_seconds,
                )
                trace.update(
                    output=estimate.description if estimate else None,
                    metadata={
                        "calories": estimate.calories if estimate else None,
                        "confidence": estimate.confidence if estimate else None,
                    },
                )
        except asyncio.TimeoutError:
            await self._send(chat_id, "That took too long. Please send the photo again.")
            return
        except PhotoError as exc:
            await self._send(chat_id, str(exc))
            return
        except TelegramError:
            await self._send(chat_id, "I could not download that photo. Please try again.")
            return
        except Exception:
            logger.exception("Photo analysis failed")
            await self._send(chat_id, "Something went wrong on my side. Please try again.")
            return

        if estimate is None:
            await self._send(chat_id, "I can't see a meal in that photo. Send a clearer photo of your food.")
            return

        draft = await self._db(
            lambda db: FoodDraftService(db, self._settings.photo_draft_hours).create(
                link.user_id, estimate
            )
        )
        macros = ", ".join(
            f"{v:g} g {label}"
            for label, v in (("protein", estimate.protein_g), ("carbs", estimate.carbs_g), ("fat", estimate.fat_g))
            if v is not None
        )
        lines = [
            f"Photo estimate (rough): {estimate.description}",
            f"About {estimate.calories} kcal" + (f", {macros}" if macros else "") + f". Confidence: {estimate.confidence}.",
            f"Log it as {meal_type}?",
        ]  # fmt: skip
        await self._client.send_message(
            chat_id,
            "\n".join(lines),
            {
                "inline_keyboard": [
                    [
                        {"text": "\u2705 Log it", "callback_data": f"pl:{draft.id}"},
                        {"text": "\u274c Discard", "callback_data": f"pd:{draft.id}"},
                    ]
                ]
            },
        )

    async def _estimate(self, file_id: str, mime: str, caption: str, limit: int) -> Any:
        assert self._vision is not None
        info = await self._client.get_file(file_id)
        path = str(info.get("file_path") or "")
        if not path:
            raise PhotoError("I could not fetch that photo.")
        if info.get("file_size") and int(info["file_size"]) > limit:
            raise PhotoError(f"That photo is too large (max {limit // 1_000_000} MB).")
        data = await self._client.download_file(path, limit)
        return await self._vision.analyze(data, mime, caption)

    async def _on_photo_callback(
        self, cq: dict[str, Any], link: LinkedUser, action: str, draft_id: str
    ) -> None:
        cq_id = str(cq.get("id", ""))
        message = cq.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        message_id = message.get("message_id")
        original = str(message.get("text") or "")
        now = self._now_fn()

        def run(db: Session) -> str:
            service = FoodDraftService(db, self._settings.photo_draft_hours)
            draft = service.get(link.user_id, draft_id)
            if draft is None:
                raise FitnessError("That estimate was not found.")
            if action == "d":
                if not service.discard(link.user_id, draft_id, now):
                    raise FitnessError("That estimate was already handled or expired.")
                return "Discarded."
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            meal = service.confirm(
                link.user_id, draft_id, today=local_today(profile.timezone, now), now=now,
                tz_name=profile.timezone,
            )  # fmt: skip
            return (
                f"Logged as {meal.meal_type}: {meal.description} (about {meal.calories} kcal, "
                "an estimate). To correct it, just tell me, e.g. 'make it 600 calories'."
            )

        try:
            result = await self._db(run)
        except FitnessError as exc:
            await self._answer(cq_id, str(exc))
            if chat_id and message_id:
                await self._edit(chat_id, message_id, f"{original}\n\n\u2139\ufe0f {exc}")
            return

        get_tracer().event(
            "photo.logged" if action == "l" else "photo.discarded", metadata={"user": f"u{link.user_id}"}
        )
        await self._answer(cq_id, "Done")
        if chat_id and message_id:
            icon = "\u2705" if action == "l" else "\u274c"
            await self._edit(chat_id, message_id, f"{icon} {result}\n\n{original.splitlines()[0] if original else ''}")

    # ----------------------------------------------------------------- invites

    def _redeem_blocked(self, telegram_id: int) -> bool:
        now = self._clock()
        recent = [
            t
            for t in self._failed_redeems.get(telegram_id, [])
            if now - t < REDEEM_WINDOW_SECONDS
        ]
        self._failed_redeems[telegram_id] = recent
        return len(recent) >= REDEEM_MAX_FAILURES

    async def _redeem(
        self, chat_id: int, telegram_id: int, name: str | None, code: str
    ) -> None:
        if self._redeem_blocked(telegram_id):
            await self._send(chat_id, "Too many attempts. Please try again later.")
            return

        try:
            await self._db(
                lambda db: TelegramUsers(db, self._settings).redeem_invite(
                    code, telegram_id, name
                )
            )
        except InviteError as exc:
            if exc.counts_as_failure:
                self._failed_redeems.setdefault(telegram_id, []).append(self._clock())
            await self._send(chat_id, exc.message)
            return

        await self._send(chat_id, "Welcome! You're all set.\n\n" + HELP_TEXT + WELCOME_TIP)

    # ---------------------------------------------------------------- commands

    async def _command(
        self, link: LinkedUser, chat_id: int, command: str, arg: str
    ) -> None:
        if command in ("/start", "/help"):
            await self._send(chat_id, HELP_TEXT + (ADMIN_HELP if link.is_admin else ""))
        elif command == "/new":
            await self._db(lambda db: ConversationService(db).clear(link.user_id))
            await self._send(chat_id, "Started a new conversation. I've forgotten the recent messages.")
        elif command == "/approvals":
            await self._list_approvals(link, chat_id)
        elif command in ("/today", "/progress", "/goal"):
            await self._fitness(link, chat_id, command, arg)
        elif command == "/plan":
            await self._plan(link, chat_id, arg)
        elif command == "/tracking":
            await self._tracking(link, chat_id, arg)
        elif command == "/review":
            await self._coach_review(link, chat_id)
        elif command == "/coach":
            await self._coach(link, chat_id, arg)
        elif command == "/reminders":
            await self._reminders(link, chat_id, arg)
        elif command == "/agenda":
            await self._agenda(link, chat_id, arg)
        elif command == "/meetings":
            await self._meetings(link, chat_id, arg)
        elif command == "/google":
            await self._google_link(link, chat_id)
        elif command == "/profile":
            await self._profile(link, chat_id)
        elif command in PROFILE_COMMANDS:
            await self._set_profile(link, chat_id, command, arg)
        elif command == "/invite" and link.is_admin:
            await self._invite(link, chat_id)
        elif command == "/users" and link.is_admin:
            rows = await self._db(lambda db: TelegramUsers(db, self._settings).list_users())
            lines = [f"{tg}  {name or '-'}  ({ext})" for tg, ext, name in rows]
            await self._send(chat_id, "Users:\n" + "\n".join(lines))
        elif command == "/remove" and link.is_admin:
            await self._remove(link, chat_id, arg)
        else:
            await self._send(chat_id, "Unknown command. Send /help to see what I can do.")

    async def _google_link(self, link: LinkedUser, chat_id: int) -> None:
        # Created for the VERIFIED Telegram user only; single-use and short-lived.
        code, _expires = await self._db(
            lambda db: LoginLinkService(db, self._settings).create(link.user_id)
        )
        base = self._settings.public_base_url.rstrip("/")
        minutes = self._settings.google_login_link_ttl_minutes
        await self._send(
            chat_id,
            "Open this link in your browser to connect your Google account "
            "(Gmail is read-only; calendar changes happen only after you approve them). "
            "It works once and expires in "
            f"{minutes} minutes. Only open links you requested yourself:\n"
            f"{base}/auth/google/login?code={code}",
        )

    async def _profile(self, link: LinkedUser, chat_id: int) -> None:
        def load(db: Session) -> tuple[Any, int]:
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            day = local_today(profile.timezone, self._now_fn())
            used = QuotaService(db, self._settings.daily_message_limit).used(link.user_id, day)
            return profile, used

        profile, used = await self._db(load)
        lines = [
            "Your profile",
            f"Timezone: {profile.timezone}",
            f"Currency: {profile.currency}",
            f"Units: {profile.unit_system}",
        ]
        if self._settings.daily_message_limit > 0:
            lines.append(f"Messages today: {used} of {self._settings.daily_message_limit}")
        lines.append(
            "Morning agenda: "
            + (f"on at {profile.agenda_time}" if profile.agenda_enabled else "off (/agenda on)")
        )
        lines.append(
            "Meeting alerts: "
            + (
                f"{profile.event_alert_minutes} minutes before"
                if profile.event_alert_minutes
                else "off (/meetings 15)"
            )
        )
        lines.append("\nChange with /timezone, /currency or /units.")
        await self._send(chat_id, "\n".join(lines))

    async def _set_profile(
        self, link: LinkedUser, chat_id: int, command: str, arg: str
    ) -> None:
        field, label, example = PROFILE_COMMANDS[command]
        if not arg:
            await self._send(chat_id, f"Usage: {example}")
            return
        try:
            profile = await self._db(
                lambda db: ProfileService(
                    db, self._settings.default_currency, self._settings.default_timezone
                ).update(link.user_id, **{field: arg})
            )
        except ProfileError as exc:
            await self._send(chat_id, exc.message)
            return
        await self._send(chat_id, f"{label} set to {getattr(profile, field)}.")

    async def _agenda(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        words = arg.split()
        service = lambda db: ProfileService(  # noqa: E731
            db, self._settings.default_currency, self._settings.default_timezone
        )
        try:
            if not words:
                profile = await self._db(lambda db: service(db).get_or_create(link.user_id))
                state = f"on, at {profile.agenda_time} ({profile.timezone})" if profile.agenda_enabled else "off"
                await self._send(
                    chat_id,
                    f"Morning agenda is {state}.\n/agenda on 08:00 turns it on, /agenda off turns it off.",
                )
                return
            command = words[0].lower()
            if command == "off":
                await self._db(lambda db: service(db).set_agenda(link.user_id, False))
                await self._send(chat_id, "Morning agenda turned off.")
            elif command == "on":
                time_text = words[1] if len(words) > 1 else None
                profile = await self._db(
                    lambda db: service(db).set_agenda(link.user_id, True, time_text)
                )
                await self._send(
                    chat_id,
                    f"Morning agenda on. You'll get it at {profile.agenda_time} "
                    f"({profile.timezone}). Check your timezone with /profile.",
                )
            else:
                await self._send(chat_id, "Usage: /agenda on 08:00   or   /agenda off")
        except ProfileError as exc:
            await self._send(chat_id, exc.message)

    async def _meetings(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        value = arg.strip().lower()
        service = lambda db: ProfileService(  # noqa: E731
            db, self._settings.default_currency, self._settings.default_timezone
        )
        try:
            if not value:
                profile = await self._db(lambda db: service(db).get_or_create(link.user_id))
                state = (
                    f"on: {profile.event_alert_minutes} minutes before each meeting"
                    if profile.event_alert_minutes
                    else "off"
                )
                await self._send(
                    chat_id,
                    f"Meeting alerts are {state}.\n/meetings 15 turns them on, /meetings off turns them off.",
                )
            elif value == "off":
                await self._db(lambda db: service(db).set_event_alerts(link.user_id, None))
                await self._send(chat_id, "Meeting alerts turned off.")
            else:
                profile = await self._db(lambda db: service(db).set_event_alerts(link.user_id, value))
                await self._send(
                    chat_id,
                    f"I'll alert you {profile.event_alert_minutes} minutes before meetings "
                    "(checked about every 2 minutes). It needs your Google account: see /google.",
                )
        except ProfileError as exc:
            await self._send(chat_id, exc.message)

    async def _reminders(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        words = arg.split()
        if words and words[0].lower() == "cancel":
            if len(words) != 2 or not words[1].isdigit():
                await self._send(chat_id, "Usage: /reminders cancel <id>")
                return
            done = await self._db(
                lambda db: ReminderService(db).cancel(link.user_id, int(words[1]))
            )
            await self._send(
                chat_id, "Reminder cancelled." if done else "I couldn't find that reminder."
            )
            return

        items = await self._db(
            lambda db: [describe(r) for r in ReminderService(db).pending(link.user_id)]
        )
        if not items:
            await self._send(chat_id, "You have no pending reminders. Try: remind me to call Ravi at 6pm")
            return
        lines = ["Your reminders:"]
        for item in items:
            repeat = f" (repeats {item['repeat']})" if item["repeat"] else ""
            lines.append(f"#{item['id']}  {item['when']}  {item['text']}{repeat}")
        lines.append("\nCancel one with /reminders cancel <id>.")
        await self._send(chat_id, "\n".join(lines))

    async def _fitness(self, link: LinkedUser, chat_id: int, command: str, arg: str) -> None:
        """Read-only views of the fitness log. Text is built by code, not by the model."""
        now = self._now_fn()

        def load(db: Session) -> str:
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            today = local_today(profile.timezone, now)
            svc = FitnessService(db)
            if command == "/today":
                day = today - timedelta(days=1) if arg.strip().lower() == "yesterday" else today
                return format_day(svc.day_summary(link.user_id, day, today))
            if command == "/progress":
                word = arg.strip()
                if word and not word.isdigit():
                    return "Usage: /progress 30   (number of days, 1-365)"
                days = int(word) if word else 30
                if not 1 <= days <= 365:
                    return "Days must be between 1 and 365."
                return format_progress(svc.progress(link.user_id, today, days))
            goal = svc.goals.get(link.user_id)
            if goal is None:
                return "No goal yet. Tell me, for example: I want to cut, target 72 kg."
            return format_goal(goal_out(goal, svc.unit(link.user_id)))

        await self._send(chat_id, await self._db(load))

    async def _plan(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        """Show the saved plan. Built by code; creating or changing it is done by chatting."""
        now = self._now_fn()
        week = arg.strip().lower() == "week"

        def load(db: Session) -> str:
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            today = local_today(profile.timezone, now)
            plan = PlanService(db).active(link.user_id)
            if plan is None:
                return (
                    "You have no plan yet. Tell me you want a workout and meal plan and I will "
                    "ask a few questions (age, height, weight, goal, health conditions...)."
                )
            return format_plan_week(plan) if week else format_plan_today(plan, today)

        await self._send(chat_id, await self._db(load))

    async def _tracking(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        words = arg.lower().split()
        usage = "Usage:\n/tracking on\n/tracking off\n/tracking lunch 13:30 (breakfast, lunch, dinner, workout)"
        try:
            if not words:
                await self._send(chat_id, await self._db(lambda db: TrackingService(db).status_text(link.user_id)))
                return
            if words == ["on"] or words == ["off"]:
                await self._db(
                    lambda db: TrackingService(db).configure(link.user_id, enabled=words[0] == "on")
                )
            elif len(words) == 2:
                await self._db(
                    lambda db: TrackingService(db).configure(
                        link.user_id, enabled=True, times={words[0]: words[1]}
                    )
                )
            else:
                await self._send(chat_id, usage)
                return
        except ProfileError as exc:
            await self._send(chat_id, f"{exc}\n{usage}")
            return
        await self._send(chat_id, await self._db(lambda db: TrackingService(db).status_text(link.user_id)))

    async def _coach_review(self, link: LinkedUser, chat_id: int) -> None:
        """The weekly-style review on demand. Built by code; the model is not involved."""

        def load(db: Session) -> str:
            profile = ProfileService(
                db, self._settings.default_currency, self._settings.default_timezone
            ).get_or_create(link.user_id)
            today = local_today(profile.timezone, self._now_fn())
            return CoachService(db).weekly_text(link.user_id, today)

        await self._send(chat_id, await self._db(load))

    async def _coach(self, link: LinkedUser, chat_id: int, arg: str) -> None:
        words = arg.lower().split()
        usage = (
            "Usage:\n/coach on 20:00 - evening check-ins at that time\n/coach off\n"
            "/coach weekly sunday - the weekly review replaces that day's check-in"
        )
        try:
            if not words:
                row = await self._db(lambda db: CoachService(db).settings.get(link.user_id))
                if row is None or not row.enabled:
                    await self._send(
                        chat_id,
                        "Coach messages are off.\n/coach on 20:00 turns on a daily evening "
                        "check-in, plus a weekly review. /review shows one now.",
                    )
                else:
                    await self._send(
                        chat_id,
                        f"Coach is on: check-in at {row.checkin_time}, weekly review on "
                        f"{WEEKDAYS[row.weekly_day].title()}. /coach off to stop.",
                    )
                return
            if words[0] == "on" and len(words) <= 2:
                row = await self._db(
                    lambda db: CoachService(db).configure(
                        link.user_id, enabled=True, time_text=words[1] if len(words) == 2 else None
                    )
                )
                await self._send(
                    chat_id,
                    f"Coach is on. I'll check in daily at {row.checkin_time} "
                    f"({WEEKDAYS[row.weekly_day].title()} is the weekly review). "
                    "Make sure your /timezone is set.",
                )
            elif words[0] == "off" and len(words) == 1:
                await self._db(lambda db: CoachService(db).configure(link.user_id, enabled=False))
                await self._send(chat_id, "Coach messages are off.")
            elif words[0] == "weekly" and len(words) == 2:
                row = await self._db(
                    lambda db: CoachService(db).configure(link.user_id, weekday=words[1])
                )
                await self._send(
                    chat_id, f"The weekly review is now on {WEEKDAYS[row.weekly_day].title()}."
                )
            else:
                await self._send(chat_id, usage)
        except ProfileError as exc:
            await self._send(chat_id, f"{exc.message}\n\n{usage}")

    async def _invite(self, admin: LinkedUser, chat_id: int) -> None:
        code, _expires = await self._db(
            lambda db: TelegramUsers(db, self._settings).create_invite(admin)
        )
        hours = self._settings.telegram_invite_ttl_hours
        text = f"Invite code (single use, valid {hours}h):\n{code}"
        if self._bot_username:
            text += f"\n\nOr share this link:\nhttps://t.me/{self._bot_username}?start={code}"
        else:
            text += "\n\nThey send: /start " + code
        await self._send(chat_id, text)

    async def _remove(self, admin: LinkedUser, chat_id: int, arg: str) -> None:
        try:
            target = int(arg)
        except ValueError:
            await self._send(chat_id, "Usage: /remove <telegram_id>")
            return
        try:
            removed = await self._db(
                lambda db: TelegramUsers(db, self._settings).remove(admin, target)
            )
        except ValueError as exc:
            await self._send(chat_id, str(exc))
            return
        await self._send(chat_id, "Removed." if removed else "No such user.")

    # ------------------------------------------------------------------- chat

    async def _chat(self, link: LinkedUser, chat_id: int, text: str) -> None:
        try:
            await self._client.send_chat_action(chat_id, "typing")
        except TelegramError:
            pass  # cosmetic only

        db = self._session_factory()
        try:
            agent = self._agent_factory(db)
            result = await asyncio.wait_for(
                agent.chat(link.external_id, text),
                timeout=self._settings.telegram_agent_timeout_seconds,
            )
        except RateLimitExceeded as exc:
            await self._send(
                chat_id,
                f"You've reached today's limit of {exc.limit} messages. "
                f"It resets at midnight in your timezone, in about "
                f"{format_wait(exc.retry_after_seconds)}.",
            )
            return
        except asyncio.TimeoutError:
            await self._send(chat_id, "That took too long. Please try again.")
            return
        except LLMError:
            await self._send(
                chat_id,
                "I'm having trouble reaching the language model right now. "
                "Please try again in a moment.",
            )
            return
        except Exception:
            logger.exception("Agent call failed")
            await self._send(chat_id, "Something went wrong on my side. Please try again.")
            return
        finally:
            await asyncio.to_thread(db.close)

        await self._send(chat_id, result.reply)
        for pending in result.pending_approvals:
            await self._send_approval(chat_id, pending)

    async def _send_approval(self, chat_id: int, pending: PendingApprovalOut) -> None:
        await self._client.send_message(
            chat_id,
            approval_text(pending.summary, pending.expires_at),
            approval_keyboard(pending.id),
        )

    async def _list_approvals(self, link: LinkedUser, chat_id: int) -> None:
        db = self._session_factory()
        try:
            pending = await self._approvals_factory(db).list_approvals(
                link.external_id, "pending"
            )
        finally:
            await asyncio.to_thread(db.close)

        if not pending:
            await self._send(chat_id, "Nothing is waiting for your approval.")
            return
        for item in pending:
            await self._client.send_message(
                chat_id,
                approval_text(item.summary, item.expires_at),
                approval_keyboard(item.id),
            )

    # --------------------------------------------------------------- callbacks

    async def _on_callback(self, cq: dict[str, Any]) -> None:
        cq_id = str(cq.get("id", ""))
        sender = cq.get("from") or {}
        data = cq.get("data") or ""
        match = CALLBACK_RE.match(data)
        photo_match = PHOTO_CB_RE.match(data)

        if (not match and not photo_match) or "id" not in sender:
            await self._answer(cq_id, "Unknown action.")
            return

        link = await self._resolve(sender["id"], _display_name(sender))
        if link is None:
            await self._answer(cq_id, "Not authorized.")
            return

        if photo_match:
            await self._on_photo_callback(cq, link, photo_match.group(1), photo_match.group(2))
            return
        assert match is not None

        kind, approval_id = match.groups()
        message = cq.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        message_id = message.get("message_id")
        original = str(message.get("text") or "")

        db = self._session_factory()
        try:
            service = self._approvals_factory(db)
            if kind == "a":
                out = await service.approve(link.external_id, approval_id)
            else:
                out = await service.reject(link.external_id, approval_id)
        except ApprovalError as exc:
            await self._answer(cq_id, exc.message)
            # Stale button (already handled / expired): remove it. For "not found"
            # (e.g. someone else's button) leave the message alone.
            if exc.status_code in (409, 410) and chat_id and message_id:
                await self._edit(chat_id, message_id, f"{original}\n\n\u2139\ufe0f {exc.message}")
            return
        finally:
            await asyncio.to_thread(db.close)

        get_tracer().event(
            f"approval.{out.status}",
            metadata={"user": f"u{link.user_id}", "summary": out.summary},
        )
        if out.status == "executed":
            text, toast = f"\u2705 Done\n{out.summary}", "Done"
        elif out.status == "rejected":
            text, toast = f"\u274c Rejected\n{out.summary}", "Rejected"
        else:
            error = str((out.result or {}).get("error", "unknown error"))[:200]
            text = f"\u26a0\ufe0f Could not complete\n{out.summary}\n{error}"
            toast = "Failed"

        await self._answer(cq_id, toast)
        if chat_id and message_id:
            await self._edit(chat_id, message_id, text)

    async def _answer(self, callback_id: str, text: str) -> None:
        try:
            await self._client.answer_callback_query(callback_id, text)
        except TelegramError as exc:
            logger.warning("answerCallbackQuery failed: %s", exc)

    async def _edit(self, chat_id: int, message_id: int, text: str) -> None:
        try:
            await self._client.edit_message_text(chat_id, message_id, text[:MAX_TELEGRAM_CHARS])
        except TelegramError as exc:
            # e.g. "message is not modified" - harmless
            logger.warning("editMessageText failed: %s", exc)