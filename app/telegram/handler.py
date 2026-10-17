import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.approvals.service import ApprovalError, ApprovalService
from app.config import Settings
from app.conversation.service import ConversationService
from app.integrations.google.links import LoginLinkService
from app.integrations.telegram.client import TelegramAPI, TelegramError
from app.llm.base import LLMError
from app.profiles.service import ProfileError, ProfileService, local_today
from app.schemas.approvals import PendingApprovalOut
from app.telegram.users import InviteError, LinkedUser, TelegramUsers
from app.usage.service import QuotaService, RateLimitExceeded

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_TELEGRAM_CHARS = 4000  # hard limit is 4096
MAX_INPUT_CHARS = 4000
CALLBACK_RE = re.compile(r"^([ar]):([0-9a-f-]{36})$")
REDEEM_WINDOW_SECONDS = 600
REDEEM_MAX_FAILURES = 5

UNAUTHORIZED = (
    "This is a private assistant. If you have an invite code, send /start <code>."
)
HELP_TEXT = (
    "I can track your expenses, read your Gmail and Calendar, and more is coming.\n\n"
    "Just write to me normally, for example:\n"
    "- I spent 250 on lunch\n"
    "- Any unread emails from my bank?\n"
    "- What's on my calendar tomorrow?\n\n"
    "Commands:\n"
    "/new - start a fresh conversation (I forget the recent messages)\n"
    "/approvals - changes waiting for your approval\n"
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
    ) -> None:
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

        if not text:
            await self._send(
                chat_id, "I can only read text messages for now. Photos are coming soon."
            )
            return

        await self._chat(link, chat_id, text[:MAX_INPUT_CHARS])

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
            "(Gmail and Calendar, read-only). It works once and expires in "
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
        match = CALLBACK_RE.match(cq.get("data") or "")

        if not match or "id" not in sender:
            await self._answer(cq_id, "Unknown action.")
            return

        link = await self._resolve(sender["id"], _display_name(sender))
        if link is None:
            await self._answer(cq_id, "Not authorized.")
            return

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