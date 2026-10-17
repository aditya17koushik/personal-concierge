import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.agent.service import AgentService
from app.approvals.service import ApprovalService
from app.config import Settings
from app.conversation.service import ConversationService
from app.database.models import Expense, PendingAction, TelegramInvite
from app.database.repositories.expenses import ExpenseRepository
from app.database.repositories.telegram import TelegramRepository
from app.database.repositories.users import UserRepository
from app.decisions.jev import Jev
from app.integrations.telegram.client import (
    TelegramClient,
    TelegramError,
    TokenRedactingFilter,
)
from app.llm.base import LLMError, LLMProvider
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.telegram.bot import TelegramBot
from app.telegram.handler import (
    MAX_TELEGRAM_CHARS,
    UNAUTHORIZED,
    TelegramHandler,
    plain_text,
    split_message,
)
from app.telegram.users import TelegramUsers, hash_code
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 9)
ADMIN = 111
BOB = 222
CAROL = 333


# ------------------------------------------------------------------ fakes


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.answers: list[tuple[str, str | None]] = []
        self.actions: list[int] = []

    async def get_me(self) -> dict[str, Any]:
        return {"username": "test_bot"}

    async def delete_webhook(self) -> None: ...

    async def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        return []

    async def send_message(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.sent.append({"chat_id": chat_id, "text": text, "markup": reply_markup})
        return {"message_id": len(self.sent)}

    async def edit_message_text(
        self, chat_id: int, message_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "text": text})

    async def answer_callback_query(self, callback_query_id: str, text: str | None = None) -> None:
        self.answers.append((callback_query_id, text))

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        self.actions.append(chat_id)

    @property
    def texts(self) -> list[str]:
        return [m["text"] for m in self.sent]


class Recorder(LLMProvider):
    name = "rec"
    model = "rec-model"

    def __init__(self, responses: list[LLMResponse | Exception] | None = None) -> None:
        self._responses = responses or [LLMResponse(content="Hello from the model", model="rec-model")]
        self.calls = 0

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        item = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class StubEvaluator:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self._response


def real_answers(cal: float, email: float, exp: float, p_high: float) -> dict[str, Any]:
    return {
        "answers": {
            "needs_calendar": {"type": "noul", "noul": cal},
            "needs_email": {"type": "noul", "noul": email},
            "needs_expense": {"type": "noul", "noul": exp},
            "needs_reminder": {"type": "noul", "noul": 0.0},
            "risk": {
                "type": "choice",
                "choice": "low",
                "probabilities": {"low": 1 - p_high, "medium": 0.0, "high": p_high},
            },
            "harmful": {"type": "noul", "noul": 0.01},
        }
    }


CHAT = real_answers(0.01, 0.02, 0.01, 0.0)  # real scores: "Hi, how are you today?"
DELETE = real_answers(0.02, 0.09, 0.98, 0.46)  # real scores: "Delete all my expenses"


def text_resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="rec-model")


def tool_resp(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id="c1", name=name, arguments=args)], model="rec-model"
    )


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "database_url": "sqlite://",
        "telegram_admin_id": ADMIN,
        "telegram_admin_external_id": "me",
        "telegram_max_users": 10,
        "telegram_agent_timeout_seconds": 5,
        "app_secret_key": "test-secret-key-not-for-production",
    }
    values.update(overrides)
    return Settings(**values)


@dataclass
class Rig:
    handler: TelegramHandler
    tg: FakeTelegram
    provider: Recorder
    factory: sessionmaker[Session]
    settings: Settings

    async def send(self, text: str, user: int = ADMIN, chat: int | None = None) -> None:
        await self.handler.handle(message(text, user, chat))

    async def press(self, data: str, user: int = ADMIN, chat: int | None = None) -> None:
        await self.handler.handle(callback(data, user, chat))


def make_rig(
    session_factory: sessionmaker[Session],
    answers: dict[str, Any] = CHAT,
    responses: list[LLMResponse | Exception] | None = None,
    **settings_overrides: Any,
) -> Rig:
    settings = make_settings(**settings_overrides)
    provider = Recorder(responses)
    tg = FakeTelegram()
    registry = build_default_registry()

    def agent_factory(db: Session) -> AgentService:
        return AgentService(
            provider,
            db,
            jev=Jev(StubEvaluator(answers)),
            registry=registry,
            approvals=ApprovalService(db, registry, today_fn=lambda: TODAY),
            daily_message_limit=settings.daily_message_limit,
            today_fn=lambda: TODAY,
        )

    def approvals_factory(db: Session) -> ApprovalService:
        return ApprovalService(db, registry, today_fn=lambda: TODAY)

    handler = TelegramHandler(
        client=tg,
        session_factory=session_factory,
        settings=settings,
        agent_factory=agent_factory,
        approvals_factory=approvals_factory,
        bot_username="test_bot",
        now_fn=lambda: datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc),  # same day as TODAY
    )
    return Rig(handler, tg, provider, session_factory, settings)


_ids = iter(range(1, 10_000))


def message(text: str | None, user: int = ADMIN, chat: int | None = None, **extra: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "message_id": 1,
        "from": {"id": user, "is_bot": False, "first_name": f"User{user}"},
        "chat": {"id": chat if chat is not None else user, "type": "private"},
    }
    if text is not None:
        msg["text"] = text
    msg.update(extra)
    return {"update_id": next(_ids), "message": msg}


def callback(data: str, user: int = ADMIN, chat: int | None = None) -> dict[str, Any]:
    return {
        "update_id": next(_ids),
        "callback_query": {
            "id": f"cq{next(_ids)}",
            "from": {"id": user, "is_bot": False, "first_name": f"User{user}"},
            "data": data,
            "message": {
                "message_id": 5,
                "chat": {"id": chat if chat is not None else user, "type": "private"},
                "text": "Approval needed:\nDelete expense #1",
            },
        },
    }


def internal_user(factory: sessionmaker[Session], external_id: str) -> int:
    with factory() as db:
        return UserRepository(db).get_or_create(external_id).id


def add_expense(factory: sessionmaker[Session], user_id: int, amount: str = "250.00") -> int:
    with factory() as db:
        return ExpenseRepository(db).add(
            user_id=user_id, amount=Decimal(amount), currency="INR",
            category="food", description="lunch", spent_on=TODAY,
        ).id


def expense_exists(factory: sessionmaker[Session], expense_id: int) -> bool:
    with factory() as db:
        return db.get(Expense, expense_id) is not None


def link_user(factory: sessionmaker[Session], telegram_id: int, external_id: str) -> int:
    uid = internal_user(factory, external_id)
    with factory() as db:
        TelegramRepository(db).create_link(telegram_id, uid, external_id)
    return uid


# ---------------------------------------------------------------- text helper


def test_split_message_respects_limit_and_loses_nothing():
    text = ("word " * 3000).strip()
    chunks = split_message(text, 100)
    assert all(0 < len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_message_prefers_line_breaks_and_handles_empty():
    assert split_message("   ") == []
    assert split_message("a\nb") == ["a\nb"]
    assert split_message("x" * 250, 100) == ["x" * 100, "x" * 100, "x" * 50]


def test_plain_text_strips_markdown_but_keeps_content():
    raw = (
        "Here are your emails:\n\n"
        "1. **From:** LinkedIn <jobs@linkedin.com>\n"
        "   **Subject:** Application sent\n"
        "## Heading\n"
        "Use `ls` here, a * b and snake_case_name stay"
    )
    assert plain_text(raw) == (
        "Here are your emails:\n\n"
        "1. From: LinkedIn <jobs@linkedin.com>\n"
        "   Subject: Application sent\n"
        "Heading\n"
        "Use ls here, a * b and snake_case_name stay"
    )


@pytest.mark.asyncio
async def test_replies_are_sent_without_markdown(session_factory):
    rig = make_rig(session_factory, responses=[text_resp("**Subject:** Hello **there**")])

    await rig.send("hi")

    assert rig.tg.texts == ["Subject: Hello there"]


# ------------------------------------------------------------- API client


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> TelegramClient:
    return TelegramClient("123:SECRET", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_client_request_shapes():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": [{"update_id": 1}]})

    client = make_client(handler)
    updates = await client.get_updates(offset=7, timeout=30)
    await client.send_message(5, "hi", {"inline_keyboard": []})
    await client.send_message(5, "plain")

    assert updates == [{"update_id": 1}]
    assert str(seen[0].url) == "https://api.telegram.org/bot123:SECRET/getUpdates"
    body = json.loads(seen[0].content)
    assert body["offset"] == 7 and body["timeout"] == 30
    assert body["allowed_updates"] == ["message", "callback_query"]
    send = json.loads(seen[1].content)
    assert send == {"chat_id": 5, "text": "hi", "reply_markup": {"inline_keyboard": []}}
    assert "parse_mode" not in send  # plain text: model output can't inject markup
    assert "reply_markup" not in json.loads(seen[2].content)


@pytest.mark.asyncio
async def test_client_maps_api_errors_and_rate_limits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 7}},
        )

    with pytest.raises(TelegramError) as exc:
        await make_client(handler).send_message(1, "x")

    assert exc.value.retry_after == 7 and exc.value.status == 429


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "not-json"])
async def test_client_errors_never_contain_the_token(failure: str, caplog: pytest.LogCaptureFixture):
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "network":
            raise httpx.ConnectError(f"cannot connect to {request.url}")
        return httpx.Response(200, content=b"<html>")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(TelegramError) as exc:
            await make_client(handler).get_me()

    assert "SECRET" not in str(exc.value)
    assert exc.value.__cause__ is None  # the original error text includes the URL
    assert "SECRET" not in caplog.text


def test_log_filter_redacts_the_token_from_httpx_style_records():
    record = logging.LogRecord(
        "httpx", logging.INFO, __file__, 1,
        'HTTP Request: %s %s "%s"',
        ("POST", "https://api.telegram.org/bot123456:ABC-def_ghi/getUpdates", "HTTP/1.1 200 OK"),
        None,
    )

    assert TokenRedactingFilter().filter(record) is True

    rendered = record.getMessage()
    assert "ABC-def_ghi" not in rendered and "123456" not in rendered
    assert "/bot<redacted>/getUpdates" in rendered


def test_log_filter_leaves_other_records_alone():
    record = logging.LogRecord("httpx", logging.INFO, __file__, 1, "plain %s", ("text",), None)
    TokenRedactingFilter().filter(record)
    assert record.getMessage() == "plain text"


# ------------------------------------------------------------ access control


@pytest.mark.asyncio
async def test_strangers_get_nothing_and_no_user_is_created(session_factory):
    rig = make_rig(session_factory)

    await rig.send("what's my balance?", user=BOB)
    await rig.send("/help", user=BOB)

    assert rig.tg.texts == [UNAUTHORIZED, UNAUTHORIZED]
    assert rig.provider.calls == 0
    with session_factory() as db:
        assert UserRepository(db).get("tg:222") is None


@pytest.mark.asyncio
async def test_admin_is_linked_to_the_existing_internal_user(session_factory):
    existing = internal_user(session_factory, "me")  # data from before Telegram existed
    rig = make_rig(session_factory)

    await rig.send("hello")

    assert rig.tg.texts == ["Hello from the model"]
    assert rig.tg.actions == [ADMIN]  # typing indicator
    with session_factory() as db:
        link = TelegramRepository(db).get_link(ADMIN)
        assert link is not None and link.user_id == existing


@pytest.mark.asyncio
async def test_unconfigured_admin_id_allows_nobody(session_factory):
    rig = make_rig(session_factory, telegram_admin_id=0)

    await rig.send("hello", user=ADMIN)

    assert rig.tg.texts == [UNAUTHORIZED]


@pytest.mark.asyncio
async def test_groups_bots_and_photos(session_factory):
    rig = make_rig(session_factory)

    group = message("hi")
    group["message"]["chat"]["type"] = "group"
    await rig.handler.handle(group)
    bot = message("hi")
    bot["message"]["from"]["is_bot"] = True
    await rig.handler.handle(bot)
    assert rig.tg.sent == []

    await rig.handler.handle(message(None, photo=[{"file_id": "x"}], caption="my lunch"))
    assert "text messages" in rig.tg.texts[0]
    assert rig.provider.calls == 0


@pytest.mark.asyncio
async def test_long_replies_are_split(session_factory):
    long_text = "line of text\n" * 1000
    rig = make_rig(session_factory, responses=[text_resp(long_text)])

    await rig.send("tell me a lot")

    assert len(rig.tg.sent) >= 3
    assert all(len(t) <= MAX_TELEGRAM_CHARS for t in rig.tg.texts)


# ------------------------------------------------------------------ invites


def invite_code(text: str) -> str:
    return next(line for line in text.splitlines() if line and " " not in line and "/" not in line)


@pytest.mark.asyncio
async def test_invite_flow_links_a_new_user_with_their_own_identity(session_factory):
    rig = make_rig(session_factory)

    await rig.send("/invite")
    reply = rig.tg.texts[-1]
    code = invite_code(reply)
    assert f"https://t.me/test_bot?start={code}" in reply

    await rig.send(f"/start {code}", user=BOB)
    assert "Welcome" in rig.tg.texts[-1]
    with session_factory() as db:
        users = TelegramUsers(db, rig.settings)
        linked = users.resolve(BOB, None)
        assert linked is not None and linked.external_id == "tg:222" and not linked.is_admin
        # only a hash of the code is stored
        assert db.query(TelegramInvite).one().code_hash == hash_code(code)

    await rig.send("hello", user=BOB)
    assert rig.tg.texts[-1] == "Hello from the model"


@pytest.mark.asyncio
async def test_invite_codes_are_single_use(session_factory):
    rig = make_rig(session_factory)
    await rig.send("/invite")
    code = invite_code(rig.tg.texts[-1])

    await rig.send(f"/start {code}", user=BOB)
    await rig.send(f"/start {code}", user=CAROL)

    assert "invalid or has expired" in rig.tg.texts[-1]
    with session_factory() as db:
        assert TelegramRepository(db).get_link(CAROL) is None


@pytest.mark.asyncio
async def test_expired_and_wrong_codes_are_rejected(session_factory):
    rig = make_rig(session_factory)
    await rig.send("/invite")
    code = invite_code(rig.tg.texts[-1])
    with session_factory() as db:
        invite = db.query(TelegramInvite).one()
        invite.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        db.commit()

    await rig.send(f"/start {code}", user=BOB)
    await rig.send("/start not-a-real-code", user=BOB)

    assert rig.tg.texts[-2:] == ["That invite code is invalid or has expired."] * 2
    with session_factory() as db:
        assert TelegramRepository(db).get_link(BOB) is None


@pytest.mark.asyncio
async def test_guessing_codes_is_throttled(session_factory):
    rig = make_rig(session_factory)
    await rig.send("/invite")
    real_code = invite_code(rig.tg.texts[-1])

    for guess in range(5):
        await rig.send(f"/start wrong{guess}", user=BOB)
    await rig.send(f"/start {real_code}", user=BOB)  # even the right code is blocked now

    assert "Too many attempts" in rig.tg.texts[-1]
    with session_factory() as db:
        assert TelegramRepository(db).get_link(BOB) is None


@pytest.mark.asyncio
async def test_user_limit_is_enforced(session_factory):
    rig = make_rig(session_factory, telegram_max_users=2)  # admin + one more
    await rig.send("hi")  # links admin
    await rig.send("/invite")
    first = invite_code(rig.tg.texts[-1])
    await rig.send("/invite")
    second = invite_code(rig.tg.texts[-1])

    await rig.send(f"/start {first}", user=BOB)
    await rig.send(f"/start {second}", user=CAROL)

    assert "user limit" in rig.tg.texts[-1]
    with session_factory() as db:
        assert TelegramRepository(db).get_link(CAROL) is None


@pytest.mark.asyncio
async def test_admin_commands_are_admin_only(session_factory):
    rig = make_rig(session_factory)
    link_user(session_factory, BOB, "tg:222")

    for command in ("/invite", "/users", "/remove 111"):
        await rig.send(command, user=BOB)

    assert all("Unknown command" in t for t in rig.tg.texts)
    with session_factory() as db:
        assert db.query(TelegramInvite).count() == 0
        assert TelegramRepository(db).get_link(ADMIN) is None  # untouched


@pytest.mark.asyncio
async def test_users_and_remove(session_factory):
    rig = make_rig(session_factory)
    link_user(session_factory, BOB, "tg:222")
    await rig.send("/users")
    assert "222" in rig.tg.texts[-1] and "tg:222" in rig.tg.texts[-1]

    await rig.send("/remove 222")
    assert rig.tg.texts[-1] == "Removed."
    await rig.send("hello", user=BOB)
    assert rig.tg.texts[-1] == UNAUTHORIZED

    await rig.send("/remove 111")
    assert "can't remove yourself" in rig.tg.texts[-1]
    await rig.send("/remove abc")
    assert "Usage" in rig.tg.texts[-1]
    await rig.send("/remove 999")
    assert rig.tg.texts[-1] == "No such user."


@pytest.mark.asyncio
async def test_help_google_and_unknown_commands(session_factory):
    rig = make_rig(session_factory)

    await rig.send("/help")
    await rig.send("/google")
    await rig.send("/nonsense")
    await rig.send("/help@test_bot")  # how Telegram addresses commands in some clients

    assert "/invite" in rig.tg.texts[0]  # admins see admin help
    assert "http://127.0.0.1:8000/auth/google/login?code=" in rig.tg.texts[1]
    assert "user_id" not in rig.tg.texts[1]  # no guessable identity in the link
    assert "works once" in rig.tg.texts[1]
    assert "Unknown command" in rig.tg.texts[2]
    assert "Commands:" in rig.tg.texts[3]


# ----------------------------------------------------------------- failures


@pytest.mark.asyncio
async def test_model_failure_gets_a_friendly_message(session_factory):
    rig = make_rig(session_factory, responses=[LLMError("secret internals")])

    await rig.send("hello")

    assert "trouble reaching the language model" in rig.tg.texts[-1]
    assert "secret internals" not in " ".join(rig.tg.texts)


@pytest.mark.asyncio
async def test_unexpected_failure_hides_details(session_factory):
    rig = make_rig(session_factory, responses=[RuntimeError("db password is hunter2")])

    await rig.send("hello")

    assert rig.tg.texts[-1] == "Something went wrong on my side. Please try again."


@pytest.mark.asyncio
async def test_slow_agent_times_out(session_factory):
    rig = make_rig(session_factory, telegram_agent_timeout_seconds=0)

    class Slow(Recorder):
        async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
            await asyncio.sleep(0.5)
            return text_resp("late")

    slow = Slow()
    rig.handler._agent_factory = lambda db: AgentService(  # type: ignore[attr-defined]
        slow, db, jev=Jev(StubEvaluator(CHAT)), today_fn=lambda: TODAY
    )

    await rig.send("hello")

    assert rig.tg.texts[-1] == "That took too long. Please try again."


# ---------------------------------------------------------------- approvals


async def propose_via_telegram(rig: Rig, expense_id: int) -> str:
    """Admin asks to delete an expense; returns the approval id from the button."""
    await rig.send("Delete my lunch expense")
    approval = rig.tg.sent[-1]
    assert approval["text"].startswith("Approval needed:\nDelete expense #")
    buttons = approval["markup"]["inline_keyboard"][0]
    assert [b["text"] for b in buttons] == ["\u2705 Approve", "\u274c Reject"]
    approve_data = buttons[0]["callback_data"]
    assert approve_data.startswith("a:") and buttons[1]["callback_data"] == "r:" + approve_data[2:]
    return approve_data[2:]


@pytest.mark.asyncio
async def test_risky_request_shows_buttons_and_executes_nothing(session_factory):
    uid = internal_user(session_factory, "me")
    eid = add_expense(session_factory, uid)
    rig = make_rig(
        session_factory, DELETE,
        [tool_resp("delete_expense", {"expense_id": eid}), text_resp("Prepared. Please approve.")],
    )

    await propose_via_telegram(rig, eid)

    assert rig.tg.texts[0] == "Prepared. Please approve."
    assert expense_exists(session_factory, eid)  # nothing deleted yet


@pytest.mark.asyncio
async def test_approve_button_executes_once(session_factory):
    uid = internal_user(session_factory, "me")
    eid = add_expense(session_factory, uid)
    rig = make_rig(session_factory, DELETE, [tool_resp("delete_expense", {"expense_id": eid}), text_resp("Prepared.")])
    approval_id = await propose_via_telegram(rig, eid)

    await rig.press(f"a:{approval_id}")

    assert not expense_exists(session_factory, eid)
    assert rig.tg.edits[-1]["text"].startswith("\u2705 Done")
    assert rig.tg.answers[-1][1] == "Done"

    await rig.press(f"a:{approval_id}")  # double tap / replay
    assert rig.tg.answers[-1][1] == "This action is already executed."
    assert "already executed" in rig.tg.edits[-1]["text"]  # stale buttons removed


@pytest.mark.asyncio
async def test_reject_button_keeps_the_data(session_factory):
    uid = internal_user(session_factory, "me")
    eid = add_expense(session_factory, uid)
    rig = make_rig(session_factory, DELETE, [tool_resp("delete_expense", {"expense_id": eid}), text_resp("Prepared.")])
    approval_id = await propose_via_telegram(rig, eid)

    await rig.press(f"r:{approval_id}")

    assert expense_exists(session_factory, eid)
    assert rig.tg.edits[-1]["text"].startswith("\u274c Rejected")
    with session_factory() as db:
        assert db.get(PendingAction, approval_id).status == "rejected"


@pytest.mark.asyncio
async def test_someone_elses_button_does_nothing(session_factory):
    uid = internal_user(session_factory, "me")
    eid = add_expense(session_factory, uid)
    link_user(session_factory, BOB, "tg:222")
    rig = make_rig(session_factory, DELETE, [tool_resp("delete_expense", {"expense_id": eid}), text_resp("Prepared.")])
    approval_id = await propose_via_telegram(rig, eid)

    await rig.press(f"a:{approval_id}", user=BOB)  # e.g. a forwarded message

    assert rig.tg.answers[-1][1] == "Approval not found."
    assert rig.tg.edits == []  # the owner's message is left alone
    assert expense_exists(session_factory, eid)


@pytest.mark.asyncio
async def test_unlinked_and_malformed_buttons_are_refused(session_factory):
    rig = make_rig(session_factory)

    await rig.press("a:" + "0" * 36, user=CAROL)  # not linked
    await rig.press("hack:1")
    await rig.press("a:../../etc")
    await rig.press("")

    assert [a[1] for a in rig.tg.answers] == [
        "Not authorized.", "Unknown action.", "Unknown action.", "Unknown action.",
    ]
    assert rig.tg.edits == []


@pytest.mark.asyncio
async def test_approvals_command_lists_pending(session_factory):
    uid = internal_user(session_factory, "me")
    eid = add_expense(session_factory, uid)
    rig = make_rig(session_factory, DELETE, [tool_resp("delete_expense", {"expense_id": eid}), text_resp("Prepared.")])

    await rig.send("/approvals")
    assert rig.tg.texts[-1] == "Nothing is waiting for your approval."

    approval_id = await propose_via_telegram(rig, eid)
    await rig.send("/approvals")

    listed = rig.tg.sent[-1]
    assert listed["text"].startswith("Approval needed:")
    assert listed["markup"]["inline_keyboard"][0][0]["callback_data"] == f"a:{approval_id}"


# -------------------------------------------------------------- polling loop


class PollingClient(FakeTelegram):
    def __init__(self, batches: list[list[dict[str, Any]] | Exception], stop: asyncio.Event) -> None:
        super().__init__()
        self.batches = batches
        self.stop = stop
        self.offsets: list[int | None] = []

    async def get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        self.offsets.append(offset)
        if not self.batches:
            self.stop.set()
            return []
        item = self.batches.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RecordingHandler:
    def __init__(self, fail_on: int | None = None, delay: float = 0.0) -> None:
        self.events: list[tuple[str, int]] = []
        self.fail_on = fail_on
        self.delay = delay

    async def handle(self, update: dict[str, Any]) -> None:
        uid = update["update_id"]
        self.events.append(("start", uid))
        await asyncio.sleep(self.delay)
        if uid == self.fail_on:
            raise RuntimeError("boom")
        self.events.append(("end", uid))


def upd(update_id: int, chat: int) -> dict[str, Any]:
    return {"update_id": update_id, "message": {"chat": {"id": chat, "type": "private"}}}


async def no_sleep(seconds: float) -> None: ...


@pytest.mark.asyncio
async def test_offset_advances_and_duplicates_are_skipped():
    stop = asyncio.Event()
    client = PollingClient([[upd(10, 1), upd(11, 1)], [upd(11, 1), upd(12, 1)]], stop)
    handler = RecordingHandler()

    await TelegramBot(client, handler, sleep=no_sleep).run(stop)

    assert client.offsets[:3] == [None, 12, 13]
    started = [uid for kind, uid in handler.events if kind == "start"]
    assert started == [10, 11, 12]  # 11 delivered twice, handled once


@pytest.mark.asyncio
async def test_one_chat_is_ordered_but_chats_run_concurrently():
    stop = asyncio.Event()
    client = PollingClient([[upd(1, 100), upd(2, 100), upd(3, 200)]], stop)
    handler = RecordingHandler(delay=0.05)

    await TelegramBot(client, handler, sleep=no_sleep).run(stop)

    events = handler.events
    assert events.index(("end", 1)) < events.index(("start", 2))  # same chat: in order
    assert events.index(("start", 3)) < events.index(("end", 1))  # other chat: not blocked


@pytest.mark.asyncio
async def test_a_failing_update_does_not_stop_the_loop():
    stop = asyncio.Event()
    client = PollingClient([[upd(1, 1), upd(2, 1), upd(3, 2)]], stop)
    handler = RecordingHandler(fail_on=1)

    await TelegramBot(client, handler, sleep=no_sleep).run(stop)

    assert ("end", 2) in handler.events and ("end", 3) in handler.events


@pytest.mark.asyncio
async def test_polling_errors_back_off_and_recover():
    stop = asyncio.Event()
    client = PollingClient(
        [TelegramError("down"), TelegramError("down"), TelegramError("slow", retry_after=9), [upd(1, 1)]],
        stop,
    )
    handler = RecordingHandler()
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    await TelegramBot(client, handler, sleep=record).run(stop)

    assert sleeps == [1.0, 2.0, 9]  # exponential backoff, then honours retry_after
    assert ("end", 1) in handler.events  # recovered and kept going


# ------------------------------------------------------- profile + limits


@pytest.mark.asyncio
async def test_profile_commands(session_factory):
    rig = make_rig(session_factory, daily_message_limit=5)

    await rig.send("/profile")
    assert "Timezone: UTC" in rig.tg.texts[-1]
    assert "Currency: " in rig.tg.texts[-1]
    assert "Messages today: 0 of 5" in rig.tg.texts[-1]

    await rig.send("/timezone asia/kolkata")
    assert rig.tg.texts[-1] == "Timezone set to Asia/Kolkata."
    await rig.send("/currency usd")
    assert rig.tg.texts[-1] == "Currency set to USD."
    await rig.send("/units imperial")
    assert rig.tg.texts[-1] == "Units set to imperial."

    await rig.send("/profile")
    assert "Timezone: Asia/Kolkata" in rig.tg.texts[-1]
    assert "Currency: USD" in rig.tg.texts[-1] and "Units: imperial" in rig.tg.texts[-1]


@pytest.mark.asyncio
async def test_invalid_profile_values_are_rejected_with_help(session_factory):
    rig = make_rig(session_factory)

    for command, expected in [
        ("/timezone Mars/Olympus", "Unknown timezone"),
        ("/currency rupees", "3-letter currency"),
        ("/units furlongs", "metric"),
        ("/timezone", "Usage: /timezone"),
        ("/currency", "Usage: /currency"),
        ("/units", "Usage: /units"),
    ]:
        await rig.send(command)
        assert expected in rig.tg.texts[-1], command

    await rig.send("/profile")
    assert "Timezone: UTC" in rig.tg.texts[-1]  # unchanged


@pytest.mark.asyncio
async def test_each_user_has_their_own_profile(session_factory):
    rig = make_rig(session_factory)
    link_user(session_factory, BOB, "tg:222")

    await rig.send("/timezone Asia/Tokyo")
    await rig.send("/profile", user=BOB)

    assert "Timezone: UTC" in rig.tg.texts[-1]  # Bob is unaffected


@pytest.mark.asyncio
async def test_the_daily_limit_is_explained_and_does_not_burn_model_calls(session_factory):
    rig = make_rig(session_factory, daily_message_limit=2)

    for _ in range(3):
        await rig.send("hello")

    assert rig.tg.texts[:2] == ["Hello from the model"] * 2
    assert "today's limit of 2 messages" in rig.tg.texts[-1]
    assert "resets at midnight" in rig.tg.texts[-1]
    assert rig.provider.calls == 2

    await rig.send("/help")  # commands are free
    assert "Commands:" in rig.tg.texts[-1]
    await rig.send("/profile")
    assert "Messages today: 2 of 2" in rig.tg.texts[-1]


@pytest.mark.asyncio
async def test_new_users_are_told_to_set_their_timezone(session_factory):
    rig = make_rig(session_factory)
    await rig.send("/invite")
    code = invite_code(rig.tg.texts[-1])

    await rig.send(f"/start {code}", user=BOB)

    assert "/timezone" in rig.tg.texts[-1]


# ----------------------------------------------------- conversation history


def remembered(factory: sessionmaker[Session], external_id: str) -> list[str]:
    uid = internal_user(factory, external_id)
    with factory() as db:
        service = ConversationService(db)
        return [m.content for m in service.recent(uid, datetime.now(timezone.utc))]


@pytest.mark.asyncio
async def test_the_conversation_is_remembered_and_new_forgets_it(session_factory):
    rig = make_rig(session_factory)

    await rig.send("hello")
    await rig.send("how are you")
    assert remembered(session_factory, "me") == [
        "hello", "Hello from the model", "how are you", "Hello from the model",
    ]

    await rig.send("/new")
    assert "new conversation" in rig.tg.texts[-1]
    assert remembered(session_factory, "me") == []


@pytest.mark.asyncio
async def test_new_only_clears_the_callers_history(session_factory):
    rig = make_rig(session_factory)
    link_user(session_factory, BOB, "tg:222")
    await rig.send("hello")
    await rig.send("hi from bob", user=BOB)

    await rig.send("/new")

    assert remembered(session_factory, "me") == []
    assert remembered(session_factory, "tg:222") == ["hi from bob", "Hello from the model"]


@pytest.mark.asyncio
async def test_new_is_listed_in_help(session_factory):
    rig = make_rig(session_factory)
    await rig.send("/help")
    assert "/new" in rig.tg.texts[-1]