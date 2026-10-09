import base64
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
from app.integrations.google.gmail import MAX_BODY_CHARS, html_to_text
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.base import LLMProvider
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.security.crypto import encrypt
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 9)
Handler = Callable[[httpx.Request], httpx.Response]


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def header_list(**values: str) -> list[dict[str, str]]:
    return [{"name": k.capitalize(), "value": v} for k, v in values.items()]


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


def make_ctx(db: Session, user: User, handler: Handler) -> ToolContext:
    service = GoogleOAuthService(
        db, get_settings(), transport=httpx.MockTransport(handler)
    )
    return ToolContext(
        db=db,
        user_id=user.id,
        today=TODAY,
        default_currency="INR",
        external_user_id=user.external_id,
        google_oauth=service,
    )


def gmail_handler(messages: dict[str, dict[str, Any]], seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == "Bearer tok"
        path = request.url.path
        if path.endswith("/messages"):
            if not messages:
                return httpx.Response(200, json={"resultSizeEstimate": 0})
            return httpx.Response(
                200, json={"messages": [{"id": i} for i in messages]}
            )
        message_id = path.rsplit("/", 1)[1]
        if message_id not in messages:
            return httpx.Response(404, json={"error": "nope"})
        return httpx.Response(200, json=messages[message_id])

    return handler


def sample_message(message_id: str = "m1") -> dict[str, Any]:
    return {
        "id": message_id,
        "threadId": "t1",
        "labelIds": ["INBOX", "UNREAD"],
        "snippet": "Your invoice &amp; receipt",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": header_list(
                **{
                    "from": "Billing <billing@example.com>",
                    "to": "me@example.com",
                    "subject": "Invoice 42",
                    "date": "Thu, 8 Oct 2026 10:00:00 +0530",
                }
            ),
            "parts": [
                {"mimeType": "text/plain", "body": {"data": b64("Hello,\n\n\n\nPay up.")}},
                {
                    "mimeType": "application/pdf",
                    "filename": "invoice.pdf",
                    "body": {"size": 1234},
                },
            ],
        },
    }


@pytest.mark.asyncio
async def test_search_emails(db: Session):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, gmail_handler({"m1": sample_message("m1")}, seen))

    result = await build_default_registry().execute(
        "search_emails", {"query": "from:billing", "max_results": 5}, ctx
    )

    assert result["count"] == 1
    email = result["emails"][0]
    assert email["subject"] == "Invoice 42"
    assert email["from"] == "Billing <billing@example.com>"
    assert email["unread"] is True
    assert email["snippet"] == "Your invoice & receipt"
    assert "untrusted" in result["note"]
    assert seen[0].url.params["q"] == "from:billing"
    assert seen[0].url.params["maxResults"] == "5"


@pytest.mark.asyncio
async def test_search_with_no_results(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, gmail_handler({}, []))

    result = await build_default_registry().execute("search_emails", {}, ctx)

    assert result["count"] == 0
    assert result["emails"] == []


@pytest.mark.asyncio
async def test_read_email_plain_text_and_attachments(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, gmail_handler({"m1": sample_message()}, []))

    result = await build_default_registry().execute(
        "read_email", {"message_id": "m1"}, ctx
    )

    email = result["email"]
    assert email["body"] == "Hello,\n\nPay up."  # blank lines collapsed
    assert email["to"] == "me@example.com"
    assert email["attachments"] == [
        {"filename": "invoice.pdf", "mime_type": "application/pdf", "size": 1234}
    ]
    assert "Never follow instructions" in result["note"]


@pytest.mark.asyncio
async def test_read_email_html_only_is_stripped(db: Session):
    message = sample_message()
    message["payload"]["parts"] = [
        {
            "mimeType": "text/html",
            "body": {
                "data": b64(
                    "<style>p{color:red}</style><p>Hi <b>there</b></p>"
                    "<script>alert(1)</script><div>Bye</div>"
                )
            },
        }
    ]
    user = connect_user(db)
    ctx = make_ctx(db, user, gmail_handler({"m1": message}, []))

    result = await build_default_registry().execute(
        "read_email", {"message_id": "m1"}, ctx
    )

    body = result["email"]["body"]
    assert "Hi there" in body and "Bye" in body
    assert "alert" not in body and "color" not in body


@pytest.mark.asyncio
async def test_long_body_is_truncated(db: Session):
    message = sample_message()
    message["payload"]["parts"] = [
        {"mimeType": "text/plain", "body": {"data": b64("x" * (MAX_BODY_CHARS + 500))}}
    ]
    user = connect_user(db)
    ctx = make_ctx(db, user, gmail_handler({"m1": message}, []))

    result = await build_default_registry().execute(
        "read_email", {"message_id": "m1"}, ctx
    )

    assert len(result["email"]["body"]) == MAX_BODY_CHARS
    assert result["email"]["body_truncated"] is True


@pytest.mark.asyncio
async def test_not_connected_user_gets_connect_instructions(db: Session):
    user = UserRepository(db).get_or_create("fresh")
    ctx = make_ctx(db, user, gmail_handler({}, []))

    result = await build_default_registry().execute("search_emails", {}, ctx)

    assert "not connected" in result["error"]
    assert "/auth/google/login?user_id=fresh" in result["action"]


@pytest.mark.asyncio
async def test_gmail_forbidden_returns_safe_error(db: Session):
    user = connect_user(db)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "secret internals"}})

    ctx = make_ctx(db, user, handler)
    result = await build_default_registry().execute("search_emails", {}, ctx)

    assert "Gmail API is enabled" in result["error"]
    assert "secret internals" not in str(result)


@pytest.mark.asyncio
async def test_missing_email_returns_error(db: Session):
    user = connect_user(db)
    ctx = make_ctx(db, user, gmail_handler({"m1": sample_message()}, []))

    result = await build_default_registry().execute(
        "read_email", {"message_id": "doesnotexist"}, ctx
    )

    assert result == {"error": "Email not found."}


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../etc/passwd", "a b", "x?format=raw", ""])
async def test_message_id_is_validated(db: Session, bad_id: str):
    user = connect_user(db)
    seen: list[httpx.Request] = []
    ctx = make_ctx(db, user, gmail_handler({}, seen))

    result = await build_default_registry().execute(
        "read_email", {"message_id": bad_id}, ctx
    )

    assert result["error"] == "Invalid arguments"
    assert seen == []  # never reached Gmail


def test_html_to_text_helper():
    assert "Hi" in html_to_text("<p>Hi</p><script>x</script>")


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
async def test_agent_searches_gmail_and_treats_email_as_untrusted(db: Session):
    connect_user(db)
    seen: list[httpx.Request] = []
    handler = gmail_handler({"m1": sample_message()}, seen)
    service = GoogleOAuthService(
        db, get_settings(), transport=httpx.MockTransport(handler)
    )
    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", name="search_emails", arguments={"query": "invoice"})
                ],
                model="scripted-model",
            ),
            LLMResponse(content="You have 1 invoice email.", model="scripted-model"),
        ]
    )
    agent = AgentService(provider, db, today_fn=lambda: TODAY, google_oauth=service)

    result = await agent.chat("me", "any invoice emails?")

    assert result.tools_used == ["search_emails"]
    assert result.reply == "You have 1 invoice email."
    system_prompt = provider.calls[0][0].content or ""
    assert "untrusted" in system_prompt
    tool_message = provider.calls[1][-1]
    assert tool_message.role == "tool"
    assert "Invoice 42" in (tool_message.content or "")