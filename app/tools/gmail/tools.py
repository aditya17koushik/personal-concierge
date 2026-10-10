import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.integrations.google.gmail import GmailClient, GmailError
from app.tools.base import Tool, ToolContext
from app.tools.google_common import UNTRUSTED_NOTE, with_google

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- search_emails


class SearchEmailsArgs(BaseModel):
    query: str = Field(
        default="",
        max_length=500,
        description=(
            "Gmail search query. Supports operators such as from:, to:, subject:, "
            "is:unread, has:attachment, newer_than:7d, after:2026/10/01, "
            "before:2026/10/31, label:. Empty returns the most recent emails."
        ),
    )
    max_results: int = Field(default=10, ge=1, le=20)


async def search_emails(ctx: ToolContext, args: SearchEmailsArgs) -> dict[str, Any]:
    async def call(
        token: str, transport: httpx.AsyncBaseTransport | None
    ) -> dict[str, Any]:
        emails = await GmailClient(token, transport).search(args.query, args.max_results)
        logger.info("gmail.search returned %d results", len(emails))
        return {"count": len(emails), "emails": emails, "note": UNTRUSTED_NOTE}

    return await with_google(ctx, call, (GmailError,))


# ------------------------------------------------------------------ read_email


class ReadEmailArgs(BaseModel):
    message_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="The id of an email, taken from search_emails results.",
    )


async def read_email(ctx: ToolContext, args: ReadEmailArgs) -> dict[str, Any]:
    async def call(
        token: str, transport: httpx.AsyncBaseTransport | None
    ) -> dict[str, Any]:
        email = await GmailClient(token, transport).read(args.message_id)
        return {"email": email, "note": UNTRUSTED_NOTE}

    return await with_google(ctx, call, (GmailError,))


GMAIL_TOOLS: list[Tool] = [
    Tool(
        name="search_emails",
        description=(
            "Search the user's Gmail (read-only). Returns sender, subject, date "
            "and a snippet for each match. Use read_email to get a full message."
        ),
        args_model=SearchEmailsArgs,
        handler=search_emails,
        domain="email",
    ),
    Tool(
        name="read_email",
        description="Read the full text of one email by id (read-only).",
        args_model=ReadEmailArgs,
        handler=read_email,
        domain="email",
    ),
]