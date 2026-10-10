"""A slow database must never freeze the event loop (and with it /health)."""

import asyncio
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.config import get_settings
from app.database.repositories.google_credentials import GoogleCredentialRepository
from app.database.repositories.users import UserRepository
from app.integrations.google.oauth import GoogleOAuthService
from app.llm.base import LLMProvider
from app.schemas.llm import LLMResponse, Message
from app.security.crypto import encrypt
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry

SLOW = 0.5  # seconds of blocking work
MAX_ALLOWED_GAP = 0.25  # the loop must never stall for anywhere near SLOW


async def measure_loop_gap(work: Any) -> tuple[Any, float]:
    """Run `work` while a ticker records the longest time the loop was stalled."""
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)  # let the ticker start
    try:
        result = await work
    finally:
        stop.set()
        await ticker_task
    return result, max(gaps)


class Empty(BaseModel):
    pass


class OneShotLLM(LLMProvider):
    name = "fake"
    model = "fake-model"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="hi", model="fake-model")


@pytest.mark.asyncio
async def test_slow_user_lookup_does_not_block_the_loop(
    db: Session, passthrough_jev, monkeypatch: pytest.MonkeyPatch
):
    real = UserRepository.get_or_create

    def slow(self: UserRepository, external_id: str):
        time.sleep(SLOW)
        return real(self, external_id)

    monkeypatch.setattr(UserRepository, "get_or_create", slow)
    agent = AgentService(OneShotLLM(), db, jev=passthrough_jev)

    result, gap = await measure_loop_gap(agent.chat("u1", "hello"))

    assert result.reply == "hi"
    assert gap < MAX_ALLOWED_GAP, f"event loop stalled for {gap:.2f}s"


@pytest.mark.asyncio
async def test_slow_sync_tool_does_not_block_the_loop(db: Session):
    def slow_handler(ctx: ToolContext, args: Any) -> dict[str, Any]:
        time.sleep(SLOW)
        return {"ok": True}

    registry = ToolRegistry(
        [
            Tool(
                name="slow",
                description="slow",
                args_model=Empty,
                handler=slow_handler,
                domain="expense",
            )
        ]
    )
    user = UserRepository(db).get_or_create("u1")
    ctx = ToolContext(db=db, user_id=user.id, today=date(2026, 10, 9), default_currency="INR")

    result, gap = await measure_loop_gap(registry.execute("slow", {}, ctx))

    assert result == {"ok": True}
    assert gap < MAX_ALLOWED_GAP, f"event loop stalled for {gap:.2f}s"


@pytest.mark.asyncio
async def test_slow_google_credential_lookup_does_not_block_the_loop(
    db: Session, monkeypatch: pytest.MonkeyPatch
):
    user = UserRepository(db).get_or_create("u1")
    GoogleCredentialRepository(db).upsert(
        user_id=user.id,
        google_email="me@example.com",
        access_token_enc=encrypt("tok"),
        refresh_token_enc=encrypt("ref"),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        scopes="x",
    )
    real = GoogleCredentialRepository.get

    def slow(self: GoogleCredentialRepository, user_id: int):
        time.sleep(SLOW)
        return real(self, user_id)

    monkeypatch.setattr(GoogleCredentialRepository, "get", slow)
    service = GoogleOAuthService(
        db, get_settings(), transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )

    token, gap = await measure_loop_gap(service.get_access_token(user.id))

    assert token == "tok"
    assert gap < MAX_ALLOWED_GAP, f"event loop stalled for {gap:.2f}s"


@pytest.mark.asyncio
async def test_slow_approval_claim_does_not_block_the_loop(
    db: Session, monkeypatch: pytest.MonkeyPatch
):
    from decimal import Decimal

    from app.approvals.service import ApprovalService
    from app.database.repositories.approvals import ApprovalRepository
    from app.database.repositories.expenses import ExpenseRepository
    from app.tools.registry import build_default_registry

    user = UserRepository(db).get_or_create("u1")
    expense = ExpenseRepository(db).add(
        user_id=user.id, amount=Decimal("1.00"), currency="INR",
        category="food", description=None, spent_on=date(2026, 10, 9),
    )
    service = ApprovalService(db, build_default_registry())
    ctx = ToolContext(db=db, user_id=user.id, today=date(2026, 10, 9), default_currency="INR")
    _, pending = await service.propose(ctx, "delete_expense", {"expense_id": expense.id}, "x")
    assert pending is not None

    real_claim = ApprovalRepository.claim

    def slow_claim(self: ApprovalRepository, approval_id: str) -> bool:
        time.sleep(SLOW)
        return real_claim(self, approval_id)

    monkeypatch.setattr(ApprovalRepository, "claim", slow_claim)

    out, gap = await measure_loop_gap(service.approve("u1", pending.id))

    assert out.status == "executed"
    assert gap < MAX_ALLOWED_GAP, f"event loop stalled for {gap:.2f}s"