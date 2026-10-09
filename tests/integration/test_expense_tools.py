from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.service import MAX_TOOL_ROUNDS, AgentService
from app.database.models import Expense
from app.database.repositories.users import UserRepository
from app.llm.base import LLMProvider
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 9)


@pytest.fixture
def ctx(db: Session) -> ToolContext:
    user = UserRepository(db).get_or_create("tester")
    return ToolContext(
        db=db,
        user_id=user.id,
        today=TODAY,
        default_currency="INR",
        external_user_id="tester",
    )


@pytest.mark.asyncio
async def test_add_expense_defaults(ctx: ToolContext):
    registry = build_default_registry()

    result = await registry.execute(
        "add_expense", {"amount": 250.5, "category": " Food "}, ctx
    )

    saved = result["saved"]
    assert saved["amount"] == "250.50"
    assert saved["currency"] == "INR"
    assert saved["category"] == "food"
    assert saved["date"] == "2026-10-09"


@pytest.mark.asyncio
async def test_list_and_summary(ctx: ToolContext):
    registry = build_default_registry()
    await registry.execute("add_expense", {"amount": 100, "category": "food"}, ctx)
    await registry.execute("add_expense", {"amount": 50, "category": "food"}, ctx)
    await registry.execute(
        "add_expense",
        {"amount": 30, "category": "transport", "spent_on": "2026-10-01"},
        ctx,
    )
    await registry.execute(
        "add_expense",
        {"amount": 999, "category": "rent", "spent_on": "2026-09-30"},
        ctx,
    )

    listed = await registry.execute("list_expenses", {"category": "food"}, ctx)
    assert listed["count"] == 2

    summary = await registry.execute("get_expense_summary", {}, ctx)  # this month
    totals = {r["category"]: r["total"] for r in summary["by_category"]}
    assert totals == {"food": "150.00", "transport": "30.00"}
    assert summary["grand_total_by_currency"] == {"INR": "180.00"}


@pytest.mark.asyncio
async def test_users_are_isolated(db: Session, ctx: ToolContext):
    registry = build_default_registry()
    await registry.execute("add_expense", {"amount": 10, "category": "food"}, ctx)

    other = UserRepository(db).get_or_create("someone-else")
    other_ctx = ToolContext(
        db=db, user_id=other.id, today=TODAY, default_currency="INR"
    )

    assert (await registry.execute("list_expenses", {}, other_ctx))["count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,args",
    [
        ("add_expense", {"amount": -5, "category": "food"}),
        ("add_expense", {"category": "food"}),
        ("list_expenses", {"start_date": "2026-10-09", "end_date": "2026-10-01"}),
        ("does_not_exist", {}),
    ],
)
async def test_bad_calls_return_errors_not_exceptions(
    ctx: ToolContext, name: str, args: dict[str, Any]
):
    result = await build_default_registry().execute(name, args, ctx)
    assert "error" in result


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
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


def tool_response(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="call_1", name=name, arguments=args)],
        model="scripted-model",
    )


@pytest.mark.asyncio
async def test_agent_runs_tool_then_answers(db: Session):
    provider = ScriptedProvider(
        [
            tool_response("add_expense", {"amount": 12, "category": "food"}),
            LLMResponse(content="Saved 12 for food.", model="scripted-model"),
        ]
    )
    service = AgentService(provider, db, today_fn=lambda: TODAY)

    result = await service.chat("u1", "I spent 12 on lunch")

    assert result.reply == "Saved 12 for food."
    assert result.tools_used == ["add_expense"]

    row = db.scalars(select(Expense)).one()
    assert row.amount == Decimal("12.00")
    assert row.spent_on == TODAY

    # second LLM call must include the assistant tool call + the tool result
    roles = [m.role for m in provider.calls[1]]
    assert roles == ["system", "user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_agent_stops_runaway_tool_loop(db: Session):
    provider = ScriptedProvider([tool_response("list_expenses", {})])
    service = AgentService(provider, db, today_fn=lambda: TODAY)

    result = await service.chat("u1", "loop forever")

    assert len(provider.calls) == MAX_TOOL_ROUNDS
    assert "couldn't complete" in result.reply