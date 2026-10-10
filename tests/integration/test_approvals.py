import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.service import (
    MAX_PROPOSALS_PER_REQUEST,
    NOT_SUPPORTED_REPLY,
    AgentService,
)
from app.api.approvals import get_approval_service
from app.approvals.service import ApprovalError, ApprovalService
from app.database.models import Expense, PendingAction
from app.database.repositories.approvals import ApprovalRepository
from app.database.repositories.expenses import ExpenseRepository
from app.database.repositories.users import UserRepository
from app.decisions.jev import Jev
from app.llm.base import LLMProvider
from app.main import app
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.tools.base import Tool, ToolContext
from app.tools.registry import ToolRegistry, build_default_registry

TODAY = date(2026, 10, 9)
START = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ helpers


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(db: Session, clock: Clock) -> ApprovalService:
    return ApprovalService(
        db, build_default_registry(), ttl_minutes=30, today_fn=lambda: TODAY, now_fn=clock
    )


def make_user(db: Session, external_id: str = "me") -> int:
    return UserRepository(db).get_or_create(external_id).id


def make_expense(db: Session, user_id: int, amount: str = "250.00", **kw: Any) -> int:
    expense = ExpenseRepository(db).add(
        user_id=user_id,
        amount=Decimal(amount),
        currency="INR",
        category=kw.get("category", "food"),
        description=kw.get("description", "lunch"),
        spent_on=TODAY,
    )
    return expense.id


def ctx_for(db: Session, user_id: int, external_id: str = "me") -> ToolContext:
    return ToolContext(
        db=db,
        user_id=user_id,
        today=TODAY,
        default_currency="INR",
        external_user_id=external_id,
    )


def utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def row(db: Session, approval_id: str) -> PendingAction:
    db.expire_all()
    found = db.get(PendingAction, approval_id)
    assert found is not None
    return found


def expense_exists(db: Session, expense_id: int) -> bool:
    db.expire_all()
    return db.get(Expense, expense_id) is not None


async def propose_delete(
    service: ApprovalService, db: Session, user_id: int, expense_id: int
) -> str:
    result, pending = await service.propose(
        ctx_for(db, user_id), "delete_expense", {"expense_id": expense_id}, "delete it"
    )
    assert pending is not None, result
    return pending.id


# ------------------------------------------------------------------ propose


@pytest.mark.asyncio
async def test_propose_stores_a_pending_action_and_executes_nothing(
    db: Session, service: ApprovalService, clock: Clock
):
    uid = make_user(db)
    eid = make_expense(db, uid)

    result, pending = await service.propose(
        ctx_for(db, uid), "delete_expense", {"expense_id": eid}, "delete my lunch"
    )

    assert pending is not None
    assert result["status"] == "pending_approval"
    assert result["approval_id"] == pending.id
    # the summary is built by code from the database, not by the model
    assert pending.summary == "Delete expense #1: 250.00 INR, food, 2026-10-09 (lunch)"
    assert utc(pending.expires_at) == clock.now + timedelta(minutes=30)

    saved = row(db, pending.id)
    assert saved.status == "pending"
    assert saved.arguments == {"expense_id": eid}
    assert saved.request_text == "delete my lunch"
    assert expense_exists(db, eid)  # nothing was deleted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,args",
    [
        ("delete_expense", {"expense_id": 999}),  # does not exist
        ("delete_expense", {"expense_id": -1}),  # invalid
        ("delete_expense", {}),  # missing
        ("list_expenses", {}),  # not an approval-gated tool
        ("nope", {}),
    ],
)
async def test_bad_proposals_are_rejected_and_store_nothing(
    db: Session, service: ApprovalService, tool: str, args: dict[str, Any]
):
    uid = make_user(db)

    result, pending = await service.propose(ctx_for(db, uid), tool, args, "x")

    assert pending is None and "error" in result
    assert db.scalars(select(PendingAction)).all() == []


@pytest.mark.asyncio
async def test_cannot_propose_deleting_another_users_expense(
    db: Session, service: ApprovalService
):
    owner = make_user(db, "owner")
    other = make_user(db, "other")
    eid = make_expense(db, owner)

    result, pending = await service.propose(
        ctx_for(db, other, "other"), "delete_expense", {"expense_id": eid}, "x"
    )

    assert pending is None and result == {"error": "Expense not found."}


# ------------------------------------------------------------------ approve


@pytest.mark.asyncio
async def test_approve_executes_and_records_the_result(
    db: Session, service: ApprovalService
):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)

    out = await service.approve("me", approval_id)

    assert out.status == "executed"
    assert out.result is not None and out.result["deleted"]["id"] == eid
    assert out.decided_at is not None
    assert not expense_exists(db, eid)


@pytest.mark.asyncio
async def test_an_action_can_only_run_once(db: Session, service: ApprovalService):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)
    await service.approve("me", approval_id)

    with pytest.raises(ApprovalError) as exc:
        await service.approve("me", approval_id)

    assert exc.value.status_code == 409
    assert row(db, approval_id).status == "executed"  # not re-run, not "failed"


def test_claim_is_atomic(db: Session):
    uid = make_user(db)
    action = ApprovalRepository(db).create(
        uid, "delete_expense", {"expense_id": 1}, "s", None, START + timedelta(minutes=5)
    )
    repo = ApprovalRepository(db)

    assert repo.claim(action.id) is True
    assert repo.claim(action.id) is False  # second caller loses


@pytest.mark.asyncio
async def test_reject_keeps_the_data_and_blocks_later_approval(
    db: Session, service: ApprovalService
):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)

    out = await service.reject("me", approval_id)

    assert out.status == "rejected"
    assert expense_exists(db, eid)
    with pytest.raises(ApprovalError) as exc:
        await service.approve("me", approval_id)
    assert exc.value.status_code == 409
    assert expense_exists(db, eid)


@pytest.mark.asyncio
async def test_expired_approvals_cannot_run(
    db: Session, service: ApprovalService, clock: Clock
):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)

    clock.now = START + timedelta(minutes=31)
    with pytest.raises(ApprovalError) as exc:
        await service.approve("me", approval_id)

    assert exc.value.status_code == 410
    assert row(db, approval_id).status == "expired"
    assert expense_exists(db, eid)


@pytest.mark.asyncio
async def test_listing_marks_stale_pending_actions_expired(
    db: Session, service: ApprovalService, clock: Clock
):
    uid = make_user(db)
    approval_id = await propose_delete(service, db, uid, make_expense(db, uid))

    assert [a.id for a in await service.list_approvals("me", "pending")] == [approval_id]

    clock.now = START + timedelta(hours=1)
    assert await service.list_approvals("me", "pending") == []
    assert [a.status for a in await service.list_approvals("me", "expired")] == ["expired"]


@pytest.mark.asyncio
async def test_other_users_cannot_decide_my_approvals(
    db: Session, service: ApprovalService
):
    uid = make_user(db, "me")
    make_user(db, "intruder")
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)

    for action in (service.approve, service.reject):
        with pytest.raises(ApprovalError) as exc:
            await action("intruder", approval_id)
        assert exc.value.status_code == 404

    assert await service.list_approvals("intruder", None) == []
    assert row(db, approval_id).status == "pending"
    assert expense_exists(db, eid)


@pytest.mark.asyncio
async def test_unknown_user_or_id_is_not_found(db: Session, service: ApprovalService):
    with pytest.raises(ApprovalError) as exc:
        await service.approve("ghost", "no-such-id")
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_approving_after_the_target_vanished_marks_failed(
    db: Session, service: ApprovalService
):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)
    ExpenseRepository(db).delete(uid, eid)  # deleted by other means meanwhile

    out = await service.approve("me", approval_id)

    assert out.status == "failed"
    assert out.result == {"error": "Expense not found."}


@pytest.mark.asyncio
async def test_list_filters_by_status(db: Session, service: ApprovalService):
    uid = make_user(db)
    first = await propose_delete(service, db, uid, make_expense(db, uid))
    second = await propose_delete(service, db, uid, make_expense(db, uid, "10.00"))
    await service.reject("me", first)

    assert [a.id for a in await service.list_approvals("me", "pending")] == [second]
    assert [a.id for a in await service.list_approvals("me", "rejected")] == [first]
    assert len(await service.list_approvals("me", None)) == 2


def test_registry_refuses_gated_tools_without_a_summary():
    from pydantic import BaseModel

    class Empty(BaseModel):
        pass

    with pytest.raises(ValueError, match="requires approval"):
        ToolRegistry(
            [
                Tool(
                    name="danger",
                    description="d",
                    args_model=Empty,
                    handler=lambda ctx, args: {},
                    domain="x",
                    risk="high",
                    side_effects=True,
                )
            ]
        )


# ---------------------------------------------------------------------- API


@pytest.fixture
def client(db: Session, service: ApprovalService) -> Iterator[TestClient]:
    app.dependency_overrides[get_approval_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_api_list_approve_and_replay(
    db: Session, service: ApprovalService, client: TestClient
):
    uid = make_user(db)
    eid = make_expense(db, uid)
    approval_id = await propose_delete(service, db, uid, eid)

    listed = client.get("/approvals", params={"user_id": "me"}).json()
    assert [a["id"] for a in listed] == [approval_id]
    assert listed[0]["summary"].startswith("Delete expense #")

    done = client.post(f"/approvals/{approval_id}/approve", params={"user_id": "me"})
    assert done.status_code == 200 and done.json()["status"] == "executed"
    assert not expense_exists(db, eid)

    again = client.post(f"/approvals/{approval_id}/approve", params={"user_id": "me"})
    assert again.status_code == 409


@pytest.mark.asyncio
async def test_api_reject_and_errors(
    db: Session, service: ApprovalService, client: TestClient, clock: Clock
):
    uid = make_user(db)
    rejected_id = await propose_delete(service, db, uid, make_expense(db, uid))
    expired_id = await propose_delete(service, db, uid, make_expense(db, uid, "5.00"))

    res = client.post(f"/approvals/{rejected_id}/reject", params={"user_id": "me"})
    assert res.status_code == 200 and res.json()["status"] == "rejected"

    clock.now = START + timedelta(hours=1)
    assert client.post(
        f"/approvals/{expired_id}/approve", params={"user_id": "me"}
    ).status_code == 410

    assert client.post(
        "/approvals/nope/approve", params={"user_id": "me"}
    ).status_code == 404
    assert client.post("/approvals/nope/approve").status_code == 422  # user_id required


# ------------------------------------------------------- agent: propose mode


def real_answers(cal: float, email: float, exp: float, p_high: float) -> dict[str, Any]:
    return {
        "answers": {
            "needs_calendar": {"type": "noul", "noul": cal},
            "needs_email": {"type": "noul", "noul": email},
            "needs_expense": {"type": "noul", "noul": exp},
            "risk": {
                "type": "choice",
                "choice": "low",
                "probabilities": {"low": 1 - p_high, "medium": 0.0, "high": p_high},
            },
            "harmful": {"type": "noul", "noul": 0.01},
        }
    }


DELETE_REQUEST = real_answers(0.02, 0.09, 0.98, 0.46)  # real scores: "Delete all my expenses"
LOW_RISK_EXPENSE = real_answers(0.03, 0.06, 0.88, 0.0)  # real scores: "I spent 250 on lunch"


class StubEvaluator:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    async def evaluate(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self._response


@dataclass
class Call:
    messages: list[Message]
    tools: list[dict[str, Any]] | None

    @property
    def tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in (self.tools or [])}


class Recorder(LLMProvider):
    name = "rec"
    model = "rec-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[Call] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(Call(list(messages), tools))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="rec-model")


def calls(*specs: tuple[str, dict[str, Any]]) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(id=f"c{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(specs)
        ],
        model="rec-model",
    )


def make_agent(
    provider: Recorder,
    db: Session,
    answers: dict[str, Any],
    clock: Clock,
    with_approvals: bool = True,
) -> AgentService:
    registry = build_default_registry()
    approvals = (
        ApprovalService(db, registry, today_fn=lambda: TODAY, now_fn=clock)
        if with_approvals
        else None
    )
    return AgentService(
        provider,
        db,
        jev=Jev(StubEvaluator(answers)),
        registry=registry,
        approvals=approvals,
        today_fn=lambda: TODAY,
    )


@pytest.mark.asyncio
async def test_delete_request_is_proposed_not_executed(db: Session, clock: Clock):
    uid = make_user(db, "u1")
    eid = make_expense(db, uid)
    provider = Recorder(
        [
            calls(("list_expenses", {})),
            calls(("delete_expense", {"expense_id": eid})),
            text("I've prepared the deletion. Please approve it."),
        ]
    )

    result = await make_agent(provider, db, DELETE_REQUEST, clock).chat(
        "u1", "Delete my lunch expense"
    )

    assert result.decision is not None and result.decision.action == "needs_approval"
    assert len(result.pending_approvals) == 1
    pending = result.pending_approvals[0]
    assert pending.tool == "delete_expense"
    assert pending.summary == "Delete expense #1: 250.00 INR, food, 2026-10-09 (lunch)"
    assert result.tools_used == ["list_expenses"]  # only the read tool really ran
    assert expense_exists(db, eid)  # NOTHING was deleted
    assert row(db, pending.id).status == "pending"

    # the model saw read tools + the proposal tool, and no side-effect tool
    assert provider.calls[0].tool_names == {
        "list_expenses", "get_expense_summary", "delete_expense",
    }
    system_prompt = provider.calls[0].messages[0].content or ""
    assert "needs their approval" in system_prompt


@pytest.mark.asyncio
async def test_proposal_can_then_be_approved_end_to_end(db: Session, clock: Clock):
    uid = make_user(db, "u1")
    eid = make_expense(db, uid)
    provider = Recorder(
        [calls(("delete_expense", {"expense_id": eid})), text("Prepared.")]
    )
    agent = make_agent(provider, db, DELETE_REQUEST, clock)
    result = await agent.chat("u1", "Delete my lunch expense")

    approvals = ApprovalService(db, build_default_registry(), today_fn=lambda: TODAY, now_fn=clock)
    out = await approvals.approve("u1", result.pending_approvals[0].id)

    assert out.status == "executed"
    assert not expense_exists(db, eid)


@pytest.mark.asyncio
async def test_side_effect_tools_are_not_reachable_in_propose_mode(db: Session, clock: Clock):
    make_user(db, "u1")
    provider = Recorder(
        [
            calls(("add_expense", {"amount": 999, "category": "scam"})),
            text("Done."),
        ]
    )

    result = await make_agent(provider, db, DELETE_REQUEST, clock).chat("u1", "delete stuff")

    assert result.tools_used == []
    assert db.scalars(select(Expense)).all() == []
    assert "not available" in (provider.calls[1].messages[-1].content or "")


@pytest.mark.asyncio
async def test_gated_tool_cannot_run_in_a_normal_tool_flow(db: Session, clock: Clock):
    uid = make_user(db, "u1")
    eid = make_expense(db, uid)
    provider = Recorder(
        [calls(("delete_expense", {"expense_id": eid})), text("ok")]
    )

    result = await make_agent(provider, db, LOW_RISK_EXPENSE, clock).chat(
        "u1", "I spent 250 on lunch"
    )

    assert "delete_expense" not in provider.calls[0].tool_names  # never even offered
    assert result.pending_approvals == []
    assert result.tools_used == []
    assert expense_exists(db, eid)
    assert db.scalars(select(PendingAction)).all() == []


@pytest.mark.asyncio
async def test_proposing_a_missing_expense_stores_nothing(db: Session, clock: Clock):
    make_user(db, "u1")
    provider = Recorder([calls(("delete_expense", {"expense_id": 42})), text("Not found.")])

    result = await make_agent(provider, db, DELETE_REQUEST, clock).chat("u1", "delete #42")

    assert result.pending_approvals == []
    assert "Expense not found" in (provider.calls[1].messages[-1].content or "")


@pytest.mark.asyncio
async def test_proposals_per_request_are_capped(db: Session, clock: Clock):
    uid = make_user(db, "u1")
    ids = [make_expense(db, uid, f"{n}.00") for n in range(1, MAX_PROPOSALS_PER_REQUEST + 2)]
    provider = Recorder(
        [calls(*[("delete_expense", {"expense_id": i}) for i in ids]), text("Prepared.")]
    )

    result = await make_agent(provider, db, DELETE_REQUEST, clock).chat(
        "u1", "Delete all my expenses"
    )

    assert len(result.pending_approvals) == MAX_PROPOSALS_PER_REQUEST
    assert all(expense_exists(db, i) for i in ids)
    assert "Too many proposals" in (provider.calls[1].messages[-1].content or "")


@pytest.mark.asyncio
async def test_without_an_approval_service_gated_requests_are_declined(
    db: Session, clock: Clock
):
    provider = Recorder([text("should never be called")])

    result = await make_agent(
        provider, db, DELETE_REQUEST, clock, with_approvals=False
    ).chat("u1", "Delete all my expenses")

    assert result.reply == NOT_SUPPORTED_REPLY
    assert provider.calls == []