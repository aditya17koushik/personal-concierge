"""Real token authentication (nothing here overrides get_current_user)."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.admin.__main__ import create_token, revoke_token
from app.agent.service import AgentService
from app.api.agent import get_agent_service, get_conversation_service
from app.api.approvals import get_approval_service
from app.approvals.service import ApprovalService
from app.conversation.service import ConversationService
from app.database.models import ApiToken
from app.database.repositories.api_tokens import ApiTokenRepository
from app.database.repositories.expenses import ExpenseRepository
from app.database.repositories.users import UserRepository
from app.database.session import get_db
from app.llm.base import LLMProvider
from app.main import app
from app.schemas.llm import LLMResponse, Message
from app.security.api_tokens import TOKEN_PREFIX, generate_token, hash_token
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 9)


def user_id_of(db: Session, external_id: str) -> int:
    user = UserRepository(db).get(external_id)
    assert user is not None
    return user.id


def last_used(db: Session) -> datetime:
    stamp = db.query(ApiToken).one().last_used_at
    assert stamp is not None
    return stamp.replace(tzinfo=timezone.utc)


class FakeProvider(LLMProvider):
    name = "fake"
    model = "fake-model"

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="ok", model="fake-model")


@pytest.fixture
def client(db: Session, passthrough_jev) -> Iterator[TestClient]:
    def override_db() -> Iterator[Session]:
        yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_agent_service] = lambda: AgentService(
        FakeProvider(), db, jev=passthrough_jev, today_fn=lambda: TODAY
    )
    app.dependency_overrides[get_approval_service] = lambda: ApprovalService(
        db, build_default_registry(), today_fn=lambda: TODAY
    )
    app.dependency_overrides[get_conversation_service] = lambda: ConversationService(db)
    yield TestClient(app)
    app.dependency_overrides.clear()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ------------------------------------------------------------------- tokens


def test_tokens_are_random_prefixed_and_only_hashes_are_stored(db: Session):
    plain, hashed = generate_token()
    other, _ = generate_token()

    assert plain.startswith(TOKEN_PREFIX) and plain != other
    assert hashed == hash_token(plain) == hash_token(f"  {plain}\n")
    assert plain not in hashed and len(hashed) == 64

    token = create_token(db, "alice", "test")
    stored = db.query(ApiToken).one()
    assert token not in stored.token_hash and stored.token_hash == hash_token(token)


def test_last_used_is_not_written_on_every_request(db: Session):
    token = create_token(db, "alice", "test")
    repo = ApiTokenRepository(db)
    t0 = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)

    repo.authenticate(hash_token(token), t0)
    repo.authenticate(hash_token(token), t0 + timedelta(seconds=30))
    db.expire_all()
    assert last_used(db) == t0

    repo.authenticate(hash_token(token), t0 + timedelta(minutes=5))
    db.expire_all()
    assert last_used(db) == t0 + timedelta(minutes=5)


# ------------------------------------------------------------------ the gate


def test_health_is_public(client: TestClient):
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer pa_not-a-real-token"},
        {"Authorization": "Basic YTpi"},
        {"Authorization": "pa_missing-scheme"},
    ],
)
def test_protected_endpoints_reject_missing_or_bad_tokens(
    client: TestClient, headers: dict[str, str]
):
    calls = [
        client.post("/agent/chat", json={"message": "hi"}, headers=headers),
        client.get("/approvals", headers=headers),
        client.post("/approvals/x/approve", headers=headers),
        client.post("/approvals/x/reject", headers=headers),
        client.get("/auth/google/status", headers=headers),
        client.delete("/auth/google", headers=headers),
        client.post("/auth/google/link", headers=headers),
        client.delete("/agent/history", headers=headers),
    ]
    for response in calls:
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


def test_a_valid_token_works_and_a_revoked_one_stops(db: Session, client: TestClient):
    token = create_token(db, "alice", "test")
    assert client.post("/agent/chat", json={"message": "hi"}, headers=bearer(token)).status_code == 200

    token_id = db.query(ApiToken).one().id
    assert revoke_token(db, token_id) is True

    assert client.post("/agent/chat", json={"message": "hi"}, headers=bearer(token)).status_code == 401
    assert revoke_token(db, token_id) is False  # already revoked


def test_identity_comes_from_the_token_not_the_request(db: Session, client: TestClient):
    alice = create_token(db, "alice", "test")

    client.post(
        "/agent/chat?user_id=victim",
        json={"message": "hi", "user_id": "victim"},
        headers={**bearer(alice), "X-User-Id": "victim"},
    )

    assert UserRepository(db).get("alice") is not None
    assert UserRepository(db).get("victim") is None


# ---------------------------------------------------------------- isolation


@pytest.mark.asyncio
async def test_one_user_cannot_see_or_decide_anothers_approvals(
    db: Session, client: TestClient
):
    alice_token = create_token(db, "alice", "a")
    bob_token = create_token(db, "bob", "b")
    alice_id = user_id_of(db, "alice")
    expense = ExpenseRepository(db).add(
        user_id=alice_id, amount=Decimal("5.00"), currency="INR",
        category="food", description=None, spent_on=TODAY,
    )
    service = ApprovalService(db, build_default_registry(), today_fn=lambda: TODAY)
    ctx = ToolContext(db=db, user_id=alice_id, today=TODAY, default_currency="INR", external_user_id="alice")
    _, pending = await service.propose(ctx, "delete_expense", {"expense_id": expense.id}, "x")
    assert pending is not None

    assert client.get("/approvals", headers=bearer(bob_token)).json() == []
    for action in ("approve", "reject"):
        res = client.post(f"/approvals/{pending.id}/{action}", headers=bearer(bob_token))
        assert res.status_code == 404
    assert db.get(type(expense), expense.id) is not None

    mine = client.get("/approvals", headers=bearer(alice_token)).json()
    assert [a["id"] for a in mine] == [pending.id]
    done = client.post(f"/approvals/{pending.id}/approve", headers=bearer(alice_token))
    assert done.status_code == 200 and done.json()["status"] == "executed"


# --------------------------------------------------------------- rate limits


def test_the_daily_limit_returns_429_with_retry_after(db: Session, client: TestClient, passthrough_jev):
    app.dependency_overrides[get_agent_service] = lambda: AgentService(
        FakeProvider(), db, jev=passthrough_jev, daily_message_limit=2, today_fn=lambda: TODAY
    )
    token = create_token(db, "alice", "test")

    codes = [
        client.post("/agent/chat", json={"message": "hi"}, headers=bearer(token))
        for _ in range(3)
    ]

    assert [r.status_code for r in codes] == [200, 200, 429]
    assert "limit of 2" in codes[2].json()["detail"]
    assert int(codes[2].headers["retry-after"]) > 0

    other = create_token(db, "bob", "test")  # limits are per user
    assert client.post("/agent/chat", json={"message": "hi"}, headers=bearer(other)).status_code == 200


def test_history_can_be_cleared_by_its_owner_only(db: Session, client: TestClient):
    alice_token = create_token(db, "alice", "a")
    bob_token = create_token(db, "bob", "b")
    alice_id = user_id_of(db, "alice")
    bob_id = user_id_of(db, "bob")
    now = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
    service = ConversationService(db)
    service.add_turn(alice_id, "alice q", "alice a", False, now)
    service.add_turn(bob_id, "bob q", "bob a", False, now)

    res = client.delete("/agent/history", headers=bearer(alice_token))

    assert res.status_code == 200 and res.json() == {"deleted": 2}
    assert service.recent(alice_id, now) == []
    assert len(service.recent(bob_id, now)) == 2  # untouched
    assert client.delete("/agent/history", headers=bearer(bob_token)).json() == {"deleted": 2}
    assert client.delete("/agent/history", headers=bearer(bob_token)).json() == {"deleted": 0}