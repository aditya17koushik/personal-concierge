from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.agent.service import AgentService
from app.database.models import Expense
from app.database.repositories.profiles import ProfileRepository
from app.database.repositories.usage import UsageRepository
from app.database.repositories.users import UserRepository
from app.llm.base import LLMProvider
from app.profiles.service import (
    ProfileError,
    ProfileService,
    local_today,
    normalize_currency,
    normalize_timezone,
    normalize_units,
    seconds_until_local_midnight,
)
from app.schemas.llm import LLMResponse, Message, ToolCall
from app.usage.service import QuotaService, RateLimitExceeded

NOW = datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc)  # 01:30 on Oct 10 in India


# ---------------------------------------------------------------- validation


def test_timezone_names_are_canonicalised_and_validated():
    assert normalize_timezone("asia/kolkata") == "Asia/Kolkata"
    assert normalize_timezone("  Europe/London ") == "Europe/London"
    assert normalize_timezone("UTC") == "UTC"
    for bad in ["", "Mars/Olympus", "../etc/passwd", "IST", "x" * 200]:
        with pytest.raises(ProfileError):
            normalize_timezone(bad)


def test_currency_and_units_are_validated():
    assert normalize_currency(" inr ") == "INR"
    for bad in ["rupees", "US", "12A", "", "I-N"]:
        with pytest.raises(ProfileError):
            normalize_currency(bad)
    assert normalize_units("Imperial") == "imperial"
    with pytest.raises(ProfileError):
        normalize_units("furlongs")


def test_local_date_follows_the_users_timezone():
    assert local_today("UTC", NOW) == date(2026, 10, 9)
    assert local_today("Asia/Kolkata", NOW) == date(2026, 10, 10)
    assert local_today("America/Los_Angeles", NOW) == date(2026, 10, 9)


def test_seconds_until_local_midnight():
    assert seconds_until_local_midnight("Asia/Kolkata", NOW) == 22 * 3600 + 30 * 60
    assert seconds_until_local_midnight("UTC", NOW) == 4 * 3600
    almost = datetime(2026, 10, 9, 23, 59, 59, 999999, tzinfo=timezone.utc)
    assert seconds_until_local_midnight("UTC", almost) >= 1


# ------------------------------------------------------------------ service


def test_profiles_are_created_lazily_with_defaults_and_are_stable(db: Session):
    uid = UserRepository(db).get_or_create("alice").id
    service = ProfileService(db, default_currency="inr", default_timezone="asia/kolkata")

    first = service.get_or_create(uid)
    again = service.get_or_create(uid)

    assert first == again
    assert (first.timezone, first.currency, first.unit_system) == ("Asia/Kolkata", "INR", "metric")


def test_updates_persist_and_invalid_ones_change_nothing(db: Session):
    uid = UserRepository(db).get_or_create("alice").id
    service = ProfileService(db)

    updated = service.update(uid, timezone="europe/london", currency="gbp", unit_system="imperial")
    assert (updated.timezone, updated.currency, updated.unit_system) == ("Europe/London", "GBP", "imperial")

    with pytest.raises(ProfileError):
        service.update(uid, timezone="Mars/Olympus", currency="USD")  # nothing applied
    assert service.get_or_create(uid) == updated


def test_bad_configured_defaults_fall_back_safely(db: Session):
    uid = UserRepository(db).get_or_create("alice").id
    profile = ProfileService(db, default_currency="rupees", default_timezone="Nowhere/Land").get_or_create(uid)
    assert (profile.timezone, profile.currency) == ("UTC", "USD")


def test_a_racing_profile_creation_returns_the_existing_one(session_factory):
    # two requests from a brand-new user, each in its own session
    with session_factory() as first_session, session_factory() as second_session:
        uid = UserRepository(first_session).get_or_create("alice").id
        first = ProfileRepository(second_session).create(uid, "UTC", "INR", "metric")

        second = ProfileRepository(first_session).create(uid, "Asia/Tokyo", "JPY", "metric")

        assert second.user_id == first.user_id
        assert (second.timezone, second.currency) == ("UTC", "INR")  # the first one wins


# --------------------------------------------------------------- daily limit


def test_the_limit_allows_exactly_n_messages_per_user_per_day(db: Session):
    a = UserRepository(db).get_or_create("a").id
    b = UserRepository(db).get_or_create("b").id
    repo = UsageRepository(db)
    day, next_day = date(2026, 10, 9), date(2026, 10, 10)

    assert [repo.try_consume(a, day, 3) for _ in range(5)] == [True, True, True, False, False]
    assert repo.count(a, day) == 3  # denied attempts are not counted
    assert repo.try_consume(b, day, 3) is True  # per user
    assert repo.try_consume(a, next_day, 3) is True  # resets each day


def test_a_limit_of_zero_disables_it(db: Session):
    uid = UserRepository(db).get_or_create("a").id
    quota = QuotaService(db, 0)
    for _ in range(10):
        quota.consume(uid, date(2026, 10, 9), 60)
    assert quota.used(uid, date(2026, 10, 9)) == 0


def test_exceeding_the_limit_reports_when_it_resets(db: Session):
    uid = UserRepository(db).get_or_create("a").id
    quota = QuotaService(db, 1)
    quota.consume(uid, date(2026, 10, 9), 120)

    with pytest.raises(RateLimitExceeded) as exc:
        quota.consume(uid, date(2026, 10, 9), 120)

    assert exc.value.limit == 1 and exc.value.retry_after_seconds == 120


# ------------------------------------------------------------ agent behaviour


class Scripted(LLMProvider):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self._responses = responses or [LLMResponse(content="ok", model="scripted-model")]
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
async def test_the_agent_uses_the_users_profile_for_date_and_currency(db: Session, passthrough_jev):
    uid = UserRepository(db).get_or_create("alice").id
    ProfileService(db).update(uid, timezone="Asia/Kolkata", currency="usd")
    provider = Scripted(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="c1", name="add_expense", arguments={"amount": 5, "category": "food"})],
                model="scripted-model",
            ),
            LLMResponse(content="Saved.", model="scripted-model"),
        ]
    )
    agent = AgentService(provider, db, jev=passthrough_jev, now_fn=lambda: NOW)

    await agent.chat("alice", "I spent 5 on tea")

    prompt = provider.calls[0][0].content or ""
    assert "Today's date is 2026-10-10 (the user's timezone is Asia/Kolkata)" in prompt
    assert "default currency is USD" in prompt
    expense = db.query(Expense).one()
    assert expense.currency == "USD" and expense.spent_on == date(2026, 10, 10)


@pytest.mark.asyncio
async def test_new_users_get_the_configured_defaults(db: Session, passthrough_jev):
    provider = Scripted()
    agent = AgentService(
        provider, db, jev=passthrough_jev,
        default_currency="EUR", default_timezone="Europe/Paris", now_fn=lambda: NOW,
    )

    await agent.chat("newcomer", "hello")

    prompt = provider.calls[0][0].content or ""
    assert "Europe/Paris" in prompt and "default currency is EUR" in prompt


class CountingJev:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0

    async def decide(self, message: str, registry: Any) -> Any:
        self.calls += 1
        return await self.inner.decide(message, registry)


@pytest.mark.asyncio
async def test_a_user_over_the_limit_costs_nothing(db: Session, passthrough_jev):
    provider = Scripted()
    jev = CountingJev(passthrough_jev)
    agent = AgentService(
        provider, db, jev=jev, daily_message_limit=2, today_fn=lambda: date(2026, 10, 9), now_fn=lambda: NOW
    )

    await agent.chat("alice", "one")
    await agent.chat("alice", "two")
    with pytest.raises(RateLimitExceeded) as exc:
        await agent.chat("alice", "three")

    assert (jev.calls, len(provider.calls)) == (2, 2)  # no Jev call, no model call
    assert exc.value.limit == 2 and exc.value.retry_after_seconds > 0

    await agent.chat("bob", "hello")  # someone else is unaffected
    assert jev.calls == 3


@pytest.mark.asyncio
async def test_the_limit_resets_on_the_users_local_day(db: Session, passthrough_jev):
    uid = UserRepository(db).get_or_create("alice").id
    ProfileService(db).update(uid, timezone="Asia/Kolkata")
    provider = Scripted()
    times = [datetime(2026, 10, 9, 17, 0, tzinfo=timezone.utc)]  # 22:30 IST, Oct 9
    agent = AgentService(
        provider, db, jev=passthrough_jev, daily_message_limit=1, now_fn=lambda: times[0]
    )

    await agent.chat("alice", "late evening")
    with pytest.raises(RateLimitExceeded):
        await agent.chat("alice", "again")

    times[0] = datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc)  # 00:30 IST, Oct 10
    await agent.chat("alice", "after local midnight")  # a new local day: allowed