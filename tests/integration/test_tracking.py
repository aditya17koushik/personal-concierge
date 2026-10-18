"""Phase 19d: nudges when a planned meal or workout has not been logged."""

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database.repositories.profiles import ProfileRepository
from app.database.repositories.telegram import TelegramRepository
from app.database.repositories.tracking import TrackingRepository
from app.fitness.plan_service import PlanService
from app.fitness.service import FitnessService
from app.profiles.service import ProfileError
from app.scheduler.runner import SchedulerRunner
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry
from app.tracking.service import TrackingService, normalize_slot
from tests.integration.test_coach import FakeTelegram, at_ist
from tests.integration.test_plans import PlanLLM
from tests.integration.test_targets import prepare

SAT = date(2026, 10, 17)  # the generated plan trains Mon, Wed, Fri, Sat
SUN = SAT + timedelta(days=1)  # rest day
IST = "Asia/Kolkata"


def runner(factory: sessionmaker[Session], tg: FakeTelegram, now: datetime) -> SchedulerRunner:
    settings = Settings(database_url="sqlite://", app_secret_key="test-secret-key-not-for-production",
                        tracking_window_hours=2)  # fmt: skip
    return SchedulerRunner(tg, factory, settings, now_fn=lambda: now)  # type: ignore[arg-type]


async def setup_user(factory: sessionmaker[Session], with_plan: bool = True, enabled: bool = True) -> int:
    with factory() as db:
        uid = prepare(db, mode="cut")
        ProfileRepository(db).create(uid, IST, "INR", "metric")
        TelegramRepository(db).create_link(555, uid, "Test")
        if with_plan:
            await PlanService(db).create(uid, SAT, PlanLLM())
        TrackingService(db).configure(uid, enabled=enabled)
    return uid


def log_meal(factory: sessionmaker[Session], uid: int, meal_type: str, day: date = SAT) -> None:
    with factory() as db:
        FitnessService(db).log_meal(uid, description="food", calories=500, protein_g=20, carbs_g=None,
                                    fat_g=None, meal_type=meal_type, eaten_on=day, estimated=False,
                                    today=day, now=datetime.now(timezone.utc), tz_name=IST)  # fmt: skip


# -------------------------------------------------------------------- settings


def test_settings_default_off_and_validate(db: Session):
    uid = prepare(db)
    svc = TrackingService(db)
    assert svc.status(uid) == {"enabled": False, "times": {}}
    svc.configure(uid, enabled=True, times={"lunch": "13:30", "gym": "6:45"})
    st = svc.status(uid)
    assert st["enabled"] and st["times"]["lunch"] == "13:30" and st["times"]["workout"] == "06:45"
    assert st["times"]["breakfast"] == "10:00"
    with pytest.raises(ProfileError):
        svc.configure(uid, times={"lunch": "noon"})
    with pytest.raises(ProfileError):
        svc.configure(uid, times={"brunch": "11:00"})
    assert "on" in svc.status_text(uid)


def test_slot_aliases():
    assert normalize_slot("Rest") == "workout" and normalize_slot(" Lunch ") == "lunch"


# --------------------------------------------------------------------- nudges


@pytest.mark.asyncio
async def test_a_missing_meal_is_nudged_once_with_the_planned_meal(session_factory):
    uid = await setup_user(session_factory)
    tg = FakeTelegram()
    r = runner(session_factory, tg, at_ist(14, 45))  # lunch is 14:30
    await r.tracking_pass()
    await r.tracking_pass()
    assert len(tg.sent) == 1 and tg.sent[0][0] == 555
    assert "No lunch logged" in tg.texts[0] and "The plan has" in tg.texts[0]
    assert "skipped lunch" in tg.texts[0]
    with session_factory() as db:
        assert TrackingRepository(db).events(uid, SAT) == {"lunch": "nudged"}


@pytest.mark.asyncio
async def test_no_nudge_before_the_time_or_after_the_window(session_factory):
    await setup_user(session_factory)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(14, 0)).tracking_pass()
    await runner(session_factory, tg, at_ist(16, 45)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_logged_meal_means_no_nudge(session_factory):
    uid = await setup_user(session_factory)
    log_meal(session_factory, uid, "lunch")
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(14, 45)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_different_meal_does_not_satisfy_the_slot(session_factory):
    uid = await setup_user(session_factory)
    log_meal(session_factory, uid, "breakfast")
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(14, 45)).tracking_pass()
    assert len(tg.sent) == 1 and "lunch" in tg.texts[0]


@pytest.mark.asyncio
async def test_skipped_means_no_nudge(session_factory):
    uid = await setup_user(session_factory)
    with session_factory() as db:
        TrackingService(db).skip(uid, "lunch", SAT)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(14, 45)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_the_workout_is_nudged_on_a_training_day_only(session_factory):
    uid = await setup_user(session_factory)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(19, 40)).tracking_pass()  # Saturday: training
    assert len(tg.sent) == 1 and "planned workout" in tg.texts[0] and "rest today" in tg.texts[0]

    tg2 = FakeTelegram()
    await runner(session_factory, tg2, at_ist(19, 40, SUN)).tracking_pass()  # Sunday: rest
    assert tg2.sent == []
    assert uid


@pytest.mark.asyncio
async def test_a_logged_workout_or_rest_today_stops_the_nudge(session_factory):
    uid = await setup_user(session_factory)
    with session_factory() as db:
        FitnessService(db).log_workout(uid, kind="strength", description="weights", duration_minutes=45,
                                       calories_burned=None, done_on=SAT, today=SAT)  # fmt: skip
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(19, 40)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_rest_today_stops_the_workout_nudge(session_factory):
    uid = await setup_user(session_factory)
    with session_factory() as db:
        TrackingService(db).skip(uid, "rest", SAT)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(19, 40)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_no_plan_means_no_workout_nudge_but_meals_still_work(session_factory):
    await setup_user(session_factory, with_plan=False)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(19, 40)).tracking_pass()
    assert tg.sent == []
    await runner(session_factory, tg, at_ist(21, 10)).tracking_pass()
    assert len(tg.sent) == 1 and "dinner" in tg.texts[0] and "The plan has" not in tg.texts[0]


@pytest.mark.asyncio
async def test_nudges_are_opt_in(session_factory):
    await setup_user(session_factory, enabled=False)
    tg = FakeTelegram()
    await runner(session_factory, tg, at_ist(14, 45)).tracking_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_failed_send_is_retried_on_the_next_poll(session_factory):
    from app.integrations.telegram.client import TelegramError

    uid = await setup_user(session_factory)
    tg = FakeTelegram()
    tg.errors.append(TelegramError("boom"))
    r = runner(session_factory, tg, at_ist(14, 45))
    await r.tracking_pass()
    assert tg.sent == []
    with session_factory() as db:
        assert TrackingRepository(db).events(uid, SAT) == {}
    await r.tracking_pass()
    assert len(tg.sent) == 1


@pytest.mark.asyncio
async def test_the_claim_is_atomic(session_factory):
    uid = await setup_user(session_factory)
    with session_factory() as db:
        repo = TrackingRepository(db)
        assert repo.claim_nudge(uid, SAT, "lunch") is True
        assert repo.claim_nudge(uid, SAT, "lunch") is False
        repo.skip(uid, SAT, "lunch")
        assert repo.events(uid, SAT) == {"lunch": "skipped"}
        repo.release_nudge(uid, SAT, "lunch")  # only removes a 'nudged' claim
        assert repo.events(uid, SAT) == {"lunch": "skipped"}


@pytest.mark.asyncio
async def test_each_user_uses_their_own_timezone(session_factory):
    await setup_user(session_factory)
    tg = FakeTelegram()
    # 09:15 UTC is 14:45 in Kolkata
    await runner(session_factory, tg, datetime(2026, 10, 17, 9, 15, tzinfo=timezone.utc)).tracking_pass()
    assert len(tg.sent) == 1


# ----------------------------------------------------------------------- tools


def ctx(db: Session, uid: int, llm: Any = None) -> ToolContext:
    return ToolContext(db=db, user_id=uid, today=SAT, default_currency="INR", llm=llm)


@pytest.mark.asyncio
async def test_skip_and_configure_through_the_tools(db: Session):
    uid = prepare(db)
    registry = build_default_registry()
    for name in ("skip_tracking", "set_tracking_reminders", "get_tracking_reminders"):
        assert not registry.get(name).requires_approval  # type: ignore[union-attr]

    out = await registry.execute("skip_tracking", {"slot": "lunch"}, ctx(db, uid))
    assert out["skipped"] == "lunch"
    assert TrackingRepository(db).events(uid, SAT) == {"lunch": "skipped"}

    bad = await registry.execute("set_tracking_reminders", {"lunch_time": "noon"}, ctx(db, uid))
    assert "24-hour" in bad["error"]
    ok = await registry.execute("set_tracking_reminders", {"enabled": True, "lunch_time": "13:00"}, ctx(db, uid))
    assert ok["enabled"] and ok["times"]["lunch"] == "13:00"
    assert (await registry.execute("get_tracking_reminders", {}, ctx(db, uid)))["enabled"] is True


@pytest.mark.asyncio
async def test_the_first_plan_switches_reminders_on_but_respects_a_prior_choice(db: Session):
    uid = prepare(db)
    registry = build_default_registry()
    out = await registry.execute("create_plan", {}, ctx(db, uid, PlanLLM()))
    assert out["tracking_reminders"].startswith("ON") and TrackingService(db).status(uid)["enabled"]

    TrackingService(db).configure(uid, enabled=False)
    upd = await registry.execute("update_plan", {}, ctx(db, uid, PlanLLM()))
    assert upd["version"] == 2 and TrackingService(db).status(uid)["enabled"] is False


@pytest.mark.asyncio
async def test_reminders_are_private(db: Session):
    alice = prepare(db)
    from tests.integration.test_targets import make_user

    bob = make_user(db, "bob")
    TrackingService(db).configure(alice, enabled=True)
    assert TrackingService(db).status(bob)["enabled"] is False