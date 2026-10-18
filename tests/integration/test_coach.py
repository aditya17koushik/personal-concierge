"""Phase 18: the coach's rules, the messages built from them, the opt-in scheduler and commands."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.coach.rules import (
    CoachData,
    DayStat,
    Flag,
    consecutive_training_days,
    evaluate,
)
from app.coach.service import CoachService, normalize_weekday
from app.config import Settings
from app.database.repositories.coach import CoachSettingsRepository
from app.database.repositories.fitness import MealRepository, WeightRepository, WorkoutRepository
from app.database.repositories.profiles import ProfileRepository
from app.database.repositories.telegram import TelegramRepository
from app.database.repositories.users import UserRepository
from app.fitness.service import FitnessService
from app.integrations.telegram.client import TelegramError
from app.profiles.service import ProfileError
from app.scheduler.runner import SchedulerRunner
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry

TODAY = date(2026, 10, 17)  # a Saturday


# ------------------------------------------------------------------ builders


def week(
    calories: list[int] | None = None,
    protein: list[float] | None = None,
    workouts: list[int] | None = None,
    minutes: int = 45,
    today_meals: int | None = None,
) -> list[DayStat]:
    """Seven days, oldest first. `calories` has one value per PAST day (6); 0 means not logged."""
    cals = calories if calories is not None else [0] * 6
    prot = protein if protein is not None else [0.0] * 6
    work = workouts if workouts is not None else [0] * 7
    days: list[DayStat] = []
    for i in range(7):
        day = TODAY - timedelta(days=6 - i)
        if i < 6:
            c, p = cals[i], prot[i]
            days.append(DayStat(day, c, 1 if c else 0, p, True, work[i], minutes if work[i] else 0))
        else:
            meals = today_meals or 0
            days.append(DayStat(day, 0, meals, 0.0, True, work[i], minutes if work[i] else 0))
    return days


def data(
    days: list[DayStat] | None = None,
    mode: str | None = "cut",
    calories: int | None = 2000,
    protein: int | None = None,
    workouts: int | None = None,
    weights: list[tuple[date, float]] | None = None,
    goal_age: int = 20,
    hour: int | None = None,
    unit: str = "kg",
) -> CoachData:
    return CoachData(
        today=TODAY,
        days=days or week(),
        weights=weights or [],
        mode=mode,
        target_calories=calories,
        target_protein_g=protein,
        weekly_workouts=workouts,
        goal_started_on=TODAY - timedelta(days=goal_age) if mode else None,
        local_hour=hour,
        unit=unit,
    )


def codes(review: Any) -> list[str]:
    return [f.code for f in review.flags]


def flag(review: Any, code: str) -> Flag:
    found = [f for f in review.flags if f.code == code]
    assert found, f"{code} not in {codes(review)}"
    return found[0]


def weigh(*points: tuple[int, float]) -> list[tuple[date, float]]:
    """(days ago, kg) pairs, any order -> oldest first."""
    return sorted((TODAY - timedelta(days=ago), kg) for ago, kg in points)


# ------------------------------------------------------------------- goals


def test_without_a_goal_the_coach_asks_for_one():
    review = evaluate(data(mode=None, calories=None))
    assert codes(review) == ["no_goal"]


def test_a_goal_without_calories_says_it_cannot_judge_eating():
    review = evaluate(data(calories=None, days=week([1800] * 6)))
    assert "no_calorie_target" in codes(review)
    assert not {"under_eating", "over_eating", "intake_on_target"} & set(codes(review))


# ------------------------------------------------------------------ eating


def test_under_eating_on_a_cut():
    review = evaluate(data(days=week([1300] * 6)))
    f = flag(review, "under_eating")
    assert f.level == "warn" and "65%" in f.text and "1300" in f.text
    assert f.suggest is not None and "Add about 500 kcal" in f.suggest


def test_over_eating_on_a_cut_says_the_cut_will_stall():
    f = flag(evaluate(data(days=week([2400] * 6))), "over_eating")
    assert "120%" in f.text and "cut will stall" in (f.suggest or "")
    assert "Trim about 400" in (f.suggest or "")


def test_intake_on_target_is_praised():
    f = flag(evaluate(data(days=week([1950, 2050, 2000, 1900, 2100, 2000]))), "intake_on_target")
    assert f.level == "good"


@pytest.mark.parametrize(
    "mode,avg,expected",
    [
        ("bulk", 2000, "under_eating"),  # 80% of 2500: not enough to gain
        ("bulk", 3100, "over_eating"),  # 124%
        ("bulk", 2400, "intake_on_target"),  # 96%
        ("maintain", 2300, "over_eating"),  # 115%
        ("maintain", 2100, "intake_on_target"),
        ("cut", 1550, "intake_on_target"),  # 77.5% of 2000 is above the 75% line
    ],
)
def test_thresholds_depend_on_the_goal(mode: str, avg: int, expected: str):
    target = 2500 if mode == "bulk" else 2000
    review = evaluate(data(mode=mode, calories=target, days=week([avg] * 6)))
    assert expected in codes(review)


def test_too_few_logged_days_means_no_verdict_on_intake():
    review = evaluate(data(days=week([1300, 1300, 0, 0, 0, 0])))
    assert not {"under_eating", "over_eating", "intake_on_target"} & set(codes(review))


def test_today_is_not_counted_in_the_intake_average():
    days = week([2000] * 6, today_meals=1)  # today has a meal but 0 kcal recorded: ignored
    assert "intake_on_target" in codes(evaluate(data(days=days)))


def test_sparse_logging_is_pointed_out():
    review = evaluate(data(days=week([2000, 2000, 2000, 0, 0, 0])))
    assert "sparse_logging" in codes(review)


def test_low_protein_needs_complete_protein_numbers():
    days = week([2000] * 6, protein=[80.0] * 6)
    f = flag(evaluate(data(days=days, protein=150)), "low_protein")
    assert "about 80 g" in f.text and "70 g short" in (f.suggest or "")

    unknown = [DayStat(d.day, d.calories, d.meals, d.protein_g, False, d.workouts, d.minutes) for d in days]
    assert "low_protein" not in codes(evaluate(data(days=unknown, protein=150)))
    ok = week([2000] * 6, protein=[140.0] * 6)
    assert "low_protein" not in codes(evaluate(data(days=ok, protein=150)))


# ------------------------------------------------------------------ safety


def test_very_low_intake_stops_all_pushing_and_points_to_a_professional():
    days = week([1000] * 6, workouts=[1, 1, 0, 0, 0, 0, 0])
    review = evaluate(data(days=days, workouts=5, weights=weigh((14, 80.0), (0, 80.0))))
    assert codes(review) == ["very_low_intake"]
    f = review.flags[0]
    assert f.level == "alert" and "very low" in f.text
    assert "doctor or dietitian" in (f.suggest or "")
    assert review.facts["avg_calories"] == 1000


def test_under_training_is_not_pushed_when_intake_is_very_low():
    days = week([900] * 6, workouts=[0] * 7)
    assert "under_training" not in codes(evaluate(data(days=days, workouts=4)))


def test_a_rest_day_is_still_urged_when_intake_is_very_low_and_training_is_daily():
    days = week([1000] * 6, workouts=[1] * 7)
    review = evaluate(data(days=days))
    assert codes(review) == ["very_low_intake", "training_streak"] or set(codes(review)) == {
        "very_low_intake",
        "training_streak",
    }
    assert "even more" in flag(review, "training_streak").text


# ---------------------------------------------------------------- training


def test_training_streak_and_rest_day():
    days = week(workouts=[0, 1, 1, 1, 1, 1, 1])  # six days up to and including today
    f = flag(evaluate(data(days=days)), "training_streak")
    assert f.level == "warn" and "6 days in a row" in f.text and "rest day" in (f.suggest or "")


def test_a_full_week_without_rest_is_an_alert():
    assert flag(evaluate(data(days=week(workouts=[1] * 7))), "training_streak").level == "alert"


def test_streak_does_not_count_a_not_yet_trained_today():
    days = week(workouts=[1, 1, 1, 1, 1, 1, 0])
    assert consecutive_training_days(days) == 6
    assert consecutive_training_days(week(workouts=[1, 1, 0, 1, 1, 1, 0])) == 3
    assert consecutive_training_days(week(workouts=[1, 1, 0, 1, 1, 0, 0])) == 0  # yesterday was a rest day
    assert consecutive_training_days(week()) == 0


def test_a_heavy_week_without_a_streak_is_flagged():
    days = week(workouts=[2, 0, 2, 2, 0, 2, 2], minutes=150)  # 5 days, 10 sessions, 750 min
    f = flag(evaluate(data(days=days, workouts=4)), "high_training_load")
    assert "10 sessions" in f.text and "750 minutes" in f.text


def test_under_training_against_the_weekly_target():
    days = week(workouts=[0, 1, 0, 0, 0, 0, 0])
    f = flag(evaluate(data(days=days, workouts=4)), "under_training")
    assert "1 session in the last 7 days against a target of 4" in f.text


def test_a_new_goal_is_not_called_under_training_yet():
    days = week(workouts=[0] * 7)
    assert "under_training" not in codes(evaluate(data(days=days, workouts=4, goal_age=2)))


def test_training_on_target_is_praised():
    days = week(workouts=[1, 0, 1, 0, 1, 0, 1])
    assert flag(evaluate(data(days=days, workouts=4)), "training_on_track").level == "good"


def test_heavy_training_with_low_intake_is_a_recovery_risk():
    days = week([1400] * 6, workouts=[0, 1, 1, 1, 1, 1, 1])
    review = evaluate(data(days=days))
    assert {"under_eating", "training_streak", "recovery_risk"} <= set(codes(review))
    assert review.flags[0].code == "recovery_risk"  # alerts sort first


# ------------------------------------------------------------------ weight


@pytest.mark.parametrize(
    "mode,points,expected",
    [
        ("cut", [(14, 82.0), (0, 80.9)], "weight_on_track"),  # -0.55 kg/week, -0.7%
        ("cut", [(14, 80.0), (0, 81.5)], "weight_wrong_way"),
        ("cut", [(14, 80.0), (0, 80.1)], "cut_stalled"),
        ("cut", [(7, 85.0), (0, 83.8)], "losing_too_fast"),
        ("bulk", [(14, 70.0), (0, 70.6)], "weight_on_track"),
        ("bulk", [(14, 70.0), (0, 69.0)], "weight_wrong_way"),
        ("bulk", [(7, 70.0), (0, 70.7)], "bulking_too_fast"),
        ("maintain", [(14, 70.0), (0, 70.1)], "weight_on_track"),
        ("maintain", [(14, 70.0), (0, 71.2)], "weight_drifting"),
    ],
)
def test_weight_trend_verdicts(mode: str, points: list[tuple[int, float]], expected: str):
    review = evaluate(data(mode=mode, weights=weigh(*points)))
    assert expected in codes(review)


def test_weight_needs_two_weigh_ins_a_week_apart():
    for points in ([(0, 80.0)], [(3, 80.0), (0, 79.0)]):
        review = evaluate(data(weights=weigh(*points)))
        assert not {"weight_on_track", "cut_stalled", "weight_wrong_way", "losing_too_fast"} & set(codes(review))


def test_a_stalled_cut_with_high_intake_blames_the_intake():
    days = week([2500] * 6)
    f = flag(evaluate(data(days=days, weights=weigh((14, 80.0), (0, 80.1)))), "cut_stalled")
    assert "intake is above target" in (f.suggest or "")


def test_a_stalled_cut_with_on_target_intake_says_check_logging_first():
    days = week([2000] * 6)
    f = flag(evaluate(data(days=days, weights=weigh((14, 80.0), (0, 80.1)))), "cut_stalled")
    assert "logging is complete" in (f.suggest or "")


def test_weights_are_shown_in_the_users_unit():
    f = flag(evaluate(data(unit="lb", weights=weigh((14, 80.0), (0, 81.5)))), "weight_wrong_way")
    assert "lb/week" in f.text


def test_a_stale_weigh_in_is_mentioned():
    review = evaluate(data(weights=weigh((12, 80.0))))
    assert "weigh_in_stale" in codes(review)
    assert "weigh_in_stale" not in codes(evaluate(data(weights=weigh((3, 80.0)))))


# ------------------------------------------------------------------- today


def today_review(**kw: Any) -> Any:
    return evaluate(data(**kw), scope="today")


def test_nothing_logged_by_the_afternoon():
    assert "nothing_logged_today" in codes(today_review(hour=15))
    assert "nothing_logged_today" not in codes(today_review(hour=11))


def test_a_big_calorie_gap_in_the_evening():
    days = week([2000] * 6)
    days[-1] = DayStat(TODAY, 800, 2)
    f = flag(today_review(days=days, hour=20), "calories_short_today")
    assert "800 kcal of 2000" in f.text and "1200 short" in f.text


def test_a_big_day_is_not_shamed():
    days = week([2000] * 6)
    days[-1] = DayStat(TODAY, 2500, 3)
    f = flag(today_review(days=days, hour=20), "calories_over_today")
    assert f.level == "info" and "One day does not change your trend" in (f.suggest or "")


def test_todays_protein_gap_in_the_evening():
    days = week([2000] * 6)
    days[-1] = DayStat(TODAY, 1500, 3, 60.0, True)
    assert "protein_gap_today" in codes(today_review(days=days, protein=150, hour=20))
    assert "protein_gap_today" not in codes(today_review(days=days, protein=150, hour=12))


def test_the_trend_review_has_no_same_day_flags():
    assert "nothing_logged_today" not in codes(evaluate(data(hour=20)))


def test_alerts_come_before_warnings_before_good_news():
    days = week([1400] * 6, workouts=[0, 1, 1, 1, 1, 1, 1])
    levels = [f.level for f in evaluate(data(days=days)).flags]
    assert levels == sorted(levels, key=["alert", "warn", "info", "good"].index)


# ---------------------------------------------------------- service and text


def make_user(db: Session, external_id: str = "me") -> int:
    return UserRepository(db).get_or_create(external_id).id


def log_days(db: Session, uid: int, calories: list[int], protein: float | None = None) -> None:
    """One meal a day for the 6 days before TODAY (oldest first)."""
    meals = MealRepository(db)
    for i, kcal in enumerate(calories):
        if kcal:
            meals.add(uid, TODAY - timedelta(days=6 - i), "lunch", "meal", kcal,
                      Decimal(str(protein)) if protein else None, None, None, True)  # fmt: skip


def set_goal(db: Session, uid: int, **kw: Any) -> None:
    args: dict[str, Any] = dict(mode="cut", target_weight=None, unit=None, daily_calories=2000,
                                daily_protein_g=None, weekly_workouts=None,
                                today=TODAY - timedelta(days=20))  # fmt: skip
    args.update(kw)
    FitnessService(db).set_goal(uid, **args)


def test_service_builds_seven_days_of_data(db: Session):
    uid = make_user(db)
    log_days(db, uid, [1800, 0, 2200, 2000, 0, 1900])
    WorkoutRepository(db).add(uid, TODAY - timedelta(days=1), "run", "5k", 30, None)
    WeightRepository(db).upsert(uid, TODAY - timedelta(days=10), Decimal("80.00"))
    WeightRepository(db).upsert(uid, TODAY, Decimal("79.00"))
    set_goal(db, uid, daily_protein_g=120, weekly_workouts=3)

    d = CoachService(db).data(uid, TODAY)
    assert [s.day for s in d.days][0] == TODAY - timedelta(days=6) and d.days[-1].day == TODAY
    assert [s.calories for s in d.days] == [1800, 0, 2200, 2000, 0, 1900, 0]
    assert d.days[-2].workouts == 1 and d.days[-2].minutes == 30
    assert d.weights == [(TODAY - timedelta(days=10), 80.0), (TODAY, 79.0)]
    assert (d.mode, d.target_calories, d.target_protein_g, d.weekly_workouts) == ("cut", 2000, 120, 3)


def test_the_weekly_text_has_numbers_and_a_fix_first_line(db: Session):
    uid = make_user(db)
    log_days(db, uid, [1300] * 6)
    set_goal(db, uid, weekly_workouts=3)
    WeightRepository(db).upsert(uid, TODAY - timedelta(days=14), Decimal("80.00"))
    WeightRepository(db).upsert(uid, TODAY, Decimal("79.00"))

    text = CoachService(db).weekly_text(uid, TODAY)
    assert text.startswith("Weekly review (last 7 days)")
    assert "Weight: 79.0 kg (-0.5 kg/week)" in text
    assert "Food: average 1300 kcal on 6 logged days (target 2000)" in text
    assert "Training: 0 sessions, 0 min, 7 rest days (target 3)" in text
    assert "Fix first: Under-eating" in text


def test_the_weekly_text_with_no_data_asks_for_logging(db: Session):
    uid = make_user(db)
    text = CoachService(db).weekly_text(uid, TODAY)
    assert "not enough logged days" in text and "You have not set a goal yet." in text


def test_the_checkin_shows_today_against_the_target(db: Session):
    uid = make_user(db)
    set_goal(db, uid, daily_protein_g=150, weekly_workouts=4)
    MealRepository(db).add(uid, TODAY, "lunch", "rice", 700, Decimal("30"), None, None, True)
    text = CoachService(db).checkin_text(uid, TODAY, 20)
    assert text.startswith("Evening check-in")
    assert "Today: 700 kcal of 2000 (1300 left)" in text
    assert "Protein: 30 g of 150 g" in text
    assert "No workout logged today. This week: 0 (target 4)." in text
    assert "700 kcal of 2000 today, 1300 short" in text


def test_the_checkin_on_a_good_day_just_encourages(db: Session):
    uid = make_user(db)
    set_goal(db, uid)
    log_days(db, uid, [2000] * 6)
    MealRepository(db).add(uid, TODAY, "dinner", "dal rice", 1900, None, None, None, False)
    text = CoachService(db).checkin_text(uid, TODAY, 20)
    assert "Going well:" in text and "Nothing needs fixing" in text


def test_the_review_for_the_model_carries_flags_facts_and_a_safety_switch(db: Session):
    uid = make_user(db)
    log_days(db, uid, [1000] * 6)
    set_goal(db, uid)
    out = CoachService(db).review_out(uid, TODAY)
    assert out["safety"] is True
    assert out["flags"][0]["code"] == "very_low_intake"
    assert out["facts"]["avg_calories"] == 1000 and out["facts"]["calorie_target"] == 2000
    assert "do not push" in out["note"]


@pytest.mark.asyncio
async def test_get_coaching_tool_uses_the_users_own_data(db: Session):
    alice, bob = make_user(db, "alice"), make_user(db, "bob")
    log_days(db, alice, [1300] * 6)
    set_goal(db, alice)
    registry = build_default_registry()
    assert registry.get("get_coaching") is not None and not registry.get("get_coaching").requires_approval  # type: ignore[union-attr]

    def ctx(uid: int) -> ToolContext:
        return ToolContext(db=db, user_id=uid, today=TODAY, default_currency="INR")

    mine = await registry.execute("get_coaching", {}, ctx(alice))
    assert "under_eating" in [f["code"] for f in mine["flags"]]
    theirs = await registry.execute("get_coaching", {}, ctx(bob))
    assert [f["code"] for f in theirs["flags"]] == ["no_goal"]


# -------------------------------------------------------------- goal merging


def test_changing_one_target_keeps_the_rest(db: Session):
    uid = make_user(db)
    svc = FitnessService(db)
    set_goal(db, uid, daily_calories=2000, daily_protein_g=150, weekly_workouts=4, target_weight=72)
    goal, _, _ = svc.set_goal(uid, mode=None, target_weight=None, unit=None, daily_calories=1900,
                              daily_protein_g=None, weekly_workouts=None, today=TODAY)  # fmt: skip
    assert goal.mode == "cut" and goal.daily_calories == 1900
    assert goal.daily_protein_g == 150 and goal.weekly_workouts == 4
    assert goal.target_weight_kg == Decimal("72.00")


def test_a_new_mode_drops_the_old_target_weight_and_a_goal_needs_a_mode(db: Session):
    from app.fitness.service import FitnessError

    uid = make_user(db)
    svc = FitnessService(db)
    set_goal(db, uid, target_weight=72)
    goal, _, _ = svc.set_goal(uid, mode="bulk", target_weight=None, unit=None, daily_calories=None,
                              daily_protein_g=None, weekly_workouts=None, today=TODAY)  # fmt: skip
    assert goal.mode == "bulk" and goal.target_weight_kg is None and goal.daily_calories == 2000

    with pytest.raises(FitnessError):
        svc.set_goal(make_user(db, "new"), mode=None, target_weight=None, unit=None,
                     daily_calories=1800, daily_protein_g=None, weekly_workouts=None, today=TODAY)  # fmt: skip


# ----------------------------------------------------------------- settings


def test_coach_settings_are_validated(db: Session):
    uid = make_user(db)
    svc = CoachService(db)
    row = svc.configure(uid, enabled=True, time_text="7:30")
    assert row.enabled and row.checkin_time == "07:30" and row.weekly_day == 6
    assert svc.configure(uid, weekday="Wed").weekly_day == 2
    assert normalize_weekday("sunday") == 6 and normalize_weekday("mon") == 0
    for bad in ("25:00", "noon", "8"):
        with pytest.raises(ProfileError):
            svc.configure(uid, time_text=bad)
    with pytest.raises(ProfileError):
        svc.configure(uid, weekday="someday")
    assert svc.configure(uid, enabled=False).enabled is False


def test_the_daily_claim_is_atomic(db: Session):
    uid = make_user(db)
    CoachService(db).configure(uid, enabled=True)
    repo = CoachSettingsRepository(db)
    assert repo.claim_send(uid, TODAY) is True
    assert repo.claim_send(uid, TODAY) is False
    assert repo.claim_send(uid, TODAY + timedelta(days=1)) is True
    repo.release_send(uid, TODAY)
    assert repo.claim_send(uid, TODAY + timedelta(days=1)) is True


# ---------------------------------------------------------------- scheduler

IST = "Asia/Kolkata"


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.errors: list[TelegramError] = []

    async def send_message(self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append((chat_id, text))
        return {"message_id": len(self.sent)}

    @property
    def texts(self) -> list[str]:
        return [t for _, t in self.sent]


def at_ist(hour: int, minute: int = 0, day: date = TODAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def coach_runner(factory: sessionmaker[Session], tg: FakeTelegram, now: datetime) -> SchedulerRunner:
    settings = Settings(
        database_url="sqlite://",
        app_secret_key="test-secret-key-not-for-production",
        coach_window_hours=2,
    )
    return SchedulerRunner(tg, factory, settings, now_fn=lambda: now)  # type: ignore[arg-type]


@pytest.fixture
def coach_user(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as db:
        uid = make_user(db)
        ProfileRepository(db).create(uid, IST, "INR", "metric")
        TelegramRepository(db).create_link(555, uid, "Test")
        CoachService(db).configure(uid, enabled=True, time_text="20:00")
        set_goal(db, uid)
    return uid


@pytest.mark.asyncio
async def test_the_checkin_is_sent_once_inside_its_window(session_factory, coach_user):
    tg = FakeTelegram()
    runner = coach_runner(session_factory, tg, at_ist(20, 10))  # a Saturday: not the weekly day
    await runner.coach_pass()
    await runner.coach_pass()

    assert len(tg.sent) == 1 and tg.sent[0][0] == 555
    assert tg.texts[0].startswith("Evening check-in")


@pytest.mark.asyncio
async def test_no_message_before_the_time_or_after_the_window(session_factory, coach_user):
    tg = FakeTelegram()
    await coach_runner(session_factory, tg, at_ist(19, 30)).coach_pass()
    await coach_runner(session_factory, tg, at_ist(22, 30)).coach_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_the_weekly_review_replaces_the_checkin_on_its_day(session_factory, coach_user):
    sunday = TODAY + timedelta(days=1)
    tg = FakeTelegram()
    await coach_runner(session_factory, tg, at_ist(20, 5, sunday)).coach_pass()
    assert len(tg.sent) == 1 and tg.texts[0].startswith("Weekly review")


@pytest.mark.asyncio
async def test_coach_messages_are_opt_in(session_factory, coach_user):
    with session_factory() as db:
        CoachService(db).configure(coach_user, enabled=False)
    tg = FakeTelegram()
    await coach_runner(session_factory, tg, at_ist(20, 10)).coach_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_user_without_a_telegram_link_is_skipped(session_factory):
    with session_factory() as db:
        uid = make_user(db, "web-only")
        ProfileRepository(db).create(uid, IST, "INR", "metric")
        CoachService(db).configure(uid, enabled=True, time_text="20:00")
    tg = FakeTelegram()
    await coach_runner(session_factory, tg, at_ist(20, 10)).coach_pass()
    assert tg.sent == []


@pytest.mark.asyncio
async def test_a_failed_send_is_tried_again_on_the_next_poll(session_factory, coach_user):
    tg = FakeTelegram()
    runner = coach_runner(session_factory, tg, at_ist(20, 10))
    tg.errors = [TelegramError("down", status=502)]
    await runner.coach_pass()
    assert tg.sent == []
    await runner.coach_pass()
    assert len(tg.sent) == 1


@pytest.mark.asyncio
async def test_the_scheduler_tick_runs_the_coach_pass(session_factory, coach_user):
    tg = FakeTelegram()
    await coach_runner(session_factory, tg, at_ist(20, 10)).tick()
    assert any(t.startswith("Evening check-in") for t in tg.texts)