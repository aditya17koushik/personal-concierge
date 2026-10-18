"""Phase 19b: 7-day workout/meal plan generation, validation, versioning and adaptation."""

import copy
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.coach.rules import CoachData, DayStat, Flag, Review, evaluate
from app.database.repositories.fitness import GoalRepository, PlanRepository, WeightRepository
from app.fitness.plan_service import (
    PlanService,
    adaptation_for,
    format_plan_today,
    format_plan_week,
)
from app.fitness.plans import DAYS, PlanError, PlanGenerator, PlanInputs
from app.llm.base import LLMError, LLMProvider
from app.schemas.llm import LLMResponse, Message
from app.tools.base import ToolContext
from app.tools.registry import build_default_registry
from tests.integration.test_targets import full_profile, make_user, prepare

TODAY = date(2026, 10, 17)  # a Saturday


# ----------------------------------------------------------------- fixtures


def good_workout(days: int = 4) -> dict[str, Any]:
    train = {0, 1, 3, 4, 2, 5}  # which weekdays train, in order of preference
    order = [0, 2, 4, 5, 1, 3]
    chosen = set(order[:days])
    out = []
    for i, name in enumerate(DAYS):
        if i in chosen:
            out.append({"day": name, "focus": "Full body", "rest": False, "minutes": 55, "exercises": [
                {"name": "Warm-up jog in place", "sets": 1, "reps": "5 min"},
                {"name": "Goblet squat", "sets": 3, "reps": "8-10"},
                {"name": "Push-up", "sets": 3, "reps": "10"},
                {"name": "Dumbbell row", "sets": 3, "reps": "10"},
            ]})  # fmt: skip
        else:
            out.append({"day": name, "focus": "Rest", "rest": True, "minutes": None, "exercises": []})
    assert train
    return {"summary": "Full body, simple.", "progression": "Add a rep each week.", "days": out}


def good_meals(calories: int = 2200, protein: int = 165) -> dict[str, Any]:
    def day(name: str) -> dict[str, Any]:
        share = [0.23, 0.32, 0.32, 0.13]
        types = ["breakfast", "lunch", "dinner", "snack"]
        return {"day": name, "meals": [
            {"type": t, "name": f"{t.title()} bowl", "items": "dal, rice, tofu, vegetables",
             "calories": round(calories * s), "protein_g": round(protein * s)}
            for t, s in zip(types, share)
        ]}  # fmt: skip

    return {"notes": "Simple Indian vegetarian.", "days": [day(d) for d in DAYS]}


class PlanLLM(LLMProvider):
    """Answers workout and meal prompts from separate queues (the two run concurrently)."""

    name = "plan"
    model = "plan-model"

    def __init__(self, workouts: list[Any] | None = None, meals: list[Any] | None = None) -> None:
        self.workouts = workouts or [good_workout()]
        self.meals = meals or [good_meals()]
        self.workout_calls = 0
        self.meal_calls = 0
        self.prompts: list[str] = []

    async def chat(self, messages: list[Message], tools: Any = None, temperature: float = 0.2,
                   max_tokens: int | None = None) -> LLMResponse:  # fmt: skip
        self.prompts.append(messages[-1].content or "")
        if "workout plan" in (messages[0].content or ""):
            queue, i = self.workouts, self.workout_calls
            self.workout_calls += 1
        else:
            queue, i = self.meals, self.meal_calls
            self.meal_calls += 1
        item = queue[min(i, len(queue) - 1)]
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResponse(content=text, model="plan-model")


def inputs(**kw: Any) -> PlanInputs:
    base: dict[str, Any] = dict(sex="male", age=30, height_cm=180, weight_kg=80.0, mode="cut",
                                experience="intermediate", days_per_week=4, equipment="gym",
                                diet_pref="vegetarian", calories=2200, protein_g=165, carbs_g=220,
                                fat_g=70)  # fmt: skip
    base.update(kw)
    return PlanInputs(**base)


# ---------------------------------------------------------------- generator


@pytest.mark.asyncio
async def test_a_good_plan_passes_first_time():
    llm = PlanLLM()
    workout, meals = await PlanGenerator(llm).both(inputs())
    assert len(workout["days"]) == 7 and len(meals["days"]) == 7
    assert (llm.workout_calls, llm.meal_calls) == (1, 1)


@pytest.mark.asyncio
async def test_bad_json_is_retried_with_the_error_fed_back():
    llm = PlanLLM(workouts=["Sure! here is your plan", good_workout()])
    workout, _ = await PlanGenerator(llm).both(inputs())
    assert llm.workout_calls == 2 and len(workout["days"]) == 7
    assert "REJECTED" in llm.prompts[-1] or any("REJECTED" in p for p in llm.prompts)


@pytest.mark.asyncio
async def test_json_inside_code_fences_is_accepted():
    llm = PlanLLM(workouts=["```json\n" + json.dumps(good_workout()) + "\n```"])
    await PlanGenerator(llm).workout(inputs())
    assert llm.workout_calls == 1


@pytest.mark.asyncio
async def test_wrong_number_of_training_days_is_rejected():
    llm = PlanLLM(workouts=[good_workout(5), good_workout(4)])
    await PlanGenerator(llm).workout(inputs(days_per_week=4))
    assert llm.workout_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "diet,food",
    [("vegetarian", "grilled chicken"), ("vegetarian", "boiled eggs"), ("vegan", "paneer"),
     ("vegan", "curd"), ("eggetarian", "fish curry")],
)  # fmt: skip
async def test_diet_violations_are_rejected(diet: str, food: str):
    bad = good_meals()
    bad["days"][2]["meals"][1]["items"] = food
    llm = PlanLLM(meals=[bad, good_meals()])
    await PlanGenerator(llm).meals(inputs(diet_pref=diet))
    assert llm.meal_calls == 2


@pytest.mark.asyncio
async def test_vegan_safe_phrases_are_not_false_positives():
    ok = good_meals()
    ok["days"][0]["meals"][0]["items"] = "oats, peanut butter, coconut milk"
    llm = PlanLLM(meals=[ok])
    await PlanGenerator(llm).meals(inputs(diet_pref="vegan"))
    assert llm.meal_calls == 1


@pytest.mark.asyncio
async def test_allergens_are_rejected():
    bad = good_meals()
    bad["days"][4]["meals"][0]["items"] = "peanut chutney and idli"
    llm = PlanLLM(meals=[bad, good_meals()])
    await PlanGenerator(llm).meals(inputs(allergies="peanuts, soy"))
    assert llm.meal_calls == 2


@pytest.mark.asyncio
async def test_calories_and_protein_must_fit_the_target():
    too_big = good_meals(calories=3000)
    low_protein = good_meals(protein=60)
    llm = PlanLLM(meals=[too_big, low_protein, good_meals()])
    await PlanGenerator(llm).meals(inputs())
    assert llm.meal_calls == 3


@pytest.mark.asyncio
async def test_equipment_is_enforced():
    bad = good_workout()
    bad["days"][0]["exercises"][1]["name"] = "Barbell back squat"
    llm = PlanLLM(workouts=[bad, good_workout()])
    await PlanGenerator(llm).workout(inputs(equipment="home_dumbbells"))
    assert llm.workout_calls == 2


@pytest.mark.asyncio
async def test_injuries_limit_exercises():
    bad = good_workout()
    bad["days"][0]["exercises"][1]["name"] = "Jump squat"
    llm = PlanLLM(workouts=[bad, good_workout()])
    await PlanGenerator(llm).workout(inputs(movement_notes=["knee"]))
    assert llm.workout_calls == 2


@pytest.mark.asyncio
async def test_rest_days_must_be_empty_and_sessions_not_too_long():
    bad = good_workout()
    rest = next(d for d in bad["days"] if d["rest"])
    rest["exercises"] = [{"name": "Plank", "sets": 3, "reps": "30 s"}]
    long = good_workout()
    long["days"][0]["minutes"] = 150
    llm = PlanLLM(workouts=[bad, long, good_workout()])
    await PlanGenerator(llm).workout(inputs())
    assert llm.workout_calls == 3


@pytest.mark.asyncio
async def test_three_failures_give_an_honest_error():
    llm = PlanLLM(meals=[good_meals(calories=4000)])
    with pytest.raises(PlanError) as info:
        await PlanGenerator(llm).meals(inputs())
    assert "3 tries" in str(info.value) and llm.meal_calls == 3


@pytest.mark.asyncio
async def test_provider_failure_is_a_plan_error():
    llm = PlanLLM(workouts=[LLMError("down")])
    with pytest.raises(PlanError):
        await PlanGenerator(llm).workout(inputs())


# ------------------------------------------------------------------ service


@pytest.mark.asyncio
async def test_creating_a_plan_stores_it_and_aligns_the_goal(db: Session):
    uid = prepare(db, mode="cut")
    plan = await PlanService(db).create(uid, TODAY, PlanLLM())
    assert plan.version == 1 and plan.status == "active"
    assert plan.targets["calories"] == 2200 and plan.targets["protein_g"] == 160
    goal = GoalRepository(db).get(uid)
    assert goal is not None and goal.daily_calories == 2200 and goal.weekly_workouts == 4
    assert goal.mode == "cut"


@pytest.mark.asyncio
async def test_meals_are_checked_against_the_computed_targets(db: Session):
    uid = prepare(db, mode="cut")
    llm = PlanLLM(meals=[good_meals(calories=2800), good_meals(calories=2200, protein=160)])
    plan = await PlanService(db).create(uid, TODAY, llm)
    assert llm.meal_calls == 2 and plan.version == 1


@pytest.mark.asyncio
async def test_a_refused_profile_gets_no_plan(db: Session):
    uid = prepare(db, conditions="eating disorder", age=24)
    llm = PlanLLM()
    with pytest.raises(PlanError):
        await PlanService(db).create(uid, TODAY, llm)
    assert llm.workout_calls == 0 and PlanRepository(db).active(uid) is None


@pytest.mark.asyncio
async def test_an_incomplete_profile_gets_no_plan(db: Session):
    uid = make_user(db)
    with pytest.raises(PlanError) as info:
        await PlanService(db).create(uid, TODAY, PlanLLM())
    assert "age" in str(info.value)


@pytest.mark.asyncio
async def test_injury_notes_reach_the_generator(db: Session):
    uid = prepare(db, conditions="knee pain")
    llm = PlanLLM()
    await PlanService(db).create(uid, TODAY, llm)
    assert any("knee" in p.lower() for p in llm.prompts)


@pytest.mark.asyncio
async def test_a_failed_generation_leaves_no_half_plan(db: Session):
    uid = prepare(db)
    with pytest.raises(PlanError):
        await PlanService(db).create(uid, TODAY, PlanLLM(meals=[good_meals(calories=4000)]))
    assert PlanRepository(db).active(uid) is None
    assert GoalRepository(db).get(uid).daily_calories is None  # type: ignore[union-attr]


def backdate(db: Session, uid: int, days: int) -> None:
    plan = PlanRepository(db).active(uid)
    assert plan is not None
    plan.created_at = datetime.now(timezone.utc) - timedelta(days=days)
    db.commit()


@pytest.mark.asyncio
async def test_update_archives_the_old_plan_and_versions_up(db: Session):
    uid = prepare(db)
    svc = PlanService(db)
    await svc.create(uid, TODAY, PlanLLM())
    plan, changes = await svc.update(uid, TODAY, PlanLLM(), datetime.now(timezone.utc))
    history = PlanRepository(db).history(uid)
    assert plan.version == 2 and [p.status for p in history] == ["active", "archived"]
    assert any("only 0 days old" in c for c in changes)  # too early to adapt from progress


@pytest.mark.asyncio
async def test_update_needs_an_existing_plan(db: Session):
    uid = prepare(db)
    with pytest.raises(PlanError):
        await PlanService(db).update(uid, TODAY, PlanLLM(), datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_a_stalled_cut_lowers_calories_after_a_week(db: Session):
    uid = prepare(db, mode="cut")
    svc = PlanService(db)
    await svc.create(uid, TODAY, PlanLLM())
    WeightRepository(db).upsert(uid, TODAY - timedelta(days=14), Decimal("80.0"))
    backdate(db, uid, 10)
    llm = PlanLLM(meals=[good_meals(calories=2075, protein=160)])
    plan, changes = await svc.update(uid, TODAY, llm, datetime.now(timezone.utc))
    assert plan.targets["calories"] == 2075
    assert any("2200 -> 2075" in c for c in changes) and any("stalled" in c for c in changes)
    assert "stalled" in plan.reason


@pytest.mark.asyncio
async def test_a_very_low_intake_blocks_adaptation(db: Session):
    from app.fitness.service import FitnessService

    uid = prepare(db, mode="cut")
    svc = PlanService(db)
    await svc.create(uid, TODAY, PlanLLM())
    for back in range(1, 5):
        FitnessService(db).log_meal(uid, description="tiny", calories=600, protein_g=20,
                                    meal_type="lunch", eaten_on=TODAY - timedelta(days=back),
                                    estimated=False, today=TODAY, carbs_g=None, fat_g=None,
                                    now=datetime.now(timezone.utc), tz_name="UTC")  # fmt: skip
    backdate(db, uid, 10)
    with pytest.raises(PlanError) as info:
        await svc.update(uid, TODAY, PlanLLM(), datetime.now(timezone.utc))
    assert "doctor" in str(info.value)
    plan = PlanRepository(db).active(uid)
    assert plan is not None and plan.version == 1


# ------------------------------------------------------- adaptation (pure)


def review(*codes: str, mode: str = "cut") -> Review:
    return Review(flags=[Flag(c, "warn", "t" if c != "weight_drifting" else "Weight is drifting up") for c in codes],
                  facts={"mode": mode})  # fmt: skip


def test_adaptation_rules():
    assert adaptation_for(review("cut_stalled")).calorie_delta == -125
    assert adaptation_for(review("cut_stalled", "over_eating")).calorie_delta == 0  # fix behaviour first
    assert adaptation_for(review("weight_wrong_way")).calorie_delta == -150
    assert adaptation_for(review("weight_wrong_way", mode="bulk")).calorie_delta == 200
    assert adaptation_for(review("bulk_stalled", mode="bulk")).calorie_delta == 150
    assert adaptation_for(review("losing_too_fast")).calorie_delta == 150
    assert adaptation_for(review("bulking_too_fast", mode="bulk")).calorie_delta == -150
    assert adaptation_for(review("weight_drifting", mode="maintain")).calorie_delta == -125


def test_heavy_load_or_missed_sessions_remove_a_training_day():
    for code in ("training_streak", "high_training_load", "recovery_risk", "under_training"):
        assert adaptation_for(review(code)).days_delta == -1


def test_the_safety_alert_blocks_every_change():
    out = adaptation_for(review("very_low_intake", "cut_stalled"))
    assert out.blocked and out.calorie_delta == 0 and out.days_delta == 0


def test_good_progress_changes_nothing():
    out = adaptation_for(review("weight_on_track", "training_on_track"))
    assert (out.calorie_delta, out.days_delta, out.reasons) == (0, 0, [])


# ------------------------------------------------------------ coach flags


def data(**kw: Any) -> CoachData:
    days = [DayStat(TODAY - timedelta(days=6 - i), calories=2200, meals=3, protein_g=160) for i in range(7)]
    base: dict[str, Any] = dict(today=TODAY, days=days, weights=[(TODAY - timedelta(days=14), 80.0), (TODAY, 80.0)],
                                mode="cut", target_calories=2200, target_protein_g=160, weekly_workouts=4)  # fmt: skip
    base.update(kw)
    return CoachData(**base)


def codes(review: Review) -> set[str]:
    return {f.code for f in review.flags}


def test_an_old_plan_with_a_problem_suggests_an_update():
    out = codes(evaluate(data(plan_age_days=15)))
    assert "cut_stalled" in out and "plan_update_suggested" in out


def test_a_young_plan_or_good_progress_does_not_nag():
    assert "plan_update_suggested" not in codes(evaluate(data(plan_age_days=5)))
    ok = data(plan_age_days=30, weights=[(TODAY - timedelta(days=14), 80.0), (TODAY, 79.0)])
    assert "plan_update_suggested" not in codes(evaluate(ok))
    assert "plan_update_suggested" not in codes(evaluate(data()))  # no plan at all


def test_a_stalled_bulk_is_noticed():
    d = data(mode="bulk", target_calories=2900)
    assert "bulk_stalled" in codes(evaluate(d))


def test_the_safety_rule_still_suppresses_the_suggestion():
    thin = [DayStat(TODAY - timedelta(days=6 - i), calories=700, meals=1) for i in range(7)]
    assert "plan_update_suggested" not in codes(evaluate(data(days=thin, plan_age_days=40)))


# ----------------------------------------------------------------- views


@pytest.mark.asyncio
async def test_today_and_week_views(db: Session):
    uid = prepare(db)
    plan = await PlanService(db).create(uid, TODAY, PlanLLM())
    today_text = format_plan_today(plan, TODAY)
    assert "Sat" in today_text and "Plan v1" in today_text and "not medical advice" in today_text
    week = format_plan_week(plan)
    assert all(d in week for d in DAYS) and "BMR 1780" in week and "Progression" in week


# ------------------------------------------------------------------ tools


def ctx_for(db: Session, uid: int, llm: LLMProvider | None) -> ToolContext:
    return ToolContext(db=db, user_id=uid, today=TODAY, default_currency="INR", llm=llm)


@pytest.mark.asyncio
async def test_plan_tools_end_to_end(db: Session):
    uid = prepare(db)
    registry = build_default_registry()
    for name in ("create_plan", "update_plan", "get_plan"):
        assert not registry.get(name).requires_approval  # type: ignore[union-attr]

    none_yet = await registry.execute("get_plan", {}, ctx_for(db, uid, None))
    assert "no plan" in none_yet["error"].lower()

    out = await registry.execute("create_plan", {}, ctx_for(db, uid, PlanLLM()))
    assert out["version"] == 1 and len(out["workout"]["days"]) == 7 and "disclaimer" in out

    again = await registry.execute("create_plan", {}, ctx_for(db, uid, PlanLLM()))
    assert "already has plan" in again["error"]

    today = await registry.execute("get_plan", {}, ctx_for(db, uid, None))
    assert today["today"]["day"] == "Sat"
    week = await registry.execute("get_plan", {"scope": "week"}, ctx_for(db, uid, None))
    assert len(week["meals"]["days"]) == 7

    upd = await registry.execute("update_plan", {}, ctx_for(db, uid, PlanLLM()))
    assert upd["version"] == 2 and upd["what_changed"]


@pytest.mark.asyncio
async def test_plan_tools_without_a_model_fail_cleanly(db: Session):
    uid = prepare(db)
    out = await build_default_registry().execute("create_plan", {}, ctx_for(db, uid, None))
    assert "not available" in out["error"]


@pytest.mark.asyncio
async def test_plans_are_private(db: Session):
    alice = prepare(db)
    bob = make_user(db, "bob")
    registry = build_default_registry()
    await registry.execute("create_plan", {}, ctx_for(db, alice, PlanLLM()))
    out = await registry.execute("get_plan", {}, ctx_for(db, bob, None))
    assert "no plan" in out["error"].lower()


def test_copy_import_is_used():
    assert copy.deepcopy(full_profile())["age"] == 30