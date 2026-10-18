"""Creating, adapting and showing the user's plan.

Adaptation is decided by code from the coach's flags (a stalled cut lowers calories a little, a
heavy week removes a training day, losing too fast adds calories back), bounded by safe floors,
and applied only when the user asks, with at least a week between changes.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.coach.rules import Review
from app.coach.service import CoachService
from app.database.models import FitnessPlan
from app.database.repositories.fitness import PlanRepository
from app.fitness.plans import DAYS, PlanError, PlanGenerator, PlanInputs
from app.fitness.profile import Computed, FitnessProfileService
from app.fitness.service import FitnessError, FitnessService
from app.llm.base import LLMProvider
from app.observability import get_tracer

MIN_ADAPT_DAYS = 7  # a plan needs a week of data before it is adapted
STEP_DOWN = 125
STEP_UP = 150
DISCLAIMER = "This is general guidance, not medical advice."


@dataclass
class Adaptation:
    calorie_delta: int = 0
    days_delta: int = 0
    reasons: list[str] = field(default_factory=list)
    blocked: str | None = None


def adaptation_for(review: Review) -> Adaptation:
    """What the plan should change, from the coach's verdicts. Conservative on purpose."""
    found = {f.code for f in review.flags}
    out = Adaptation()

    if "very_low_intake" in found:
        out.blocked = (
            "Your logged intake is very low, so I will not change your plan. Please speak to a "
            "doctor or dietitian, or log all your meals first."
        )
        return out

    behaviour_issue = "over_eating" in found  # eating above target explains a stall: fix that first
    mode = review.facts.get("mode")
    if "cut_stalled" in found and not behaviour_issue:
        out.calorie_delta -= STEP_DOWN
        out.reasons.append(f"your cut stalled while eating on target (-{STEP_DOWN} kcal)")
    if "weight_wrong_way" in found and not behaviour_issue:
        if mode == "bulk":
            out.calorie_delta += 200
            out.reasons.append("weight is falling during a bulk (+200 kcal)")
        else:
            out.calorie_delta -= STEP_UP
            out.reasons.append(f"weight is rising during a cut (-{STEP_UP} kcal)")
    if "bulk_stalled" in found:
        out.calorie_delta += STEP_UP
        out.reasons.append(f"your bulk stalled (+{STEP_UP} kcal)")
    if "losing_too_fast" in found:
        out.calorie_delta += STEP_UP
        out.reasons.append(f"you are losing weight too fast (+{STEP_UP} kcal)")
    if "bulking_too_fast" in found:
        out.calorie_delta -= STEP_UP
        out.reasons.append(f"you are gaining too fast (-{STEP_UP} kcal)")
    if "weight_drifting" in found and mode == "maintain":
        direction = -STEP_DOWN if any("up" in f.text for f in review.flags if f.code == "weight_drifting") else STEP_DOWN
        out.calorie_delta += direction
        out.reasons.append(f"weight is drifting while maintaining ({direction:+d} kcal)")

    if found & {"training_streak", "high_training_load", "recovery_risk"}:
        out.days_delta -= 1
        out.reasons.append("your training load has been too heavy (one fewer training day)")
    elif "under_training" in found:
        out.days_delta -= 1
        out.reasons.append("you have been doing fewer sessions than planned (one fewer, easier to keep)")
    return out


class PlanService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self.profile = FitnessProfileService(db)
        self.plans = PlanRepository(db)
        self.fitness = FitnessService(db)

    # --------------------------------------------------------------- inputs

    def _inputs(
        self, user_id: int, today: date, calorie_adjust: int = 0, days_delta: int = 0
    ) -> tuple[PlanInputs, Computed]:
        try:
            computed = self.profile.compute(user_id, today, calorie_adjust=calorie_adjust)
        except FitnessError as exc:
            raise PlanError(str(exc)) from exc
        if computed.status == "refuse" or computed.targets is None:
            raise PlanError(" ".join(computed.screen.reasons))
        row = self.profile.get(user_id)
        assert row is not None and row.sex and row.height_cm and row.experience
        assert row.equipment and row.diet_pref and row.days_per_week is not None
        age = self.profile.view(user_id, today)["profile"]["age"]
        t = computed.targets
        days = max(1, min(6, row.days_per_week + days_delta))
        inputs = PlanInputs(
            sex=row.sex,
            age=int(age),
            height_cm=row.height_cm,
            weight_kg=computed.weight_kg,
            mode=computed.mode,
            experience=row.experience,
            days_per_week=days,
            equipment=row.equipment,
            diet_pref=row.diet_pref,
            calories=t.calories,
            protein_g=t.protein_g,
            carbs_g=t.carbs_g,
            fat_g=t.fat_g,
            session_minutes=row.session_minutes or 60,
            muscle_level=row.muscle_level,
            preferred_workouts=row.preferred_workouts,
            allergies=row.allergies,
            cuisine=row.cuisine,
            movement_notes=list(computed.screen.movement_notes),
        )
        return inputs, computed

    def _store(
        self, user_id: int, today: date, inputs: PlanInputs, computed: Computed,
        workout: dict[str, Any], meals: dict[str, Any], reason: str,
    ) -> FitnessPlan:  # fmt: skip
        t = computed.targets
        assert t is not None
        plan = self.plans.add(
            user_id,
            workout,
            meals,
            {
                "bmr": t.bmr, "tdee": t.tdee, "calories": t.calories, "protein_g": t.protein_g,
                "carbs_g": t.carbs_g, "fat_g": t.fat_g, "mode": computed.mode,
                "status": computed.status, "reasons": list(computed.screen.reasons),
                "see_a_doctor_first": computed.screen.clearance,
            },
            inputs.snapshot(),
            reason,
        )  # fmt: skip
        # the coach now judges the user against what the plan asks of them
        self.fitness.set_goal(
            user_id,
            mode=computed.mode,
            target_weight=None,
            unit=None,
            daily_calories=t.calories,
            daily_protein_g=t.protein_g,
            weekly_workouts=inputs.days_per_week,
            today=today,
        )
        return plan

    # ------------------------------------------------------------- creating

    async def create(
        self, user_id: int, today: date, provider: LLMProvider, reason: str = "First plan"
    ) -> FitnessPlan:
        import asyncio

        inputs, computed = await asyncio.to_thread(self._inputs, user_id, today)
        with get_tracer().span("plan.create", metadata={"user": f"u{user_id}", "reason": reason}) as span:
            workout, meals = await PlanGenerator(provider).both(inputs)
            plan = await asyncio.to_thread(
                self._store, user_id, today, inputs, computed, workout, meals, reason
            )
            span.update(metadata={"version": plan.version})
            return plan

    async def update(
        self, user_id: int, today: date, provider: LLMProvider, now: datetime
    ) -> tuple[FitnessPlan, list[str]]:
        """Rebuild from the latest stats and progress. Returns the new plan and what changed."""
        import asyncio

        current = await asyncio.to_thread(self.plans.active, user_id)
        if current is None:
            raise PlanError("There is no plan to update yet. Ask me to create one first.")

        def prepare() -> tuple[PlanInputs, Computed, Adaptation, int]:
            review, _ = CoachService(self._db).review(user_id, today)
            change = adaptation_for(review)
            if change.blocked:
                raise PlanError(change.blocked)
            created = current.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (now - created).days
            if age_days < MIN_ADAPT_DAYS:
                change = Adaptation(reasons=[])  # too early to read a trend: refresh only
            inputs, computed = self._inputs(user_id, today, change.calorie_delta, change.days_delta)
            return inputs, computed, change, age_days

        inputs, computed, change, age_days = await asyncio.to_thread(prepare)
        with get_tracer().span(
            "plan.update", metadata={"user": f"u{user_id}", "adaptation": change.reasons}
        ):
            workout, meals = await PlanGenerator(provider).both(inputs)

        old = current.targets
        old_days = current.inputs.get("days_per_week")
        assert computed.targets is not None
        diffs: list[str] = []
        if computed.targets.calories != old.get("calories"):
            diffs.append(f"Calories: {old.get('calories')} -> {computed.targets.calories} a day")
        if inputs.days_per_week != old_days:
            diffs.append(f"Training days: {old_days} -> {inputs.days_per_week} a week")
        if computed.mode != old.get("mode"):
            diffs.append(f"Goal mode: {old.get('mode')} -> {computed.mode}")
        if not diffs:
            diffs.append("Numbers stayed the same; the plan itself was refreshed.")
        if age_days < MIN_ADAPT_DAYS:
            diffs.append(f"The plan is only {age_days} days old, so I did not change targets from progress yet.")

        reason = "Updated: " + ("; ".join(change.reasons) if change.reasons else "refreshed from your latest stats")
        plan = await asyncio.to_thread(
            self._store, user_id, today, inputs, computed, workout, meals, reason
        )
        return plan, diffs + [f"Why: {r}" for r in change.reasons]

    # ---------------------------------------------------------------- views

    def active(self, user_id: int) -> FitnessPlan | None:
        return self.plans.active(user_id)

    def plan_age_days(self, user_id: int, now: datetime) -> int | None:
        plan = self.plans.active(user_id)
        if plan is None:
            return None
        created = plan.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).days


def day_name(day: date) -> str:
    return DAYS[day.weekday()]


def plan_day(plan: FitnessPlan, day: date) -> dict[str, Any]:
    name = day_name(day)
    workout = next((d for d in plan.workout["days"] if d["day"] == name), None)
    meals = next((d for d in plan.meals["days"] if d["day"] == name), None)
    return {"day": name, "workout": workout, "meals": meals}


def plan_out(plan: FitnessPlan, scope: str, today: date) -> dict[str, Any]:
    out: dict[str, Any] = {
        "version": plan.version,
        "reason": plan.reason,
        "targets": plan.targets,
        "days_per_week": plan.inputs.get("days_per_week"),
        "disclaimer": DISCLAIMER,
    }
    if scope == "week":
        out["workout"] = plan.workout
        out["meals"] = plan.meals
    else:
        out["today"] = plan_day(plan, today)
    return out


def _workout_lines(day: dict[str, Any] | None) -> list[str]:
    if day is None:
        return ["  (no workout planned)"]
    if day["rest"]:
        return ["  Rest day. A walk or stretching is fine."]
    mins = f" (~{day['minutes']} min)" if day.get("minutes") else ""
    lines = [f"  {day['focus']}{mins}"]
    for ex in day["exercises"]:
        lines.append(f"  - {ex['name']}: {ex['sets']} x {ex['reps']}")
    return lines


def _meal_lines(day: dict[str, Any] | None) -> list[str]:
    if day is None:
        return ["  (no meals planned)"]
    lines = [
        f"  {m['type'].title()}: {m['name']} - {m['items']} ({m['calories']} kcal, {m['protein_g']:g} g protein)"
        for m in day["meals"]
    ]
    total = sum(m["calories"] for m in day["meals"])
    lines.append(f"  Total: {total} kcal, {sum(m['protein_g'] for m in day['meals']):.0f} g protein")
    return lines


def format_plan_today(plan: FitnessPlan, today: date) -> str:
    view = plan_day(plan, today)
    t = plan.targets
    lines = [
        f"Plan v{plan.version} - {view['day']}",
        f"Targets: {t['calories']} kcal, {t['protein_g']} g protein ({t['mode']})",
        "",
        "Workout:",
        *_workout_lines(view["workout"]),
        "",
        "Meals:",
        *_meal_lines(view["meals"]),
        "",
        "/plan week shows all 7 days. " + DISCLAIMER,
    ]
    return "\n".join(lines)


def format_plan_week(plan: FitnessPlan) -> str:
    t = plan.targets
    lines = [
        f"Plan v{plan.version} ({plan.reason})",
        f"BMR {t['bmr']} kcal, maintenance {t['tdee']} kcal. Target {t['calories']} kcal, "
        f"{t['protein_g']} g protein, {t['carbs_g']} g carbs, {t['fat_g']} g fat ({t['mode']}).",
    ]
    for reason in t.get("reasons", []):
        lines.append("Note: " + reason)
    lines += ["", "WORKOUTS - " + plan.workout["summary"]]
    for d in plan.workout["days"]:
        lines.append(f"{d['day']}")
        lines += _workout_lines(d)
    lines += ["", "Progression: " + plan.workout["progression"], "", "MEALS - " + plan.meals["notes"]]
    for d in plan.meals["days"]:
        lines.append(f"{d['day']}")
        lines += _meal_lines(d)
    lines += ["", DISCLAIMER]
    return "\n".join(lines)