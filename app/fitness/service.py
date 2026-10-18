"""Fitness rules: meals, workouts, weigh-ins, goals, and the summaries built from them.

Everything the user sees as a number or a verdict is computed HERE, in code. The model only
chooses which tool to call and words the answer; it never does the arithmetic.
"""

import re
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.database.models import BodyWeight, FitnessGoal, Meal, Workout
from app.database.repositories.fitness import (
    GoalRepository,
    MealRepository,
    WeightRepository,
    WorkoutRepository,
)
from app.database.repositories.profiles import ProfileRepository

MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")
WORKOUT_KINDS = ("strength", "cardio", "walk", "run", "cycle", "yoga", "sport", "other")
GOAL_MODES = ("cut", "bulk", "maintain")

KG_PER_LB = Decimal("0.45359237")
MIN_WEIGHT_KG = Decimal("25")
MAX_WEIGHT_KG = Decimal("400")
MAX_MEAL_CALORIES = 5000
MAX_DAILY_CALORIES_TARGET = 6000
MIN_DAILY_CALORIES_TARGET = 800
LOW_CALORIE_WARNING = 1200
MAX_BACKDATE_DAYS = 400
MAX_PROGRESS_DAYS = 365

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class FitnessError(Exception):
    """A known, user-safe problem. The message is shown to the model."""


def clean(text: str, limit: int = 300) -> str:
    out = " ".join(_CONTROL.sub(" ", text).split())
    if not out:
        raise FitnessError("Text must not be empty.")
    return out[:limit]


# ------------------------------------------------------------------- units


def unit_for(unit_system: str | None, explicit: str | None = None) -> str:
    """'kg' or 'lb'. An explicit unit wins; otherwise the user's profile decides."""
    if explicit:
        return "lb" if explicit.lower() in ("lb", "lbs") else "kg"
    return "lb" if unit_system == "imperial" else "kg"


def to_kg(value: float | Decimal, unit: str) -> Decimal:
    amount = Decimal(str(value))
    kg = amount * KG_PER_LB if unit == "lb" else amount
    # Two decimals, so 180 lb -> 81.65 kg -> 180.0 lb again (one decimal would drift to 179.9).
    kg = kg.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if not (MIN_WEIGHT_KG <= kg <= MAX_WEIGHT_KG):
        raise FitnessError(
            f"That weight looks wrong ({value} {unit}). Ask the user to check the number and unit."
        )
    return kg


def from_kg(kg: Decimal | float, unit: str) -> float:
    value = Decimal(str(kg))
    if unit == "lb":
        value = value / KG_PER_LB
    return float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def weight_out(kg: Decimal | float | None, unit: str) -> dict[str, Any] | None:
    if kg is None:
        return None
    return {"value": from_kg(kg, unit), "unit": unit}


def _dec(value: float | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def default_meal_type(now: datetime, tz_name: str) -> str:
    hour = now.astimezone(ZoneInfo(tz_name)).hour
    if hour < 11:
        return "breakfast"
    if hour < 16:
        return "lunch"
    if hour < 21:
        return "dinner"
    return "snack"


def check_date(day: date, today: date, what: str) -> None:
    if day > today:
        raise FitnessError(f"{what} can't be in the future (today is {today.isoformat()}).")
    if (today - day).days > MAX_BACKDATE_DAYS:
        raise FitnessError(f"{what} is too far in the past.")


# -------------------------------------------------------------- serializers


def meal_out(m: Meal) -> dict[str, Any]:
    return {
        "id": m.id,
        "date": m.eaten_on.isoformat(),
        "meal": m.meal_type,
        "description": m.description,
        "calories": m.calories,
        "protein_g": float(m.protein_g) if m.protein_g is not None else None,
        "carbs_g": float(m.carbs_g) if m.carbs_g is not None else None,
        "fat_g": float(m.fat_g) if m.fat_g is not None else None,
        "estimated": m.estimated,
    }


def workout_out(w: Workout) -> dict[str, Any]:
    return {
        "id": w.id,
        "date": w.done_on.isoformat(),
        "kind": w.kind,
        "description": w.description,
        "duration_minutes": w.duration_minutes,
        "calories_burned": w.calories_burned,
    }


def goal_out(g: FitnessGoal, unit: str) -> dict[str, Any]:
    return {
        "mode": g.mode,
        "target_weight": weight_out(g.target_weight_kg, unit),
        "start_weight": weight_out(g.start_weight_kg, unit),
        "daily_calories": g.daily_calories,
        "daily_protein_g": g.daily_protein_g,
        "weekly_workouts": g.weekly_workouts,
        "started_on": g.started_on.isoformat(),
    }


# ----------------------------------------------------------------- service


class FitnessService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self.meals = MealRepository(db)
        self.workouts = WorkoutRepository(db)
        self.weights = WeightRepository(db)
        self.goals = GoalRepository(db)

    def unit(self, user_id: int, explicit: str | None = None) -> str:
        profile = ProfileRepository(self._db).get(user_id)
        return unit_for(profile.unit_system if profile else None, explicit)

    # ---- meals

    def log_meal(
        self,
        user_id: int,
        *,
        description: str,
        calories: int,
        protein_g: float | None,
        carbs_g: float | None,
        fat_g: float | None,
        meal_type: str | None,
        eaten_on: date | None,
        estimated: bool,
        today: date,
        now: datetime,
        tz_name: str,
    ) -> Meal:
        if not (0 <= calories <= MAX_MEAL_CALORIES):
            raise FitnessError(f"Calories for one meal must be between 0 and {MAX_MEAL_CALORIES}.")
        day = eaten_on or today
        check_date(day, today, "A meal date")
        return self.meals.add(
            user_id,
            day,
            meal_type or default_meal_type(now, tz_name),
            clean(description),
            calories,
            _dec(protein_g),
            _dec(carbs_g),
            _dec(fat_g),
            estimated,
        )

    def update_meal(
        self,
        user_id: int,
        meal_id: int,
        *,
        description: str | None,
        calories: int | None,
        protein_g: float | None,
        carbs_g: float | None,
        fat_g: float | None,
        meal_type: str | None,
    ) -> Meal:
        meal = self.meals.get(user_id, meal_id)
        if meal is None:
            raise FitnessError("Meal not found.")
        if calories is not None:
            if not (0 <= calories <= MAX_MEAL_CALORIES):
                raise FitnessError(
                    f"Calories for one meal must be between 0 and {MAX_MEAL_CALORIES}."
                )
            meal.calories = calories
            meal.estimated = False  # the user corrected it, so it is no longer a guess
        if description is not None:
            meal.description = clean(description)
        if protein_g is not None:
            meal.protein_g = _dec(protein_g)
        if carbs_g is not None:
            meal.carbs_g = _dec(carbs_g)
        if fat_g is not None:
            meal.fat_g = _dec(fat_g)
        if meal_type is not None:
            meal.meal_type = meal_type
        return self.meals.save(meal)

    # ---- workouts and weight

    def log_workout(
        self,
        user_id: int,
        *,
        kind: str,
        description: str,
        duration_minutes: int | None,
        calories_burned: int | None,
        done_on: date | None,
        today: date,
    ) -> Workout:
        day = done_on or today
        check_date(day, today, "A workout date")
        return self.workouts.add(
            user_id, day, kind, clean(description), duration_minutes, calories_burned
        )

    def log_weight(
        self,
        user_id: int,
        *,
        weight: float,
        unit: str | None,
        measured_on: date | None,
        today: date,
    ) -> tuple[BodyWeight, str]:
        chosen = self.unit(user_id, unit)
        day = measured_on or today
        check_date(day, today, "A weigh-in date")
        return self.weights.upsert(user_id, day, to_kg(weight, chosen)), chosen

    # ---- goals

    def set_goal(
        self,
        user_id: int,
        *,
        mode: str | None,
        target_weight: float | None,
        unit: str | None,
        daily_calories: int | None,
        daily_protein_g: int | None,
        weekly_workouts: int | None,
        today: date,
    ) -> tuple[FitnessGoal, str, list[str]]:
        """Create or UPDATE the goal. Anything left out keeps its current value."""
        chosen = self.unit(user_id, unit)
        warnings: list[str] = []
        existing = self.goals.get(user_id)
        latest = self.weights.latest(user_id, today)

        if mode is None:
            if existing is None:
                raise FitnessError("Which goal: cut, bulk or maintain? Ask the user.")
            mode = existing.mode
        mode_changed = existing is not None and existing.mode != mode

        target_kg = to_kg(target_weight, chosen) if target_weight is not None else None
        if target_kg is None and existing is not None and not mode_changed:
            target_kg = existing.target_weight_kg  # keep it; a new mode needs a new target

        if target_weight is not None and latest is not None:
            if mode == "cut" and target_kg is not None and target_kg >= latest.weight_kg:
                raise FitnessError(
                    "A cut needs a target below the current weight "
                    f"({from_kg(latest.weight_kg, chosen)} {chosen}). Ask the user to confirm."
                )
            if mode == "bulk" and target_kg is not None and target_kg <= latest.weight_kg:
                raise FitnessError(
                    "A bulk needs a target above the current weight "
                    f"({from_kg(latest.weight_kg, chosen)} {chosen}). Ask the user to confirm."
                )
        if mode == "maintain":
            if target_weight is not None:
                raise FitnessError("A maintain goal has no target weight. Leave it out.")
            target_kg = None

        def keep(new: int | None, old: int | None) -> int | None:
            return new if new is not None else old

        calories = keep(daily_calories, existing.daily_calories if existing else None)
        protein = keep(daily_protein_g, existing.daily_protein_g if existing else None)
        workouts = keep(weekly_workouts, existing.weekly_workouts if existing else None)

        if calories is not None:
            if not (MIN_DAILY_CALORIES_TARGET <= calories <= MAX_DAILY_CALORIES_TARGET):
                raise FitnessError(
                    f"Daily calories must be between {MIN_DAILY_CALORIES_TARGET} and "
                    f"{MAX_DAILY_CALORIES_TARGET}."
                )
            if calories < LOW_CALORIE_WARNING:
                warnings.append(
                    f"{calories} kcal a day is very low. Suggest checking it with a "
                    "doctor or dietitian."
                )
        if protein is not None and not (20 <= protein <= 400):
            raise FitnessError("Daily protein must be between 20 and 400 g.")
        if workouts is not None and not (0 <= workouts <= 14):
            raise FitnessError("Weekly workouts must be between 0 and 14.")

        start_kg = (
            latest.weight_kg
            if (existing is None or mode_changed) and latest is not None
            else (existing.start_weight_kg if existing else None)
        )
        goal = self.goals.set(
            user_id, mode, target_kg, start_kg, calories, protein, workouts, today
        )
        return goal, chosen, warnings

    # ---- deletes (the tool that calls these is approval-gated)

    def entry(self, user_id: int, kind: str, entry_id: int) -> Meal | Workout | BodyWeight | None:
        if kind == "meal":
            return self.meals.get(user_id, entry_id)
        if kind == "workout":
            return self.workouts.get(user_id, entry_id)
        return self.weights.get(user_id, entry_id)

    def delete_entry(self, user_id: int, kind: str, entry_id: int) -> bool:
        if kind == "meal":
            return self.meals.delete(user_id, entry_id)
        if kind == "workout":
            return self.workouts.delete(user_id, entry_id)
        return self.weights.delete(user_id, entry_id)

    # ---- summaries

    def day_summary(self, user_id: int, day: date, today: date) -> dict[str, Any]:
        unit = self.unit(user_id)
        goal = self.goals.get(user_id)
        meals = list(reversed(self.meals.find(user_id, day, day, limit=100)))
        workouts = list(reversed(self.workouts.find(user_id, day, day, limit=100)))
        latest = self.weights.latest(user_id, day)

        calories = sum(m.calories for m in meals)
        protein = sum(float(m.protein_g) for m in meals if m.protein_g is not None)
        carbs = sum(float(m.carbs_g) for m in meals if m.carbs_g is not None)
        fat = sum(float(m.fat_g) for m in meals if m.fat_g is not None)
        burned = sum(w.calories_burned or 0 for w in workouts)

        out: dict[str, Any] = {
            "date": day.isoformat(),
            "is_today": day == today,
            "meals": [meal_out(m) for m in meals],
            "calories": calories,
            "protein_g": round(protein, 1),
            "carbs_g": round(carbs, 1),
            "fat_g": round(fat, 1),
            "some_estimated": any(m.estimated for m in meals),
            "macros_incomplete": any(
                m.protein_g is None or m.carbs_g is None or m.fat_g is None for m in meals
            ),
            "workouts": [workout_out(w) for w in workouts],
            "calories_burned_logged": burned,
            "weight": weight_out(latest.weight_kg, unit) if latest else None,
            "goal": goal_out(goal, unit) if goal else None,
        }
        if goal and goal.daily_calories:
            out["calories_remaining"] = goal.daily_calories - calories
        if goal and goal.daily_protein_g:
            out["protein_remaining_g"] = round(goal.daily_protein_g - protein, 1)
        return out

    def progress(self, user_id: int, today: date, days: int) -> dict[str, Any]:
        days = max(1, min(days, MAX_PROGRESS_DAYS))
        unit = self.unit(user_id)
        goal = self.goals.get(user_id)
        start = today - timedelta(days=days - 1)

        weights = self.weights.find(user_id, start, today)
        meal_days = self.meals.daily_totals(user_id, start, today)
        meals = self.meals.find(user_id, start, today, limit=2000)
        workouts = self.workouts.find(user_id, start, today, limit=1000)
        weeks = days / 7

        out: dict[str, Any] = {
            "from": start.isoformat(),
            "to": today.isoformat(),
            "days": days,
            "goal": goal_out(goal, unit) if goal else None,
        }

        # --- weight
        weight_part: dict[str, Any] = {"weigh_ins": len(weights)}
        per_week_kg: Decimal | None = None
        if weights:
            first, last = weights[0], weights[-1]
            change = last.weight_kg - first.weight_kg
            span = (last.measured_on - first.measured_on).days
            weight_part.update(
                first=weight_out(first.weight_kg, unit),
                latest=weight_out(last.weight_kg, unit),
                latest_on=last.measured_on.isoformat(),
                change=weight_out(change, unit),
            )
            if len(weights) >= 2 and span >= 7:
                per_week_kg = change / (Decimal(span) / 7)
                weight_part["change_per_week"] = weight_out(per_week_kg, unit)
        if goal and goal.target_weight_kg is not None and weights:
            weight_part["to_target"] = weight_out(
                goal.target_weight_kg - weights[-1].weight_kg, unit
            )
        out["weight"] = weight_part

        # --- food (averaged over days that have meals logged; unlogged days are not zeros)
        logged = len(meal_days)
        protein_vals = [float(m.protein_g) for m in meals if m.protein_g is not None]
        food: dict[str, Any] = {"days_logged": logged}
        if logged:
            avg = round(sum(meal_days.values()) / logged)
            food["avg_calories_per_logged_day"] = avg
            if goal and goal.daily_calories:
                food["calorie_target"] = goal.daily_calories
                food["avg_vs_target"] = avg - goal.daily_calories
            if protein_vals and len(protein_vals) == len(meals):
                food["avg_protein_g_per_logged_day"] = round(sum(protein_vals) / logged, 1)
        out["food"] = food

        # --- training
        count = len(workouts)
        training: dict[str, Any] = {
            "workouts": count,
            "minutes": sum(w.duration_minutes or 0 for w in workouts),
            "per_week": round(count / weeks, 1) if weeks else count,
        }
        if goal and goal.weekly_workouts is not None:
            training["weekly_target"] = goal.weekly_workouts
        out["training"] = training

        out["notes"] = progress_notes(
            goal, weights, per_week_kg, logged, days, count, weeks, unit
        )
        return out


def progress_notes(
    goal: FitnessGoal | None,
    weights: list[BodyWeight],
    per_week_kg: Decimal | None,
    days_logged: int,
    days: int,
    workouts: int,
    weeks: float,
    unit: str,
) -> list[str]:
    """Plain, factual observations built from the numbers. Not medical advice."""
    notes: list[str] = []
    if goal is None:
        notes.append("No goal set yet. Say 'I want to cut', 'bulk' or 'maintain' to set one.")

    if per_week_kg is None:
        notes.append(
            "Not enough weigh-ins for a trend yet: two weigh-ins at least a week apart are needed."
        )
    elif goal is not None and weights:
        body = weights[-1].weight_kg
        pct = float(per_week_kg / body * 100)  # % of body weight per week
        rate = f"{abs(from_kg(per_week_kg, unit)):.1f} {unit}/week"
        if goal.mode == "cut":
            if pct <= -1.0:
                notes.append(f"Losing {rate}: faster than about 1% of body weight a week.")
            elif pct <= -0.2:
                notes.append(f"Losing {rate}: a steady pace for a cut.")
            elif pct < 0.2:
                notes.append("Weight is flat over this period, so the cut has stalled.")
            else:
                notes.append(f"Weight is going UP ({rate}) while the goal is a cut.")
        elif goal.mode == "bulk":
            if pct >= 0.75:
                notes.append(f"Gaining {rate}: quicker than about 0.5% a week, so more may be fat.")
            elif pct >= 0.1:
                notes.append(f"Gaining {rate}: a steady pace for a bulk.")
            elif pct > -0.1:
                notes.append("Weight is flat over this period, so the bulk has stalled.")
            else:
                notes.append(f"Weight is going DOWN ({rate}) while the goal is a bulk.")
        else:
            if abs(pct) < 0.5:
                notes.append("Weight is stable, as intended for maintenance.")
            else:
                direction = "up" if pct > 0 else "down"
                notes.append(f"Weight is drifting {direction} ({rate}) while maintaining.")

    if days_logged and days_logged < max(3, days // 2):
        notes.append(
            f"Meals were logged on only {days_logged} of {days} days, so food averages are rough."
        )
    if goal is not None and goal.weekly_workouts is not None and weeks >= 1:
        per_week = workouts / weeks
        if per_week + 0.05 < goal.weekly_workouts:
            notes.append(
                f"Training {per_week:.1f} sessions a week against a target of {goal.weekly_workouts}."
            )
        else:
            notes.append(f"Hitting the training target ({goal.weekly_workouts} a week).")
    return notes


# ------------------------------------------------------------ text for chats


def _w(w: dict[str, Any] | None) -> str:
    return f"{w['value']} {w['unit']}" if w else "-"


def format_day(s: dict[str, Any]) -> str:
    lines = [f"{'Today' if s['is_today'] else s['date']}: {s['calories']} kcal eaten"]
    goal = s.get("goal")
    if goal and goal.get("daily_calories"):
        remaining = s["calories_remaining"]
        lines[0] += f" of {goal['daily_calories']} ({abs(remaining)} {'left' if remaining >= 0 else 'over'})"
    macro = f"Protein {s['protein_g']} g, carbs {s['carbs_g']} g, fat {s['fat_g']} g"
    if s["macros_incomplete"]:
        macro += " (some meals have no macros)"
    lines.append(macro)
    if goal and goal.get("daily_protein_g"):
        left = s["protein_remaining_g"]
        lines.append(f"Protein target {goal['daily_protein_g']} g ({abs(left)} g {'to go' if left >= 0 else 'over'})")
    for m in s["meals"]:
        mark = " (est.)" if m["estimated"] else ""
        lines.append(f"  #{m['id']} {m['meal']}: {m['description']} - {m['calories']} kcal{mark}")
    if not s["meals"]:
        lines.append("  No meals logged.")
    for w in s["workouts"]:
        extra = f", {w['duration_minutes']} min" if w["duration_minutes"] else ""
        lines.append(f"  #{w['id']} workout: {w['description']}{extra}")
    if s["weight"]:
        lines.append(f"Weight: {_w(s['weight'])}")
    return "\n".join(lines)


def format_progress(p: dict[str, Any]) -> str:
    goal = p.get("goal")
    head = f"Progress, last {p['days']} days"
    if goal:
        head += f" (goal: {goal['mode']})"
    lines = [head]
    wt = p["weight"]
    if wt.get("latest"):
        line = f"Weight: {_w(wt['first'])} -> {_w(wt['latest'])} (change {_w(wt['change'])})"
        if wt.get("change_per_week"):
            line += f", {_w(wt['change_per_week'])}/week"
        lines.append(line)
        if wt.get("to_target"):
            lines.append(f"To target: {_w(wt['to_target'])}")
    else:
        lines.append("Weight: no weigh-ins yet")
    food = p["food"]
    if food["days_logged"]:
        line = f"Food: avg {food['avg_calories_per_logged_day']} kcal on {food['days_logged']} logged days"
        if "avg_vs_target" in food:
            line += f" ({food['avg_vs_target']:+d} vs target {food['calorie_target']})"
        lines.append(line)
    else:
        lines.append("Food: no meals logged")
    tr = p["training"]
    line = f"Training: {tr['workouts']} workouts, {tr['minutes']} min"
    if "weekly_target" in tr:
        line += f" (target {tr['weekly_target']}/week)"
    lines.append(line)
    lines += [f"- {n}" for n in p["notes"]]
    return "\n".join(lines)


def format_goal(g: dict[str, Any]) -> str:
    lines = [f"Goal: {g['mode']} (since {g['started_on']})"]
    if g["start_weight"]:
        lines.append(f"Started at: {_w(g['start_weight'])}")
    if g["target_weight"]:
        lines.append(f"Target weight: {_w(g['target_weight'])}")
    if g["daily_calories"]:
        lines.append(f"Calories: {g['daily_calories']} kcal a day")
    if g["daily_protein_g"]:
        lines.append(f"Protein: {g['daily_protein_g']} g a day")
    if g["weekly_workouts"] is not None:
        lines.append(f"Workouts: {g['weekly_workouts']} a week")
    return "\n".join(lines)