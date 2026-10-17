from datetime import date as date_type
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.fitness.service import (
    FitnessError,
    FitnessService,
    format_day,
    goal_out,
    meal_out,
    weight_out,
    workout_out,
)
from app.tools.base import Tool, ToolContext, ToolError

MealType = Literal["breakfast", "lunch", "dinner", "snack"]
WorkoutKind = Literal["strength", "cardio", "walk", "run", "cycle", "yoga", "sport", "other"]
WeightUnit = Literal["kg", "lb"]


def _svc(ctx: ToolContext) -> FitnessService:
    return FitnessService(ctx.db)


# ------------------------------------------------------------------- meals


class LogMealArgs(BaseModel):
    description: str = Field(
        min_length=1, max_length=300, description="What was eaten, e.g. '2 rotis, dal and curd'."
    )
    calories: int = Field(
        ge=0,
        le=5000,
        description=(
            "Total calories (kcal). Use the number the user gave. If they gave none, ESTIMATE "
            "a reasonable figure yourself and set estimated=true. Do not ask for numbers."
        ),
    )
    protein_g: float | None = Field(default=None, ge=0, le=500, description="Protein in grams.")
    carbs_g: float | None = Field(default=None, ge=0, le=1000, description="Carbs in grams.")
    fat_g: float | None = Field(default=None, ge=0, le=500, description="Fat in grams.")
    meal_type: MealType | None = Field(
        default=None, description="Omit to pick from the time of day."
    )
    eaten_on: date_type | None = Field(
        default=None, description="YYYY-MM-DD. Omit for today. Never in the future."
    )
    estimated: bool = Field(
        default=True,
        description="true if YOU estimated the numbers; false only if the user stated them.",
    )


def log_meal(ctx: ToolContext, args: LogMealArgs) -> dict[str, Any]:
    try:
        meal = _svc(ctx).log_meal(
            ctx.user_id,
            description=args.description,
            calories=args.calories,
            protein_g=args.protein_g,
            carbs_g=args.carbs_g,
            fat_g=args.fat_g,
            meal_type=args.meal_type,
            eaten_on=args.eaten_on,
            estimated=args.estimated,
            today=ctx.today,
            now=ctx.now,
            tz_name=ctx.timezone,
        )
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    day = _svc(ctx).day_summary(ctx.user_id, meal.eaten_on, ctx.today)
    return {
        "saved": meal_out(meal),
        "day_total_calories": day["calories"],
        "calories_remaining": day.get("calories_remaining"),
    }


class UpdateMealArgs(BaseModel):
    meal_id: int = Field(gt=0, description="Id of the meal, from list_meals or an earlier reply.")
    description: str | None = Field(default=None, max_length=300)
    calories: int | None = Field(default=None, ge=0, le=5000)
    protein_g: float | None = Field(default=None, ge=0, le=500)
    carbs_g: float | None = Field(default=None, ge=0, le=1000)
    fat_g: float | None = Field(default=None, ge=0, le=500)
    meal_type: MealType | None = None

    @model_validator(mode="after")
    def _something(self) -> "UpdateMealArgs":
        changed = (
            self.description, self.calories, self.protein_g,
            self.carbs_g, self.fat_g, self.meal_type,
        )  # fmt: skip
        if all(v is None for v in changed):
            raise ValueError("Give at least one thing to change")
        return self


def update_meal(ctx: ToolContext, args: UpdateMealArgs) -> dict[str, Any]:
    try:
        meal = _svc(ctx).update_meal(
            ctx.user_id,
            args.meal_id,
            description=args.description,
            calories=args.calories,
            protein_g=args.protein_g,
            carbs_g=args.carbs_g,
            fat_g=args.fat_g,
            meal_type=args.meal_type,
        )
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    return {"updated": meal_out(meal)}


class RangeArgs(BaseModel):
    start_date: date_type | None = Field(default=None, description="YYYY-MM-DD. Omit for today.")
    end_date: date_type | None = Field(default=None, description="YYYY-MM-DD. Omit for today.")
    limit: int = Field(default=30, ge=1, le=100)

    @model_validator(mode="after")
    def _order(self) -> "RangeArgs":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


def list_meals(ctx: ToolContext, args: RangeArgs) -> dict[str, Any]:
    start = args.start_date or args.end_date or ctx.today
    end = args.end_date or ctx.today
    meals = _svc(ctx).meals.find(ctx.user_id, start, end, args.limit)
    return {"count": len(meals), "meals": [meal_out(m) for m in meals]}


# ---------------------------------------------------------------- workouts


class LogWorkoutArgs(BaseModel):
    kind: WorkoutKind = Field(description="Type of session.")
    description: str = Field(
        min_length=1,
        max_length=300,
        description="What was done, e.g. 'Push day: bench 3x8 at 60kg' or '5 km run'.",
    )
    duration_minutes: int | None = Field(default=None, ge=1, le=600)
    calories_burned: int | None = Field(
        default=None, ge=0, le=5000, description="Only if the user stated it. Do not guess."
    )
    done_on: date_type | None = Field(default=None, description="YYYY-MM-DD. Omit for today.")


def log_workout(ctx: ToolContext, args: LogWorkoutArgs) -> dict[str, Any]:
    try:
        workout = _svc(ctx).log_workout(
            ctx.user_id,
            kind=args.kind,
            description=args.description,
            duration_minutes=args.duration_minutes,
            calories_burned=args.calories_burned,
            done_on=args.done_on,
            today=ctx.today,
        )
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    return {"saved": workout_out(workout)}


def list_workouts(ctx: ToolContext, args: RangeArgs) -> dict[str, Any]:
    start = args.start_date or args.end_date or ctx.today
    end = args.end_date or ctx.today
    rows = _svc(ctx).workouts.find(ctx.user_id, start, end, args.limit)
    return {"count": len(rows), "workouts": [workout_out(w) for w in rows]}


# ------------------------------------------------------------------ weight


class LogWeightArgs(BaseModel):
    weight: float = Field(gt=0, le=1000, description="Body weight as the user said it.")
    unit: WeightUnit | None = Field(
        default=None, description="kg or lb, only if the user said which. Omit otherwise."
    )
    measured_on: date_type | None = Field(default=None, description="YYYY-MM-DD. Omit for today.")


def log_weight(ctx: ToolContext, args: LogWeightArgs) -> dict[str, Any]:
    svc = _svc(ctx)
    try:
        row, unit = svc.log_weight(
            ctx.user_id,
            weight=args.weight,
            unit=args.unit,
            measured_on=args.measured_on,
            today=ctx.today,
        )
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "saved": {
            "id": row.id,
            "date": row.measured_on.isoformat(),
            "weight": weight_out(row.weight_kg, unit),
        }
    }


class WeightHistoryArgs(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="How many days back.")


def list_weights(ctx: ToolContext, args: WeightHistoryArgs) -> dict[str, Any]:
    from datetime import timedelta

    svc = _svc(ctx)
    unit = svc.unit(ctx.user_id)
    rows = svc.weights.find(ctx.user_id, ctx.today - timedelta(days=args.days - 1), ctx.today)
    return {
        "count": len(rows),
        "weights": [
            {"id": r.id, "date": r.measured_on.isoformat(), "weight": weight_out(r.weight_kg, unit)}
            for r in rows
        ],
    }


# ------------------------------------------------------------------- goals


class SetGoalArgs(BaseModel):
    mode: Literal["cut", "bulk", "maintain"] = Field(
        description="cut = lose fat, bulk = gain muscle/weight, maintain = hold weight."
    )
    target_weight: float | None = Field(
        default=None, gt=0, le=1000, description="Target body weight. Not for maintain."
    )
    unit: WeightUnit | None = Field(
        default=None, description="kg or lb, only if the user said which."
    )
    daily_calories: int | None = Field(
        default=None, description="Daily calorie target. Only if the user gave one."
    )
    daily_protein_g: int | None = Field(
        default=None, description="Daily protein target in grams. Only if the user gave one."
    )
    weekly_workouts: int | None = Field(
        default=None, description="Workouts per week. Only if the user gave one."
    )


def set_goal(ctx: ToolContext, args: SetGoalArgs) -> dict[str, Any]:
    try:
        goal, unit, warnings = _svc(ctx).set_goal(
            ctx.user_id,
            mode=args.mode,
            target_weight=args.target_weight,
            unit=args.unit,
            daily_calories=args.daily_calories,
            daily_protein_g=args.daily_protein_g,
            weekly_workouts=args.weekly_workouts,
            today=ctx.today,
        )
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    out: dict[str, Any] = {"goal": goal_out(goal, unit)}
    if warnings:
        out["warnings"] = warnings
    return out


class NoArgs(BaseModel):
    pass


def get_goal(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    svc = _svc(ctx)
    goal = svc.goals.get(ctx.user_id)
    if goal is None:
        return {"goal": None, "hint": "No goal set. Ask whether they want to cut, bulk or maintain."}
    return {"goal": goal_out(goal, svc.unit(ctx.user_id))}


# --------------------------------------------------------------- summaries


class DaySummaryArgs(BaseModel):
    date: date_type | None = Field(default=None, description="YYYY-MM-DD. Omit for today.")


def get_day_summary(ctx: ToolContext, args: DaySummaryArgs) -> dict[str, Any]:
    summary = _svc(ctx).day_summary(ctx.user_id, args.date or ctx.today, ctx.today)
    summary["text"] = format_day(summary)
    return summary


class ProgressArgs(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="Window length in days.")


def get_progress(ctx: ToolContext, args: ProgressArgs) -> dict[str, Any]:
    return _svc(ctx).progress(ctx.user_id, ctx.today, args.days)


# ----------------------------------------------------------------- deletes


class DeleteEntryArgs(BaseModel):
    kind: Literal["meal", "workout", "weight"] = Field(description="Which log the entry is in.")
    entry_id: int = Field(gt=0, description="Id from list_meals, list_workouts or list_weights.")


def summarize_delete_entry(ctx: ToolContext, args: DeleteEntryArgs) -> str:
    entry = _svc(ctx).entry(ctx.user_id, args.kind, args.entry_id)
    if entry is None:
        raise ToolError("Entry not found.")
    if args.kind == "meal":
        m = meal_out(entry)  # type: ignore[arg-type]
        return f"Delete meal #{m['id']}: {m['description']} ({m['calories']} kcal, {m['date']})"
    if args.kind == "workout":
        w = workout_out(entry)  # type: ignore[arg-type]
        return f"Delete workout #{w['id']}: {w['description']} ({w['date']})"
    unit = _svc(ctx).unit(ctx.user_id)
    weight = weight_out(entry.weight_kg, unit)  # type: ignore[union-attr]
    return (
        f"Delete weigh-in #{entry.id}: {weight['value']} {weight['unit']} "  # type: ignore[index]
        f"({entry.measured_on.isoformat()})"  # type: ignore[union-attr]
    )


def delete_entry(ctx: ToolContext, args: DeleteEntryArgs) -> dict[str, Any]:
    svc = _svc(ctx)
    if svc.entry(ctx.user_id, args.kind, args.entry_id) is None:
        return {"error": "Entry not found."}
    svc.delete_entry(ctx.user_id, args.kind, args.entry_id)
    return {"deleted": {"kind": args.kind, "id": args.entry_id}}


FITNESS_TOOLS: list[Tool] = [
    Tool(
        name="log_meal",
        description=(
            "Record something the user ate, with calories (estimated if they gave none) and "
            "optional protein/carbs/fat."
        ),
        args_model=LogMealArgs,
        handler=log_meal,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="update_meal",
        description="Correct a logged meal (calories, macros, description or meal type).",
        args_model=UpdateMealArgs,
        handler=update_meal,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="list_meals",
        description="List logged meals for a date or date range, newest first.",
        args_model=RangeArgs,
        handler=list_meals,
        domain="fitness",
    ),
    Tool(
        name="log_workout",
        description="Record a workout or activity the user did.",
        args_model=LogWorkoutArgs,
        handler=log_workout,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="list_workouts",
        description="List logged workouts for a date or date range, newest first.",
        args_model=RangeArgs,
        handler=list_workouts,
        domain="fitness",
    ),
    Tool(
        name="log_weight",
        description="Record the user's body weight. One per day; a new one replaces that day's.",
        args_model=LogWeightArgs,
        handler=log_weight,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="list_weights",
        description="List the user's recent weigh-ins, oldest first.",
        args_model=WeightHistoryArgs,
        handler=list_weights,
        domain="fitness",
    ),
    Tool(
        name="set_goal",
        description=(
            "Set or replace the user's goal: cut, bulk or maintain, with optional target weight, "
            "daily calories, daily protein and weekly workouts."
        ),
        args_model=SetGoalArgs,
        handler=set_goal,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="get_goal",
        description="Show the user's current goal and targets.",
        args_model=NoArgs,
        handler=get_goal,
        domain="fitness",
    ),
    Tool(
        name="get_day_summary",
        description=(
            "Everything for one day: meals, calories and macros eaten, what is left against the "
            "goal, workouts and weight."
        ),
        args_model=DaySummaryArgs,
        handler=get_day_summary,
        domain="fitness",
    ),
    Tool(
        name="get_progress",
        description=(
            "Progress over the last N days: weight change and weekly rate, average calories, "
            "training count, and factual notes against the goal."
        ),
        args_model=ProgressArgs,
        handler=get_progress,
        domain="fitness",
    ),
    Tool(
        name="delete_log_entry",
        description=(
            "Delete one logged meal, workout or weigh-in by id. This only PREPARES the deletion; "
            "it happens only after the user approves it. To correct a meal use update_meal."
        ),
        args_model=DeleteEntryArgs,
        handler=delete_entry,
        domain="fitness",
        risk="high",
        side_effects=True,
        summarize=summarize_delete_entry,
    ),
]