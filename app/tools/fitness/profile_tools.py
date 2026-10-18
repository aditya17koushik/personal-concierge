from typing import Any

from pydantic import BaseModel, Field

from app.fitness.profile import FitnessProfileService
from app.fitness.service import FitnessError
from app.tools.base import Tool, ToolContext, ToolError


class SaveProfileArgs(BaseModel):
    age: int | None = Field(default=None, ge=10, le=100, description="Age in years.")
    sex: str | None = Field(default=None, description="male, female or other.")
    height_cm: float | None = Field(
        default=None, gt=0, le=300, description="Height in centimetres. Convert feet/inches yourself."
    )
    activity_level: str | None = Field(
        default=None,
        description="sedentary, light, moderate or very_active (daily activity outside workouts).",
    )
    experience: str | None = Field(default=None, description="beginner, intermediate or advanced.")
    muscle_level: str | None = Field(default=None, description="low, average or high.")
    conditions: str | None = Field(
        default=None,
        max_length=500,
        description="Health conditions, injuries or surgeries in the user's words, or 'none'.",
    )
    preferred_workouts: str | None = Field(
        default=None, max_length=200, description="e.g. 'weights, running, yoga'."
    )
    days_per_week: int | None = Field(default=None, ge=1, le=6, description="Training days a week.")
    session_minutes: int | None = Field(default=None, ge=15, le=180)
    equipment: str | None = Field(
        default=None, description="gym, home_dumbbells, home_minimal or none."
    )
    diet_pref: str | None = Field(
        default=None, description="vegetarian, eggetarian, non_vegetarian, pescatarian or vegan (map the user's words)."
    )
    allergies: str | None = Field(default=None, max_length=200, description="Foods to avoid.")
    cuisine: str | None = Field(default=None, max_length=100)


def save_fitness_profile(ctx: ToolContext, args: SaveProfileArgs) -> dict[str, Any]:
    svc = FitnessProfileService(ctx.db)
    try:
        svc.save(ctx.user_id, ctx.today, **args.model_dump())
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc
    return svc.view(ctx.user_id, ctx.today)


class NoArgs(BaseModel):
    pass


def get_fitness_profile(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    return FitnessProfileService(ctx.db).view(ctx.user_id, ctx.today)


def calculate_targets(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    try:
        return FitnessProfileService(ctx.db).computed_out(ctx.user_id, ctx.today)
    except FitnessError as exc:
        raise ToolError(str(exc)) from exc


FITNESS_PROFILE_TOOLS: list[Tool] = [
    Tool(
        name="save_fitness_profile",
        description=(
            "Save what the user told you about themselves (age, sex, height, activity level, "
            "experience, health conditions, days per week, equipment, diet, allergies...). Pass "
            "only what they just said; earlier answers are kept. Returns what is still missing."
        ),
        args_model=SaveProfileArgs,
        handler=save_fitness_profile,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="get_fitness_profile",
        description="Show the user's saved profile and which required answers are still missing.",
        args_model=NoArgs,
        handler=get_fitness_profile,
        domain="fitness",
    ),
    Tool(
        name="calculate_targets",
        description=(
            "Work out BMR, maintenance calories, the daily calorie and macro targets for the "
            "user's goal, and the health screen result. All numbers are computed in code."
        ),
        args_model=NoArgs,
        handler=calculate_targets,
        domain="fitness",
    ),
]