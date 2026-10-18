from typing import Any, Literal

from pydantic import BaseModel, Field

from app.profiles.service import ProfileError
from app.tools.base import Tool, ToolContext, ToolError
from app.tracking.service import TrackingService


class SkipArgs(BaseModel):
    slot: Literal["breakfast", "lunch", "dinner", "workout"] = Field(
        description="What the user is skipping today. 'rest today' / 'no gym today' = workout."
    )


class TrackingArgs(BaseModel):
    enabled: bool | None = Field(default=None, description="Turn tracking reminders on or off.")
    breakfast_time: str | None = Field(default=None, description="HH:MM, 24-hour.")
    lunch_time: str | None = Field(default=None, description="HH:MM, 24-hour.")
    dinner_time: str | None = Field(default=None, description="HH:MM, 24-hour.")
    workout_time: str | None = Field(default=None, description="HH:MM, 24-hour.")


def skip_tracking(ctx: ToolContext, args: SkipArgs) -> dict[str, Any]:
    svc = TrackingService(ctx.db)
    svc.skip(ctx.user_id, args.slot, ctx.today)
    return {"skipped": args.slot, "day": ctx.today.isoformat(),
            "note": "No more reminders for this today. Do not scold; keep it brief and kind."}  # fmt: skip


def set_tracking(ctx: ToolContext, args: TrackingArgs) -> dict[str, Any]:
    times = {
        slot: value
        for slot, value in (
            ("breakfast", args.breakfast_time), ("lunch", args.lunch_time),
            ("dinner", args.dinner_time), ("workout", args.workout_time),
        )  # fmt: skip
        if value
    }
    svc = TrackingService(ctx.db)
    try:
        svc.configure(ctx.user_id, enabled=args.enabled, times=times)
    except ProfileError as exc:
        raise ToolError(str(exc)) from exc
    return svc.status(ctx.user_id)


def get_tracking(ctx: ToolContext, args: BaseModel) -> dict[str, Any]:
    return TrackingService(ctx.db).status(ctx.user_id)


class NoArgs(BaseModel):
    pass


FITNESS_TRACKING_TOOLS: list[Tool] = [
    Tool(
        name="skip_tracking",
        description=(
            "The user says they skipped breakfast, lunch or dinner, or that today is a rest day "
            "or they are not working out: stop tracking reminders for that slot today."
        ),
        args_model=SkipArgs,
        handler=skip_tracking,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="set_tracking_reminders",
        description=(
            "Turn the meal and workout tracking reminders on or off, or change their times "
            "(breakfast, lunch, dinner, workout). Times are in the user's timezone."
        ),
        args_model=TrackingArgs,
        handler=set_tracking,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="get_tracking_reminders",
        description="Show whether tracking reminders are on and at what times.",
        args_model=NoArgs,
        handler=get_tracking,
        domain="fitness",
    ),
]