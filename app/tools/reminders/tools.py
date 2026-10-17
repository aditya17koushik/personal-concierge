from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.reminders.service import ReminderError, ReminderService, describe
from app.tools.base import Tool, ToolContext, ToolError

LOCAL_TIME_HELP = (
    "Local time in the user's timezone, as YYYY-MM-DDTHH:MM, with no UTC offset."
)


def _service(ctx: ToolContext) -> ReminderService:
    from app.config import get_settings

    settings = get_settings()
    return ReminderService(
        ctx.db,
        max_pending=settings.reminder_max_pending,
        max_days=settings.reminder_max_days_ahead,
    )


class CreateReminderArgs(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=300,
        description="What to remind the user about, in a few words, e.g. 'Call Ravi'.",
    )
    at: datetime | None = Field(default=None, description=LOCAL_TIME_HELP)
    in_minutes: int | None = Field(
        default=None,
        ge=1,
        le=60 * 24 * 366,
        description="Remind after this many minutes from now. Use this for 'in 30 minutes'.",
    )
    repeat: Literal["daily", "weekdays", "weekly"] | None = Field(
        default=None, description="Repeat at the same local time. Omit for a one-time reminder."
    )

    @field_validator("at")
    @classmethod
    def _naive(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            raise ValueError("Give local time without a UTC offset, e.g. 2026-10-20T18:00")
        return value

    @model_validator(mode="after")
    def _one_time(self) -> "CreateReminderArgs":
        if (self.at is None) == (self.in_minutes is None):
            raise ValueError("Give either at or in_minutes, not both and not neither")
        return self


def create_reminder(ctx: ToolContext, args: CreateReminderArgs) -> dict[str, Any]:
    try:
        reminder = _service(ctx).create(
            ctx.user_id,
            args.text,
            ctx.timezone,
            at=args.at,
            in_minutes=args.in_minutes,
            repeat=args.repeat,
            now=ctx.now,
        )
    except ReminderError as exc:
        raise ToolError(str(exc)) from exc
    return {
        "created": describe(reminder),
        "note": "It will be sent to the user in Telegram at that time.",
    }


class ListRemindersArgs(BaseModel):
    pass


def list_reminders(ctx: ToolContext, args: ListRemindersArgs) -> dict[str, Any]:
    items = [describe(r) for r in _service(ctx).pending(ctx.user_id)]
    return {"count": len(items), "reminders": items}


class CancelReminderArgs(BaseModel):
    reminder_id: int = Field(gt=0, description="Id from list_reminders.")


def cancel_reminder(ctx: ToolContext, args: CancelReminderArgs) -> dict[str, Any]:
    if not _service(ctx).cancel(ctx.user_id, args.reminder_id):
        raise ToolError("Reminder not found (it may have been sent or cancelled already).")
    return {"cancelled": args.reminder_id}


REMINDER_TOOLS: list[Tool] = [
    Tool(
        name="create_reminder",
        description=(
            "Set a reminder: the user is messaged in Telegram at that time. Give "
            "EITHER at (local time) OR in_minutes, plus optional repeat. Only for "
            "reminders the user asked for."
        ),
        args_model=CreateReminderArgs,
        handler=create_reminder,
        domain="reminder",
        side_effects=True,
    ),
    Tool(
        name="list_reminders",
        description="List the user's pending reminders with their ids.",
        args_model=ListRemindersArgs,
        handler=list_reminders,
        domain="reminder",
    ),
    Tool(
        name="cancel_reminder",
        description="Cancel one pending reminder by id (from list_reminders).",
        args_model=CancelReminderArgs,
        handler=cancel_reminder,
        domain="reminder",
        side_effects=True,
    ),
]