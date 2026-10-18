import asyncio
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.fitness.plan_service import PlanService, plan_out
from app.fitness.plans import PlanError
from app.tracking.service import TrackingService
from app.tools.base import Tool, ToolContext, ToolError


class NoArgs(BaseModel):
    pass


class CreatePlanArgs(BaseModel):
    confirm: bool = Field(
        default=True,
        description="Always true. Only call after the profile is complete and targets are calculated.",
    )


class GetPlanArgs(BaseModel):
    scope: Literal["today", "week"] = Field(
        default="today", description="today = today's workout and meals; week = all 7 days."
    )


def _need_llm(ctx: ToolContext) -> Any:
    if ctx.llm is None:
        raise ToolError("Plan generation is not available right now.")
    return ctx.llm


async def create_plan(ctx: ToolContext, args: CreatePlanArgs) -> dict[str, Any]:
    llm = _need_llm(ctx)
    svc = PlanService(ctx.db)
    existing = await asyncio.to_thread(svc.active, ctx.user_id)
    if existing is not None:
        raise ToolError(
            f"The user already has plan v{existing.version}. Use update_plan only if they asked "
            "to change it."
        )
    try:
        plan = await svc.create(ctx.user_id, ctx.today, llm)
    except PlanError as exc:
        raise ToolError(f"{exc} Do not call create_plan again in this reply: tell the user.") from exc
    out = await asyncio.to_thread(_view, ctx, plan.version, "week")
    out["tracking_reminders"] = await asyncio.to_thread(_enable_tracking, ctx)
    return out


def _enable_tracking(ctx: ToolContext) -> str:
    """First plan: switch on meal/workout reminders unless the user already chose."""
    svc = TrackingService(ctx.db)
    if svc.repo.get(ctx.user_id) is None:
        svc.configure(ctx.user_id, enabled=True)
        return (
            "ON: I will remind the user if breakfast, lunch, dinner or a planned workout is not "
            "logged. Tell them, and that they can say 'turn off tracking reminders' or "
            "'skipped lunch'."
        )
    return svc.status_text(ctx.user_id)


async def update_plan(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    llm = _need_llm(ctx)
    svc = PlanService(ctx.db)
    try:
        plan, changes = await svc.update(ctx.user_id, ctx.today, llm, ctx.now)
    except PlanError as exc:
        raise ToolError(str(exc)) from exc
    out = await asyncio.to_thread(_view, ctx, plan.version, "week")
    out["what_changed"] = changes
    return out


def _view(ctx: ToolContext, version: int, scope: str) -> dict[str, Any]:
    plan = PlanService(ctx.db).active(ctx.user_id)
    if plan is None or plan.version != version:
        raise ToolError("No active plan.")
    return plan_out(plan, scope, ctx.today)


def get_plan(ctx: ToolContext, args: GetPlanArgs) -> dict[str, Any]:
    plan = PlanService(ctx.db).active(ctx.user_id)
    if plan is None:
        raise ToolError("The user has no plan yet. Collect their profile, then call create_plan.")
    return plan_out(plan, args.scope, ctx.today)


FITNESS_PLAN_TOOLS: list[Tool] = [
    Tool(
        name="create_plan",
        description=(
            "Create the user's first 7-day workout and meal plan from their saved profile and "
            "calculated targets. Takes up to a minute. Only once, at the start."
        ),
        args_model=CreatePlanArgs,
        handler=create_plan,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="update_plan",
        description=(
            "Rebuild the plan from the user's latest stats and progress. Only when the user "
            "asks to update or change their plan."
        ),
        args_model=NoArgs,
        handler=update_plan,
        domain="fitness",
        side_effects=True,
    ),
    Tool(
        name="get_plan",
        description="Show the user's current plan: today's workout and meals, or the whole week.",
        args_model=GetPlanArgs,
        handler=get_plan,
        domain="fitness",
    ),
]