import asyncio
from datetime import date, datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user
from app.config import get_settings
from app.database.session import get_db
from app.fitness.service import FitnessService, format_day, format_progress
from app.profiles.service import ProfileService, local_today

router = APIRouter(prefix="/fitness", tags=["fitness"])


def _today(db: Session, user_id: int) -> date:
    settings = get_settings()
    profile = ProfileService(db, settings.default_currency, settings.default_timezone).get_or_create(
        user_id
    )
    return local_today(profile.timezone, datetime.now(timezone.utc))


@router.get("/day")
async def day(
    on: date | None = Query(default=None, description="YYYY-MM-DD. Omit for today."),
    current: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Meals, calories, workouts and weight for one day. Log them by chatting."""

    def load() -> dict[str, Any]:
        today = _today(db, current.user_id)
        summary = FitnessService(db).day_summary(current.user_id, on or today, today)
        summary["text"] = format_day(summary)
        return summary

    return await asyncio.to_thread(load)


@router.get("/progress")
async def progress(
    days: int = Query(default=30, ge=1, le=365),
    current: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    def load() -> dict[str, Any]:
        result = FitnessService(db).progress(current.user_id, _today(db, current.user_id), days)
        result["text"] = format_progress(result)
        return result

    return await asyncio.to_thread(load)


@router.get("/goal")
async def goal(
    current: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    from app.fitness.service import goal_out

    def load() -> dict[str, Any] | None:
        svc = FitnessService(db)
        found = svc.goals.get(current.user_id)
        return goal_out(found, svc.unit(current.user_id)) if found else None

    result = await asyncio.to_thread(load)
    if result is None:
        raise HTTPException(status_code=404, detail="No goal set. Chat: 'I want to cut'.")
    return result