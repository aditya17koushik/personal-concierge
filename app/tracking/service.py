"""Tracking nudges: remind the user to log a planned meal or workout they have not logged.

Rules are code. A meal nudge fires once per day per slot, only after the slot's time and within a
window, only if no meal of that type is logged and the user did not say they skipped it. The
workout nudge fires only on the plan's training days. The text mentions what the plan has for
that slot and tells the user how to say "skipped" / "rest today".
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.database.models import TrackingSettings
from app.database.repositories.fitness import MealRepository, WorkoutRepository
from app.database.repositories.tracking import SLOT_COLUMNS, TrackingRepository
from app.fitness.plan_service import PlanService, plan_day
from app.profiles.service import ProfileError, normalize_time

SLOTS = ("breakfast", "lunch", "dinner", "workout")


@dataclass(frozen=True)
class Nudge:
    slot: str
    text: str


def normalize_slot(word: str) -> str:
    w = word.strip().lower()
    aliases = {"rest": "workout", "gym": "workout", "training": "workout", "exercise": "workout"}
    w = aliases.get(w, w)
    if w not in SLOTS:
        raise ProfileError("Slot must be breakfast, lunch, dinner or workout.")
    return w


def times_of(row: TrackingSettings) -> dict[str, str]:
    return {slot: getattr(row, col) for slot, col in SLOT_COLUMNS.items()}


class TrackingService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self.repo = TrackingRepository(db)

    def configure(
        self, user_id: int, *, enabled: bool | None = None, times: dict[str, str] | None = None
    ) -> TrackingSettings:
        clean: dict[str, str] = {}
        for slot, value in (times or {}).items():
            clean[normalize_slot(slot)] = normalize_time(value)
        return self.repo.set(user_id, enabled=enabled, times=clean)

    def status(self, user_id: int) -> dict[str, Any]:
        row = self.repo.get(user_id)
        if row is None:
            return {"enabled": False, "times": {}}
        return {"enabled": row.enabled, "times": times_of(row)}

    def status_text(self, user_id: int) -> str:
        st = self.status(user_id)
        if not st["enabled"]:
            return (
                "Tracking reminders are off. /tracking on turns them on: I will nudge you if a "
                "meal or planned workout is not logged. Times: /tracking lunch 13:30"
            )
        t = st["times"]
        return (
            f"Tracking reminders are on: breakfast {t['breakfast']}, lunch {t['lunch']}, dinner "
            f"{t['dinner']}, workout {t['workout']} (training days only). /tracking off to stop. "
            "Say 'skipped lunch' or 'rest today' and I will stop asking for that day."
        )

    def skip(self, user_id: int, slot: str, day: date) -> None:
        self.repo.skip(user_id, day, normalize_slot(slot))

    def due(self, user_id: int, local: datetime, window: timedelta) -> list[Nudge]:
        """The nudges that should go out now (not yet claimed)."""
        row = self.repo.get(user_id)
        if row is None or not row.enabled:
            return []
        day = local.date()
        events = self.repo.events(user_id, day)
        plan = PlanService(self._db).active(user_id)
        view = plan_day(plan, day) if plan is not None else None
        out: list[Nudge] = []

        for slot, hhmm in times_of(row).items():
            if slot in events:
                continue
            hour, minute = (int(x) for x in hhmm.split(":"))
            start = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if not (start <= local < start + window):
                continue
            if slot == "workout":
                text = self._workout_nudge(user_id, day, view)
            else:
                text = self._meal_nudge(user_id, day, slot, view)
            if text:
                out.append(Nudge(slot, text))
        return out

    # ---------------------------------------------------------------- texts

    def _meal_nudge(self, user_id: int, day: date, slot: str, view: dict[str, Any] | None) -> str | None:
        meals = MealRepository(self._db).find(user_id, day, day, limit=50)
        if any(m.meal_type == slot for m in meals):
            return None
        planned = ""
        if view and view["meals"]:
            match = next((m for m in view["meals"]["meals"] if m["type"] == slot), None)
            if match:
                planned = f" The plan has: {match['name']} ({match['calories']} kcal)."
        return (
            f"No {slot} logged yet today.{planned} Tell me what you had, even roughly "
            f"(for example: 2 rotis and dal), or say 'skipped {slot}'."
        )

    def _workout_nudge(self, user_id: int, day: date, view: dict[str, Any] | None) -> str | None:
        if view is None or view["workout"] is None or view["workout"]["rest"]:
            return None  # no plan, or a rest day: nothing to chase
        if WorkoutRepository(self._db).find(user_id, day, day, limit=1):
            return None
        w = view["workout"]
        mins = f", about {w['minutes']} min" if w.get("minutes") else ""
        return (
            f"Today's planned workout: {w['focus']}{mins}. Not logged yet. Log it when you "
            "finish (for example: did 45 min of weights), or say 'rest today' if you are skipping."
        )