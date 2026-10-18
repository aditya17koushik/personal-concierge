"""Builds the coach's data, runs the rules, and writes the messages (text is built by code)."""

from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.coach.rules import (
    CoachData,
    DayStat,
    Flag,
    Review,
    Scope,
    evaluate,
    last_seven,
)
from app.database.models import CoachSettings
from app.database.repositories.coach import CoachSettingsRepository
from app.database.repositories.fitness import PlanRepository
from app.fitness.service import FitnessService, from_kg
from app.profiles.service import ProfileError, normalize_time

WEIGHT_WINDOW_DAYS = 28
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def flag_out(flag: Flag) -> dict[str, Any]:
    return {
        "code": flag.code,
        "level": flag.level,
        "text": flag.text,
        "suggest": flag.suggest,
        "scope": flag.scope,
    }


def normalize_weekday(value: str) -> int:
    text = value.strip().lower()
    for index, name in enumerate(WEEKDAYS):
        if text in (name, name[:3]):
            return index
    raise ProfileError("Use a weekday like sunday or mon.")


class CoachService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self.fitness = FitnessService(db)
        self.settings = CoachSettingsRepository(db)

    # ------------------------------------------------------------------ data

    def data(self, user_id: int, today: date, local_hour: int | None = None) -> CoachData:
        fit = self.fitness
        start = today - timedelta(days=6)
        meals = fit.meals.find(user_id, start, today, limit=2000)
        workouts = fit.workouts.find(user_id, start, today, limit=1000)

        days: list[DayStat] = []
        for day in last_seven(today):
            todays_meals = [m for m in meals if m.eaten_on == day]
            todays_work = [w for w in workouts if w.done_on == day]
            days.append(
                DayStat(
                    day=day,
                    calories=sum(m.calories for m in todays_meals),
                    meals=len(todays_meals),
                    protein_g=sum(float(m.protein_g) for m in todays_meals if m.protein_g is not None),
                    protein_known=all(m.protein_g is not None for m in todays_meals),
                    workouts=len(todays_work),
                    minutes=sum(w.duration_minutes or 0 for w in todays_work),
                )
            )

        weights = [
            (w.measured_on, float(w.weight_kg))
            for w in fit.weights.find(user_id, today - timedelta(days=WEIGHT_WINDOW_DAYS - 1), today)
        ]
        goal = fit.goals.get(user_id)
        plan = PlanRepository(self._db).active(user_id)
        plan_age: int | None = None
        if plan is not None:
            created = plan.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            plan_age = (datetime.now(timezone.utc) - created).days
        return CoachData(
            plan_age_days=plan_age,
            today=today,
            days=days,
            weights=weights,
            mode=goal.mode if goal else None,
            target_calories=goal.daily_calories if goal else None,
            target_protein_g=goal.daily_protein_g if goal else None,
            weekly_workouts=goal.weekly_workouts if goal else None,
            goal_started_on=goal.started_on if goal else None,
            local_hour=local_hour,
            unit=fit.unit(user_id),
        )

    def review(
        self, user_id: int, today: date, scope: Scope = "trend", local_hour: int | None = None
    ) -> tuple[Review, CoachData]:
        data = self.data(user_id, today, local_hour)
        return evaluate(data, scope), data

    def review_out(self, user_id: int, today: date) -> dict[str, Any]:
        """What the get_coaching tool returns: flags and the facts behind them."""
        review, data = self.review(user_id, today)
        facts = dict(review.facts)
        facts["unit"] = data.unit
        if "weight_latest_kg" in facts:
            facts["weight_latest"] = from_kg(float(facts.pop("weight_latest_kg")), data.unit)  # type: ignore[arg-type]
        if "weight_per_week_kg" in facts:
            facts["weight_change_per_week"] = from_kg(float(facts.pop("weight_per_week_kg")), data.unit)  # type: ignore[arg-type]
        if data.mode:
            facts["calorie_target"] = data.target_calories
            facts["protein_target_g"] = data.target_protein_g
            facts["weekly_workout_target"] = data.weekly_workouts
        return {
            "flags": [flag_out(f) for f in review.flags],
            "facts": facts,
            "safety": any(f.level == "alert" for f in review.flags),
            "note": (
                "Flags are computed from the user's logs. Report them honestly; use only these "
                "numbers. If safety is true, do not push the user to eat less or train more."
            ),
        }

    # -------------------------------------------------------------- messages

    def checkin_text(self, user_id: int, today: date, local_hour: int) -> str:
        review, data = self.review(user_id, today, "today", local_hour)
        stat = next(d for d in data.days if d.day == today)
        lines = ["Evening check-in"]

        eaten = f"Today: {stat.calories} kcal"
        if data.target_calories:
            left = data.target_calories - stat.calories
            eaten += f" of {data.target_calories} ({abs(left)} {'left' if left >= 0 else 'over'})"
        lines.append(eaten)
        if data.target_protein_g and stat.protein_known and stat.meals:
            lines.append(f"Protein: {round(stat.protein_g)} g of {data.target_protein_g} g")
        if stat.workouts:
            lines.append(f"Workout: {stat.workouts} logged ({stat.minutes} min)")
        elif data.weekly_workouts is not None:
            done = sum(d.workouts for d in data.days)
            lines.append(f"No workout logged today. This week: {done} (target {data.weekly_workouts}).")

        return "\n".join(lines + [""] + _flag_lines(review, limit=3))

    def weekly_text(self, user_id: int, today: date) -> str:
        review, data = self.review(user_id, today)
        lines = ["Weekly review (last 7 days)"]

        per_week = review.facts.get("weight_per_week_kg")
        latest = review.facts.get("weight_latest_kg")
        if latest is not None:
            line = f"Weight: {from_kg(float(latest), data.unit)} {data.unit}"  # type: ignore[arg-type]
            if per_week is not None:
                change = from_kg(float(per_week), data.unit)  # type: ignore[arg-type]
                line += f" ({change:+.1f} {data.unit}/week)"
            lines.append(line)
        avg = review.facts.get("avg_calories")
        if avg is not None:
            line = f"Food: average {avg} kcal on {review.facts['logged_days']} logged days"
            if data.target_calories:
                line += f" (target {data.target_calories})"
            lines.append(line)
        else:
            lines.append("Food: not enough logged days to judge")
        sessions = review.facts["sessions_7d"]
        minutes = review.facts["training_minutes_7d"]
        rest = 7 - int(review.facts["training_days_7d"])  # type: ignore[arg-type]
        line = f"Training: {sessions} sessions, {minutes} min, {rest} rest days"
        if data.weekly_workouts is not None:
            line += f" (target {data.weekly_workouts})"
        lines.append(line)

        return "\n".join(lines + [""] + _flag_lines(review, limit=5, focus=True))

    # -------------------------------------------------------------- settings

    def configure(
        self,
        user_id: int,
        *,
        enabled: bool | None = None,
        time_text: str | None = None,
        weekday: str | None = None,
    ) -> CoachSettings:
        return self.settings.set(
            user_id,
            enabled=enabled,
            checkin_time=normalize_time(time_text) if time_text else None,
            weekly_day=normalize_weekday(weekday) if weekday else None,
        )


def _flag_lines(review: Review, limit: int, focus: bool = False) -> list[str]:
    problems = review.by_level("alert", "warn")
    info = review.by_level("info")
    good = review.by_level("good")
    lines: list[str] = []

    if good:
        lines.append("Going well: " + good[0].text)
    if problems:
        if focus:
            lines.append("Fix first: " + problems[0].line())
            for flag in problems[1:limit]:
                lines.append("Also: " + flag.line())
        else:
            for flag in problems[:limit]:
                lines.append(flag.line())
    elif not good:
        lines.append("Nothing needs fixing right now.")
    else:
        lines.append("Nothing needs fixing. Keep it going.")
    for flag in info[: max(0, limit - len(problems))]:
        lines.append(flag.line())
    return lines