from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import (
    BodyWeight,
    FitnessGoal,
    FitnessPlan,
    FitnessProfile,
    Meal,
    Workout,
)


class MealRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        user_id: int,
        eaten_on: date,
        meal_type: str,
        description: str,
        calories: int,
        protein_g: Decimal | None,
        carbs_g: Decimal | None,
        fat_g: Decimal | None,
        estimated: bool,
    ) -> Meal:
        meal = Meal(
            user_id=user_id,
            eaten_on=eaten_on,
            meal_type=meal_type,
            description=description,
            calories=calories,
            protein_g=protein_g,
            carbs_g=carbs_g,
            fat_g=fat_g,
            estimated=estimated,
        )
        self._db.add(meal)
        self._db.commit()
        self._db.refresh(meal)
        return meal

    def get(self, user_id: int, meal_id: int) -> Meal | None:
        return self._db.scalar(select(Meal).where(Meal.id == meal_id, Meal.user_id == user_id))

    def save(self, meal: Meal) -> Meal:
        self._db.commit()
        self._db.refresh(meal)
        return meal

    def find(self, user_id: int, start: date, end: date, limit: int = 50) -> list[Meal]:
        return list(
            self._db.scalars(
                select(Meal)
                .where(Meal.user_id == user_id, Meal.eaten_on >= start, Meal.eaten_on <= end)
                .order_by(Meal.eaten_on.desc(), Meal.id.desc())
                .limit(limit)
            )
        )

    def delete(self, user_id: int, meal_id: int) -> bool:
        result = self._db.execute(delete(Meal).where(Meal.id == meal_id, Meal.user_id == user_id))
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]

    def daily_totals(self, user_id: int, start: date, end: date) -> dict[date, int]:
        rows = self._db.execute(
            select(Meal.eaten_on, func.sum(Meal.calories))
            .where(Meal.user_id == user_id, Meal.eaten_on >= start, Meal.eaten_on <= end)
            .group_by(Meal.eaten_on)
        ).all()
        return {day: int(total or 0) for day, total in rows}


class WorkoutRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def add(
        self,
        user_id: int,
        done_on: date,
        kind: str,
        description: str,
        duration_minutes: int | None,
        calories_burned: int | None,
    ) -> Workout:
        workout = Workout(
            user_id=user_id,
            done_on=done_on,
            kind=kind,
            description=description,
            duration_minutes=duration_minutes,
            calories_burned=calories_burned,
        )
        self._db.add(workout)
        self._db.commit()
        self._db.refresh(workout)
        return workout

    def get(self, user_id: int, workout_id: int) -> Workout | None:
        return self._db.scalar(
            select(Workout).where(Workout.id == workout_id, Workout.user_id == user_id)
        )

    def find(self, user_id: int, start: date, end: date, limit: int = 50) -> list[Workout]:
        return list(
            self._db.scalars(
                select(Workout)
                .where(Workout.user_id == user_id, Workout.done_on >= start, Workout.done_on <= end)
                .order_by(Workout.done_on.desc(), Workout.id.desc())
                .limit(limit)
            )
        )

    def delete(self, user_id: int, workout_id: int) -> bool:
        result = self._db.execute(
            delete(Workout).where(Workout.id == workout_id, Workout.user_id == user_id)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]


class WeightRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def upsert(self, user_id: int, measured_on: date, weight_kg: Decimal) -> BodyWeight:
        """One weigh-in per day: a second one for the same day replaces the first."""
        existing = self._db.scalar(
            select(BodyWeight).where(
                BodyWeight.user_id == user_id, BodyWeight.measured_on == measured_on
            )
        )
        if existing is not None:
            existing.weight_kg = weight_kg
            self._db.commit()
            self._db.refresh(existing)
            return existing
        row = BodyWeight(user_id=user_id, measured_on=measured_on, weight_kg=weight_kg)
        self._db.add(row)
        try:
            self._db.commit()
        except IntegrityError:  # two quick messages for the same day
            self._db.rollback()
            again = self._db.scalar(
                select(BodyWeight).where(
                    BodyWeight.user_id == user_id, BodyWeight.measured_on == measured_on
                )
            )
            if again is None:
                raise
            again.weight_kg = weight_kg
            self._db.commit()
            self._db.refresh(again)
            return again
        self._db.refresh(row)
        return row

    def get(self, user_id: int, weight_id: int) -> BodyWeight | None:
        return self._db.scalar(
            select(BodyWeight).where(BodyWeight.id == weight_id, BodyWeight.user_id == user_id)
        )

    def find(self, user_id: int, start: date, end: date, limit: int = 400) -> list[BodyWeight]:
        """Oldest first, so trends read left to right."""
        return list(
            self._db.scalars(
                select(BodyWeight)
                .where(
                    BodyWeight.user_id == user_id,
                    BodyWeight.measured_on >= start,
                    BodyWeight.measured_on <= end,
                )
                .order_by(BodyWeight.measured_on, BodyWeight.id)
                .limit(limit)
            )
        )

    def latest(self, user_id: int, on_or_before: date) -> BodyWeight | None:
        return self._db.scalar(
            select(BodyWeight)
            .where(BodyWeight.user_id == user_id, BodyWeight.measured_on <= on_or_before)
            .order_by(BodyWeight.measured_on.desc(), BodyWeight.id.desc())
            .limit(1)
        )

    def delete(self, user_id: int, weight_id: int) -> bool:
        result = self._db.execute(
            delete(BodyWeight).where(BodyWeight.id == weight_id, BodyWeight.user_id == user_id)
        )
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]


class GoalRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> FitnessGoal | None:
        return self._db.get(FitnessGoal, user_id)

    def set(
        self,
        user_id: int,
        mode: str,
        target_weight_kg: Decimal | None,
        start_weight_kg: Decimal | None,
        daily_calories: int | None,
        daily_protein_g: int | None,
        weekly_workouts: int | None,
        started_on: date,
    ) -> FitnessGoal:
        goal = self.get(user_id)
        if goal is None:
            goal = FitnessGoal(user_id=user_id, started_on=started_on, mode=mode)
            self._db.add(goal)
        elif goal.mode != mode:
            goal.started_on = started_on  # a new kind of goal starts a new progress window
        goal.mode = mode
        goal.target_weight_kg = target_weight_kg
        goal.start_weight_kg = start_weight_kg
        goal.daily_calories = daily_calories
        goal.daily_protein_g = daily_protein_g
        goal.weekly_workouts = weekly_workouts
        self._db.commit()
        self._db.refresh(goal)
        return goal

    def clear(self, user_id: int) -> bool:
        result = self._db.execute(delete(FitnessGoal).where(FitnessGoal.user_id == user_id))
        self._db.commit()
        return result.rowcount == 1  # type: ignore[attr-defined]


class FitnessProfileRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get(self, user_id: int) -> FitnessProfile | None:
        return self._db.get(FitnessProfile, user_id)

    def save(self, user_id: int, **fields: object) -> FitnessProfile:
        """Create or update. Only the given fields change."""
        row = self.get(user_id)
        if row is None:
            row = FitnessProfile(user_id=user_id)
            self._db.add(row)
        for name, value in fields.items():
            setattr(row, name, value)
        self._db.commit()
        self._db.refresh(row)
        return row


class PlanRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def active(self, user_id: int) -> FitnessPlan | None:
        return self._db.scalar(
            select(FitnessPlan).where(FitnessPlan.user_id == user_id, FitnessPlan.status == "active")
        )

    def history(self, user_id: int, limit: int = 10) -> list[FitnessPlan]:
        return list(
            self._db.scalars(
                select(FitnessPlan)
                .where(FitnessPlan.user_id == user_id)
                .order_by(FitnessPlan.version.desc())
                .limit(limit)
            )
        )

    def add(
        self,
        user_id: int,
        workout: dict[str, Any],
        meals: dict[str, Any],
        targets: dict[str, Any],
        inputs: dict[str, Any],
        reason: str,
    ) -> FitnessPlan:
        """The new plan becomes active; the old one is archived, in one transaction."""
        latest = self._db.scalar(
            select(func.max(FitnessPlan.version)).where(FitnessPlan.user_id == user_id)
        )
        self._db.execute(
            update(FitnessPlan)
            .where(FitnessPlan.user_id == user_id, FitnessPlan.status == "active")
            .values(status="archived")
        )
        plan = FitnessPlan(
            user_id=user_id,
            version=(latest or 0) + 1,
            status="active",
            workout=workout,
            meals=meals,
            targets=targets,
            inputs=inputs,
            reason=reason[:300],
        )
        self._db.add(plan)
        self._db.commit()
        self._db.refresh(plan)
        return plan