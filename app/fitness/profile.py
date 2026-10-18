"""The user's stats and preferences: intake, completeness, and the targets computed from them."""

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from app.database.models import FitnessProfile
from app.database.repositories.fitness import FitnessProfileRepository
from app.fitness.service import FitnessError, FitnessService, clean, weight_out
from app.fitness.targets import (
    ACTIVITY_FACTORS,
    DIET_PREFS,
    EQUIPMENT,
    EXPERIENCE,
    MUSCLE_LEVELS,
    SEXES,
    Screen,
    Targets,
    bmi,
    compute_targets,
    screen,
)

# Without these we do not generate numbers or plans. 'conditions' is required on purpose:
# "none" is a valid answer, but it must be an answer, because it decides how careful to be.
REQUIRED = (
    "age", "sex", "height_cm", "activity_level", "experience", "conditions",
    "days_per_week", "equipment", "diet_pref",
)  # fmt: skip
OPTIONAL = ("muscle_level", "preferred_workouts", "session_minutes", "allergies", "cuisine")

QUESTIONS = {
    "age": "age in years",
    "sex": "sex (male, female or other), used by the calorie formula",
    "height_cm": "height (convert to cm)",
    "activity_level": "daily activity outside workouts: sedentary (desk job), light (light exercise "
    "1-3 days), moderate (3-5 days) or very_active (6-7 days or a physical job)",
    "experience": "training experience: beginner, intermediate or advanced",
    "conditions": "any health conditions, injuries or surgeries (the answer 'none' is fine)",
    "days_per_week": "how many days a week they can train (1-6)",
    "equipment": "where they train: gym, home_dumbbells, home_minimal or none",
    "diet_pref": "diet: vegetarian, eggetarian, non_vegetarian or vegan",
    "weight": "current weight (log it with log_weight)",
    "goal": "goal: cut, bulk or maintain (set it with set_goal)",
    "muscle_level": "current muscle level: low, average or high (optional)",
    "preferred_workouts": "preferred workout types, e.g. weights, running, yoga (optional)",
    "allergies": "food allergies or foods to avoid (optional)",
    "cuisine": "cuisine they like, e.g. North Indian (optional)",
}


@dataclass(frozen=True)
class Computed:
    status: str  # ok | conservative | refuse
    mode: str
    weight_kg: float
    bmi: float
    screen: Screen
    targets: Targets | None


def age_now(row: FitnessProfile, today: date) -> int | None:
    if row.age is None:
        return None
    if row.age_set_on is None:
        return row.age
    return row.age + (today - row.age_set_on).days // 365


class FitnessProfileService:
    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = FitnessProfileRepository(db)
        self._fitness = FitnessService(db)

    def save(self, user_id: int, today: date, **given: Any) -> FitnessProfile:
        """Merge the given answers (None means 'not given', not 'clear')."""
        fields: dict[str, Any] = {}
        for name, value in given.items():
            if value is None:
                continue
            fields[name] = self._validate(name, value)
        if "age" in fields:
            fields["age_set_on"] = today
        return self._repo.save(user_id, **fields)

    @staticmethod
    def _validate(name: str, value: Any) -> Any:
        if name == "age":
            if not 10 <= int(value) <= 100:
                raise FitnessError("Age must be between 10 and 100.")
            return int(value)
        if name == "height_cm":
            if not 120 <= float(value) <= 230:
                raise FitnessError("Height should be between 120 and 230 cm. Ask the user to check.")
            return int(round(float(value)))
        if name == "days_per_week":
            if not 1 <= int(value) <= 6:
                raise FitnessError("Training days must be between 1 and 6 a week.")
            return int(value)
        if name == "session_minutes":
            if not 15 <= int(value) <= 180:
                raise FitnessError("Session length must be between 15 and 180 minutes.")
            return int(value)
        choices = {
            "sex": SEXES,
            "activity_level": tuple(ACTIVITY_FACTORS),
            "experience": EXPERIENCE,
            "muscle_level": MUSCLE_LEVELS,
            "equipment": EQUIPMENT,
            "diet_pref": DIET_PREFS,
        }
        if name in choices:
            text = str(value).strip().lower()
            if text not in choices[name]:
                raise FitnessError(f"{name} must be one of: {', '.join(choices[name])}.")
            return text
        limits = {"conditions": 500, "preferred_workouts": 200, "allergies": 200, "cuisine": 100}
        if name in limits:
            return clean(str(value), limits[name])
        raise FitnessError(f"Unknown field: {name}")

    def get(self, user_id: int) -> FitnessProfile | None:
        return self._repo.get(user_id)

    def missing(self, user_id: int, today: date) -> list[str]:
        row = self._repo.get(user_id)
        gaps = [f for f in REQUIRED if row is None or getattr(row, f) is None]
        if self._fitness.weights.latest(user_id, today) is None:
            gaps.append("weight")
        if self._fitness.goals.get(user_id) is None:
            gaps.append("goal")
        return gaps

    def view(self, user_id: int, today: date) -> dict[str, Any]:
        row = self._repo.get(user_id)
        unit = self._fitness.unit(user_id)
        latest = self._fitness.weights.latest(user_id, today)
        out: dict[str, Any] = {
            "profile": {},
            "weight": weight_out(latest.weight_kg, unit) if latest else None,
            "missing_required": self.missing(user_id, today),
            "optional_not_given": [
                f for f in OPTIONAL if row is None or getattr(row, f) is None
            ],
        }
        if row is not None:
            out["profile"] = {
                "age": age_now(row, today),
                "sex": row.sex,
                "height_cm": row.height_cm,
                "activity_level": row.activity_level,
                "experience": row.experience,
                "muscle_level": row.muscle_level,
                "conditions": row.conditions,
                "preferred_workouts": row.preferred_workouts,
                "days_per_week": row.days_per_week,
                "session_minutes": row.session_minutes,
                "equipment": row.equipment,
                "diet_pref": row.diet_pref,
                "allergies": row.allergies,
                "cuisine": row.cuisine,
            }
        out["ask_for"] = {k: QUESTIONS[k] for k in out["missing_required"]}
        return out

    def compute(
        self, user_id: int, today: date, mode_override: str | None = None, calorie_adjust: int = 0
    ) -> Computed:
        gaps = self.missing(user_id, today)
        if gaps:
            raise FitnessError(
                "Missing before targets can be worked out: " + ", ".join(gaps) + ". Ask the user."
            )
        row = self._repo.get(user_id)
        latest = self._fitness.weights.latest(user_id, today)
        goal = self._fitness.goals.get(user_id)
        assert row is not None and latest is not None and goal is not None
        age = age_now(row, today)
        assert age is not None and row.height_cm is not None
        assert row.sex and row.activity_level and row.experience
        weight = float(latest.weight_kg)
        mode = mode_override or goal.mode

        check = screen(age, row.conditions, weight, float(row.height_cm), mode)
        if check.mode_override:
            mode = check.mode_override
        if check.status == "refuse":
            return Computed("refuse", mode, weight, bmi(weight, row.height_cm), check, None)
        targets = compute_targets(
            sex=row.sex,
            age=age,
            height_cm=float(row.height_cm),
            weight_kg=weight,
            activity=row.activity_level,
            experience=row.experience,
            mode=mode,
            conservative=check.status == "conservative",
            calorie_adjust=calorie_adjust,
        )
        return Computed(check.status, mode, weight, bmi(weight, row.height_cm), check, targets)

    def computed_out(self, user_id: int, today: date) -> dict[str, Any]:
        c = self.compute(user_id, today)
        unit = self._fitness.unit(user_id)
        out: dict[str, Any] = {
            "status": c.status,
            "goal_mode_used": c.mode,
            "weight": weight_out(c.weight_kg, unit),
            "bmi": round(c.bmi, 1),
            "reasons": list(c.screen.reasons),
            "see_a_doctor_first": c.screen.clearance,
            "training_limits": list(c.screen.movement_notes),
        }
        if c.targets is None:
            out["targets"] = None
            out["note"] = "No numbers are given. Be kind, explain why, and point to a professional."
            return out
        t = c.targets
        out["targets"] = {
            "bmr_kcal": t.bmr,
            "maintenance_kcal": t.tdee,
            "daily_calories": t.calories,
            "protein_g": t.protein_g,
            "fat_g": t.fat_g,
            "carbs_g": t.carbs_g,
            "expected_weight_change_per_week": weight_out(t.change_per_week_kg, unit),
            "notes": list(t.notes),
        }
        return out