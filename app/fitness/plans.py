"""Generating a 7-day workout plan and a 7-day meal plan.

The model writes the exercises and foods. CODE decides everything that matters for safety and
fit: the calorie and protein targets are inputs, and every plan is validated against them, against
the training days, the equipment, injuries, diet type and allergies. A plan that breaks a rule is
rejected and regenerated with the errors fed back; after three tries the user gets an honest error.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.llm.base import LLMError, LLMProvider
from app.schemas.llm import Message

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DayName = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_ATTEMPTS = 3
CALORIE_TOLERANCE = 0.10  # each day within +-10% of the target
PROTEIN_MIN_RATIO = 0.85


class PlanError(Exception):
    """User-safe reason a plan could not be made."""


# ------------------------------------------------------------------- schemas


class Exercise(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    sets: int = Field(ge=1, le=10)
    reps: str = Field(min_length=1, max_length=20)  # "8-10", "30 s"
    notes: str | None = Field(default=None, max_length=120)


class WorkoutDay(BaseModel):
    day: DayName
    focus: str = Field(max_length=60)
    rest: bool
    minutes: int | None = Field(default=None, ge=0, le=240)
    exercises: list[Exercise] = Field(default_factory=list)


class WorkoutPlan(BaseModel):
    summary: str = Field(max_length=300)
    progression: str = Field(max_length=300)
    days: list[WorkoutDay] = Field(min_length=7, max_length=7)


class MealItem(BaseModel):
    type: Literal["breakfast", "lunch", "dinner", "snack"]
    name: str = Field(min_length=1, max_length=80)
    items: str = Field(min_length=1, max_length=200)
    calories: int = Field(ge=0, le=2500)
    protein_g: float = Field(ge=0, le=200)


class MealDay(BaseModel):
    day: DayName
    meals: list[MealItem] = Field(min_length=3, max_length=6)


class MealPlan(BaseModel):
    notes: str = Field(max_length=300)
    days: list[MealDay] = Field(min_length=7, max_length=7)


@dataclass(frozen=True)
class PlanInputs:
    sex: str
    age: int
    height_cm: int
    weight_kg: float
    mode: str
    experience: str
    days_per_week: int
    equipment: str
    diet_pref: str
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    session_minutes: int = 60
    muscle_level: str | None = None
    preferred_workouts: str | None = None
    allergies: str | None = None
    cuisine: str | None = None
    movement_notes: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------- validation tables

GYM_ONLY = ("barbell", "machine", "cable", "lat pulldown", "leg press", "smith", "pec deck", "treadmill")
EQUIPMENT_BANS = {
    "gym": (),
    "home_dumbbells": GYM_ONLY,
    "home_minimal": GYM_ONLY + ("dumbbell", "kettlebell"),
    "none": GYM_ONLY + ("dumbbell", "kettlebell", "band"),
}
MOVEMENT_BANS = {
    "knee": ("jump", "lunge", "box", "burpee", "sprint", "run"),
    "shoulder": ("overhead", "snatch", "handstand", "military press", "upright row"),
    "back pain": ("deadlift", "good morning", "sit-up", "situp", "jump"),
    "slip disc": ("deadlift", "good morning", "sit-up", "situp", "jump"),
    "sciatica": ("deadlift", "good morning", "sit-up", "situp", "jump"),
    "injur": ("jump", "burpee", "sprint"),
    "hernia": ("deadlift", "sit-up", "crunch"),
    "arthritis": ("jump", "burpee"),
    "osteopor": ("jump", "twist", "sit-up", "crunch"),
    "low-impact only": ("jump", "burpee", "sprint", "run", "plyo"),
    "surgery": ("jump", "burpee", "sprint"),
    "dizz": ("sprint", "hiit"),
    "heart": ("sprint", "hiit", "1rm", "max effort"),
    "cardiac": ("sprint", "hiit", "1rm", "max effort"),
    "hypertension": ("sprint", "hiit", "1rm", "max effort"),
    "blood pressure": ("sprint", "hiit", "1rm", "max effort"),
    "asthma": ("sprint",),
    "fainting": ("sprint", "hiit"),
}
MEAT = (
    "chicken", "mutton", "lamb", "fish", "prawn", "shrimp", "beef", "pork", "bacon", "ham",
    "turkey", "salmon", "tuna", "meat", "crab", "keema", "sausage",
)  # fmt: skip
EGG = ("egg", "eggs", "omelette", "omelet")
DAIRY = (
    "paneer", "milk", "curd", "yogurt", "yoghurt", "cheese", "ghee", "butter", "whey",
    "honey", "cream", "buttermilk", "lassi", "dahi", "khoya", "mawa",
)  # fmt: skip
VEGAN_SAFE_PHRASES = (
    "peanut butter", "almond butter", "nut butter", "cocoa butter", "butter beans",
    "coconut milk", "almond milk", "soy milk", "oat milk", "coconut cream", "soya milk",
)  # fmt: skip


def _has(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}", text) is not None


def _whole(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def diet_violations(text: str, diet: str) -> list[str]:
    t = text.lower()
    if diet == "vegan":
        for phrase in VEGAN_SAFE_PHRASES:
            t = t.replace(phrase, " ")
    found = [w for w in MEAT if _whole(t, w)]
    if diet in ("vegetarian", "vegan"):
        found += [w for w in EGG if _whole(t, w)]
    if diet == "vegan":
        found += [w for w in DAIRY if _whole(t, w)]
    return found


def allergen_tokens(allergies: str | None) -> list[str]:
    if not allergies:
        return []
    parts = re.split(r"[,;/\n]| and | or | & ", allergies.lower())
    tokens = []
    for part in parts:
        word = part.strip(" .")
        if word in ("", "none", "no", "nil", "n/a", "na", "nothing") or len(word) < 3:
            continue
        tokens.append(word)
    return tokens


# ---------------------------------------------------------------- validation


def validate_workout(plan: WorkoutPlan, inputs: PlanInputs) -> list[str]:
    errors: list[str] = []
    if [d.day for d in plan.days] != list(DAYS):
        errors.append("days must be exactly Mon, Tue, Wed, Thu, Fri, Sat, Sun in that order")
    training = [d for d in plan.days if not d.rest]
    if len(training) != inputs.days_per_week:
        errors.append(
            f"there must be exactly {inputs.days_per_week} training days; you gave {len(training)}"
        )
    bans = list(EQUIPMENT_BANS.get(inputs.equipment, ()))
    for note in inputs.movement_notes:
        bans += list(MOVEMENT_BANS.get(note, ()))
    cap = inputs.session_minutes + 15
    for d in plan.days:
        if d.rest and d.exercises:
            errors.append(f"{d.day} is a rest day and must have no exercises")
        if not d.rest and not (3 <= len(d.exercises) <= 10):
            errors.append(f"{d.day} needs between 3 and 10 exercises")
        if not d.rest and d.minutes is not None and d.minutes > cap:
            errors.append(f"{d.day} is {d.minutes} minutes; the limit is {cap}")
        for ex in d.exercises:
            text = f"{ex.name} {ex.notes or ''}".lower()
            for word in bans:
                if _has(text, word):
                    errors.append(f"{d.day} '{ex.name}' is not allowed (equipment or injury: '{word}')")
                    break
    return errors


def validate_meals(plan: MealPlan, inputs: PlanInputs) -> list[str]:
    errors: list[str] = []
    if [d.day for d in plan.days] != list(DAYS):
        errors.append("days must be exactly Mon, Tue, Wed, Thu, Fri, Sat, Sun in that order")
    low = inputs.calories * (1 - CALORIE_TOLERANCE)
    high = inputs.calories * (1 + CALORIE_TOLERANCE)
    allergens = allergen_tokens(inputs.allergies)
    for d in plan.days:
        total = sum(m.calories for m in d.meals)
        protein = sum(m.protein_g for m in d.meals)
        types = {m.type for m in d.meals}
        if not {"breakfast", "lunch", "dinner"} <= types:
            errors.append(f"{d.day} must include breakfast, lunch and dinner")
        if not (low <= total <= high):
            errors.append(
                f"{d.day} totals {total} kcal; it must be between {round(low)} and {round(high)} "
                f"(target {inputs.calories})"
            )
        if protein < inputs.protein_g * PROTEIN_MIN_RATIO:
            errors.append(
                f"{d.day} has {round(protein)} g protein; it needs at least "
                f"{round(inputs.protein_g * PROTEIN_MIN_RATIO)} g (target {inputs.protein_g})"
            )
        for m in d.meals:
            text = f"{m.name} {m.items}".lower()
            bad = diet_violations(text, inputs.diet_pref)
            if bad:
                errors.append(f"{d.day} {m.type} '{m.name}' breaks the {inputs.diet_pref} diet: {', '.join(bad)}")
            for token in allergens:
                if _has(text, token) or _has(text, token.rstrip("s")):
                    errors.append(f"{d.day} {m.type} '{m.name}' contains '{token}', which the user avoids")
    return errors


# ------------------------------------------------------------------- prompts

WORKOUT_SHAPE = (
    '{"summary": "one sentence", "progression": "how to progress week to week", "days": '
    '[{"day": "Mon", "focus": "Upper body", "rest": false, "minutes": 55, "exercises": '
    '[{"name": "Dumbbell bench press", "sets": 3, "reps": "8-10", "notes": "optional"}]}, '
    '... one entry for each of Mon..Sun in order; rest days have "rest": true and "exercises": []]}'
)
MEAL_SHAPE = (
    '{"notes": "one sentence", "days": [{"day": "Mon", "meals": [{"type": "breakfast", '
    '"name": "Veg poha with curd", "items": "1.5 cups poha, 1 cup curd, peanuts", '
    '"calories": 450, "protein_g": 18}, ... breakfast, lunch, dinner and optional snacks]}, '
    '... one entry for each of Mon..Sun in order]}'
)


def _profile_lines(i: PlanInputs) -> str:
    lines = [
        f"- {i.sex}, {i.age} years, {i.height_cm} cm, {i.weight_kg:.1f} kg, goal: {i.mode}",
        f"- training experience: {i.experience}; equipment: {i.equipment}",
        f"- diet: {i.diet_pref.replace('_', ' ')}",
    ]
    if i.muscle_level:
        lines.append(f"- current muscle level: {i.muscle_level}")
    if i.preferred_workouts:
        lines.append(f"- likes: {i.preferred_workouts}")
    if i.cuisine:
        lines.append(f"- cuisine: {i.cuisine}")
    if i.allergies and allergen_tokens(i.allergies):
        lines.append(f"- NEVER include: {i.allergies}")
    if i.movement_notes:
        lines.append("- health limits (keep training gentle and safe): " + ", ".join(i.movement_notes))
    return "\n".join(lines)


def workout_messages(i: PlanInputs, errors: list[str] | None = None) -> list[Message]:
    system = (
        "You are a careful strength and conditioning coach writing a 7-day workout plan. "
        "You are not a doctor. Reply with ONLY one JSON object, no prose and no code fences."
    )
    user = (
        f"Person:\n{_profile_lines(i)}\n\n"
        f"Write a plan with EXACTLY {i.days_per_week} training days and "
        f"{7 - i.days_per_week} rest days, spread sensibly through Mon-Sun. Sessions should "
        f"take about {i.session_minutes} minutes. Use 3 to 10 exercises per training day, suited "
        f"to a {i.experience}. Use only the listed equipment ({i.equipment}). Include a warm-up "
        f"as the first exercise. Keep it simple and sustainable.\n\nJSON shape:\n{WORKOUT_SHAPE}"
    )
    return _with_errors(system, user, errors)


def meal_messages(i: PlanInputs, errors: list[str] | None = None) -> list[Message]:
    system = (
        "You are a careful nutrition coach writing a 7-day meal plan. You are not a doctor. "
        "Reply with ONLY one JSON object, no prose and no code fences."
    )
    user = (
        f"Person:\n{_profile_lines(i)}\n\n"
        f"Daily targets: about {i.calories} kcal (each day within 5% of that), at least "
        f"{i.protein_g} g protein, around {i.carbs_g} g carbs and {i.fat_g} g fat. Every day has "
        f"breakfast, lunch and dinner, plus snacks if needed to reach the target. Use everyday, "
        f"easy-to-cook foods and vary the days. Give honest calorie and protein numbers for each "
        f"meal; they are added up and checked.\n\nJSON shape:\n{MEAL_SHAPE}"
    )
    return _with_errors(system, user, errors)


def _with_errors(system: str, user: str, errors: list[str] | None) -> list[Message]:
    if errors:
        user += (
            "\n\nYour previous answer was REJECTED for these reasons. Fix every one and return "
            "the complete JSON again:\n- " + "\n- ".join(errors[:12])
        )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def parse_json(text: str | None) -> dict[str, Any]:
    if not text:
        raise ValueError("empty answer")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


# ----------------------------------------------------------------- generator


class PlanGenerator:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def workout(self, inputs: PlanInputs) -> dict[str, Any]:
        return await self._generate(
            lambda errs: workout_messages(inputs, errs),
            WorkoutPlan,
            lambda plan: validate_workout(plan, inputs),
            "workout",
        )

    async def meals(self, inputs: PlanInputs) -> dict[str, Any]:
        return await self._generate(
            lambda errs: meal_messages(inputs, errs),
            MealPlan,
            lambda plan: validate_meals(plan, inputs),
            "meal",
        )

    async def both(self, inputs: PlanInputs) -> tuple[dict[str, Any], dict[str, Any]]:
        workout, meals = await asyncio.gather(self.workout(inputs), self.meals(inputs))
        return workout, meals

    async def _generate(self, build: Any, model: type[BaseModel], check: Any, label: str) -> dict[str, Any]:
        errors: list[str] | None = None
        for _attempt in range(MAX_ATTEMPTS):
            try:
                response = await self._provider.chat(
                    build(errors), temperature=0.4, max_tokens=4500
                )
            except LLMError as exc:
                raise PlanError("The assistant could not build the plan right now. Try again in a minute.") from exc
            try:
                plan = model.model_validate(parse_json(response.content))
            except (ValueError, ValidationError) as exc:
                errors = [f"the JSON was not in the required shape: {str(exc)[:300]}"]
                continue
            errors = check(plan)
            if not errors:
                return plan.model_dump()
        raise PlanError(
            f"I could not produce a {label} plan that fits your numbers and limits after "
            f"{MAX_ATTEMPTS} tries. Please try again."
        )