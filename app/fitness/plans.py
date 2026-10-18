"""Generating a 7-day workout plan and a 7-day meal plan.

The model writes the exercises and foods. CODE decides everything that matters for safety and
fit: the calorie and protein targets are inputs, and every plan is validated against them, against
the training days, the equipment, injuries, diet type and allergies. A plan that breaks a rule is
rejected and regenerated with the errors fed back. Rules split in two: HARD rules (diet, allergies,
equipment, injuries, the week's shape, wildly wrong numbers) are never broken; SOFT rules (calories a
little off, protein a little low, long sessions) are tolerated after three tries, and the plan says so.
So the user gets a safe plan instead of an error whenever one is possible.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.observability.tracing import llm_task
from app.llm.base import LLMError, LLMProvider
from app.schemas.llm import Message

logger = logging.getLogger(__name__)

DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
DayName = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_ATTEMPTS = 3  # tries before a plan that only breaks soft rules is accepted
MAX_HARD_ATTEMPTS = 5  # extra tries while a hard rule is still broken
CALORIE_TOLERANCE = 0.10  # each day within +-10% of the target
PROTEIN_MIN_RATIO = 0.80
CALORIE_HARD_TOLERANCE = 0.25  # beyond this a day is not usable at all
PROTEIN_HARD_RATIO = 0.60


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
FISH = ("fish", "prawn", "shrimp", "salmon", "tuna", "crab")
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
    found: list[str] = []
    if diet != "non_vegetarian":  # non_vegetarian may eat anything; the rest never eat meat
        allowed = FISH if diet == "pescatarian" else ()  # pescatarians eat fish and seafood
        found += [w for w in MEAT if w not in allowed and _whole(t, w)]
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


def workout_issues(plan: WorkoutPlan, inputs: PlanInputs) -> tuple[list[str], list[str]]:
    """(hard, soft) problems. Hard: the week's shape, equipment and injury limits."""
    hard: list[str] = []
    soft: list[str] = []
    if [d.day for d in plan.days] != list(DAYS):
        hard.append("days must be exactly Mon, Tue, Wed, Thu, Fri, Sat, Sun in that order")
    training = [d for d in plan.days if not d.rest]
    if len(training) != inputs.days_per_week:
        hard.append(
            f"there must be exactly {inputs.days_per_week} training days; you gave {len(training)}"
        )
    bans = list(EQUIPMENT_BANS.get(inputs.equipment, ()))
    for note in inputs.movement_notes:
        bans += list(MOVEMENT_BANS.get(note, ()))
    cap = inputs.session_minutes + 15
    for d in plan.days:
        if d.rest and d.exercises:
            hard.append(f"{d.day} is a rest day and must have no exercises")
        if not d.rest and not (3 <= len(d.exercises) <= 10):
            soft.append(f"{d.day} needs between 3 and 10 exercises")
        if not d.rest and d.minutes is not None and d.minutes > cap:
            soft.append(f"{d.day} is {d.minutes} minutes; the limit is {cap}")
        for ex in d.exercises:
            text = f"{ex.name} {ex.notes or ''}".lower()
            for word in bans:
                if _has(text, word):
                    hard.append(f"{d.day} '{ex.name}' is not allowed (equipment or injury: '{word}')")
                    break
    return hard, soft


def validate_workout(plan: WorkoutPlan, inputs: PlanInputs) -> list[str]:
    hard, soft = workout_issues(plan, inputs)
    return hard + soft


def hard_workout_errors(plan: WorkoutPlan, inputs: PlanInputs) -> list[str]:
    return workout_issues(plan, inputs)[0]


def meal_day_issues(d: MealDay, inputs: PlanInputs) -> tuple[list[str], list[str]]:
    """(hard, soft) problems of one day. Hard: diet, allergies, missing meals, numbers far off."""
    hard: list[str] = []
    soft: list[str] = []
    low = inputs.calories * (1 - CALORIE_TOLERANCE)
    high = inputs.calories * (1 + CALORIE_TOLERANCE)
    total = sum(m.calories for m in d.meals)
    protein = sum(m.protein_g for m in d.meals)
    if not {"breakfast", "lunch", "dinner"} <= {m.type for m in d.meals}:
        hard.append(f"{d.day} must include breakfast, lunch and dinner")
    if not (low <= total <= high):
        msg = (
            f"{d.day} totals {total} kcal; it must be between {round(low)} and {round(high)} "
            f"(target {inputs.calories})"
        )
        far = abs(total - inputs.calories) > inputs.calories * CALORIE_HARD_TOLERANCE
        (hard if far else soft).append(msg)
    if protein < inputs.protein_g * PROTEIN_MIN_RATIO:
        msg = (
            f"{d.day} has {round(protein)} g protein; it needs at least "
            f"{round(inputs.protein_g * PROTEIN_MIN_RATIO)} g (target {inputs.protein_g})"
        )
        (hard if protein < inputs.protein_g * PROTEIN_HARD_RATIO else soft).append(msg)
    allergens = allergen_tokens(inputs.allergies)
    for m in d.meals:
        text = f"{m.name} {m.items}".lower()
        bad = diet_violations(text, inputs.diet_pref)
        if bad:
            hard.append(f"{d.day} {m.type} '{m.name}' breaks the {inputs.diet_pref} diet: {', '.join(bad)}")
        for token in allergens:
            if _has(text, token) or _has(text, token.rstrip("s")):
                hard.append(f"{d.day} {m.type} '{m.name}' contains '{token}', which the user avoids")
    return hard, soft


def validate_meal_day(d: MealDay, inputs: PlanInputs) -> list[str]:
    hard, soft = meal_day_issues(d, inputs)
    return hard + soft


def hard_meal_errors(plan: MealPlan, inputs: PlanInputs) -> list[str]:
    errors: list[str] = []
    if [d.day for d in plan.days] != list(DAYS):
        errors.append("days must be exactly Mon, Tue, Wed, Thu, Fri, Sat, Sun in that order")
    for d in plan.days:
        errors += meal_day_issues(d, inputs)[0]
    return errors


def hard_failing_days(plan: MealPlan, inputs: PlanInputs) -> list[str]:
    return [d.day for d in plan.days if meal_day_issues(d, inputs)[0]]


def tolerance_note(plan: MealPlan, inputs: PlanInputs) -> str:
    """One honest sentence about days that are a bit off the targets (empty when none)."""
    off = [d for d in plan.days if meal_day_issues(d, inputs)[1]]
    if not off:
        return ""
    worst = max(abs(sum(m.calories for m in d.meals) - inputs.calories) for d in off)
    return (
        f" Note: {len(off)} day(s) are slightly off your targets (calories by up to {round(worst)} kcal, "
        f"or protein a little low). Ask me to adjust it if you like."
    )


def validate_meals(plan: MealPlan, inputs: PlanInputs) -> list[str]:
    errors: list[str] = []
    if [d.day for d in plan.days] != list(DAYS):
        errors.append("days must be exactly Mon, Tue, Wed, Thu, Fri, Sat, Sun in that order")
    for d in plan.days:
        errors += validate_meal_day(d, inputs)
    return errors


def failing_days(plan: MealPlan, inputs: PlanInputs) -> list[str]:
    """The days (in order) that break a rule."""
    return [d.day for d in plan.days if validate_meal_day(d, inputs)]


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


def _budget(i: PlanInputs) -> str:
    def part(share: float) -> str:
        return f"~{round(i.calories * share / 5) * 5} kcal and ~{round(i.protein_g * share)} g protein"

    return (
        f"breakfast {part(0.25)}; lunch {part(0.32)}; dinner {part(0.30)}; "
        f"snack {part(0.13)}."
    )


def meal_fix_messages(i: PlanInputs, days: list[str], errors: list[str]) -> list[Message]:
    system = (
        "You are a careful nutrition coach fixing days of a meal plan. You are not a doctor. "
        "Reply with ONLY one JSON object, no prose and no code fences."
    )
    user = (
        f"Person:\n{_profile_lines(i)}\n\n"
        f"Rewrite ONLY these days: {', '.join(days)}. Each day must total about {i.calories} kcal "
        f"(never outside {round(i.calories * 0.93)}-{round(i.calories * 1.07)}) and at least "
        f"{i.protein_g} g protein, with breakfast, lunch and dinner. Budget: {_budget(i)}\n"
        "Add up your meal numbers before answering. They were rejected for:\n- "
        + "\n- ".join(errors[:12])
        + '\n\nJSON shape: {"days": [{"day": "Mon", "meals": [{"type": "breakfast", "name": "...", '
        '"items": "...", "calories": 600, "protein_g": 40}, ...]}]}'
    )
    return [Message(role="system", content=system), Message(role="user", content=user)]


def meal_messages(i: PlanInputs, errors: list[str] | None = None) -> list[Message]:
    system = (
        "You are a careful nutrition coach writing a 7-day meal plan. You are not a doctor. "
        "Reply with ONLY one JSON object, no prose and no code fences."
    )
    user = (
        f"Person:\n{_profile_lines(i)}\n\n"
        f"Daily targets: about {i.calories} kcal (each day within 5% of that), at least "
        f"{i.protein_g} g protein, around {i.carbs_g} g carbs and {i.fat_g} g fat. Every day has "
        f"breakfast, lunch and dinner, plus snacks if needed to reach the target.\n"
        f"Budget per day so the totals work: {_budget(i)}\n"
        f"Use everyday, easy-to-cook foods and vary the days. Give honest calorie and protein "
        f"numbers for each meal; they are ADDED UP and checked, so make each day's meals sum to "
        f"the targets. Choose high-protein foods (eggs, chicken, fish, paneer, dal, curd, tofu, "
        f"whey) to reach the protein.\n\nJSON shape:\n{MEAL_SHAPE}"
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
            lambda plan: hard_workout_errors(plan, inputs),
        )

    async def meals(self, inputs: PlanInputs) -> dict[str, Any]:
        """Generate all 7 days once; then repair only the days that break a rule (smaller, faster)."""
        plan: MealPlan | None = None
        errors: list[str] | None = None
        bad_days: list[str] = []
        for attempt in range(1, MAX_HARD_ATTEMPTS + 1):
            messages = (
                meal_messages(inputs, errors)
                if plan is None
                else meal_fix_messages(inputs, bad_days, errors or [])
            )
            try:
                with llm_task("plan.meals"):
                    response = await self._provider.chat(messages, temperature=0.4, max_tokens=4500)
            except LLMError as exc:
                raise PlanError("The assistant could not build the plan right now. Try again in a minute.") from exc
            try:
                data = parse_json(response.content)
                if plan is None:
                    plan = MealPlan.model_validate(data)
                else:
                    plan = MealPlan(notes=plan.notes, days=_merge_days(plan, data))
            except (ValueError, ValidationError, KeyError) as exc:
                errors = [f"the JSON was not in the required shape: {str(exc)[:300]}"]
                logger.info("meal plan attempt %s: bad shape: %s", attempt, errors[0])
                continue
            errors = validate_meals(plan, inputs)
            if not errors:
                return plan.model_dump()
            hard = hard_meal_errors(plan, inputs)
            if attempt >= MAX_ATTEMPTS and not hard:
                # Only soft problems (a few days a little off): a usable plan beats an error.
                logger.info("meal plan accepted after %s attempts with soft issues: %s", attempt, "; ".join(errors[:3]))
                plan = MealPlan(notes=(plan.notes + tolerance_note(plan, inputs))[:300], days=plan.days)
                return plan.model_dump()
            if attempt >= MAX_ATTEMPTS:
                errors = hard  # keep fixing only what must not be broken
                bad_days = hard_failing_days(plan, inputs) or list(DAYS)
            else:
                bad_days = failing_days(plan, inputs) or list(DAYS)
            logger.info("meal plan attempt %s rejected (%s): %s", attempt, ",".join(bad_days), "; ".join(errors[:4]))
        raise PlanError(
            "I could not build a meal plan that respects your diet, allergies and numbers. "
            "Tell me a few foods you like and eat regularly and I will try again."
        )

    async def both(self, inputs: PlanInputs) -> tuple[dict[str, Any], dict[str, Any]]:
        workout, meals = await asyncio.gather(self.workout(inputs), self.meals(inputs))
        return workout, meals

    async def _generate(
        self, build: Any, model: type[BaseModel], check: Any, label: str, hard: Any = None
    ) -> dict[str, Any]:
        errors: list[str] | None = None
        for attempt in range(1, MAX_HARD_ATTEMPTS + 1):
            try:
                with llm_task(f"plan.{label}"):
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
            hard_errors = hard(plan) if hard else errors
            logger.info("%s plan attempt %s rejected: %s", label, attempt, "; ".join(errors[:4]))
            if attempt >= MAX_ATTEMPTS and not hard_errors:
                out = plan.model_dump()  # only soft problems (e.g. a session a bit long): accept it
                if "summary" in out:
                    out["summary"] = (str(out["summary"]) + " (Some sessions may run a little long.)")[:300]
                return out
            if attempt >= MAX_ATTEMPTS:
                errors = hard_errors
        raise PlanError(
            f"I could not build a {label} plan that respects your equipment and health limits. "
            "Tell me what you can use and what to avoid, and I will try again."
        )


def _merge_days(plan: MealPlan, data: dict[str, Any]) -> list[MealDay]:
    """The current plan with the days in `data["days"]` replaced. Unknown days are ignored."""
    fixed = {d["day"]: MealDay.model_validate(d) for d in data.get("days", []) if d.get("day") in DAYS}
    if not fixed:
        raise ValueError("no usable days in the answer")
    return [fixed.get(d.day, d) for d in plan.days]