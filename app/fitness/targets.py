"""BMR, daily calories, macros and the health screen. Pure functions: no database, no model.

The numbers a plan is built on are decided HERE. The model never chooses a calorie target.
Formulas: Mifflin-St Jeor for BMR; standard activity multipliers for TDEE. This is general
guidance, not medical advice, and the screen sends higher-risk people to a professional.
"""

import re
from dataclasses import dataclass, field

ACTIVITY_FACTORS = {
    "sedentary": 1.2,  # desk job, little exercise
    "light": 1.375,  # light exercise 1-3 days a week
    "moderate": 1.55,  # moderate exercise 3-5 days a week
    "very_active": 1.725,  # hard exercise 6-7 days a week or a physical job
}
SEXES = ("male", "female", "other")
EXPERIENCE = ("beginner", "intermediate", "advanced")
MUSCLE_LEVELS = ("low", "average", "high")
EQUIPMENT = ("gym", "home_dumbbells", "home_minimal", "none")
DIET_PREFS = ("vegetarian", "eggetarian", "non_vegetarian", "vegan")

KCAL_PER_KG_FAT = 7700
FLOOR_MALE = 1500
FLOOR_OTHER = 1200
MIN_AGE = 18

_REFUSE = ("eating disorder", "anorexi", "bulimi", "binge", "pregnan")
_CAREFUL_DIET = (
    "diabet", "kidney", "liver", "thyroid", "pcos", "cancer", "gout",
    "breastfeed", "lactat", "heart", "cardiac", "hypertension", "blood pressure",
)  # fmt: skip
_MOVEMENT = (
    "injur", "surgery", "back pain", "slip disc", "sciatica", "knee", "shoulder",
    "hernia", "arthritis", "osteopor", "asthma", "epilep", "heart", "cardiac",
    "hypertension", "blood pressure", "dizz", "fainting",
)  # fmt: skip
_NONE_WORDS = re.compile(r"^\s*(none|no|nil|nothing|n/?a|na|healthy|no issues|none at all|-)\s*\.?\s*$", re.I)


@dataclass(frozen=True)
class Screen:
    status: str  # ok | conservative | refuse
    reasons: list[str] = field(default_factory=list)
    clearance: bool = False  # suggest a doctor's OK before starting
    movement_notes: list[str] = field(default_factory=list)  # things the workout plan must respect
    mode_override: str | None = None


@dataclass(frozen=True)
class Targets:
    bmr: int
    tdee: int
    calories: int
    protein_g: int
    fat_g: int
    carbs_g: int
    mode: str
    change_per_week_kg: float  # expected, from the calorie gap
    notes: list[str] = field(default_factory=list)


def bmr_mifflin(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    if sex == "male":
        return base + 5
    if sex == "female":
        return base - 161
    return base - 78  # 'other': the midpoint of the two formulas


def calorie_floor(sex: str, bmr: float) -> float:
    """The lowest daily calories we will ever plan for a cut."""
    return max(FLOOR_MALE if sex == "male" else FLOOR_OTHER, 0.9 * bmr)


def bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def _round(value: float, step: int) -> int:
    return int(round(value / step) * step)


def screen(age: int, conditions: str | None, weight_kg: float, height_cm: float, mode: str) -> Screen:
    """Decide how careful to be. Conservative on purpose: a false alarm costs one sentence."""
    text = (conditions or "").lower()
    reasons: list[str] = []
    if age < MIN_AGE:
        return Screen(
            "refuse",
            ["Diet and training plans for under-18s need a doctor or a qualified coach."],
            clearance=True,
        )
    if any(word in text for word in _REFUSE):
        return Screen(
            "refuse",
            ["This needs a doctor or dietitian, not an app. I will not set calorie targets."],
            clearance=True,
        )

    status = "ok"
    clearance = False
    override: str | None = None
    notes: list[str] = []
    if text and not _NONE_WORDS.match(text):
        if any(word in text for word in _CAREFUL_DIET):
            status = "conservative"
            clearance = True
            reasons.append(
                "A health condition you mentioned can change what is safe to eat, so I will keep "
                "calories near maintenance. Please check the plan with your doctor or dietitian."
            )
        for word in _MOVEMENT:
            if word in text:
                notes.append(word)
        if notes:
            clearance = True
            reasons.append(
                "Because of what you mentioned (" + ", ".join(sorted(set(notes))) + "), the "
                "training will be low-impact, and a doctor's OK before starting is wise."
            )
    if age >= 65:
        status = "conservative"
        clearance = True
        reasons.append("At 65+ I keep plans gentle. Please get a doctor's OK first.")
    body_mass_index = bmi(weight_kg, height_cm)
    if body_mass_index < 18.5 and mode == "cut":
        status = "conservative"
        override = "maintain"
        reasons.append(
            f"Your BMI is {body_mass_index:.1f}, which is underweight, so a cut is not "
            "appropriate. I will plan for maintenance."
        )
    if body_mass_index >= 40:
        status = "conservative"
        clearance = True
        reasons.append("At this weight, low-impact training and a doctor's OK are the safe start.")
        notes.append("low-impact only")
    return Screen(status, reasons, clearance, sorted(set(notes)), override)


def compute_targets(
    *,
    sex: str,
    age: int,
    height_cm: float,
    weight_kg: float,
    activity: str,
    experience: str,
    mode: str,
    conservative: bool = False,
    calorie_adjust: int = 0,
) -> Targets:
    """`calorie_adjust` is the plan-adaptation nudge (kcal). It can never push a cut above
    maintenance or below the safe floor, or a bulk below maintenance."""
    notes: list[str] = []
    base = bmr_mifflin(sex, weight_kg, height_cm, age)
    maintenance = base * ACTIVITY_FACTORS[activity]

    if mode == "cut":
        deficit = min(0.20 * maintenance, weight_kg * 0.0075 * KCAL_PER_KG_FAT / 7)
        if conservative:
            deficit = min(deficit, 0.10 * maintenance)
        calories = maintenance - deficit
        floor = calorie_floor(sex, base)
        if calories < floor:
            calories = min(floor, maintenance)
            notes.append("The deficit is limited by a safe minimum, so progress will be slower.")
    elif mode == "bulk":
        surplus = min(max(0.10 * maintenance, 200), 450)
        if conservative:
            surplus = min(surplus, 0.05 * maintenance)
        calories = maintenance + surplus
    else:
        calories = maintenance

    if calorie_adjust:
        calories += calorie_adjust
        floor = calorie_floor(sex, base)
        if mode == "cut":
            lowest = max(floor, 0.9 * maintenance if conservative else 0)
            calories = min(max(calories, lowest), maintenance)
        elif mode == "bulk":
            calories = max(calories, maintenance)
        else:
            calories = min(max(calories, maintenance - 300), maintenance + 300)
    calories_i = _round(calories, 25)

    height_m = height_cm / 100
    basis = min(weight_kg, 27 * height_m**2)  # protein is set from a lean-ish weight
    if experience == "beginner":
        per_kg = 1.6
    else:
        per_kg = {"cut": 2.0, "bulk": 1.8}.get(mode, 1.6)
    protein = min(_round(per_kg * basis, 5), int(0.35 * calories_i / 4))
    fat = max(_round(0.8 * basis, 5), _round(0.25 * calories_i / 9, 5))
    carbs = max(_round((calories_i - 4 * protein - 9 * fat) / 4, 5), 50)

    return Targets(
        bmr=int(round(base)),
        tdee=int(round(maintenance)),
        calories=calories_i,
        protein_g=protein,
        fat_g=fat,
        carbs_g=carbs,
        mode=mode,
        change_per_week_kg=round((calories_i - maintenance) * 7 / KCAL_PER_KG_FAT, 2),
        notes=notes,
    )