"""Turn whatever the user typed into one of the allowed intake values.

The model usually maps words itself, but it can be wrong or give up ("mix of veg and non-veg",
"dumbbells at home", "desk job"). This is the safety net: common phrasings are mapped by code, close
spellings are corrected, and only a truly unknown answer is rejected (with the list of options).
"""

import difflib
import re

ALIASES: dict[str, dict[str, str]] = {
    "diet_pref": {
        "veg": "vegetarian", "pure veg": "vegetarian", "vegetarian": "vegetarian", "jain": "vegetarian",
        "lacto vegetarian": "vegetarian", "lacto-vegetarian": "vegetarian", "no meat": "vegetarian",
        "egg": "eggetarian", "eggs": "eggetarian", "eggetarian": "eggetarian", "ovo vegetarian": "eggetarian",
        "veg with eggs": "eggetarian", "veg and eggs": "eggetarian", "vegetarian with eggs": "eggetarian",
        "non veg": "non_vegetarian", "non-veg": "non_vegetarian", "nonveg": "non_vegetarian",
        "non vegetarian": "non_vegetarian", "non-vegetarian": "non_vegetarian", "meat": "non_vegetarian",
        "omnivore": "non_vegetarian", "everything": "non_vegetarian", "anything": "non_vegetarian",
        "none": "non_vegetarian", "no restrictions": "non_vegetarian", "no restriction": "non_vegetarian",
        "halal": "non_vegetarian", "both": "non_vegetarian", "mix": "non_vegetarian",
        "mixed": "non_vegetarian", "veg and non veg": "non_vegetarian", "veg and non-veg": "non_vegetarian",
        "veg non veg mix": "non_vegetarian", "non vegetarian and vegetarian mix": "non_vegetarian",
        "pescatarian": "pescatarian", "pescetarian": "pescatarian", "fish": "pescatarian",
        "vegan": "vegan", "plant based": "vegan", "plant-based": "vegan",
    },
    "equipment": {
        "gym": "gym", "commercial gym": "gym", "fitness center": "gym", "fitness centre": "gym",
        "full gym": "gym", "weights": "gym", "barbell": "gym",
        "dumbbells": "home_dumbbells", "dumbbell": "home_dumbbells", "home dumbbells": "home_dumbbells",
        "home with dumbbells": "home_dumbbells", "dumbbells at home": "home_dumbbells",
        "home": "home_minimal", "at home": "home_minimal", "home workout": "home_minimal",
        "home minimal": "home_minimal", "bands": "home_minimal", "resistance bands": "home_minimal",
        "none": "none", "no equipment": "none", "bodyweight": "none", "body weight": "none", "calisthenics": "none",
    },
    "experience": {
        "beginner": "beginner", "new": "beginner", "novice": "beginner", "starting": "beginner",
        "just starting": "beginner", "never trained": "beginner", "less than a year": "beginner",
        "intermediate": "intermediate", "medium": "intermediate", "some experience": "intermediate",
        "1-3 years": "intermediate", "2 years": "intermediate", "3 years": "intermediate",
        "advanced": "advanced", "expert": "advanced", "experienced": "advanced", "pro": "advanced",
    },
    "activity_level": {
        "sedentary": "sedentary", "desk job": "sedentary", "desk": "sedentary", "little": "sedentary",
        "none": "sedentary", "inactive": "sedentary", "low": "sedentary",
        "light": "light", "lightly active": "light", "light exercise": "light",
        "moderate": "moderate", "moderately active": "moderate", "medium": "moderate", "average": "moderate",
        "very active": "very_active", "very_active": "very_active", "active": "very_active",
        "heavy": "very_active", "physical job": "very_active", "high": "very_active",
    },
    "sex": {
        "male": "male", "m": "male", "man": "male", "boy": "male",
        "female": "female", "f": "female", "woman": "female", "girl": "female",
        "other": "other", "non binary": "other", "non-binary": "other", "nonbinary": "other",
    },
    "muscle_level": {
        "low": "low", "little": "low", "skinny": "low", "average": "average", "medium": "average",
        "normal": "average", "high": "high", "muscular": "high", "a lot": "high",
    },
}


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]", " ", text.lower().replace("_", " "))).strip()


def normalize_choice(field: str, value: object, choices: tuple[str, ...]) -> str | None:
    """The allowed value for what the user said, or None when it cannot be mapped."""
    text = _key(str(value))
    if not text:
        return None
    for choice in choices:  # exact (also "home dumbbells" for home_dumbbells)
        if text == _key(choice):
            return choice
    table = {_key(k): v for k, v in ALIASES.get(field, {}).items()}
    if text in table and table[text] in choices:
        return table[text]
    # a phrase that CONTAINS a known alias ("i eat both veg and non veg"): longest alias wins
    for alias in sorted(table, key=len, reverse=True):
        if len(alias) >= 4 and re.search(rf"\b{re.escape(alias)}\b", text) and table[alias] in choices:
            return table[alias]
    close = difflib.get_close_matches(text, list(table), n=1, cutoff=0.8)
    if close and table[close[0]] in choices:
        return table[close[0]]
    return None