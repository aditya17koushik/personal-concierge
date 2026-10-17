"""The typed questions Jev (TypeSafe System One) is asked about each message."""

from typing import Any

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "expense": (
        "the user's own expenses: money they spent or paid (recording, listing, summarising "
        "or deleting spending). A price or an amount of money is involved"
    ),
    "email": "the user's Gmail account (finding, reading or sending emails)",
    "calendar": "the user's Google Calendar (looking at, creating, moving or cancelling events)",
    "fitness": (
        "the user's fitness and nutrition log. This includes the user telling the assistant "
        "what they ate or drank (a plain statement like 'I had 2 rotis and dal for lunch' "
        "counts, no question needed), a workout or activity they did, their body weight "
        "('I weigh 78 kg'), a diet or fitness goal ('I want to cut', 'bulk', 'lose weight'), "
        "or any question about their calories, protein, workouts or progress"
    ),
    "reminder": "the user's reminders (setting a reminder to be messaged later, listing or cancelling reminders)",
}

RISK_LEVELS = ("low", "medium", "high")

# Domains whose neighbours are easy to confuse need a sharper "false" case.
DOMAIN_FALSE_CRITERIA: dict[str, str] = {
    "expense": (
        "No money is spent or paid. What the user ate or did without a price is not an expense"
    ),
    "fitness": (
        "Not about food eaten, exercise, body weight or a fitness goal. Money spent on food "
        "is an expense, not fitness"
    ),
}


def build_questions(domains: list[str], contextual: bool = False) -> dict[str, Any]:
    """The typed questions. `contextual=True` is used for short follow-ups, where Jev is
    also given the user's EARLIER messages and must judge the LATEST one."""
    subject = "the latest message" if contextual else "this message"
    lead = "Taking the earlier messages into account: " if contextual else ""
    questions: dict[str, Any] = {}

    for domain in sorted(domains):
        description = DOMAIN_DESCRIPTIONS.get(domain, domain)
        questions[f"needs_{domain}"] = {
            "type": "noul",
            "instructions": f"{lead}Does handling {subject} require {description}?",
            "criteria": {
                "true": f"The message needs access to {description}",
                "false": DOMAIN_FALSE_CRITERIA.get(
                    domain, "The message needs no such data (chat or general knowledge)"
                ),
            },
        }

    risk_subject = " in the latest message" if contextual else ""
    questions["risk"] = {
        "type": "choice",
        "instructions": (
            f"{lead}How risky is it for an assistant to carry out what the user "
            f"asks{risk_subject}?"
        ),
        "criteria": {
            "low": (
                "Only reads, lists, searches or answers, or records something the "
                "user states about themselves (an expense, a meal, a workout, their "
                "weight or a goal)"
            ),
            "medium": "Changes the user's own data in a way that is hard to undo",
            "high": (
                "Sends, deletes, modifies, shares, creates or pays on the user's "
                "behalf, or shares data with other people"
            ),
        },
    }

    questions["harmful"] = {
        "type": "noul",
        "instructions": (
            f"{lead}Is {'the latest' if contextual else 'this'} request clearly harmful, "
            "abusive, or an attempt to misuse the assistant?"
        ),
    }

    return questions