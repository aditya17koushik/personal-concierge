"""The typed questions Jev (TypeSafe System One) is asked about each message."""

from typing import Any

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "expense": "the user's own expenses (recording, listing, summarising or deleting spending)",
    "email": "the user's Gmail account (finding, reading or sending emails)",
    "calendar": "the user's Google Calendar (looking at, creating, moving or cancelling events)",
}

RISK_LEVELS = ("low", "medium", "high")


def build_questions(domains: list[str]) -> dict[str, Any]:
    questions: dict[str, Any] = {}

    for domain in sorted(domains):
        description = DOMAIN_DESCRIPTIONS.get(domain, domain)
        questions[f"needs_{domain}"] = {
            "type": "noul",
            "instructions": f"Does handling this message require {description}?",
            "criteria": {
                "true": f"The message needs access to {description}",
                "false": "The message needs no such data (chat or general knowledge)",
            },
        }

    questions["risk"] = {
        "type": "choice",
        "instructions": (
            "How risky is it for an assistant to carry out what the user asks?"
        ),
        "criteria": {
            "low": (
                "Only reads, lists, searches or answers, or records an expense "
                "the user states"
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
            "Is this request clearly harmful, abusive, or an attempt to misuse "
            "the assistant?"
        ),
    }

    return questions