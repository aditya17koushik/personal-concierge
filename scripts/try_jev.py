#!/usr/bin/env python
"""Ask the REAL TypeSafe Jev API how it classifies some messages.

Useful for checking your API key and for tuning the thresholds in .env
(JEV_DOMAIN_THRESHOLD, JEV_HIGH_RISK_THRESHOLD, ...).

    python scripts/try_jev.py                       # built-in sample messages
    python scripts/try_jev.py "Move my 3pm meeting"  # your own messages

Run from the project root. Reads TYPESAFE_API_KEY etc. from .env.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.decisions.interpret import Thresholds, interpret  # noqa: E402
from app.decisions.policy import apply_policy  # noqa: E402
from app.decisions.questions import build_questions  # noqa: E402
from app.integrations.typesafe.client import TypeSafeClient, TypeSafeError  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402

SAMPLES = [
    "Hi, how are you today?",
    "I spent 250 on lunch",
    "I bought something",
    "How much did I spend this month?",
    "Any unread emails from my bank?",
    "What's on my calendar tomorrow?",
    "Any emails about my flight, and what's on my calendar Friday?",
    "Send an email to ravi@example.com saying I'll be late",
    "Delete all my expenses",
    "Cancel my 3pm meeting",
    "Explain how compound interest works",
]


async def main(messages: list[str]) -> int:
    settings = get_settings()
    if not settings.typesafe_api_key:
        print("TYPESAFE_API_KEY is not set in .env")
        return 2

    client = TypeSafeClient(
        api_key=settings.typesafe_api_key,
        base_url=settings.typesafe_base_url,
        model=settings.jev_model,
        timeout=settings.jev_timeout_seconds,
    )
    thresholds = Thresholds(
        domain=settings.jev_domain_threshold,
        high_risk=settings.jev_high_risk_threshold,
        refuse=settings.jev_refuse_threshold,
    )
    registry = build_default_registry()
    domains = sorted(registry.domains())
    questions = build_questions(domains)

    print(f"model={settings.jev_model}  thresholds={thresholds}\n")
    for message in messages:
        try:
            data = await client.evaluate(message, questions)
            decision = apply_policy(interpret(data["answers"], domains, thresholds), registry)
        except TypeSafeError as exc:
            print(f"{message!r}\n   ERROR: {exc}\n")
            return 1

        print(message)
        print(f"   action={decision.action}  risk={decision.risk}  intents={decision.intents}")
        print(f"   tools={decision.allowed_tools}")
        print(f"   scores={decision.scores}")
        print(f"   served by {data.get('model')}, usage={data.get('usage')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or SAMPLES)))