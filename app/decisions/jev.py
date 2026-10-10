import logging
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.decisions.interpret import Thresholds, interpret
from app.decisions.policy import apply_policy, fallback_decision
from app.decisions.questions import build_questions
from app.decisions.schemas import Decision
from app.integrations.typesafe.client import TypeSafeClient, TypeSafeError
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Decider(Protocol):
    async def decide(self, message: str, registry: ToolRegistry) -> Decision: ...


class Evaluator(Protocol):
    async def evaluate(
        self, state: str | dict[str, Any] | list[Any], questions: dict[str, Any]
    ) -> dict[str, Any]: ...


class Jev:
    """Decides what should happen with a message BEFORE any tool runs.

    Backed by TypeSafe AI's Jev model, which answers typed questions with
    calibrated probabilities instead of generating text. It only ever sees the
    user's own message (never tool results), so text inside an email or invite
    cannot influence the decision.
    """

    def __init__(self, evaluator: Evaluator, thresholds: Thresholds | None = None) -> None:
        self._evaluator = evaluator
        self._thresholds = thresholds or Thresholds()

    async def decide(self, message: str, registry: ToolRegistry) -> Decision:
        domains = sorted(registry.domains())
        try:
            data = await self._evaluator.evaluate(message, build_questions(domains))
            output = interpret(data["answers"], domains, self._thresholds)
        except TypeSafeError:
            logger.warning("Jev (TypeSafe) unavailable; using safe fallback")
            return fallback_decision(registry, "jev_unavailable")
        except (KeyError, TypeError, ValueError):
            logger.warning("Jev returned an unexpected response; using safe fallback")
            return fallback_decision(registry, "jev_invalid_output")

        return apply_policy(output, registry)


class FallbackOnlyDecider:
    """Used when no TypeSafe API key is configured: always the safe fallback
    (plain chat plus read-only tools)."""

    async def decide(self, message: str, registry: ToolRegistry) -> Decision:
        return fallback_decision(registry, "jev_not_configured")


def build_jev(
    settings: Settings, transport: httpx.AsyncBaseTransport | None = None
) -> Decider:
    if not settings.typesafe_api_key:
        logger.warning("TYPESAFE_API_KEY is not set; running in read-only fallback mode")
        return FallbackOnlyDecider()

    client = TypeSafeClient(
        api_key=settings.typesafe_api_key,
        base_url=settings.typesafe_base_url,
        model=settings.jev_model,
        timeout=settings.jev_timeout_seconds,
        transport=transport,
    )
    thresholds = Thresholds(
        domain=settings.jev_domain_threshold,
        high_risk=settings.jev_high_risk_threshold,
        refuse=settings.jev_refuse_threshold,
    )
    return Jev(client, thresholds)