"""Turns Jev's calibrated probabilities into a proposal. Thresholds live in
config, so tuning behaviour never means editing a prompt."""

from dataclasses import dataclass
from typing import Any

from app.decisions.questions import RISK_LEVELS
from app.decisions.schemas import JevOutput


class JevResponseError(ValueError):
    """Jev's response did not have the expected shape."""


@dataclass(frozen=True)
class Thresholds:
    # Defaults tuned on real Jev scores: reads scored P(high)<=0.06, writes >=0.45;
    # true domain matches scored >=0.74, false ones <=0.49.
    domain: float = 0.6
    high_risk: float = 0.3
    refuse: float = 0.8


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevResponseError("probability is not a number")
    if not 0.0 <= value <= 1.0:
        raise JevResponseError("probability out of range")
    return float(value)


def _noul(answers: dict[str, Any], key: str) -> float:
    answer = answers[key]
    if not isinstance(answer, dict) or answer.get("type") != "noul":
        raise JevResponseError(f"'{key}' is not a noul answer")
    return _probability(answer.get("noul"))


def _risk(answers: dict[str, Any], thresholds: Thresholds) -> tuple[str, float]:
    answer = answers["risk"]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevResponseError("'risk' is not a choice answer")

    choice = answer.get("choice")
    if choice not in RISK_LEVELS:
        raise JevResponseError("'risk' has an unknown choice")

    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevResponseError("'risk' has no probabilities")
    p_high = _probability(probabilities.get("high", 0.0))

    # Err on the side of caution: a meaningful chance of "high" counts as high,
    # even when another level has the single highest probability.
    if choice == "high" or p_high >= thresholds.high_risk:
        return "high", p_high
    return str(choice), p_high


def interpret(
    answers: dict[str, Any], domains: list[str], thresholds: Thresholds
) -> JevOutput:
    scores: dict[str, float] = {}

    needed: list[str] = []
    for domain in sorted(domains):
        p = _noul(answers, f"needs_{domain}")
        scores[f"needs_{domain}"] = round(p, 3)
        if p >= thresholds.domain:
            needed.append(domain)

    risk, p_high = _risk(answers, thresholds)
    harmful = _noul(answers, "harmful")
    scores.update(risk_high=round(p_high, 3), harmful=round(harmful, 3))

    if harmful >= thresholds.refuse:
        action = "refuse"
    elif risk == "high":
        # Risk is judged on its own, not only when a domain matched: a request
        # like "send an email" can be high risk even if no read domain fits.
        action = "needs_approval"
    elif needed:
        action = "use_tools"
    else:
        action = "respond"

    reason = (
        "needs=" + (",".join(needed) or "none")
        + f" risk={risk}(p_high={p_high:.2f}) harmful={harmful:.2f}"
    )

    return JevOutput(
        intents=needed or ["general"],
        action=action,  # type: ignore[arg-type]
        risk=risk,  # type: ignore[arg-type]
        reason=reason,
        scores=scores,
    )