"""Deterministic rules applied to whatever Jev proposes. Jev proposes,
this code decides: it can only ever narrow what is allowed, never widen it."""

from app.decisions.schemas import RISK_ORDER, Decision, JevOutput
from app.tools.registry import ToolRegistry


def apply_policy(output: JevOutput, registry: ToolRegistry) -> Decision:
    known = registry.domains() | {"general"}
    intents = [i for i in dict.fromkeys(output.intents) if i in known] or ["general"]
    domains = {i for i in intents if i != "general"}

    candidates = [t for t in registry.tools if t.domain in domains]
    blocked = [t.name for t in candidates if t.requires_approval]
    allowed = [t.name for t in candidates if not t.requires_approval]

    action = output.action
    if action == "use_tools":
        if RISK_ORDER[output.risk] >= RISK_ORDER["high"]:
            action = "needs_approval"
        elif not allowed and blocked:
            action = "needs_approval"
        elif not allowed:
            action = "respond"  # nothing to run

    if action != "use_tools":
        allowed = []  # tools are exposed only for use_tools
    return Decision(
        intents=intents,
        action=action,
        risk=output.risk,
        reason=output.reason,
        allowed_tools=allowed,
        blocked_tools=blocked,
        scores=output.scores,
    )


def fallback_decision(registry: ToolRegistry, reason: str) -> Decision:
    """Used when Jev itself fails: plain chat plus read-only tools only."""
    read_only = [t for t in registry.tools if not t.side_effects and not t.requires_approval]
    return Decision(
        intents=sorted({t.domain for t in read_only}) or ["general"],
        action="use_tools" if read_only else "respond",
        risk="medium",
        reason=reason,
        allowed_tools=[t.name for t in read_only],
        blocked_tools=[t.name for t in registry.tools if t not in read_only],
        fallback=True,
    )