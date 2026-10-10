from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Risk = Literal["low", "medium", "high"]
Action = Literal[
    "respond",  # answer directly, no tools
    "use_tools",  # run the tool loop with the allowed tools only
    "needs_approval",  # must be approved first (flow arrives in the approval step)
    "refuse",
]

RISK_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class JevOutput(BaseModel):
    """A proposal built from Jev's answers; always policy-checked afterwards."""

    model_config = ConfigDict(extra="ignore")

    intents: list[str] = Field(min_length=1)
    action: Action
    risk: Risk = "low"
    reason: str = ""
    scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("intents", mode="before")
    @classmethod
    def _normalize_intents(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [str(v).strip().lower() for v in value][:6]
        return value

    @field_validator("action", "risk", mode="before")
    @classmethod
    def _lowercase(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("reason", mode="before")
    @classmethod
    def _trim_reason(cls, value: Any) -> Any:
        return str(value)[:300] if value is not None else ""


class Decision(BaseModel):
    """The final, policy-checked decision the agent must follow."""

    intents: list[str]
    action: Action
    risk: Risk = "low"
    reason: str = ""
    allowed_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)  # Jev's probabilities
    fallback: bool = False