from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.decisions.schemas import Risk

if TYPE_CHECKING:
    from app.integrations.google.oauth import GoogleOAuthService


@dataclass
class ToolContext:
    """Per-request information handed to every tool handler."""

    db: Session
    user_id: int  # internal users.id, not the external id
    today: date
    default_currency: str
    external_user_id: str = ""
    google_oauth: "GoogleOAuthService | None" = None


ToolResult = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    # handlers may be plain functions or coroutines
    handler: Callable[[ToolContext, Any], ToolResult | Awaitable[ToolResult]]

    # Used by the Jev decision layer
    domain: str  # "expense", "email", "calendar", ...
    risk: Risk = "low"
    side_effects: bool = False  # changes data or the outside world

    @property
    def requires_approval(self) -> bool:
        """Such tools are never exposed to the model without an approval flow."""
        return self.risk == "high" or (self.side_effects and self.risk == "medium")

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }