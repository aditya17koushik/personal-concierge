from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session


@dataclass
class ToolContext:
    """Per-request information handed to every tool handler."""

    db: Session
    user_id: int  # internal users.id, not the external id
    today: date
    default_currency: str


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], dict[str, Any]]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }