from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PendingApprovalOut(BaseModel):
    """What the chat response tells the client to ask the user about."""

    id: str
    tool: str
    summary: str
    expires_at: datetime


class ApprovalOut(BaseModel):
    id: str
    tool: str
    summary: str
    status: str
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    result: dict[str, Any] | None = None