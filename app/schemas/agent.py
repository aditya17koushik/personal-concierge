from pydantic import BaseModel, Field, field_validator

from app.decisions.schemas import Decision


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=255)
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ChatResponse(BaseModel):
    reply: str
    provider: str
    model: str
    tools_used: list[str] = Field(default_factory=list)
    decision: Decision | None = None