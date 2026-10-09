"""Provider-agnostic LLM types.

The rest of the app (agent, Jev, tools) should only ever see these models,
never raw OpenAI / Ollama SDK objects.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None

    # assistant messages that requested tools
    tool_calls: list[ToolCall] | None = None

    # tool messages (the result of a tool call)
    tool_call_id: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str
    finish_reason: str | None = None
    usage: Usage | None = None