from abc import ABC, abstractmethod
from typing import Any

from app.schemas.llm import LLMResponse, Message


class LLMError(Exception):
    """Raised for any provider failure so callers don't depend on SDK errors."""


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a conversation and return a normalized response.

        `tools` uses the OpenAI function-calling schema:
        [{"type": "function", "function": {"name", "description", "parameters"}}]
        """