import json
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.llm.base import LLMError, LLMProvider
from app.schemas.llm import LLMResponse, Message, ToolCall, Usage

logger = logging.getLogger(__name__)


def message_to_openai(message: Message) -> dict[str, Any]:
    """Convert our Message into the dict format the OpenAI-style API expects."""
    data: dict[str, Any] = {"role": message.role, "content": message.content}

    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in message.tool_calls
        ]

    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id

    return data


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Model returned invalid tool arguments: %r", raw)
        return {}


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [message_to_openai(m) for m in messages],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            logger.exception("OpenAI request failed")
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0]
        raw_calls = choice.message.tool_calls or []

        return LLMResponse(
            content=choice.message.content,
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_parse_arguments(tc.function.arguments),
                )
                for tc in raw_calls
            ],
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=(
                Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                if response.usage
                else None
            ),
        )