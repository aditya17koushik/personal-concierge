import json
from types import SimpleNamespace
from typing import Any, cast
import pytest

from app.llm.base import LLMError
from app.llm.openai_provider import OpenAIProvider, message_to_openai
from app.schemas.llm import Message, ToolCall


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def make_provider(fake: FakeCompletions) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test", model="gpt-4o-mini")
    provider._client = cast(
        Any,
        SimpleNamespace(chat=SimpleNamespace(completions=fake)),
    )
    return provider


def make_response(content=None, tool_calls=None):
    return SimpleNamespace(
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


@pytest.mark.asyncio
async def test_plain_chat():
    fake = FakeCompletions(make_response(content="hello"))
    provider = make_provider(fake)

    result = await provider.chat([Message(role="user", content="hi")])

    assert result.content == "hello"
    assert result.tool_calls == []
    assert result.usage is not None
    assert result.usage.total_tokens == 3
    assert fake.kwargs["model"] == "gpt-4o-mini"
    assert "tools" not in fake.kwargs


@pytest.mark.asyncio
async def test_tool_call_parsing():
    raw = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="add_expense", arguments='{"amount": 12.5}'),
    )
    fake = FakeCompletions(make_response(tool_calls=[raw]))
    provider = make_provider(fake)

    result = await provider.chat(
        [Message(role="user", content="spent 12.5")],
        tools=[{"type": "function", "function": {"name": "add_expense"}}],
    )

    assert result.tool_calls[0].name == "add_expense"
    assert result.tool_calls[0].arguments == {"amount": 12.5}
    assert "tools" in fake.kwargs


@pytest.mark.asyncio
async def test_invalid_tool_arguments_do_not_crash():
    raw = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="x", arguments="{not json")
    )
    provider = make_provider(FakeCompletions(make_response(tool_calls=[raw])))

    result = await provider.chat([Message(role="user", content="hi")])

    assert result.tool_calls[0].arguments == {}


@pytest.mark.asyncio
async def test_provider_error_is_wrapped():
    from openai import OpenAIError

    provider = make_provider(FakeCompletions(error=OpenAIError("boom")))

    with pytest.raises(LLMError):
        await provider.chat([Message(role="user", content="hi")])


def test_message_conversion_with_tool_calls():
    msg = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="f", arguments={"a": 1})],
    )
    data = message_to_openai(msg)

    assert data["tool_calls"][0]["function"]["arguments"] == json.dumps({"a": 1})

    tool_msg = message_to_openai(Message(role="tool", content="ok", tool_call_id="c1"))
    assert tool_msg["tool_call_id"] == "c1"