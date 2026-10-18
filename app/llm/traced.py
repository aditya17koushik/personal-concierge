"""Wraps any LLMProvider so every model call becomes a Langfuse generation: model, prompt, reply,
tool calls, token counts (Langfuse prices them from the model name) and latency. Images are never
sent: only a size."""

from typing import Any

from app.llm.base import LLMProvider
from app.observability import EXTERNAL_REDACTED, current_task, external_seen, get_tracer
from app.schemas.llm import LLMResponse, Message


def messages_for_trace(messages: list[Message], hide_external: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        hidden = hide_external and m.role in ("tool", "assistant")
        item: dict[str, Any] = {"role": m.role, "content": EXTERNAL_REDACTED if hidden else m.content}
        if m.tool_calls:
            item["tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]
        if m.images:
            item["images"] = [f"[image, {len(url)} chars of base64 not sent]" for url in m.images]
        out.append(item)
    return out


class TracedProvider(LLMProvider):
    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        self.name = inner.name
        self.model = inner.model

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        tracer = get_tracer()
        hide = external_seen()
        params = {"temperature": temperature, "max_tokens": max_tokens, "tools_offered": len(tools or [])}
        with tracer.generation(
            current_task(), model=self._inner.model, input=messages_for_trace(messages, hide), params=params
        ) as gen:
            response = await self._inner.chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
            usage = None
            if response.usage:
                usage = {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                }
            gen.update(
                output={
                    "content": EXTERNAL_REDACTED if hide else response.content,
                    "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in response.tool_calls],
                },
                model=response.model,
                usage=usage,
                metadata={"finish_reason": response.finish_reason, "tool_calls": len(response.tool_calls)},
            )
            return response