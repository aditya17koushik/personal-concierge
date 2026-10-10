import asyncio
import inspect
import json
import logging
from typing import Any, Collection

from pydantic import ValidationError

from app.tools.base import Tool, ToolContext, ToolError

logger = logging.getLogger(__name__)


def _validation_details(exc: ValidationError) -> Any:
    return json.loads(
        exc.json(include_url=False, include_context=False, include_input=False)
    )


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        for tool in tools:
            if tool.requires_approval and tool.summarize is None:
                raise ValueError(
                    f"Tool '{tool.name}' requires approval but has no summarize function"
                )
        self._tools = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def domains(self) -> set[str]:
        return {tool.domain for tool in self._tools.values()}

    def schemas(self, names: Collection[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI tool schemas, optionally limited to the given tool names."""
        return [
            tool.to_openai()
            for tool in self._tools.values()
            if names is None or tool.name in names
        ]

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        """Run a tool. Never raises: errors are returned so the LLM can recover."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}

        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "Invalid arguments", "details": _validation_details(exc)}

        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(ctx, args)
            else:
                # Sync handlers do blocking database I/O. Run them in a worker
                # thread so a slow database can never freeze the event loop.
                result = await asyncio.to_thread(tool.handler, ctx, args)
                if inspect.isawaitable(result):
                    result = await result
            return result
        except ToolError as exc:
            return {"error": str(exc)}
        except Exception:
            logger.exception("Tool %s failed", name)
            await asyncio.to_thread(ctx.db.rollback)
            return {"error": "Tool failed unexpectedly"}

    async def prepare(
        self, name: str, arguments: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        """Validate a PROPOSED call and build its summary. Executes nothing.

        Returns {"arguments": <json-safe validated args>, "summary": str},
        or {"error": ...}.
        """
        tool = self._tools.get(name)
        if tool is None or tool.summarize is None:
            return {"error": f"Unknown tool: {name}"}

        try:
            args = tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            return {"error": "Invalid arguments", "details": _validation_details(exc)}

        try:
            if inspect.iscoroutinefunction(tool.summarize):
                summary = await tool.summarize(ctx, args)
            else:
                summary = await asyncio.to_thread(tool.summarize, ctx, args)
                if inspect.isawaitable(summary):
                    summary = await summary
        except ToolError as exc:
            return {"error": str(exc)}
        except Exception:
            logger.exception("Preparing %s failed", name)
            await asyncio.to_thread(ctx.db.rollback)
            return {"error": "Could not prepare this action."}

        return {"arguments": args.model_dump(mode="json"), "summary": str(summary)}


def build_default_registry() -> ToolRegistry:
    from app.tools.calendar.tools import CALENDAR_TOOLS
    from app.tools.expenses.tools import EXPENSE_TOOLS
    from app.tools.gmail.tools import GMAIL_TOOLS

    return ToolRegistry([*EXPENSE_TOOLS, *GMAIL_TOOLS, *CALENDAR_TOOLS])