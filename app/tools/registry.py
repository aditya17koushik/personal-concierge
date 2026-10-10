import asyncio
import inspect
import json
import logging
from typing import Any, Collection

from pydantic import ValidationError

from app.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

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
            return {
                "error": "Invalid arguments",
                "details": json.loads(
                    exc.json(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    )
                ),
            }

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
        except Exception:
            logger.exception("Tool %s failed", name)
            await asyncio.to_thread(ctx.db.rollback)
            return {"error": "Tool failed unexpectedly"}


def build_default_registry() -> ToolRegistry:
    from app.tools.calendar.tools import CALENDAR_TOOLS
    from app.tools.expenses.tools import EXPENSE_TOOLS
    from app.tools.gmail.tools import GMAIL_TOOLS

    return ToolRegistry([*EXPENSE_TOOLS, *GMAIL_TOOLS, *CALENDAR_TOOLS])