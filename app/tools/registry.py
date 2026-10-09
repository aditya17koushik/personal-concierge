import json
import logging
from typing import Any

from pydantic import ValidationError

from app.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai() for tool in self._tools.values()]

    def execute(
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
            return tool.handler(ctx, args)
        except Exception:
            logger.exception("Tool %s failed", name)
            ctx.db.rollback()
            return {"error": "Tool failed unexpectedly"}


def build_default_registry() -> ToolRegistry:
    from app.tools.expenses.tools import EXPENSE_TOOLS

    return ToolRegistry([*EXPENSE_TOOLS])