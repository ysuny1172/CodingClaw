from __future__ import annotations

from typing import Any

from codingclaw.errors import ToolError
from .base import Tool, ToolContext, ToolResult


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult.failure("ToolNotFound", f"Tool not found: {name}")
        try:
            self._validate_args(tool, args)
            return tool.execute(args, self.context)
        except Exception as error:
            return ToolResult.failure(error.__class__.__name__, str(error))

    def _validate_args(self, tool: Tool, args: dict[str, Any]) -> None:
        schema = tool.parameters
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for key in required:
            if key not in args:
                raise ToolError(f"Missing required argument: {key}")
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected == "string" and not isinstance(value, str):
                raise ToolError(f"Argument {key} must be a string")
            if expected == "integer" and not isinstance(value, int):
                raise ToolError(f"Argument {key} must be an integer")
            if expected == "number" and not isinstance(value, (int, float)):
                raise ToolError(f"Argument {key} must be a number")
