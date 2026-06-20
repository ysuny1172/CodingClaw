from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from codingclaw.unicode import sanitize_json_value


@dataclass(frozen=True)
class ToolContext:
    workspace_root: Path


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any = None
    error: dict[str, str] | None = None

    @classmethod
    def success(cls, data: Any) -> "ToolResult":
        return cls(ok=True, data=data, error=None)

    @classmethod
    def failure(cls, error_type: str, message: str) -> "ToolResult":
        return cls(ok=False, data=None, error={"type": error_type, "message": message})

    def to_dict(self) -> dict[str, Any]:
        return sanitize_json_value({"ok": self.ok, "data": self.data, "error": self.error})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolResult:
        ...

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
