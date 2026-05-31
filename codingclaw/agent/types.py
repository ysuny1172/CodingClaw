from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Message = dict[str, Any]
AgentEvent = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai(cls, raw: dict[str, Any], index: int = 0) -> "ToolCall":
        function = raw.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            import json

            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_raw": arguments}
            arguments = parsed
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        return cls(
            id=str(raw.get("id") or f"call_{index}"),
            name=str(function.get("name") or raw.get("name") or ""),
            arguments=arguments,
        )

    def to_openai(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(frozen=True)
class AssistantResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None


class LLMClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AssistantResponse:
        ...


EventListener = Protocol
StopReason = Literal["stop", "max_steps"]
