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
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    is_estimate: bool = False

    @classmethod
    def from_openai(cls, raw: dict[str, Any] | None) -> "TokenUsage | None":
        if not isinstance(raw, dict):
            return None
        return cls(
            prompt_tokens=_int_or_none(raw.get("prompt_tokens")),
            completion_tokens=_int_or_none(raw.get("completion_tokens")),
            total_tokens=_int_or_none(raw.get("total_tokens")),
        )

    @classmethod
    def estimate(cls, prompt_tokens: int) -> "TokenUsage":
        return cls(prompt_tokens=prompt_tokens, is_estimate=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "is_estimate": self.is_estimate,
        }


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


@dataclass(frozen=True)
class AssistantResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: TokenUsage | None = None


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
