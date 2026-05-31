from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from codingclaw.agent.types import Message


@dataclass(frozen=True)
class BeforeToolCallContext:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    messages: list[Message]
    workspace_root: Path


@dataclass(frozen=True)
class ToolDecision:
    allow: bool = True
    reason: str | None = None
    arguments: dict[str, Any] | None = None


BeforeToolCallHook = Callable[[BeforeToolCallContext], ToolDecision | None]


class HookRegistry:
    def __init__(self) -> None:
        self._before_tool_call: list[BeforeToolCallHook] = []

    def before_tool_call(self, hook: BeforeToolCallHook) -> Callable[[], None]:
        self._before_tool_call.append(hook)

        def unregister() -> None:
            if hook in self._before_tool_call:
                self._before_tool_call.remove(hook)

        return unregister

    def run_before_tool_call(self, context: BeforeToolCallContext) -> ToolDecision:
        arguments = dict(context.arguments)
        for hook in list(self._before_tool_call):
            decision = hook(context)
            if decision is None:
                continue
            if decision.arguments is not None:
                arguments = dict(decision.arguments)
                context = BeforeToolCallContext(
                    tool_call_id=context.tool_call_id,
                    tool_name=context.tool_name,
                    arguments=arguments,
                    messages=context.messages,
                    workspace_root=context.workspace_root,
                )
            if not decision.allow:
                return ToolDecision(allow=False, reason=decision.reason, arguments=arguments)
        return ToolDecision(allow=True, arguments=arguments)
