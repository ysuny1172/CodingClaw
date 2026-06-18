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


@dataclass(frozen=True)
class BeforeCompactionContext:
    reason: str
    usage: Any
    messages_to_summarize: list[Message]
    kept_messages: list[Message]
    previous_summary: str | None
    details: dict[str, Any]
    workspace_root: Path
    instructions: str | None = None


@dataclass(frozen=True)
class CompactionDecision:
    allow: bool = True
    reason: str | None = None
    summary: str | None = None
    details: dict[str, Any] | None = None


BeforeToolCallHook = Callable[[BeforeToolCallContext], ToolDecision | None]
BeforeCompactionHook = Callable[[BeforeCompactionContext], CompactionDecision | None]


class HookRegistry:
    def __init__(self) -> None:
        self._before_tool_call: list[BeforeToolCallHook] = []
        self._before_compaction: list[BeforeCompactionHook] = []

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

    def before_compaction(self, hook: BeforeCompactionHook) -> Callable[[], None]:
        self._before_compaction.append(hook)

        def unregister() -> None:
            if hook in self._before_compaction:
                self._before_compaction.remove(hook)

        return unregister

    def register_before_compaction(self, hook: BeforeCompactionHook) -> Callable[[], None]:
        return self.before_compaction(hook)

    def run_before_compaction(self, context: BeforeCompactionContext) -> CompactionDecision:
        details = dict(context.details)
        summary: str | None = None
        for hook in list(self._before_compaction):
            decision = hook(context)
            if decision is None:
                continue
            if decision.details is not None:
                details = _merge_details(details, decision.details)
                context = BeforeCompactionContext(
                    reason=context.reason,
                    usage=context.usage,
                    messages_to_summarize=context.messages_to_summarize,
                    kept_messages=context.kept_messages,
                    previous_summary=context.previous_summary,
                    details=details,
                    workspace_root=context.workspace_root,
                    instructions=context.instructions,
                )
            if decision.summary is not None:
                summary = decision.summary
            if not decision.allow:
                return CompactionDecision(allow=False, reason=decision.reason, summary=summary, details=details)
        return CompactionDecision(allow=True, summary=summary, details=details)


def _merge_details(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(merged.get(key), list) and isinstance(value, list):
            existing = [str(item) for item in merged[key]]
            for item in value:
                item_text = str(item)
                if item_text not in existing:
                    existing.append(item_text)
            merged[key] = existing
        else:
            merged[key] = value
    return merged
