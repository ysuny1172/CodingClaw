from __future__ import annotations

from pathlib import Path
from typing import Callable

from codingclaw.agent import Agent
from codingclaw.agent.types import AgentEvent, LLMClient, TokenUsage
from codingclaw.agent.events import make_event
from codingclaw.config import Config
from codingclaw.hooks import HookRegistry
from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.command_tools import RunCommandTool
from codingclaw.tools.file_tools import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from codingclaw.trace import TraceLogger
from codingclaw.unicode import sanitize_text
from .compaction import CompactionReason, CompactionResult
from .context import ContextManager, ContextUsage
from .resources import ResourceLoader
from .session_store import SessionStore
from .system_prompt import build_system_prompt


class Session:
    """High-level coding-agent session abstraction."""

    def __init__(self, *, config: Config, llm: LLMClient, store: SessionStore | None = None) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace).resolve()
        self.llm = llm
        self.store = store or SessionStore(self.workspace_root)
        self.trace = TraceLogger(self.workspace_root, run_id=self.store.session_id)
        self.resources = ResourceLoader(self.workspace_root)
        self.tools = self._create_tools()
        self.hooks = HookRegistry()
        self.trace_errors: list[str] = []
        self._listeners: list[Callable[[AgentEvent], None]] = []
        self.agent = Agent(
            llm=self.llm,
            model=config.model,
            tools=self.tools,
            hooks=self.hooks,
            max_steps=config.max_steps,
        )
        self.agent.state.messages = self.store.load_messages()
        self.agent.subscribe(self._handle_agent_event)
        self.context = ContextManager(
            config=self.config,
            llm=self.llm,
            store=self.store,
            hooks=self.hooks,
            workspace_root=self.workspace_root,
            get_system_prompt=lambda: self.agent.state.system_prompt,
            get_messages=lambda: self.agent.state.messages,
            set_messages=self._set_agent_messages,
            get_tool_schemas=self.tools.openai_schemas,
            emit=self._emit_session_event,
        )
        self.store.append_model_change(config.model)
        self._log_trace(
            {
                "type": "run_created" if self.store.is_new else "session_resumed",
                "model": config.model,
                "workspace": str(self.workspace_root),
                "message_count": len(self.agent.state.messages),
            }
        )
        if self.store.unicode_repairs:
            self._log_trace(
                {
                    "type": "session_unicode_repaired",
                    "entry_count": self.store.unicode_repairs,
                    "session_path": str(self.store.path),
                }
            )

    def prompt(self, text: str) -> str:
        text = sanitize_text(text)
        self._refresh_system_prompt()
        try:
            self.context.maybe_compact_before_prompt(text)
        except Exception:
            pass
        try:
            final_text = self.agent.prompt(text)
        except Exception as error:
            if not self.context.is_context_overflow_error(error):
                raise
            try:
                result = self.context.compact(reason="overflow")
            except Exception:
                raise error
            if not result:
                raise error
            final_text = self.agent.continue_run()
        try:
            self.context.maybe_auto_compact()
        except Exception:
            pass
        return final_text

    def _refresh_system_prompt(self) -> None:
        loaded = self.resources.load()
        self.agent.state.system_prompt = build_system_prompt(
            workspace_root=self.workspace_root,
            tools=self.tools,
            resources=loaded,
        )
        self._log_trace(
            {
                "type": "resources_loaded",
                "skills": [skill.name for skill in loaded.skills],
                "context_files": [str(item.path) for item in loaded.context_files],
                "diagnostics": [diagnostic.__dict__ for diagnostic in loaded.diagnostics],
            }
        )

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def context_token_estimate(self) -> int:
        return self.context.context_token_estimate()

    def context_usage(self) -> ContextUsage:
        return self.context.context_usage()

    def context_tokens_label(self) -> str:
        return self.context.context_tokens_label()

    def compact(self, reason: CompactionReason = "manual", instructions: str | None = None) -> CompactionResult | None:
        return self.context.compact(reason=reason, instructions=instructions)

    def latest_usage_label(self) -> str | None:
        return self.context.latest_usage_label()

    def _create_tools(self) -> ToolRegistry:
        registry = ToolRegistry(ToolContext(workspace_root=self.workspace_root))
        registry.register(ListFilesTool())
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(RunCommandTool())
        return registry

    def _handle_agent_event(self, event: AgentEvent) -> None:
        self._log_trace(event)
        usage = event.get("usage")
        if event.get("type") in {"llm_request", "llm_response"} and isinstance(usage, dict):
            self.context.record_usage(
                TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
                    completion_tokens=usage.get("completion_tokens")
                    if isinstance(usage.get("completion_tokens"), int)
                    else None,
                    total_tokens=usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
                    is_estimate=bool(usage.get("is_estimate")),
                )
            )
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict):
                self.store.append_message(message)
        self._notify_listeners(event)

    def _emit_session_event(self, event: AgentEvent) -> None:
        self._log_trace(event)
        self._notify_listeners(event)

    def _notify_listeners(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def _set_agent_messages(self, messages: list[dict]) -> None:
        self.agent.state.messages = messages

    def _log_trace(self, event: dict) -> None:
        try:
            self.trace.log(event)
        except Exception as error:
            self.trace_errors.append(str(error))
