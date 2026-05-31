from __future__ import annotations

from pathlib import Path
from typing import Callable

from codingclaw.agent import Agent
from codingclaw.agent.types import AgentEvent, LLMClient, TokenUsage
from codingclaw.config import Config
from codingclaw.hooks import HookRegistry
from codingclaw.llm.tokens import estimate_prompt_tokens
from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.command_tools import RunCommandTool
from codingclaw.tools.file_tools import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from codingclaw.trace import TraceLogger
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
        self.latest_usage: TokenUsage | None = None
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
        self.store.append_model_change(config.model)
        self.trace.log(
            {
                "type": "run_created" if self.store.is_new else "session_resumed",
                "model": config.model,
                "workspace": str(self.workspace_root),
                "message_count": len(self.agent.state.messages),
            }
        )

    def prompt(self, text: str) -> str:
        loaded = self.resources.load()
        self.agent.state.system_prompt = build_system_prompt(
            workspace_root=self.workspace_root,
            tools=self.tools,
            resources=loaded,
        )
        self.trace.log(
            {
                "type": "resources_loaded",
                "skills": [skill.name for skill in loaded.skills],
                "context_files": [str(item.path) for item in loaded.context_files],
                "diagnostics": [diagnostic.__dict__ for diagnostic in loaded.diagnostics],
            }
        )
        return self.agent.prompt(text)

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def context_token_estimate(self) -> int:
        loaded = self.resources.load()
        system_prompt = build_system_prompt(
            workspace_root=self.workspace_root,
            tools=self.tools,
            resources=loaded,
        )
        return estimate_prompt_tokens(
            system_prompt=system_prompt,
            messages=self.agent.state.messages,
            tools=self.tools.openai_schemas(),
        )

    def context_tokens_label(self) -> str:
        return f"~{self.context_token_estimate():,} tokens"

    def latest_usage_label(self) -> str | None:
        if not self.latest_usage or self.latest_usage.prompt_tokens is None:
            return None
        prefix = "~" if self.latest_usage.is_estimate else ""
        parts = [f"{prefix}{self.latest_usage.prompt_tokens:,} prompt"]
        if self.latest_usage.completion_tokens is not None:
            parts.append(f"{self.latest_usage.completion_tokens:,} completion")
        if self.latest_usage.total_tokens is not None:
            parts.append(f"{self.latest_usage.total_tokens:,} total")
        return " / ".join(parts) + " tokens"

    def _create_tools(self) -> ToolRegistry:
        registry = ToolRegistry(ToolContext(workspace_root=self.workspace_root))
        registry.register(ListFilesTool())
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(EditFileTool())
        registry.register(RunCommandTool())
        return registry

    def _handle_agent_event(self, event: AgentEvent) -> None:
        self.trace.log(event)
        usage = event.get("usage")
        if event.get("type") in {"llm_request", "llm_response"} and isinstance(usage, dict):
            self.latest_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
                completion_tokens=usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
                total_tokens=usage.get("total_tokens") if isinstance(usage.get("total_tokens"), int) else None,
                is_estimate=bool(usage.get("is_estimate")),
            )
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict):
                self.store.append_message(message)
        for listener in list(self._listeners):
            listener(event)
