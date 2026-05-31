from __future__ import annotations

from pathlib import Path

from codingclaw.agent import Agent
from codingclaw.agent.types import AgentEvent, LLMClient
from codingclaw.config import Config
from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.command_tools import RunCommandTool
from codingclaw.tools.file_tools import ListFilesTool, ReadFileTool, WriteFileTool
from codingclaw.trace import TraceLogger
from .resources import ResourceLoader
from .session_store import SessionStore
from .system_prompt import build_system_prompt


class Session:
    """High-level coding-agent session abstraction."""

    def __init__(self, *, config: Config, llm: LLMClient) -> None:
        self.config = config
        self.workspace_root = Path(config.workspace).resolve()
        self.llm = llm
        self.store = SessionStore(self.workspace_root)
        self.trace = TraceLogger(self.workspace_root, run_id=self.store.session_id)
        self.resources = ResourceLoader(self.workspace_root)
        self.tools = self._create_tools()
        self.agent = Agent(
            llm=self.llm,
            model=config.model,
            tools=self.tools,
            max_steps=config.max_steps,
        )
        self.agent.subscribe(self._handle_agent_event)
        self.store.append_model_change(config.model)
        self.trace.log({"type": "run_created", "model": config.model, "workspace": str(self.workspace_root)})

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

    def _create_tools(self) -> ToolRegistry:
        registry = ToolRegistry(ToolContext(workspace_root=self.workspace_root))
        registry.register(ListFilesTool())
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())
        registry.register(RunCommandTool())
        return registry

    def _handle_agent_event(self, event: AgentEvent) -> None:
        self.trace.log(event)
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict):
                self.store.append_message(message)
