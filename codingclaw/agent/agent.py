from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .events import make_event
from .loop import run_agent_loop
from .types import AgentEvent, LLMClient, Message
from codingclaw.tools.registry import ToolRegistry


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    model: str = ""


class Agent:
    """Thin stateful wrapper around the core agent loop."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        tools: ToolRegistry,
        system_prompt: str = "",
        max_steps: int = 20,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.state = AgentState(system_prompt=system_prompt, model=model)
        self._listeners: list[Callable[[AgentEvent], None]] = []

    def subscribe(self, listener: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def emit(self, event: AgentEvent) -> None:
        for listener in list(self._listeners):
            listener(event)

    def prompt(self, text: str | list[Message]) -> str:
        prompt_messages = text if isinstance(text, list) else [{"role": "user", "content": text}]

        self.emit(make_event("agent_start"))
        for message in prompt_messages:
            self.state.messages.append(message)
            self.emit(make_event("message_start", message=message))
            self.emit(make_event("message_end", message=message))

        final_text, next_messages = run_agent_loop(
            llm=self.llm,
            model=self.state.model,
            system_prompt=self.state.system_prompt,
            messages=self.state.messages,
            tools=self.tools,
            max_steps=self.max_steps,
            emit=self.emit,
        )
        self.state.messages = next_messages
        return final_text

    def continue_run(self) -> str:
        self.emit(make_event("agent_start"))
        final_text, next_messages = run_agent_loop(
            llm=self.llm,
            model=self.state.model,
            system_prompt=self.state.system_prompt,
            messages=self.state.messages,
            tools=self.tools,
            max_steps=self.max_steps,
            emit=self.emit,
        )
        self.state.messages = next_messages
        return final_text
