from __future__ import annotations

from typing import Callable

from .events import make_event
from .types import AgentEvent, AssistantResponse, LLMClient, Message
from codingclaw.tools.registry import ToolRegistry


EventSink = Callable[[AgentEvent], None]


def run_agent_loop(
    *,
    llm: LLMClient,
    model: str,
    system_prompt: str,
    messages: list[Message],
    tools: ToolRegistry,
    max_steps: int,
    emit: EventSink,
) -> tuple[str, list[Message]]:
    """Run the LLM/tool loop until the assistant stops or max_steps is reached."""
    final_text = ""

    for step in range(max_steps):
        emit(make_event("llm_request", step=step, message_count=len(messages)))
        response: AssistantResponse = llm.chat(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools.openai_schemas(),
        )
        emit(
            make_event(
                "llm_response",
                step=step,
                finish_reason=response.finish_reason,
                tool_call_count=len(response.tool_calls),
                raw=response.raw,
            )
        )

        assistant_message: Message = {
            "role": "assistant",
            "content": response.content or "",
        }
        if response.tool_calls:
            assistant_message["tool_calls"] = [call.to_openai() for call in response.tool_calls]
        messages.append(assistant_message)
        emit(make_event("message_start", message=assistant_message))
        emit(make_event("message_end", message=assistant_message))

        final_text = response.content or final_text
        if not response.tool_calls:
            emit(make_event("agent_end", reason="stop", messages=messages))
            return final_text, messages

        for call in response.tool_calls:
            emit(make_event("tool_call_start", tool_call=call.__dict__))
            result = tools.execute(call.name, call.arguments)
            emit(
                make_event(
                    "tool_call_end",
                    tool_call=call.__dict__,
                    result=result.to_dict(),
                    is_error=not result.ok,
                )
            )
            tool_message: Message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result.to_json(),
            }
            messages.append(tool_message)
            emit(make_event("message_start", message=tool_message))
            emit(make_event("message_end", message=tool_message))

    emit(make_event("agent_end", reason="max_steps", messages=messages))
    return final_text or "Stopped after reaching max_steps.", messages
