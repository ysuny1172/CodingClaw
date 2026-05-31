from __future__ import annotations

from typing import Callable

from .events import make_event
from .types import AgentEvent, AssistantResponse, LLMClient, Message
from codingclaw.hooks import BeforeToolCallContext, HookRegistry
from codingclaw.tools import ToolResult
from codingclaw.tools.registry import ToolRegistry


EventSink = Callable[[AgentEvent], None]


def run_agent_loop(
    *,
    llm: LLMClient,
    model: str,
    system_prompt: str,
    messages: list[Message],
    tools: ToolRegistry,
    hooks: HookRegistry,
    max_steps: int,
    emit: EventSink,
) -> tuple[str, list[Message]]:
    """Run the LLM/tool loop until the assistant stops or max_steps is reached."""
    final_text = ""

    for step in range(max_steps):
        if step > 0:
            emit(make_event("turn_start", turn_index=step))

        tool_schemas = tools.openai_schemas()
        emit(make_event("llm_request", step=step, message_count=len(messages)))
        response: AssistantResponse = llm.chat(
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tools=tool_schemas,
        )
        emit(
            make_event(
                "llm_response",
                step=step,
                finish_reason=response.finish_reason,
                tool_call_count=len(response.tool_calls),
                usage=response.usage.to_dict() if response.usage else None,
                raw=response.raw,
            )
        )

        tool_results: list[Message] = []
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
            emit(make_event("turn_end", turn_index=step, message=assistant_message, tool_results=tool_results))
            emit(make_event("agent_end", reason="stop", messages=messages))
            return final_text, messages

        for call in response.tool_calls:
            args = call.arguments
            emit(
                make_event(
                    "tool_execution_start",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    args=args,
                    tool_call=call.__dict__,
                )
            )
            try:
                decision = hooks.run_before_tool_call(
                    BeforeToolCallContext(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        arguments=args,
                        messages=list(messages),
                        workspace_root=tools.context.workspace_root,
                    )
                )
                if decision.arguments is not None:
                    args = decision.arguments
                if decision.allow:
                    result = tools.execute(call.name, args)
                else:
                    result = ToolResult.failure("ToolBlocked", decision.reason or "Tool execution was blocked")
            except Exception as error:
                result = ToolResult.failure("HookError", str(error))

            emit(
                make_event(
                    "tool_execution_end",
                    tool_call_id=call.id,
                    tool_name=call.name,
                    args=args,
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
            tool_results.append(tool_message)
            messages.append(tool_message)
            emit(make_event("message_start", message=tool_message))
            emit(make_event("message_end", message=tool_message))

        emit(make_event("turn_end", turn_index=step, message=assistant_message, tool_results=tool_results))

    emit(make_event("agent_end", reason="max_steps", messages=messages))
    return final_text or "Stopped after reaching max_steps.", messages
