import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent import Agent
from codingclaw.agent.types import AssistantResponse, ToolCall
from codingclaw.hooks import HookRegistry, ToolDecision
from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.base import ToolResult
from codingclaw.tools.file_tools import ListFilesTool


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, *, model, system_prompt, messages, tools):
        self.calls.append({"model": model, "system_prompt": system_prompt, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


class EchoTool:
    name = "echo"
    description = "Echo text."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self):
        self.calls = []

    def execute(self, args, context):
        self.calls.append(args)
        return ToolResult.success({"text": args["text"]})

    def openai_schema(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def event_labels(events):
    labels = []
    for event in events:
        if event["type"] in {"message_start", "message_end"}:
            labels.append(f"{event['type']}:{event['message']['role']}")
        elif event["type"] in {"tool_execution_start", "tool_execution_end"}:
            labels.append(f"{event['type']}:{event['tool_name']}")
        elif event["type"] == "llm_request":
            labels.append("llm_request")
        elif event["type"] == "llm_response":
            labels.append("llm_response")
        else:
            labels.append(event["type"])
    return labels


class AgentLoopTest(unittest.TestCase):
    def test_event_order_for_single_prompt(self):
        registry = ToolRegistry(ToolContext(Path(".")))
        llm = FakeLLM([AssistantResponse(content="hello")])
        agent = Agent(llm=llm, model="fake", tools=registry, system_prompt="test")
        events = []
        agent.subscribe(events.append)

        result = agent.prompt("hi")

        self.assertEqual(result, "hello")
        self.assertEqual(
            event_labels(events),
            [
                "agent_start",
                "turn_start",
                "message_start:user",
                "message_end:user",
                "llm_request",
                "llm_response",
                "message_start:assistant",
                "message_end:assistant",
                "turn_end",
                "agent_end",
            ],
        )

    def test_tool_call_then_final_answer(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "hello.txt").write_text("hi", encoding="utf-8")
            registry = ToolRegistry(ToolContext(Path(tmp)))
            registry.register(ListFilesTool())
            llm = FakeLLM(
                [
                    AssistantResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="list_files", arguments={"path": "."})],
                    ),
                    AssistantResponse(content="There is a hello.txt file."),
                ]
            )
            agent = Agent(llm=llm, model="fake", tools=registry, system_prompt="test")
            events = []
            agent.subscribe(events.append)
            result = agent.prompt("what files are here?")

            self.assertIn("hello.txt", json.dumps(agent.state.messages))
            self.assertEqual(result, "There is a hello.txt file.")
            self.assertEqual(len(llm.calls), 2)
            self.assertEqual(
                event_labels(events),
                [
                    "agent_start",
                    "turn_start",
                    "message_start:user",
                    "message_end:user",
                    "llm_request",
                    "llm_response",
                    "message_start:assistant",
                    "message_end:assistant",
                    "tool_execution_start:list_files",
                    "tool_execution_end:list_files",
                    "message_start:tool",
                    "message_end:tool",
                    "turn_end",
                    "turn_start",
                    "llm_request",
                    "llm_response",
                    "message_start:assistant",
                    "message_end:assistant",
                    "turn_end",
                    "agent_end",
                ],
            )
            turn_ends = [event for event in events if event["type"] == "turn_end"]
            self.assertEqual(turn_ends[0]["message"]["role"], "assistant")
            self.assertEqual(turn_ends[0]["tool_results"][0]["role"], "tool")

    def test_before_tool_call_allows_and_can_rewrite_arguments(self):
        registry = ToolRegistry(ToolContext(Path(".")))
        tool = EchoTool()
        registry.register(tool)
        hooks = HookRegistry()
        hooks.before_tool_call(lambda context: ToolDecision(arguments={"text": context.arguments["text"].upper()}))
        llm = FakeLLM(
            [
                AssistantResponse(content="", tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]),
                AssistantResponse(content="done"),
            ]
        )
        agent = Agent(llm=llm, model="fake", tools=registry, hooks=hooks)

        result = agent.prompt("start")

        self.assertEqual(result, "done")
        self.assertEqual(tool.calls, [{"text": "HI"}])

    def test_before_tool_call_blocks_execution(self):
        registry = ToolRegistry(ToolContext(Path(".")))
        tool = EchoTool()
        registry.register(tool)
        hooks = HookRegistry()
        hooks.before_tool_call(lambda _context: ToolDecision(allow=False, reason="unsafe"))
        llm = FakeLLM(
            [
                AssistantResponse(content="", tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]),
                AssistantResponse(content="done"),
            ]
        )
        agent = Agent(llm=llm, model="fake", tools=registry, hooks=hooks)
        events = []
        agent.subscribe(events.append)

        result = agent.prompt("start")

        self.assertEqual(result, "done")
        self.assertEqual(tool.calls, [])
        tool_end = next(event for event in events if event["type"] == "tool_execution_end")
        self.assertTrue(tool_end["is_error"])
        self.assertEqual(tool_end["result"]["error"]["type"], "ToolBlocked")

    def test_before_tool_call_error_becomes_tool_error(self):
        registry = ToolRegistry(ToolContext(Path(".")))
        tool = EchoTool()
        registry.register(tool)
        hooks = HookRegistry()

        def broken(_context):
            raise RuntimeError("policy unavailable")

        hooks.before_tool_call(broken)
        llm = FakeLLM(
            [
                AssistantResponse(content="", tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]),
                AssistantResponse(content="done"),
            ]
        )
        agent = Agent(llm=llm, model="fake", tools=registry, hooks=hooks)
        events = []
        agent.subscribe(events.append)

        result = agent.prompt("start")

        self.assertEqual(result, "done")
        self.assertEqual(tool.calls, [])
        tool_end = next(event for event in events if event["type"] == "tool_execution_end")
        self.assertTrue(tool_end["is_error"])
        self.assertEqual(tool_end["result"]["error"]["type"], "HookError")

    def test_before_tool_call_hooks_stop_after_first_block(self):
        registry = ToolRegistry(ToolContext(Path(".")))
        tool = EchoTool()
        registry.register(tool)
        hooks = HookRegistry()
        calls = []
        hooks.before_tool_call(lambda _context: calls.append("first") or ToolDecision(allow=False, reason="blocked"))
        hooks.before_tool_call(lambda _context: calls.append("second") or ToolDecision())
        llm = FakeLLM(
            [
                AssistantResponse(content="", tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hi"})]),
                AssistantResponse(content="done"),
            ]
        )
        agent = Agent(llm=llm, model="fake", tools=registry, hooks=hooks)

        agent.prompt("start")

        self.assertEqual(calls, ["first"])
        self.assertEqual(tool.calls, [])


if __name__ == "__main__":
    unittest.main()
