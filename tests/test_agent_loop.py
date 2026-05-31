import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent import Agent
from codingclaw.agent.types import AssistantResponse, ToolCall
from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.file_tools import ListFilesTool


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, *, model, system_prompt, messages, tools):
        self.calls.append({"model": model, "system_prompt": system_prompt, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


class AgentLoopTest(unittest.TestCase):
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
            result = agent.prompt("what files are here?")

            self.assertIn("hello.txt", json.dumps(agent.state.messages))
            self.assertEqual(result, "There is a hello.txt file.")
            self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
