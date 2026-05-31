import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent.types import AssistantResponse, ToolCall
from codingclaw.config import Config
from codingclaw.session import Session
from codingclaw.session.session_store import SessionStore


class FakeLLM:
    def chat(self, *, model, system_prompt, messages, tools):
        self.system_prompt = system_prompt
        self.messages = list(messages)
        return AssistantResponse(content="done")


class ToolCallingFakeLLM:
    def __init__(self):
        self.responses = [
            AssistantResponse(content="", tool_calls=[ToolCall(id="call_1", name="list_files", arguments={"path": "."})]),
            AssistantResponse(content="done"),
        ]

    def chat(self, *, model, system_prompt, messages, tools):
        return self.responses.pop(0)


class SessionTest(unittest.TestCase):
    def test_session_persists_messages_and_trace(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "AGENTS.md").write_text("Project rule.", encoding="utf-8")
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            llm = FakeLLM()
            session = Session(config=config, llm=llm)
            events = []
            session.subscribe(events.append)

            result = session.prompt("hello")

            self.assertEqual(result, "done")
            self.assertIn("Project rule.", llm.system_prompt)
            session_lines = session.store.path.read_text(encoding="utf-8").splitlines()
            trace_lines = session.trace.path.read_text(encoding="utf-8").splitlines()
            session_entry_types = [json.loads(line)["type"] for line in session_lines]
            trace_entry_types = [json.loads(line)["type"] for line in trace_lines]
            self.assertGreaterEqual(len(session_lines), 4)
            self.assertEqual(session_entry_types.count("message"), 2)
            self.assertIn("llm_response", trace_entry_types)
            self.assertIn("turn_start", trace_entry_types)
            self.assertIn("turn_end", trace_entry_types)
            self.assertIn("agent_end", [event["type"] for event in events])

    def test_session_can_resume_persisted_messages(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            first = Session(config=config, llm=FakeLLM())

            first.prompt("hello")

            store = SessionStore.open(config.workspace, first.store.path)
            resumed_llm = FakeLLM()
            resumed = Session(config=config, llm=resumed_llm, store=store)
            result = resumed.prompt("again")

            self.assertEqual(result, "done")
            self.assertEqual([message["role"] for message in resumed_llm.messages], ["user", "assistant", "user"])
            self.assertEqual(SessionStore.open_latest(config.workspace).path, first.store.path)

    def test_session_trace_records_tool_execution_events(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "hello.txt").write_text("hi", encoding="utf-8")
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            session = Session(config=config, llm=ToolCallingFakeLLM())

            result = session.prompt("list files")

            self.assertEqual(result, "done")
            trace_types = [json.loads(line)["type"] for line in session.trace.path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("tool_execution_start", trace_types)
            self.assertIn("tool_execution_end", trace_types)


if __name__ == "__main__":
    unittest.main()
