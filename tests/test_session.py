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


class CompactionFakeLLM:
    def __init__(self, *, overflow_once: bool = False, summary_fails: bool = False):
        self.overflow_once = overflow_once
        self.summary_fails = summary_fails
        self.normal_calls = 0
        self.summary_calls = 0

    def chat(self, *, model, system_prompt, messages, tools):
        if "context summarization assistant" in system_prompt:
            self.summary_calls += 1
            if self.summary_fails:
                raise RuntimeError("summary service unavailable")
            return AssistantResponse(content="compact summary")
        self.normal_calls += 1
        if self.overflow_once and self.normal_calls == 2:
            raise RuntimeError("context_length_exceeded: maximum context length")
        return AssistantResponse(content=f"answer {self.normal_calls}")


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

    def test_session_store_loads_compacted_context(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            store.append_message({"role": "user", "content": "old"})
            store.append_message({"role": "assistant", "content": "older"})
            first_kept = store.load_entries()[2]["id"]
            store.append_compaction("summary", first_kept, 100, "manual")
            store.append_message({"role": "user", "content": "new"})

            messages = store.load_messages()

            self.assertEqual(messages[0]["role"], "user")
            self.assertIn("<context_summary", messages[0]["content"])
            self.assertEqual([message["content"] for message in messages[1:]], ["older", "new"])

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

    def test_session_auto_compacts_after_threshold(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=50,
                reserve_tokens=0,
                keep_recent_tokens=5,
            )
            llm = CompactionFakeLLM()
            session = Session(config=config, llm=llm)
            events = []
            session.subscribe(events.append)

            result = session.prompt("old " * 100)

            self.assertEqual(result, "answer 1")
            self.assertEqual(llm.summary_calls, 1)
            self.assertTrue(any(entry["type"] == "compaction" for entry in session.store.load_entries()))
            self.assertIn("compaction_start", [event["type"] for event in events])
            self.assertIn("compaction_end", [event["type"] for event in events])

    def test_session_compacts_and_retries_after_context_overflow(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
                auto_compact=False,
            )
            llm = CompactionFakeLLM(overflow_once=True)
            session = Session(config=config, llm=llm)

            self.assertEqual(session.prompt("old context"), "answer 1")
            self.assertEqual(session.prompt("new request"), "answer 3")

            self.assertEqual(llm.summary_calls, 1)
            self.assertTrue(any(entry.get("reason") == "overflow" for entry in session.store.load_entries()))

    def test_overflow_compaction_failure_raises_original_error(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
                auto_compact=False,
            )
            llm = CompactionFakeLLM(overflow_once=True, summary_fails=True)
            session = Session(config=config, llm=llm)

            session.prompt("old context")
            with self.assertRaisesRegex(RuntimeError, "context_length_exceeded"):
                session.prompt("new request")


if __name__ == "__main__":
    unittest.main()
