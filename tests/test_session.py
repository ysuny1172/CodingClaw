import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent.types import AssistantResponse, TokenUsage, ToolCall
from codingclaw.config import Config
from codingclaw.hooks import CompactionDecision
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


class UsageFakeLLM:
    def chat(self, *, model, system_prompt, messages, tools):
        return AssistantResponse(
            content="done",
            usage=TokenUsage(prompt_tokens=123, completion_tokens=4, total_tokens=127),
        )


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

    def test_trace_failure_does_not_interrupt_session_persistence(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            session = Session(config=config, llm=FakeLLM())

            def fail_trace(_event):
                raise UnicodeEncodeError("utf-8", "\udce4", 0, 1, "surrogates not allowed")

            session.trace.log = fail_trace

            result = session.prompt("hello")

            self.assertEqual(result, "done")
            self.assertEqual(
                [(message["role"], message["content"]) for message in session.store.load_messages()],
                [("user", "hello"), ("assistant", "done")],
            )
            self.assertTrue(session.trace_errors)

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

    def test_session_store_reads_new_compaction_entry_shape(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            store.append_message({"role": "user", "content": "old"})
            store.append_message({"role": "assistant", "content": "kept"})
            first_kept = store.load_entries()[2]["id"]
            store.append_compaction(
                "summary",
                first_kept_entry_id=first_kept,
                tokens_before=100,
                reason="manual",
                details={"read_files": ["a.py"], "modified_files": ["b.py"]},
            )
            store.append_message({"role": "user", "content": "new"})

            entries = store.load_entries()
            compaction = [entry for entry in entries if entry.get("type") == "compaction"][0]
            messages = store.load_messages()

            self.assertEqual(compaction["first_kept_entry_id"], first_kept)
            self.assertEqual(compaction["details"]["read_files"], ["a.py"])
            self.assertEqual([message["content"] for message in messages[1:]], ["kept", "new"])

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

    def test_session_compacts_before_prompt_when_projected_context_crosses_threshold(self):
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
            session.store.append_message({"role": "user", "content": "old " * 100})
            session.store.append_message({"role": "assistant", "content": "kept"})
            session.agent.state.messages = session.store.load_messages()

            result = session.prompt("new request")

            self.assertEqual(result, "answer 1")
            self.assertGreaterEqual(llm.summary_calls, 1)
            entries = session.store.load_entries()
            compaction_index = next(index for index, entry in enumerate(entries) if entry.get("type") == "compaction")
            new_user_index = next(
                index
                for index, entry in enumerate(entries)
                if entry.get("type") == "message" and entry.get("message", {}).get("content") == "new request"
            )
            self.assertLess(compaction_index, new_user_index)

    def test_context_usage_prefers_provider_usage_until_compaction(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
            )
            session = Session(config=config, llm=UsageFakeLLM())

            session.prompt("hello")
            usage = session.context_usage()

            self.assertEqual(usage.tokens, 123)
            self.assertEqual(usage.source, "provider")
            self.assertEqual(session.latest_usage_label(), "123 prompt / 4 completion / 127 total tokens")

            session.compact(reason="manual")
            usage_after = session.context_usage()

            self.assertEqual(usage_after.source, "estimate")
            self.assertIsNone(session.latest_usage_label())

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

    def test_before_compaction_hook_can_block(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
            )
            llm = CompactionFakeLLM()
            session = Session(config=config, llm=llm)
            session.prompt("old context")
            session.hooks.register_before_compaction(lambda _context: CompactionDecision(allow=False, reason="blocked"))

            result = session.compact(reason="manual")

            self.assertIsNone(result)
            self.assertEqual(llm.summary_calls, 0)
            self.assertFalse(any(entry["type"] == "compaction" for entry in session.store.load_entries()))

    def test_before_compaction_hook_can_override_summary_and_details(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
            )
            llm = CompactionFakeLLM()
            session = Session(config=config, llm=llm)
            session.prompt("old context")
            session.hooks.register_before_compaction(
                lambda _context: CompactionDecision(
                    summary="hook summary",
                    details={"read_files": ["a.py"], "modified_files": []},
                )
            )

            result = session.compact(reason="manual")

            self.assertIsNotNone(result)
            self.assertTrue(result.from_hook)
            self.assertEqual(llm.summary_calls, 0)
            self.assertIn("<read-files>\na.py\n</read-files>", result.summary)

    def test_before_compaction_hook_exception_records_error_event(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(
                workspace=tmp,
                api_key="fake",
                model="fake",
                context_window=10_000,
                reserve_tokens=0,
                keep_recent_tokens=5,
            )
            session = Session(config=config, llm=CompactionFakeLLM())
            events = []
            session.subscribe(events.append)
            session.prompt("old context")

            def fail(_context):
                raise RuntimeError("hook failed")

            session.hooks.register_before_compaction(fail)

            result = session.compact(reason="manual")

            self.assertIsNone(result)
            end_events = [event for event in events if event["type"] == "compaction_end"]
            self.assertIn("hook failed", end_events[-1]["error_message"])


if __name__ == "__main__":
    unittest.main()
