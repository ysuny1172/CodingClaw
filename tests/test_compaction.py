import unittest

from codingclaw.session.compaction import generate_summary, prepare_compaction
from codingclaw.agent.types import AssistantResponse


class CapturingLLM:
    def __init__(self):
        self.calls = []

    def chat(self, *, model, system_prompt, messages, tools):
        self.calls.append({"model": model, "system_prompt": system_prompt, "messages": messages, "tools": tools})
        return AssistantResponse(content="summary")


def entry(entry_id, role, content, **extra):
    message = {"role": role, "content": content, **extra}
    return {"type": "message", "id": entry_id, "message": message}


class CompactionTest(unittest.TestCase):
    def test_prepare_compaction_returns_none_when_nothing_can_be_summarized(self):
        result = prepare_compaction(
            [entry("m1", "user", "hello")],
            previous_compaction=None,
            tokens_before=100,
            keep_recent_tokens=10,
        )

        self.assertIsNone(result)

    def test_prepare_compaction_keeps_recent_messages(self):
        entries = [
            entry("m1", "user", "old " * 20),
            entry("m2", "assistant", "older " * 20),
            entry("m3", "user", "recent"),
            entry("m4", "assistant", "done"),
        ]

        result = prepare_compaction(
            entries,
            previous_compaction=None,
            tokens_before=200,
            keep_recent_tokens=5,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.first_kept_message_id, "m4")
        self.assertEqual([message["content"] for message in result.messages_to_summarize], ["old " * 20, "older " * 20, "recent"])

    def test_prepare_compaction_does_not_keep_orphan_tool_message(self):
        entries = [
            entry("m1", "user", "old"),
            entry("m2", "assistant", "", tool_calls=[{"id": "call_1"}]),
            entry("m3", "tool", "tool result", tool_call_id="call_1", name="echo"),
            entry("m4", "assistant", "final"),
        ]

        result = prepare_compaction(
            entries,
            previous_compaction=None,
            tokens_before=200,
            keep_recent_tokens=20,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.first_kept_message_id, "m2")
        self.assertEqual(result.kept_messages[0]["role"], "assistant")

    def test_generate_summary_includes_previous_summary(self):
        llm = CapturingLLM()
        preparation = prepare_compaction(
            [entry("m1", "user", "old"), entry("m2", "assistant", "recent")],
            previous_compaction={"summary": "previous summary"},
            tokens_before=100,
            keep_recent_tokens=1,
        )

        self.assertIsNotNone(preparation)
        summary = generate_summary(llm=llm, model="fake", preparation=preparation, reserve_tokens=100)

        self.assertEqual(summary, "summary")
        prompt = llm.calls[0]["messages"][0]["content"]
        self.assertIn("<previous-summary>", prompt)
        self.assertIn("previous summary", prompt)


if __name__ == "__main__":
    unittest.main()
