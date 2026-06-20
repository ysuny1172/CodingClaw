import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codingclaw.agent.types import AssistantResponse
from codingclaw.config import Config
from codingclaw.llm.client import OpenAICompatibleClient
from codingclaw.session import Session
from codingclaw.session.session_store import SessionStore
from codingclaw.tools.base import ToolResult
from codingclaw.unicode import sanitize_json_value, sanitize_text


class FakeHTTPResponse:
    def __init__(self, content="done"):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": self.content},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode("utf-8")


class CapturingLLM:
    def chat(self, *, model, system_prompt, messages, tools):
        self.messages = list(messages)
        return AssistantResponse(content="done")


class UnicodeSafetyTest(unittest.TestCase):
    def test_sanitize_json_value_replaces_nested_surrogates(self):
        value = {"message": ["ok", {"content": "bad\udce4text"}]}

        sanitized = sanitize_json_value(value)

        self.assertEqual(sanitized["message"][1]["content"], "bad\ufffdtext")
        self.assertEqual(sanitize_text("\u4e2d\u6587"), "\u4e2d\u6587")
        self.assertEqual(sanitize_text("\ud83d\ude00"), "\U0001f600")

    def test_tool_result_sanitizes_values_before_message_serialization(self):
        result = ToolResult.success({"stdout": "bad\udce4text"})

        self.assertEqual(result.to_dict()["data"]["stdout"], "bad\ufffdtext")
        self.assertNotIn("\\udce4", result.to_json().lower())

    def test_session_store_sanitizes_new_messages_before_writing(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))

            store.append_message({"role": "tool", "content": "bad\udce4text"})

            raw = store.path.read_text(encoding="utf-8")
            messages = store.load_messages()
            self.assertNotIn("\\udce4", raw.lower())
            self.assertEqual(messages[0]["content"], "bad\ufffdtext")

    def test_session_store_recovers_polluted_legacy_session_in_memory(self):
        with TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            polluted_entry = {
                "type": "message",
                "id": "polluted",
                "message": {"role": "tool", "content": "bad\udce4text"},
            }
            with store.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(polluted_entry) + "\n")

            reopened = SessionStore.open(Path(tmp), store.path)
            messages = reopened.load_messages()

            self.assertEqual(messages[0]["content"], "bad\ufffdtext")
            self.assertEqual(reopened.unicode_repairs, 1)
            self.assertNotIn("\\udce4", reopened.path.read_text(encoding="utf-8").lower())

    def test_llm_client_sanitizes_payload_before_utf8_encoding(self):
        client = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="fake")
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse()

        with patch("codingclaw.llm.client.urllib.request.urlopen", side_effect=fake_urlopen):
            response = client.chat(
                model="fake",
                system_prompt="system\udce4",
                messages=[{"role": "user", "content": "bad\udce4text"}],
                tools=[],
            )

        self.assertEqual(response.content, "done")
        self.assertEqual(captured["payload"]["messages"][0]["content"], "system\ufffd")
        self.assertEqual(captured["payload"]["messages"][1]["content"], "bad\ufffdtext")

    def test_llm_client_sanitizes_surrogates_in_provider_response(self):
        client = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="fake")

        with patch(
            "codingclaw.llm.client.urllib.request.urlopen",
            return_value=FakeHTTPResponse(content="bad\udce4reply"),
        ):
            response = client.chat(model="fake", system_prompt="", messages=[], tools=[])

        self.assertEqual(response.content, "bad\ufffdreply")

    def test_session_sanitizes_user_input_before_agent_state(self):
        with TemporaryDirectory() as tmp:
            llm = CapturingLLM()
            session = Session(
                config=Config.from_env(workspace=tmp, api_key="fake", model="fake"),
                llm=llm,
            )

            session.prompt("bad\udce4input")

            self.assertEqual(llm.messages[0]["content"], "bad\ufffdinput")


if __name__ == "__main__":
    unittest.main()
