import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codingclaw.llm.client import OpenAICompatibleClient
from codingclaw.session.session_store import SessionStore
from codingclaw.unicode import sanitize_json_value, sanitize_text


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode("utf-8")


class UnicodeSafetyTest(unittest.TestCase):
    def test_sanitize_json_value_replaces_nested_surrogates(self):
        value = {"message": ["ok", {"content": "bad\udce4text"}]}

        sanitized = sanitize_json_value(value)

        self.assertEqual(sanitized["message"][1]["content"], "bad\ufffdtext")
        self.assertEqual(sanitize_text("正常中文"), "正常中文")

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

            messages = store.load_messages()

            self.assertEqual(messages[0]["content"], "bad\ufffdtext")

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


if __name__ == "__main__":
    unittest.main()
