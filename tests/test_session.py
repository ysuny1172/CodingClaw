import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent.types import AssistantResponse
from codingclaw.config import Config
from codingclaw.session import Session


class FakeLLM:
    def chat(self, *, model, system_prompt, messages, tools):
        self.system_prompt = system_prompt
        return AssistantResponse(content="done")


class SessionTest(unittest.TestCase):
    def test_session_persists_messages_and_trace(self):
        with TemporaryDirectory() as tmp:
            Path(tmp, "AGENTS.md").write_text("Project rule.", encoding="utf-8")
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            llm = FakeLLM()
            session = Session(config=config, llm=llm)

            result = session.prompt("hello")

            self.assertEqual(result, "done")
            self.assertIn("Project rule.", llm.system_prompt)
            session_lines = session.store.path.read_text(encoding="utf-8").splitlines()
            trace_lines = session.trace.path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(session_lines), 4)
            self.assertTrue(any(json.loads(line)["type"] == "llm_response" for line in trace_lines))


if __name__ == "__main__":
    unittest.main()
