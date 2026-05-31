import argparse
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.agent.types import AssistantResponse
from codingclaw.cli import build_parser, run_interactive, should_run_interactive
from codingclaw.config import Config
from codingclaw.session import Session


class HistoryAwareFakeLLM:
    def __init__(self):
        self.message_counts = []

    def chat(self, *, model, system_prompt, messages, tools):
        self.message_counts.append(len(messages))
        return AssistantResponse(content=f"messages={len(messages)}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    return build_parser().parse_args(argv)


class CliTest(unittest.TestCase):
    def test_no_task_enters_interactive_mode(self):
        args = parse_args([])

        self.assertTrue(should_run_interactive(args))

    def test_task_defaults_to_print_mode(self):
        args = parse_args(["hello"])

        self.assertFalse(should_run_interactive(args))

    def test_task_with_interactive_stays_interactive(self):
        args = parse_args(["--interactive", "hello"])

        self.assertTrue(should_run_interactive(args))

    def test_interactive_reuses_one_session_history(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            llm = HistoryAwareFakeLLM()
            session = Session(config=config, llm=llm)
            inputs = iter(["first", "second", "/session", "/exit"])
            output = io.StringIO()
            errors = io.StringIO()

            result = run_interactive(
                session,
                input_fn=lambda _prompt: next(inputs),
                output=output,
                error_output=errors,
            )

            self.assertEqual(result, 0)
            self.assertEqual(llm.message_counts, [1, 3])
            self.assertIn("messages=1", output.getvalue())
            self.assertIn("messages=3", output.getvalue())
            self.assertIn(str(session.store.path), output.getvalue())
            self.assertEqual(errors.getvalue(), "")

    def test_interactive_runs_initial_task_before_loop(self):
        with TemporaryDirectory() as tmp:
            config = Config.from_env(workspace=tmp, api_key="fake", model="fake")
            llm = HistoryAwareFakeLLM()
            session = Session(config=config, llm=llm)
            output = io.StringIO()

            result = run_interactive(
                session,
                initial_task="initial",
                input_fn=lambda _prompt: "/exit",
                output=output,
                error_output=io.StringIO(),
            )

            self.assertEqual(result, 0)
            self.assertEqual(llm.message_counts, [1])
            self.assertIn("messages=1", output.getvalue())
            self.assertTrue(Path(session.store.path).exists())


if __name__ == "__main__":
    unittest.main()
