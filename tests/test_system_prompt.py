import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.session.resources import LoadedResources
from codingclaw.session.system_prompt import build_system_prompt
from codingclaw.tools import ToolContext, ToolRegistry


class SystemPromptTest(unittest.TestCase):
    def test_includes_command_environment_safety_guidelines(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = build_system_prompt(
                workspace_root=workspace,
                tools=ToolRegistry(ToolContext(workspace)),
                resources=LoadedResources(),
            )

        self.assertIn("Shell commands start there but are not filesystem-sandboxed", prompt)
        self.assertIn("confirm which interpreter or runtime", prompt)
        self.assertIn("Never install dependencies into CodingClaw's own runtime", prompt)
        self.assertIn("avoid repeated dependency experiments", prompt)

    def test_includes_baseline_change_regression_workflow(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = build_system_prompt(
                workspace_root=workspace,
                tools=ToolRegistry(ToolContext(workspace)),
                resources=LoadedResources(),
            )

        baseline = prompt.index("run the smallest relevant test command before editing")
        change = prompt.index("Make the smallest source change")
        regression = prompt.index("run focused tests")

        self.assertLess(baseline, change)
        self.assertLess(change, regression)
        self.assertIn("pre-existing baseline failures", prompt)
        self.assertIn("environment failures", prompt)
        self.assertIn("regressions caused by the change", prompt)
        self.assertIn("inspect the final diff", prompt)

    def test_protects_existing_tests_without_forbidding_valid_updates(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = build_system_prompt(
                workspace_root=workspace,
                tools=ToolRegistry(ToolContext(workspace)),
                resources=LoadedResources(),
            )

        self.assertIn("Treat existing tests as protected project behavior", prompt)
        self.assertIn("Do not delete, skip, weaken, or rewrite them merely", prompt)
        self.assertIn("Modify an existing test only when", prompt)
        self.assertIn("the task explicitly changes the intended behavior", prompt)
        self.assertIn("Do not add broad skips, xfails", prompt)
        self.assertIn("Never use test changes to work around dependency", prompt)

    def test_safety_guidelines_are_appended_to_custom_system_prompt(self):
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            prompt = build_system_prompt(
                workspace_root=workspace,
                tools=ToolRegistry(ToolContext(workspace)),
                resources=LoadedResources(system_prompt="Custom project instructions."),
            )

        self.assertTrue(prompt.startswith("Custom project instructions."))
        self.assertIn("Never install dependencies into CodingClaw's own runtime", prompt)


if __name__ == "__main__":
    unittest.main()
