import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.command_tools import RunCommandTool
from codingclaw.tools.file_tools import EditFileTool, ReadFileTool, WriteFileTool


class ToolsTest(unittest.TestCase):
    def test_write_and_read_file(self):
        with TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(Path(tmp)))
            registry.register(WriteFileTool())
            registry.register(ReadFileTool())

            write = registry.execute("write_file", {"path": "a/b.txt", "content": "hello"})
            read = registry.execute("read_file", {"path": "a/b.txt"})

            self.assertTrue(write.ok)
            self.assertTrue(read.ok)
            self.assertEqual(read.data["content"], "hello")

    def test_missing_required_argument(self):
        with TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(Path(tmp)))
            registry.register(ReadFileTool())
            result = registry.execute("read_file", {})

            self.assertFalse(result.ok)
            self.assertIn("Missing required argument", json.dumps(result.to_dict()))

    def test_edit_file_replaces_exact_text_once(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "note.txt")
            path.write_text("hello old world", encoding="utf-8")
            registry = ToolRegistry(ToolContext(Path(tmp)))
            registry.register(EditFileTool())

            result = registry.execute(
                "edit_file",
                {"path": "note.txt", "old_text": "old", "new_text": "new"},
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["replacements"], 1)
            self.assertEqual(path.read_text(encoding="utf-8"), "hello new world")

    def test_edit_file_rejects_unexpected_replacement_count(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "note.txt")
            path.write_text("same same", encoding="utf-8")
            registry = ToolRegistry(ToolContext(Path(tmp)))
            registry.register(EditFileTool())

            result = registry.execute(
                "edit_file",
                {"path": "note.txt", "old_text": "same", "new_text": "changed"},
            )

            self.assertFalse(result.ok)
            self.assertIn("ReplacementCountMismatch", json.dumps(result.to_dict()))
            self.assertEqual(path.read_text(encoding="utf-8"), "same same")

    def test_run_command_replaces_invalid_utf8_bytes(self):
        with TemporaryDirectory() as tmp:
            completed = subprocess.CompletedProcess(
                args="echo test",
                returncode=0,
                stdout=b"valid\xe4invalid",
                stderr=b"error\xff",
            )
            with patch("codingclaw.tools.command_tools.subprocess.run", return_value=completed):
                result = RunCommandTool().execute({"command": "echo test"}, ToolContext(Path(tmp)))

            self.assertTrue(result.ok)
            self.assertEqual(result.data["stdout"], "valid\ufffdinvalid")
            self.assertEqual(result.data["stderr"], "error\ufffd")


if __name__ == "__main__":
    unittest.main()
