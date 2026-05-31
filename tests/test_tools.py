import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.tools import ToolContext, ToolRegistry
from codingclaw.tools.file_tools import ReadFileTool, WriteFileTool


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


if __name__ == "__main__":
    unittest.main()
