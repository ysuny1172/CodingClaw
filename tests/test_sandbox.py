import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.errors import SandboxError
from codingclaw.sandbox import SandboxPolicy


class SandboxTest(unittest.TestCase):
    def test_rejects_path_escape(self):
        with TemporaryDirectory() as tmp:
            policy = SandboxPolicy(Path(tmp))
            with self.assertRaises(SandboxError):
                policy.paths.resolve_read_path("../outside.txt")

    def test_rejects_dangerous_command(self):
        with TemporaryDirectory() as tmp:
            policy = SandboxPolicy(Path(tmp))
            with self.assertRaises(SandboxError):
                policy.commands.assert_allowed("git reset --hard HEAD")


if __name__ == "__main__":
    unittest.main()
