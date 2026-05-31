import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from codingclaw.session.skills import SkillLoader


class SkillsTest(unittest.TestCase):
    def test_loads_skill_metadata_only(self):
        with TemporaryDirectory() as tmp:
            skill_dir = Path(tmp, ".codingclaw", "skills", "code-review")
            skill_dir.mkdir(parents=True)
            Path(skill_dir, "SKILL.md").write_text(
                "---\nname: code-review\ndescription: Review code carefully.\n---\n\nSECRET BODY",
                encoding="utf-8",
            )

            skills, diagnostics = SkillLoader(Path(tmp)).load()

            self.assertEqual(diagnostics, [])
            self.assertEqual(skills[0].name, "code-review")
            self.assertEqual(skills[0].description, "Review code carefully.")


if __name__ == "__main__":
    unittest.main()
