from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    file_path: Path
    base_dir: Path


@dataclass(frozen=True)
class ResourceDiagnostic:
    type: str
    message: str
    path: str | None = None


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    values: dict[str, str] = {}
    end_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip("\"'")
    if end_index is None:
        return {}, text
    body = "\n".join(lines[end_index + 1 :])
    return values, body


class SkillLoader:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def load(self) -> tuple[list[Skill], list[ResourceDiagnostic]]:
        skills_dir = self.workspace_root / ".codingclaw" / "skills"
        skills: list[Skill] = []
        diagnostics: list[ResourceDiagnostic] = []
        if not skills_dir.exists():
            return skills, diagnostics

        for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
            try:
                text = skill_file.read_text(encoding="utf-8")
                frontmatter, _body = parse_frontmatter(text)
                name = frontmatter.get("name") or skill_file.parent.name
                description = frontmatter.get("description")
                if not description:
                    diagnostics.append(
                        ResourceDiagnostic(
                            type="warning",
                            message="Skill missing required description; skipped",
                            path=str(skill_file),
                        )
                    )
                    continue
                skills.append(Skill(name=name, description=description, file_path=skill_file, base_dir=skill_file.parent))
            except OSError as error:
                diagnostics.append(ResourceDiagnostic(type="warning", message=str(error), path=str(skill_file)))
        return skills, diagnostics
