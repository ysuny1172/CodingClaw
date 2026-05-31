from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .skills import ResourceDiagnostic, Skill, SkillLoader


@dataclass(frozen=True)
class ContextFile:
    path: Path
    content: str


@dataclass
class LoadedResources:
    system_prompt: str | None = None
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    diagnostics: list[ResourceDiagnostic] = field(default_factory=list)


class ResourceLoader:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root

    def load(self) -> LoadedResources:
        resources = LoadedResources()
        system_file = self.workspace_root / ".codingclaw" / "SYSTEM.md"
        if system_file.exists():
            try:
                resources.system_prompt = system_file.read_text(encoding="utf-8")
            except OSError as error:
                resources.diagnostics.append(ResourceDiagnostic("warning", str(error), str(system_file)))

        agents_file = self.workspace_root / "AGENTS.md"
        if agents_file.exists():
            try:
                resources.context_files.append(ContextFile(path=agents_file, content=agents_file.read_text(encoding="utf-8")))
            except OSError as error:
                resources.diagnostics.append(ResourceDiagnostic("warning", str(error), str(agents_file)))

        skills, diagnostics = SkillLoader(self.workspace_root).load()
        resources.skills = skills
        resources.diagnostics.extend(diagnostics)
        return resources
