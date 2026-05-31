from __future__ import annotations

from pathlib import Path

from codingclaw.tools.registry import ToolRegistry
from .resources import LoadedResources


def build_system_prompt(*, workspace_root: Path, tools: ToolRegistry, resources: LoadedResources) -> str:
    if resources.system_prompt:
        prompt = resources.system_prompt.strip()
    else:
        prompt = (
            "You are CodingClaw, a minimal coding agent harness. "
            "Help the user by inspecting files, running safe commands, and editing files when useful."
        )

    prompt += "\n\nAvailable tools:\n"
    for schema in tools.openai_schemas():
        function = schema["function"]
        prompt += f"- {function['name']}: {function['description']}\n"

    prompt += "\nGuidelines:\n"
    prompt += "- Prefer reading relevant files before changing them.\n"
    prompt += "- Keep changes focused on the user's task.\n"
    prompt += "- Use workspace-relative paths when calling tools.\n"
    prompt += "- If a tool fails, use the error message to correct the next step.\n"

    if resources.context_files:
        prompt += "\nProject Context:\n"
        for context_file in resources.context_files:
            rel = context_file.path.relative_to(workspace_root)
            prompt += f"\n## {rel}\n{context_file.content.strip()}\n"

    if resources.skills:
        prompt += "\nAvailable Skills:\n"
        prompt += "Only skill metadata is shown here. Read the skill's SKILL.md when the task matches.\n"
        for skill in resources.skills:
            rel = skill.file_path.relative_to(workspace_root)
            prompt += f'- <skill name="{skill.name}" path="{rel}">{skill.description}</skill>\n'

    prompt += f"\nCurrent working directory: {workspace_root}"
    return prompt
