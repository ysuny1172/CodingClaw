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
    prompt += (
        "- Treat the workspace as the project boundary. Shell commands start there but are not filesystem-sandboxed; "
        "do not intentionally read or modify paths outside it unless the user explicitly asks.\n"
    )
    prompt += (
        "- Before installing dependencies or running a language package manager, inspect the project's environment "
        "and dependency files and confirm which interpreter or runtime the command will use.\n"
    )
    prompt += (
        "- Never install dependencies into CodingClaw's own runtime or an unrelated global environment. "
        "Prefer the project's existing environment and project-declared test commands.\n"
    )
    prompt += "- If a tool fails, use the error message to correct the next step.\n"
    prompt += "\nChange and verification workflow:\n"
    prompt += (
        "- Before editing, inspect the relevant source, tests, project configuration, and current worktree changes. "
        "Preserve unrelated user changes.\n"
    )
    prompt += (
        "- When practical, run the smallest relevant test command before editing to establish a baseline. "
        "Record whether a failure already existed and whether it is caused by the project environment.\n"
    )
    prompt += (
        "- Make the smallest source change that addresses the task, then run focused tests followed by the broader "
        "relevant test suite when feasible.\n"
    )
    prompt += (
        "- For bug fixes, reproduce the reported behavior before editing when feasible and add or update a focused "
        "regression test that fails before the fix and passes after it.\n"
    )
    prompt += (
        "- Compare post-change failures with the baseline. Classify failures as pre-existing baseline failures, "
        "environment failures, or regressions caused by the change; do not treat them as interchangeable.\n"
    )
    prompt += (
        "- If the declared project environment cannot run, avoid repeated dependency experiments. Build the closest "
        "safe executable reproduction you can, and clearly report what could and could not be verified.\n"
    )
    prompt += (
        "- Before finishing, inspect the final diff and report the files changed, tests run, results, and any "
        "remaining verification limitations.\n"
    )
    prompt += "\nBehavioral verification:\n"
    prompt += (
        "- Derive the externally observable behavior the task requires before choosing an implementation. Preserve "
        "related parsing, rendering, indexing, serialization, cross-reference, and public API semantics.\n"
    )
    prompt += (
        "- Verify the public output or structure users depend on, such as a parsed tree, generated artifact, rendered "
        "text, stored value, protocol response, or command result. A helper-level check alone is not sufficient.\n"
    )
    prompt += (
        "- Do not mistake suppressing an exception, warning, or validation error for a complete fix. Confirm that the "
        "accepted input is interpreted and represented correctly after the change.\n"
    )
    prompt += (
        "- Before concluding that a fix is correct, test the reported example and at least one nearby existing case "
        "that could regress. Inspect exact assertions and node or output boundaries, not only whether code executes.\n"
    )
    prompt += "\nTest integrity:\n"
    prompt += (
        "- Treat existing tests as protected project behavior. Do not delete, skip, weaken, or rewrite them merely "
        "to make a failing implementation or incompatible environment appear successful.\n"
    )
    prompt += (
        "- Prefer fixing production code and adding a focused regression test. Modify an existing test only when "
        "the task explicitly changes the intended behavior or the test is demonstrably incorrect.\n"
    )
    prompt += (
        "- When an existing test must change, keep the change minimal and explain which requirement changed and why "
        "the previous expectation is no longer valid.\n"
    )
    prompt += (
        "- Do not add broad skips, xfails, exception swallowing, or looser assertions without explicit task "
        "justification. Never use test changes to work around dependency or interpreter incompatibility.\n"
    )

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
