# CodingClaw

CodingClaw is a minimal Python coding-agent harness inspired by Pi's layered architecture:

- `Session` owns coding-agent concerns: context, skills, tools, session history, and traces.
- `Agent` stays small: it runs the LLM/tool loop and emits events.
- Tools are OpenAI-compatible function tools.

Built-in tools:

```text
list_files   List files and directories under the workspace.
read_file    Read a UTF-8 text file.
write_file   Write a full UTF-8 text file.
edit_file    Replace an exact text segment in a UTF-8 text file.
run_command  Run an allowlisted shell command in the workspace.
```

## Install

```powershell
pip install -e .
```

## Configure

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

## Run

Single task mode:

```powershell
codingclaw "List the files in this project and explain what this repo does."
```

Interactive mode:

```powershell
codingclaw
```

Run an initial task, then keep chatting in the same session:

```powershell
codingclaw -i "List the files in this project."
```

Resume the latest session for the current workspace:

```powershell
codingclaw --continue
```

Resume a specific session file:

```powershell
codingclaw --session .codingclaw/sessions/20260531T120000Z_abc123.jsonl
```

Session history is written to `.codingclaw/sessions/`.
Trace logs are written to `.codingclaw/traces/`.

Interactive commands:

```text
/help     Show commands.
/session  Show current session and trace files.
/exit     Exit interactive mode.
/quit     Exit interactive mode.
```

## Skills

CodingClaw discovers skill metadata from:

```text
.codingclaw/
  skills/
    code-review/
      SKILL.md
```

Only the skill name and description are inserted into the system prompt. The full `SKILL.md` remains available for the model to read with `read_file`.

Example:

```markdown
---
name: code-review
description: Review Python code for correctness, maintainability, and missing tests.
---

# Code Review

Use this workflow when the user asks for a code review.
```

## Tests

```powershell
python -m unittest discover -s tests
```
