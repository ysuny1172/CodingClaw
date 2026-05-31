# CodingClaw

CodingClaw is a minimal Python coding-agent harness inspired by Pi's layered architecture:

- `Session` owns coding-agent concerns: context, skills, tools, session history, and traces.
- `Agent` stays small: it runs the LLM/tool loop and emits events.
- Tools are OpenAI-compatible function tools.

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

```powershell
codingclaw "List the files in this project and explain what this repo does."
```

Session history is written to `.codingclaw/sessions/`.
Trace logs are written to `.codingclaw/traces/`.

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
