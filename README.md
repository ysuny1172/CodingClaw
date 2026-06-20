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
$env:CODINGCLAW_CONTEXT_WINDOW="128000"
$env:CODINGCLAW_RESERVE_TOKENS="16384"
$env:CODINGCLAW_KEEP_RECENT_TOKENS="20000"
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

The prompt shows current context usage. Provider usage is preferred when available; otherwise CodingClaw shows a local estimate:

```text
claw [~1,234/128,000 tokens estimate]>
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
Both paths are relative to the active workspace. Interactive startup and `/session` print the resolved workspace and absolute files.

Interactive commands:

```text
/help     Show commands.
/session  Show current session, trace files, and token usage.
/compact [instructions]
          Compact the current session context, optionally with summary guidance.
/exit     Exit interactive mode.
/quit     Exit interactive mode.
```

Token usage display only reports token counts. It does not calculate cost. When the model returns OpenAI-compatible `usage`, CodingClaw records the latest request's prompt, completion, and total tokens. After compaction, stale provider usage is discarded and the prompt falls back to an estimate until the next model response.

## Context Compaction

CodingClaw can compact long sessions by summarizing older messages and keeping recent context. Auto-compaction checks the projected context before a request and the active context after assistant responses when usage exceeds `context_window - reserve_tokens`. If a model request fails with a context-limit error, CodingClaw compacts and retries that request once.

Configure compaction with CLI flags:

```powershell
codingclaw --context-window 128000 --reserve-tokens 16384 --keep-recent-tokens 20000
codingclaw --no-auto-compact
```

Manual compaction supports optional instructions:

```text
/compact focus on open decisions and modified files
```

Compaction appends a Pi-like `compaction` entry to the session JSONL with `summary`, `first_kept_entry_id`, `tokens_before`, `reason`, and `details`. Resume rebuilds context as summary plus kept messages; older session files using `first_kept_message_id` still load normally. Default details track cumulative `read_files` and `modified_files`, and summaries include `<read-files>` / `<modified-files>` blocks when applicable.

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

## Events and Hooks

CodingClaw uses events for lifecycle observation and hooks for pre-action decisions.

Core events include:

```text
agent_start
turn_start
message_start
message_end
llm_request
llm_response
tool_execution_start
tool_execution_end
turn_end
agent_end
```

`Session.subscribe(listener)` lets callers observe events. Tool hooks use `session.hooks.before_tool_call(...)`, which can allow, block, or rewrite a tool call before the tool executes. Context hooks use `session.hooks.register_before_compaction(...)`, which can block compaction, add details, or provide a summary override.

## Tests

```powershell
python -m unittest discover -s tests
```

## SWE-bench Pass@1 Runner

Run the SWE-bench runner inside WSL from the SWE-bench virtual environment. CodingClaw is
invoked through its own CLI, so the runner does not depend on CodingClaw's internal Python APIs.

Generate a prediction for one instance:

```bash
source ~/swebench-learning/SWE-bench/.venv/bin/activate

python ~/codingclaw/scripts/run_swebench.py generate \
  --run-id codingclaw-flask-4045-v1 \
  --instance-id pallets__flask-4045 \
  --codingclaw-executable ~/codingclaw/.venv/bin/codingclaw
```

Evaluate the saved prediction without calling the model again:

```bash
python ~/codingclaw/scripts/run_swebench.py evaluate \
  --run-id codingclaw-flask-4045-v1 \
  --swebench-root ~/swebench-learning/SWE-bench
```

Generate and evaluate the fixed five-instance sample:

```bash
python ~/codingclaw/scripts/run_swebench.py all \
  --run-id codingclaw-lite-5-v1 \
  --instances-file ~/codingclaw/evals/swebench_lite_random_5.json \
  --codingclaw-executable ~/codingclaw/.venv/bin/codingclaw \
  --swebench-root ~/swebench-learning/SWE-bench
```

Use `--resume` with `generate` or `all` to skip instances that already produced a pass@1
prediction. Run artifacts are written to `evals/swebench_runs/<run-id>/`.
