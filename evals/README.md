# CodingClaw Eval Workflow

这套 eval 的目标不是一次性证明模型“很聪明”，而是防止你每次改 prompt、工具、模型、sandbox 后，把已经做对的行为改坏。

## 文件

- `eval_cases.json`: 20 个固定回归用例。
- `context_cases.json`: 上下文压缩和 Agent 记忆用例。
- `../scripts/run_evals.py`: 本地最小 runner。
- `../scripts/run_context_evals.py`: 多轮上下文 eval runner，会显示压缩节省的 token。
- `runs/`: 每次运行后的 JSONL 报告目录，默认不提交。
- `context_runs/`: 每次上下文 eval 的 JSONL 报告目录，默认不提交。

## 实际开发流程

1. 先改代码、prompt、工具或模型配置。
2. 确认本次 shell 里有模型配置：

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

3. 跑单元测试：

```powershell
python -m unittest discover -s tests
```

4. 跑 eval 回归：

```powershell
python scripts/run_evals.py
```

5. 如果只想调一个 case：

```powershell
python scripts/run_evals.py --case-id tool_missing_file_001 --keep-workspaces
```

6. 跑上下文压缩 eval：

```powershell
python scripts/run_context_evals.py --case-id memory_fact_retention_001 --keep-workspaces
```

手动压缩 case 适合做冒烟测试。自动压缩 case 会要求真实触发 `threshold` 压缩：

```powershell
python scripts/run_context_evals.py --case-id auto_memory_fact_retention_001 --keep-workspaces
```

普通 eval 汇总会显示 `Pass rate`。上下文 eval 还会显示 `Total tokens saved` 和 `Average savings rate`。自动压缩 case 还会检查 `expected_compaction_reason`、`min_compactions` 和 `min_tokens_before`。

7. 打开 `evals/runs/*.jsonl` 或 `evals/context_runs/*.jsonl` 看失败详情：

- `stdout`: agent 的最终答案。
- `tool_calls`: trace 里实际调用过的工具。
- `failures`: 自动检查发现的问题。
- `manual_judge`: 人工复核时要看的重点。

## 怎么判通过

每个 case 分两层判断：

- 自动检查：关键词、禁用词、JSON 格式、工具是否调用。
- 人工抽检：答案是否真的符合 `manual_judge`。

早期不要追求全自动。先让自动检查抓明显退化，再人工看少量失败样本，最省时间。

## 常见问题

如果看到：

```text
OPENAI_API_KEY is required for the real LLM client.
```

说明当前 PowerShell 会话没有模型 API key。先设置 `$env:OPENAI_API_KEY`，再重新运行 eval。

## 怎么新增失败案例

每次真实使用时发现 bug，就按这个格式加入 `eval_cases.json`：

```json
{
  "id": "new_bug_001",
  "category": "tool_failure",
  "purpose": "一句话说明这个 case 防什么退化。",
  "user_input": "用户当时的原始请求",
  "setup_files": {
    "example.txt": "需要提前放进临时工作区的内容"
  },
  "expected_behavior": [
    "应该做什么",
    "不应该做什么"
  ],
  "expected_tools": ["read_file"],
  "forbidden_tools": ["write_file"],
  "must_include_any": [["必须出现的词 A", "或同义词 B"]],
  "must_not_include": ["禁止出现的词"],
  "manual_judge": "人工复核通过条件。"
}
```

新增原则：

- 一个 case 只测一个主要能力。
- 真实失败案例优先于脑补案例。
- 不要求标准答案完全一致，只约束关键行为。
- 危险操作、写文件、跑命令都必须用 `setup_files` 创建临时工作区。

## 什么时候接平台

当你已经有 50 到 100 个 case，并且开始关心跨版本趋势、多人标注、LLM judge、trace 检索时，再接 LangSmith、Langfuse 或 Braintrust。

现在这个阶段，固定测试集加 JSONL 报告就够用。
