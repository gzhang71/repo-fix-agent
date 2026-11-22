# repo-fix-agent

A bug-fixing agent you can plug onto any repo, plus a pipeline that distills a strong teacher
(~60%) and 10k engineer demos into a deployable student (~30% today).

- **[docs/DESIGN.md](docs/DESIGN.md)**: problem setup, data schema, training plan
- **[docs/DEMO_DATA_PROPOSALS.md](docs/DEMO_DATA_PROPOSALS.md)**: proposals for using the 10k demos to improve the student
- **[configs/models.yaml](configs/models.yaml)**: student/teacher candidates (Claude, ChatGPT/GPT, open-weight) and student serving config

## Quickstart

```bash
pip install -e '.[teacher,openai]'
python -m pytest -q                       # tools, sandbox, policies (docker tests skip without docker)

# Build + validate the 10-demo seed dataset (gold patch must flip hidden tests)
python scripts/build_seed_dataset.py
repo-fix validate data/demos/seed.jsonl

# Export chat-format SFT data
repo-fix export-sft data/demos/seed.jsonl runs/sft.jsonl

# Run a teacher or the student on the seed set / on your own repo
repo-fix eval data/demos/seed.jsonl --model claude-opus-5-5
repo-fix eval data/demos/seed.jsonl --model gpt-5
repo-fix eval data/demos/seed.jsonl --model student   # vLLM/SGLang server from configs/models.yaml
repo-fix fix --repo ../some-repo --issue-file issue.md --test-cmd "python -m pytest -q"
```

## Layout

```
src/repo_fix_agent/
  tools.py      shared action space: bash, view_file, str_replace, create_file, search,
                find_files, insert, undo_edit, run_tests, submit (validated, never raise)
  sandbox.py    LocalSandbox (dev) / DockerSandbox (prod: no network, mem/pid/cpu limits)
  env.py        one episode = sandbox + tool state; diff / apply_patch for grading
  agent.py      agent loop + system prompt
  policies.py   ScriptedPolicy (demos), AnthropicPolicy + OpenAIPolicy (teachers),
                LocalModelPolicy (student on vLLM/SGLang, context budget from config)
  grader.py     execution-based grading (FAIL_TO_PASS / PASS_TO_PASS)
  schema.py     Demo / Step / Action records (JSONL)
  sft.py        demo -> chat-format SFT example
  cli.py        repo-fix CLI
scripts/build_seed_dataset.py   seed repos + engineer action scripts -> recorded trajectories
data/demos/seed.jsonl           10 validated demos
data/repos/                     the 10 seed repos (buggy state)
```
