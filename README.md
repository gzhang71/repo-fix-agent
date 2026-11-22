# repo-fix-agent

A bug-fixing agent you can plug onto any repo, plus a pipeline that distills a strong teacher
(~60%) and 10k engineer demos into a deployable student (~30% today).

- **[docs/DESIGN.md](docs/DESIGN.md)**: problem setup, data schema, training plan
- **[docs/DEMO_DATA_PROPOSALS.md](docs/DEMO_DATA_PROPOSALS.md)**: proposals for using the 10k demos to improve the student
- **[configs/models.yaml](configs/models.yaml)**: student/teacher candidates (Claude, ChatGPT/GPT, open-weight) and student serving config

## Quickstart

```bash
pip install -e '.[teacher,openai]'        # optional: embed (semantic retrieval/dedup), secrets
docker build -f docker/sandbox.Dockerfile -t repo-fix-sandbox:py3.12 docker   # production sandbox
python -m pytest -q                       # docker tests skip without docker + image

# Build + validate the 100-instance seed dataset (gold patch must flip hidden tests)
python scripts/build_seed_dataset.py
repo-fix validate data/demos/seed.jsonl           # add --sandbox docker|local before the command to force a backend

# Export chat-format SFT data
repo-fix export-sft data/demos/seed.jsonl runs/sft.jsonl

# Run a teacher or the student on the seed set / on your own repo
repo-fix eval data/demos/seed.jsonl --model claude-opus-5-5
repo-fix eval data/demos/seed.jsonl --model gpt-5
repo-fix eval data/demos/seed.jsonl --model student   # vLLM/SGLang server from configs/models.yaml
repo-fix fix --repo ../some-repo --issue-file issue.md --test-cmd "python -m pytest -q"
```

## Data pipeline

Each proposal in `docs/DEMO_DATA_PROPOSALS.md` is a `repo-fix` command: `audit`, `split`,
`eval-set`, `distill`, `mix`, `sft-mix`, `aux-tasks`, `prefs`, `verifier-train`, `best-of-n`,
`rl-rollouts`, `retrieval-index`, `retrieve`, `augment`. See §7 of that doc for the typical flow.

## Layout

```
src/repo_fix_agent/
  tools.py      shared action space: bash, view_file, str_replace, create_file, search,
                find_files, insert, undo_edit, run_tests, submit (validated, never raise)
  sandbox.py    LocalSandbox (dev) / DockerSandbox (prod: no network, mem/pid/cpu limits,
                image from docker/sandbox.Dockerfile)
  env.py        one episode = sandbox + tool state; diff / apply_patch for grading
  agent.py      agent loop + system prompt
  policies.py   ScriptedPolicy (demos), AnthropicPolicy + OpenAIPolicy (teachers),
                LocalModelPolicy (student on vLLM/SGLang, context budget from config)
  grader.py     execution-based grading (FAIL_TO_PASS / PASS_TO_PASS)
  schema.py     Demo / Step / Action records (JSONL)
  sft.py        demo -> chat-format SFT example
  llm.py        single-turn TextLLM interface (Claude / GPT) for pipeline steps
  pipeline/     one module per data proposal (audit, split, distill, sft_data, aux_tasks,
                preferences, verifier, rl, retrieval, augment) + shared helpers: embed
                (hash / sentence-transformers / Voyage / OpenAI embeddings), repro (executed
                reproductions for synthetic issues), parallel (ordered thread pool for rollouts)
  cli.py        repo-fix CLI
scripts/seed_specs/batch_*.py   seed instance specs (7 batches)
scripts/build_seed_dataset.py   specs -> repos + recorded trajectories -> data/demos/seed.jsonl
scripts/eval_retrieval_paraphrase.py   retrieval backends on paraphrased queries (top-1 / MRR)
docker/sandbox.Dockerfile       DockerSandbox image
data/demos/seed.jsonl           100 validated demos
data/repos/                     the 100 seed repos (buggy state)
```
