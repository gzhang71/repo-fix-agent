# repo-fix-agent

A bug-fixing agent you can plug onto any repository, plus the tooling to make a small, cheap
**student** model as good as a strong **teacher**: evaluation, data pipelines built on 10k engineer
demos, teacher distillation (off- and on-policy), preference learning, and RL.

| Doc | What's in it |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | every attempt so far: what was tried, what was measured, what is implemented but not yet run |
| [docs/DESIGN.md](docs/DESIGN.md) | problem setup, data schema, sandbox, grading |
| [docs/BASELINE.md](docs/BASELINE.md) | measured student vs teacher accuracy on the 100 seed demos and failure analysis |
| [docs/DEMO_DATA_PROPOSALS.md](docs/DEMO_DATA_PROPOSALS.md) | proposals for using the engineer demos, with implementation status |
| [configs/models.yaml](configs/models.yaml) | student / teacher candidates, student serving + scaffold options |

## Current status

| | Model | Seed set (100) |
|---|---|---|
| Teacher | `claude-opus-5-5` | **98%** resolved, 5.5 steps, about $0.04 per instance |
| Student | `nemotron-lightning-3p5-30b-a3b` (Fireworks, 30B MoE, ~3B active) | **85%** resolved, 16.6 steps |

Student failures are mostly fixes that cover only the issue's example and miss the rest of the
behaviour, not tool misuse (see BASELINE.md). The improvements in CHANGELOG "Unreleased" target
that and are implemented and tested offline, **but not yet run against the paid APIs**.
The seed set is close to its ceiling. Add harder instances before relying on small gains.

## Setup

```bash
pip install -e '.[teacher,openai]'      # optional extras: embed, voyage, secrets
cp .env.example .env                    # fill in ANTHROPIC_API_KEY, FIREWORKS_API_KEY (gitignored)
docker build -f docker/sandbox.Dockerfile -t repo-fix-sandbox:py3.12 docker   # the sandbox
python -m pytest -q                     # 424 tests, offline (no API calls)
```

* Keys load from `.env` automatically; variables already set in your shell win.
* **Sandbox:** the agent and the hidden tests run in Docker with no network. `--sandbox local`
  (before the subcommand) is for development only.
* **API cost control:**
  * Every LLM and embedding request is recorded with its response in `.cache/llm/` and replayed
    on reruns. Repeated *sampled* requests still get distinct answers the first time.
  * `repo-fix llm-cache stats` shows what was saved. Set `REPOFIX_LLM_CACHE=off` to resample.
  * Teacher calls use Anthropic prompt caching.
  * Paid pipeline stages refuse to run without `--yes`.

## Use the agent

```bash
repo-fix fix --repo ../some-repo --issue-file issue.md --test-cmd "python -m pytest -q" [--apply]
repo-fix serve        # web UI: paste a repo URL -> verified fixes -> one PR per fix
```

For `serve`:
1. **Clone:** it uses your `gh` login (or `GITHUB_TOKEN`).
2. **Environment:** it installs dependencies once, with network access, into a sandbox image.
3. **Find bugs:** it groups failing tests (and optionally issues labeled `bug`) into bugs.
4. **Fix and verify:** it fixes each bug in an offline container. A fix counts only if its tests
   now pass and nothing that passed before breaks.
5. **Open PRs:** it opens a PR per fix after you review it, or automatically for verified fixes.
   If your token can't push to the repo, it pushes to a fork.

The server listens on localhost only. Anyone who can reach it can push with your credentials.

## Evaluate

```bash
repo-fix validate data/demos/seed.jsonl                       # gold patches flip hidden tests (free)
repo-fix eval data/demos/seed.jsonl --model claude-opus-5-5 --workers 6 --out runs/baseline/teacher.jsonl
repo-fix eval data/demos/seed.jsonl --model student --workers 8 --out runs/baseline/student.jsonl
# noise-aware: mean pass@1 over 3 samples, plus pass@3 (the best-of-n ceiling)
repo-fix eval runs/splits/dev.jsonl --model student --samples 3 --temperature 0.7
# best-of-n with the execution reranker (reports chosen vs oracle accuracy)
repo-fix eval runs/splits/dev.jsonl --model student --best-of 4 --temperature 0.7
```

Each run writes per-instance JSONL plus a `.summary.json`: accuracy by difficulty, steps, token
usage and estimated cost. `--save-rollouts` keeps every episode for training.

## Improve the student

**Without training** (on by default for the student in `configs/models.yaml`):
- **`keep_reasoning`:** keeps the model's `reasoning_content` and sends it back each turn. Before
  this, it was dropped and the student rebuilt its plan every turn.
- **`spec_check`:** the system prompt asks it to list the behaviours and edge cases the issue
  implies and check them with `scratch_*.py` repro scripts, which never ship in the patch.
- **`submit_gate`:** one nudge if it submits after an edit it never checked.
- **Best-of-n with the execution reranker:** pools every candidate's repro scripts, runs each
  patch against all of them and the test suite, and picks the best. Selection costs no model calls.

**With training**, one round:

```bash
repo-fix split data/demos/seed.jsonl --out-dir runs/splits      # frozen test split, repo-disjoint
repo-fix student-train --round r1                              # prints every stage + cost; runs nothing
repo-fix student-train --round r1 --run augment                # free stages run directly
repo-fix student-train --round r1 --run distill --yes          # paid stages need --yes
```

The stages, in order:
1. **`augment`:** synthetic bug-injected instances, tagged when their hidden tests go beyond the
   issue's example.
2. **`student-rollouts`:** 4 student episodes per instance, kept for the later stages.
3. **`weights`** (free): each instance's student pass rate p sets its bucket and training weight
   (see below).
4. **`distill`:** rejection-sampled teacher trajectories, recorded under the student's prompt. Only
   for instances with p ≤ 0.75.
5. **`onpolicy`:** the teacher relabels the states the student actually reached (DAgger), and
   also produces step-level DPO pairs. Also only for p ≤ 0.75.
6. **`aux`:** auxiliary training tasks.
7. **`prefs`:** preference pairs.
8. **`sft-mix`:** engineer, teacher, on-policy and the student's own successful runs, weighted by
   difficulty, with edge-case instances weighted 2x.
9. **`export`:** chat fine-tuning JSONL.
10. **`train-sft`, then `train-dpo`:** `scripts/train_lora.py`, on a GPU box.
11. **`deploy`:** Fireworks on-demand deployment.
12. **`eval`**
13. **`post-rollouts`, then `rl-weights`:** pass rates of the fine-tuned student, for the RL curriculum.
14. **`rl-rollouts`, then `train-rl`:** GRPO-style policy gradient with a KL penalty to the reference,
    on instances the student solves only sometimes.

**Difficulty weighting** (`repo-fix difficulty-weights`, no model calls). Training effort goes
where the student's pass rate p is low, not by which model solved an instance:

| Bucket | Student pass rate | Used for | SFT weight |
|---|---|---|---|
| gap | never (teacher solves or unknown) | teacher distillation, on-policy relabel | 1 − p (= 1) |
| learning | sometimes | RL (groups with both wins and losses), DPO pairs | 1 − p |
| mastered | always | a little replay so it isn't forgotten | 0.2 |
| unsolved | never, teacher fails too | skipped (no correct run to learn from) | 0 |

`sft-mix --instance-weights` applies the weights and caps the summed weight per instance at 3, so
an instance with many recorded runs can't dominate.

**Training runs outside this repo.** Fireworks' managed SFT/DPO stopped taking new jobs on
2026-11-09. The exported JSONL fits their Training API, or use the bundled
`scripts/train_lora.py` (transformers + PEFT; SFT, DPO and policy-gradient objectives) on a GPU.
The current student supports supervised LoRA on Fireworks, but not RL and not serverless LoRA,
so a fine-tuned version needs an on-demand deployment.

## Data pipeline (engineer demos)

One command per proposal in `docs/DEMO_DATA_PROPOSALS.md`:

| Group | Commands |
|---|---|
| Clean and split | `audit` (validation, flakiness, dedup incl. semantic, leakage, secrets/PII), `split`, `eval-set` |
| Training data | `distill`, `mix`, `sft-mix`, `aux-tasks`, `prefs`, `augment`, `onpolicy-distill`, `difficulty-weights` |
| Verification and RL | `verifier-train`, `best-of-n`, `rl-rollouts` |
| Retrieval | `retrieval-index`, `retrieve` (hybrid TF-IDF + embeddings) |
| Training | `export-chat-ft`, `fireworks-deploy`, `student-train` |

## Layout

```
src/repo_fix_agent/
  agent.py        agent loop, system prompt, AgentOptions (spec check, submit gate)
  tools.py        action space: bash, view_file, str_replace, create_file, search, find_files,
                  insert, undo_edit, run_tests, submit
  sandbox.py      DockerSandbox (no network, limits) / LocalSandbox (dev)
  env.py          episode = sandbox + tool state; patch excludes tests and scratch_* files
  policies.py     AnthropicPolicy (teacher, prompt caching), OpenAIPolicy, LocalModelPolicy
                  (student on any OpenAI-compatible server: Fireworks / vLLM / SGLang)
  grader.py       FAIL_TO_PASS / PASS_TO_PASS grading
  evaluation.py   eval: parallel, multi-sample, best-of-n, cost accounting, rollouts
  llm.py          single-turn TextLLM (Claude / GPT) for pipeline steps
  llm_cache.py    record/replay cache for paid calls
  envfile.py      .env loader
  sft.py          demo -> chat SFT example
  pipeline/       audit split distill sft_data aux_tasks preferences verifier rl retrieval augment
                  repro embed parallel rerank onpolicy training difficulty
  web/            `repo-fix serve`: jobs, GitHub forge, HTTP server, static UI
scripts/
  build_seed_dataset.py, seed_specs/     the 100 seed instances
  train_lora.py                          LoRA SFT / DPO / policy-gradient training
  eval_retrieval_paraphrase.py           retrieval backend comparison
docker/sandbox.Dockerfile                sandbox image
data/demos/seed.jsonl, data/repos/       100 validated demos + their repos
```
