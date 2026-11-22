# Repo Fix Agent: Problem Setup & Design

## 1. Problem statement

Build a bug-fixing agent that can be **plugged onto any repository**. Given an issue description,
it explores the repo, edits code, verifies, and outputs a patch.

| Asset | What we have |
|---|---|
| Student | Third-party pretrained model, **~30%** resolve rate |
| Teacher | Strong model, **~60%** resolve rate (candidates: Claude Opus 5.5, ChatGPT/GPT-5 family) |
| Human data | **10,000 engineer demos** (issue → actions → fix) |

**Goal:** a deployable student that closes as much of the 30% → 60% gap as possible, at a fraction
of the teacher's per-fix cost. **Stretch:** go beyond the teacher on our own repos' distribution,
using execution feedback.

**Success metric:** resolve rate = share of held-out instances where the agent's patch makes every
`FAIL_TO_PASS` test pass and keeps every `PASS_TO_PASS` test passing (`grader.grade`). Secondary
metrics: cost per resolved instance, median steps, patch-apply rate, regression rate.

## 2. System

```
issue + repo ──► RepoEnv (sandbox copy, baseline commit)
                    ▲            │ observation
         action     │            ▼
              Policy (student | teacher | scripted demo)
                    │
                 submit ──► git diff ──► grader (hidden tests) ──► resolved?
```

* **One action space for everyone** (`tools.py`): `bash`, `view_file`, `str_replace`, `create_file`,
  `search`, `find_files`, `insert`, `undo_edit`, `run_tests`, `submit`. Arguments are validated
  against the schema, and every failure becomes an `Error:` observation, so a bad call never crashes an episode.
  Engineer demos, teacher rollouts and student rollouts all share the schema, so they can be mixed
  freely as training data.
* **Plug onto a repo:** `repo-fix fix --repo /path/to/repo --issue-file issue.md [--apply]`.
  The agent works on a temp copy; the real checkout changes only with `--apply`.
* **Hidden tests:** `test_patch` is applied only at grading, after the candidate patch, so the
  agent can't see or edit the tests it's graded on. Edits under `tests/` are left out of the
  submitted patch.
* **Sandbox** (`sandbox.py`): `backend="docker"` runs each episode in its own container
  (`--network none`, memory/pid/CPU limits, `no-new-privileges`, host uid) from the image in
  `docker/sandbox.Dockerfile` (git + pytest baked in, since the container has no network);
  `backend="local"` is a temp-dir copy for development and is **not** a security boundary (no
  network isolation). `auto` picks docker when the daemon is up and the image exists, otherwise
  falls back with a warning; `--sandbox` / `REPOFIX_SANDBOX` force a backend. Timeouts kill the
  whole process group. All 100 seed instances validate inside the container.
* **Student serving:** `LocalModelPolicy` talks to a vLLM/SGLang OpenAI-compatible server with
  native tool calling. When the conversation exceeds the context window set in
  `configs/models.yaml`, the oldest tool outputs are elided first.

## 3. Data: `Demo` schema (`schema.py`)

| Field | Purpose |
|---|---|
| `instance_id`, `repo`, `base_commit` | Where the bug lives (local path or git URL + commit) |
| `problem_statement` | Issue text: the only task input the agent gets |
| `test_cmd`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS` | Execution-based grading |
| `patch` | Gold source fix |
| `trajectory[]` | `{thought, action{tool,args}, observation}` steps |
| `metadata` | `source` (engineer / teacher:<model> / student:<model>), bug type, difficulty, files changed |

### Converting the 10k engineer demos
1. Map each engineer's actions onto the 5 tools (shell history → `bash`, editor diffs → `str_replace`).
2. **Re-execute** them in `RepoEnv` to regenerate exact observations (as `scripts/build_seed_dataset.py`
   does). Raw logs drift from what the agent would actually see.
3. Validate with `repo-fix validate`. Drop demos whose gold patch doesn't flip the tests.
4. Write `thought`s where they're missing. You can have the teacher add rationales *conditioned on
   the known actions*, but never let it change the actions themselves.
5. Split **by repository** (train / dev / test ≈ 80 / 10 / 10) so eval measures generalization to
   new codebases, not memorized ones.

## 4. Training plan

| Stage | Data | Expected effect |
|---|---|---|
| 0. Baseline | Evaluate student candidates + teacher on dev/test with this harness | Confirms 30/60 in *our* scaffold; picks the student |
| 1. SFT on engineer demos | ~10k validated human trajectories | Learns tool format, repo navigation, verify-before-submit |
| 2. Teacher distillation | Teacher rollouts on train issues, k=4 samples, **keep only resolved** (rejection sampling) | Biggest jump: on-policy-style traces in the exact action space |
| 3. Hard-case mining | Instances the teacher fails but engineers solved: let the teacher retry with the gold patch as a hidden hint, re-verify | Covers the 40% the teacher can't do alone |
| 4. RL with execution reward | Student rollouts, reward = `grade().resolved` (+ small step penalty), GRPO/RLOO | Pushes past imitation, fixes student-specific failure modes |

Ablations to run: engineer-only vs teacher-only vs mixed SFT data; mixing ratio; effect of
including failed-but-informative trajectories in DPO pairs (resolved vs unresolved on the same issue).

**Licensing check:** before stage 2, confirm your agreement with each teacher provider (Anthropic, OpenAI) allows
training a model on its outputs. If it doesn't, use one of the open-weight teachers in `configs/models.yaml`.

## 5. Seed dataset (this repo)

`data/demos/seed.jsonl`: **100 validated instances**, each a small Python repo under `data/repos/`,
built from `scripts/seed_specs/batch_*.py` by `scripts/build_seed_dataset.py`:

| batch | ids | theme |
|---|---|---|
| 00 | 001-010 | original hand-written set |
| 01 | 011-025 | text processing & parsing |
| 02 | 026-040 | data structures & algorithms |
| 03 | 041-055 | dates, numbers, money, units |
| 04 | 056-070 | Python object model, iterators, decorators |
| 05 | 071-085 | config, CLI, serialization, paths, caching |
| 06 | 086-100 | multi-module business logic (hardest) |

Mix: 32 easy / 47 medium / 21 hard; 5-13 steps per trajectory (mean 7.9). The fix spans 2 files
in 4 instances; in about 60 the bug lives in a different module from where the symptom shows. A
few trajectories start on a wrong hypothesis and correct course. The builder rejects a spec whose
steps error, whose existing tests fail on the buggy repo, or where no hidden test fails.

These are deliberately small, to exercise the pipeline end to end. They are **not** a benchmark:
real eval instances should come from the engineer demos on your own repos.

Detailed options for using the demo data: [DEMO_DATA_PROPOSALS.md](DEMO_DATA_PROPOSALS.md). Each proposal
is implemented as a module in `src/repo_fix_agent/pipeline/` (map in §7 of that doc).

## 6. Next steps
1. Convert a 200-demo sample of the engineer data → validate → measure the drop rate.
2. Point `student.serving` at your vLLM server and run stage 0 baselines (`repo-fix eval ... --model student`).
3. Teacher rollouts on the 200, then SFT smoke test.
4. Scale to the full 10k.
