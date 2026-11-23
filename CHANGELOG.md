# Changelog

Every attempt so far, newest first. Each entry says what was tried, what was **measured**
(with numbers) and what is only **implemented** (tested offline, not yet run against paid APIs).
Accuracy is "resolved" on the 100 seed demos unless stated; per-instance results are in
`runs/` (gitignored), and the analysis is in `docs/BASELINE.md`.

## [Unreleased]: student improvements

**Measured 2026-11-23, v3** (`20261123-081218-7a82b0`): with the **repro-only submit gate**, the
student scored **88/100**, vs 83 for v2 and 85 for v1.
- **The gate:** only running a `scratch_*.py` script after the last edit counts as a check now.
  It fired in 60 episodes, and 55 of those were then solved.
- **Scratch scripts:** used in 85 of 100 episodes, up from 37.
- **Failure modes:** partial fixes fell from 8 to 3. Mean steps rose from 20.4 to 24.0.
- **Hard instances:** 86%, up from 71%.
- **A paired comparison:** at temperature 0, v3 replays v2's cached responses until the gate
  first fires (1,077 live calls vs 2,047). Every difference is downstream of the gate, so the
  +5 is attributable to it rather than to sampling noise.
- **Backtest dashboard:** `runs/dashboard.html`, regenerated after every eval.

**Measured 2026-11-23, v2** (details in `docs/BASELINE.md`):
- **Result:** the no-training scaffold scored 83/100 vs 85/100 on one sample each, which is within
  noise.
- **What worked:** reasoning replay (reasoning on 100% of turns, from 24%).
- **What didn't:** the spec check was followed in only 37% of episodes, and the submit gate never
  fired, because running the visible tests counts as a check.
- The training items below are still implemented but not run.

### RL improvements (implemented, not run)
- **Shaped reward on the hidden tests** (`rl_rewards.py`, the default):
  - Resolved scores 1.0.
  - Otherwise, 0.5 × the fraction of FAIL_TO_PASS tests passed.
  - Either way, minus 0.5 × the fraction of PASS_TO_PASS tests broken, the largest failure mode
    in v3.
  - A resolved episode always outranks an unresolved one.
- **Verified-repro bonus (+0.1):** the student's `scratch_*.py` scripts run against the untouched
  base commit and against its patched tree.
  - The bonus needs one script that fails on the base and passes on the patch.
  - A failure from not running at all (import, syntax or name errors) doesn't count.
  - Scripts never ship, so the bonus can't change what is graded.
- **Dynamic sampling:** `rl-rollouts --dynamic-sampling [--target-groups N]` drops groups whose
  rewards all tie (zero advantage, no gradient) and keeps drawing instances.
- **The update** (`train_lora.py --objective pg`):
  - Clipped policy-gradient surrogate against the starting policy's log-probs, with a looser
    upward limit (0.2 / 0.28) so rare good tokens can grow.
  - Token-level loss averaging over each accumulation window, so long episodes aren't
    down-weighted.
  - `student-train` runs it for 2 passes.
  - Also fixed: a skipped record could end a window without an optimizer step.
- **Running out of steps:**
  - "[N steps left]" is appended to observations for the last 8 steps (`step_notice` in
    `models.yaml`), identically in `run_agent`, the RL environment and SFT rendering. Already
    recorded notices aren't doubled.
  - The reward breakdown marks `truncated` episodes apart from wrong fixes.
- **Verified offline:**
  - An end-to-end episode where the student's repro script failed on the base and passed on its
    fix earned the bonus.
  - An import-error script didn't.
  - The pg update trained on a tiny model with token-level and per-episode averaging.
- **Not yet measured:** the steps-left notice changes what the student sees, so the next baseline
  run will show its effect.

### Run database
- **`runs/repofix.sqlite`:** `eval` records every run. Tables:
  - `runs`: model, scaffold settings with the system-prompt hash, dataset hash, git commit and
    dirty flag, sampling settings, status, summary, usage and cost.
  - `results`: one row per sample.
  - `episodes`: the full trajectory and patch, compressed.
  - `instances`: one row per demo.
- **Writes commit as they happen**, so a crash keeps progress. `eval --resume <run>` skips samples
  already done and summarizes the whole run.
- **`repo-fix db runs | show | compare | import | export | delete`:**
  - `compare` lists the instances that flipped between two runs.
  - `export` writes results or episodes in the format the pipeline reads.
- **Imported:** the 2026-11-22 student and teacher baselines, without a git commit, since the code
  that produced them is unknown.

### Distillation by teacher signal (implemented, not run)
- **Question it answers:** a teacher may expose its reasoning, its token log-probs or its full
  logits, and the student may or may not have a reasoning channel. Each case trains differently.
- **`distill-signal`:** reads `distillation_signal` in `models.yaml` and picks the richest mode the
  pair supports, explaining why richer modes are out:
  - `action`: only the teacher's output.
  - `reasoning`: its reasoning text too.
  - `topk`: per-token log-probs, same tokenizer.
  - `logits`: local weights, same tokenizer.
- **Teacher reasoning is now recorded apart from its visible text** (`last_reasoning` /
  `last_text` on the policies).
  - `onpolicy-distill --signal action|reasoning`: action mode trains on the visible text and tool
    call only.
  - Reasoning mode places the reasoning in the student's `reasoning_content`, or in `content` for
    a student without one, with its own loss weight: 0.5 for summarized (Claude), 1.0 for raw.
  - `train_lora.py` weights reasoning tokens apart from action tokens inside the same message.
    The chat-FT export has one weight per message, so it drops the split.
- **`token-distill`** (topk): renders student episodes with the student's chat template and gets
  the teacher's prompt log-probs (OpenAI-compatible `/completions`, `echo` + `logprobs`, cached).
  - It aligns the teacher's tokens to the student's by offset and text, and refuses if trained
    tokens don't match (a different tokenizer).
  - Dry run unless `--yes`. `--render-only` writes chat records for GKD (free).
- **`train_lora.py` objectives:**
  - `tokenkl`: per-token reverse KL as a policy gradient with reward log p_T − log p_S, plus
    forward KL on the teacher's top-k.
  - `gkd`: full-distribution KL / JSD against a local teacher; checks that the vocabularies match.
- **Verified offline:**
  - `gkd_loss` equals the reference KL in both directions and drives a toy student toward the
    teacher. `tokenkl` raises the student's probability of the tokens the teacher rates likely.
  - Both objectives trained end to end on a tiny HF model, with token-level data from a
    simulated same-tokenizer teacher (774/774 trained tokens aligned).
- **`student-train --signal auto|action|reasoning|topk|logits`:**
  - topk adds the stages `token-distill` and `train-token`.
  - logits adds `render-rollouts` and `train-gkd`.
  - An unsupported mode exits with the reason. With Claude, auto selects `reasoning`.

### Difficulty-aware data weighting (implemented, free)
- **Question it answers:** train on what the teacher solves and the student doesn't, or on both?
  Answer: both, weighted by the student's own pass rate p, measured over several samples.
- **`difficulty-weights`:** buckets each instance:
  - gap: student never solves it. Teacher data, weight 1.
  - learning: student solves it sometimes. RL and DPO, weight 1 − p.
  - mastered: student always solves it. Replay only, weight 0.2.
  - unsolved: neither model solves it. Skipped.
- **Free check on the baseline** (1 sample each, so p is only 0 or 1): 14 gap, 85 mastered,
  1 unsolved, which matches the teacher-only / both / neither split. Several samples per instance
  are needed before the learning bucket fills.
- **Wiring:**
  - `sft-mix --instance-weights` scales the weights and caps the total per instance at 3.
  - `sft-mix --student-rollouts` adds the student's shortest successful run per instance
    (self-distillation, on its own distribution).
  - `distill` and `onpolicy-distill` get `--instance-weights --max-pass-rate 0.75`, so teacher
    calls go only to instances the student doesn't always solve.
  - `--pass-rates-out` feeds the `rl-rollouts` curriculum.
  - `student-train` gains the stages `weights`, `post-rollouts` and `rl-weights`.

Motivation (from the baseline analysis below):
- **Partial fixes:** 10 of the 14 instances the student missed and the teacher solved were partial
  fixes, where the issue's example is fixed and other hidden tests fail.
- **Regressions:** 5 broke a hidden test even though every visible test passed.
- **Reasoning:** 76% of student turns had no visible reasoning.

No paid API calls were made for this entry; all of it is covered by offline tests (451 passing).

### No training needed
- **Reasoning replay:** `keep_reasoning`. The student's `reasoning_content` was silently dropped
  every turn. It is now recorded as the step's thought and sent back in later turns.
- **Spec-check prompt:** `spec_check`. The system prompt asks the model to list the behaviours and
  edge cases the issue implies and check them with `scratch_*.py` repro scripts. Scratch files are
  excluded from patches.
- **Submit gate:** `submit_gate`. The first `submit` after an edit that was never checked is
  bounced once with a "verify edge cases" nudge.
- **Same prompt and gate everywhere:** student inference, SFT rendering, the RL environment and
  teacher distillation all use them (train == inference). Teacher evaluation is unchanged.
- **Execution reranker for best-of-n:**
  - It pools every candidate's repro scripts and runs each patch against all of them and the test
    suite (CodeT-style cross-checking). Selection makes no model calls.
  - `repo-fix eval --best-of N` reports the chosen patch's accuracy and the oracle (any candidate
    correct). `best-of-n --verifier exec` is the default.
- **Noise-aware evaluation:**
  - `eval --samples k --temperature t` reports mean pass@1, pass@k, and how many instances are
    solved only sometimes (the ones RL can learn from).
  - `--save-rollouts` keeps the episodes.
- **Benchmark fix:** `seed-002`'s issue now states the behaviour its hidden test checks (the
  caller's list must not be mutated). Both models had failed it the same way. Re-validated: 100/100.

### Training
- **On-policy distillation** (`onpolicy-distill`, DAgger-style):
  - The teacher sees the task plus the student's transcript at states the *student* reached and
    gives the best next action.
  - Output: SFT records (student-format prefix, loss only on the teacher's turn) and step-level DPO
    pairs at disagreements.
  - `--dry-run` estimates calls and cost.
  - Token-level on-policy distillation (reverse KL) needs teacher log-probs and a shared tokenizer,
    and Claude provides neither, so this version relabels actions.
- **Edge-case weighting:** synthetic instances record `hidden_beyond_issue`, and
  `sft-mix --edge-case-weight 2` weights them up. `sft-mix` also accepts pre-rendered records
  (`--extra-records`) and several teacher files.
- **Export:** `export-chat-ft` writes the Fireworks/OpenAI chat fine-tuning schema (per-message
  `weight`, `tools`, `reasoning_content`), with validation.
- **`scripts/train_lora.py`:** transformers + PEFT, with three objectives:
  - `sft`: weighted tokens.
  - `dpo`: chosen vs rejected, against the reference model.
  - `pg`: REINFORCE with group-relative (GRPO) advantages from `rl-rollouts`, plus a KL penalty to
    the reference, which is the base model with the adapter switched off.
  - All three were smoke-tested on CPU with a tiny random model, on data exported by the pipeline.
- **`fireworks-deploy`:** creates an on-demand deployment for a fine-tuned model through the
  documented REST API. Dry run unless `--yes`.
- **`student-train`:** prints the whole round with per-stage cost and runs one stage at a time.
  Paid stages need `--yes`.
- Findings while building this:
  - Fireworks managed SFT/DPO and dataset uploads closed to new jobs on 2026-11-09. New training
    goes through their Training API or your own GPUs.
  - The student supports supervised LoRA on Fireworks, but not RL and not serverless LoRA.

### What RL is
GRPO-style group-relative advantages (RLOO also exported) with no value network, so it isn't PPO.
DPO is the separate, offline preference step.

### Still to do
- Measure all of the above: rerun the student baseline with the new scaffold, then pass@4, then
  best-of-4.
- Build a harder evaluation set first. The seed set is near its ceiling.

## 2026-11-22: baselines (measured)

| | Student (Fireworks `nemotron-lightning-3p5-30b-a3b`) | Teacher (`claude-opus-5-5`) |
|---|---|---|
| Resolved | **85 / 100** (easy 78%, medium 89%, hard 86%) | **98 / 100** (97% / 98% / 100%) |
| Mean steps | 16.6 | 5.5 |
| Cost | Fireworks bill (1,638 calls) | about $4.24 (575 calls, prompt caching on) |

- Overlap: 84 instances solved by both, 14 by the teacher only, 1 by the student only, 1 by neither.
- Failure analysis rebuilt from the response cache:
  - Tool errors are rare, and the student runs the tests after nearly every edit.
  - Every "regression" passed the visible suite. The hidden tests caught edge cases the repo's
    own tests don't cover.
  - Repro scripts were written in only 40% of failed episodes.
- **Limitation:** the seed set is too easy (expected about 30% vs 60%) and is not a reliable
  decision benchmark yet.

### Student model choice
- `gpt-oss-120b`: listed on Fireworks but not deployed serverless (404).
- Switched to the smallest serverless tool-calling model, `nemotron-lightning-3p5-30b-a3b`, for cost.

## 2026-11-22: configuration, API cost controls

- **Keys:** in `.env` (gitignored) with `.env.example`. A dependency-free loader runs in the CLI.
- **Student on Fireworks:** served through the OpenAI-compatible API; config in `models.yaml`.
- **Response cache** (`llm_cache.py`): every LLM and embedding request and response is stored in
  SQLite and replayed on reruns.
  - The n-th identical sampled request maps to the n-th recorded response, so samples stay
    distinct. temperature=0 requests dedupe.
  - Verified live: a replay returns in 0.00s with no API call.
  - Two bugs found and fixed in live testing: Claude replies didn't rebuild under strict
    validation, and the tests were writing into the real cache.
- **Anthropic prompt caching** for the teacher: each step re-reads earlier turns from cache.
  Measured: step 2 read 1,970 cached tokens and wrote 94 new ones.
- **`eval` upgrade:** workers, per-instance JSONL, accuracy by difficulty, token and cost accounting.

## 2026-11-22: web UI (`repo-fix serve`)

- The flow: repo URL, clone, per-repo environment image, failing tests become bugs, parallel
  fixes in sandboxes, verification, one PR per fix (review first, or automatic for verified
  fixes), fork fallback.
- Measured: end to end with Docker on a local test repo (2 of 2 fixes verified, a deliberately
  bad fix flagged as a regression), and checked in Chrome. No real PR has been opened yet.
- Fixed during testing:
  - The environment image set HOME to root's home directory, so every command printed a
    permission error.
  - The page rebuilt itself on every poll, which could swallow clicks.
  - A missing API key was only caught after clone and install. It now fails fast.

## 2026-11-22: resolve pipeline limitations (commit 8a83c42)

- **Docker sandbox:** first real runs. Built a dedicated image (no network), runs as the host
  user, and `auto` mode checks for the image. 100/100 seeds validate in the container.
- **Synthetic issues written from executed repros:** the old template had leaked hidden
  assertions in **34/146** issues. The new writer leaks in **0**, and 116/146 issues include runnable repros.
- **Embeddings:** retrieval on paraphrased queries improved from 3/10 to **8/10** top-1 with
  `bge-small`. Semantic dedup catches renamed-and-reworded duplicates that MinHash misses.
  The offline hashing embedder did **not** beat TF-IDF.
- **Secret scanner:** provider patterns, an entropy check and optional detect-secrets. False
  positives went from 8 to **0** on 246 instances.
- **Parallel rollouts** (`--workers`).

## 2026-11-22: seed set to 100 instances, demo-data pipeline (commit 8a83c42)

- 90 new instances: 32 easy, 47 medium, 21 hard, every one validated.
- One `repo-fix` command per data proposal: audit, split, distill, sft-mix, aux-tasks, prefs,
  verifier, rl-rollouts, retrieval, augment.

## 2026-11-22: scaffold (commit d47207b)

- The agent, sandbox, tools and grader, plus the policies: Claude teacher, GPT teacher candidate,
  and student on vLLM/SGLang. 10 seed instances and the design docs.
