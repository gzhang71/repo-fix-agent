# Changelog

Every attempt so far, newest first. Each entry says what was tried, what was **measured**
(with numbers) and what is only **implemented** (tested offline, not yet run against paid APIs).
Accuracy is "resolved" on the 100 seed demos unless stated; per-instance results are in
`runs/` (gitignored), and the analysis is in `docs/BASELINE.md`.

## [Unreleased]: student improvements (implemented, not run)

Motivation (from the baseline analysis below):
- **Partial fixes:** 10 of the 14 instances the student missed and the teacher solved were partial
  fixes, where the issue's example is fixed and other hidden tests fail.
- **Regressions:** 5 broke a hidden test even though every visible test passed.
- **Reasoning:** 76% of student turns had no visible reasoning.

No paid API calls were made for this entry; all of it is covered by offline tests (417 passing).

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
