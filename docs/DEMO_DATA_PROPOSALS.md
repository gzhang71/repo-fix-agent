# How to Spend the 10k Engineer Demos: Proposals

Status: proposal for decision. Companion to `docs/DESIGN.md`. All gains below are **hypotheses to
validate on our own dev split**, not promises. Published numbers from other scaffolds don't
transfer; `configs/models.yaml` already warns that scaffold differences move scores by 10-20 pts.

Setup recap: student ≈ 30% resolve rate (open-weight 20-32B coder, fine-tunable); teacher ≈ 60%
(Claude Opus/Sonnet 5.5, GPT-5 family, or open-weight Qwen3-Coder-480B / Kimi-K2 / DeepSeek-V3.1).
10k engineer demos convertible into `Demo` records (`schema.py`) with hidden tests. Metric:
`grader.grade().resolved`.

---

## 1. TL;DR

1. **Audit and split first (week 1-2).** Run `repo-fix validate` on everything, dedup, split by
   repo into train / dev / **frozen test**. The frozen test set is never trained on, never used as
   teacher hints, never used for retrieval. Without this, nothing below is measurable.
2. **The demos are worth more as *environments* than as *trajectories*.** Each validated instance
   is a (problem_statement, repo@base_commit, hidden tests) triple: a free, verifiable task. Most of
   the value comes from rolling out models on them, not from imitating engineers keystroke by keystroke.
3. **Biggest expected jump: rejection-sampled teacher distillation** (P0) on train instances,
   plus **teacher-hinted generation** (P0) to cover the instances the teacher can't solve alone.
4. **Engineer-trajectory SFT is a supporting ingredient** (P0, but small weight): good for format,
   verification habits, and diversity; expected to be weaker than teacher traces as a sole source
   because human traces skip reasoning and contain off-tool behavior.
5. **Cheap inference win: a verifier for best-of-n reranking** (P1), trained on resolved / unresolved
   rollouts we generate anyway. Ships independently of model training.
6. **Then RL with execution reward** (P1) on the instance pool, curriculum by student pass@k, with
   demo-prefix fallback for pass@k = 0 instances.
7. **Auxiliary sub-skill data, preference data, augmentation, retrieval** are P1/P2: add only if
   ablations show they move resolve rate on dev.
8. **Expected ordering of impact (hypothesis):** teacher distillation (+hinted) > RL on demo
   environments > best-of-n verifier > engineer SFT > augmentation (bug injection) > DPO >
   auxiliary tasks > retrieval.

---

## 2. What the demos contain and what they're worth

### 2.1 Audit checklist (run before any training)

| Check | How | Action / threshold |
|---|---|---|
| Validation pass rate | `repo-fix validate` (`grader.validate_instance`): F2P fails before, gold `patch` resolves, P2P green before | Drop failures. Report drop rate per repo; >20% in a repo means the env build for that repo is broken, so fix the env, don't drop the repo |
| Flakiness | Run validate 3x per instance | Drop instances where any F2P/P2P test flips across runs; flaky tests poison both SFT filtering and RL reward |
| Trajectory replay | Re-execute `trajectory[].action` in `RepoEnv`; final `git diff` should match `patch` modulo whitespace/tests | Mark `metadata.replay_ok`. Non-replaying trajectories are still valid *instances* (usable for 3b, 3c, 3g), just not for 3a |
| Exact dedup | Hash of normalized `patch` and of `problem_statement` | Keep one per hash |
| Near-dup | MinHash/embedding similarity on `problem_statement` + `patch` (e.g. same fix cherry-picked to several branches) | Cluster; keep all members of a cluster **in the same split** |
| Repo distribution | Count per `repo` | Cap any single repo at ~5-10% of train to avoid one codebase dominating; report long tail |
| Trajectory length | `len(trajectory)`, total observation tokens | Flag >40 steps (our `max_steps`) and >64k tokens (`max_seq_len`); these need truncation or exclusion |
| Patch size | Lines / files / hunks in `patch` | Bucket (1 file / 1 hunk, multi-hunk, multi-file); used for stratification and curriculum |
| Bug-type taxonomy | Teacher labels `metadata.bug_type` from issue + patch (off-by-one, null handling, API misuse, concurrency, config...) then spot-check 100 by hand | Needed for stratified eval and error analysis |
| Test leakage | Does `problem_statement` contain the test code or exact assertion? Does `trajectory` read or create files touched by `test_patch`? | Strip leaked test text from issues; drop trajectory steps that `view_file` hidden test files (they don't exist at inference) |
| Issue quality | Teacher scores whether the issue alone is sufficient to determine the fix (SWE-bench Verified-style) | Keep underspecified ones in train (realistic), exclude from the frozen test set or tag them |
| Contamination with student pretraining | Is `repo` a public GitHub repo? Was the fix commit public before the candidate's training cutoff? Probe: give the student the issue only and measure exact/near-exact patch reproduction | Tag `metadata.public=true/false`. Report all metrics split public vs private; private-repo numbers are the ones we trust |
| PII / secrets | Secret scanners (gitleaks/trufflehog-style) over `observation`, `patch`, `problem_statement`; email/name regexes | Redact before anything leaves the secure environment, especially before sending to a third-party teacher API |

Deliverable: a one-page audit report (counts after each filter, per-repo table) and a cleaned
`demos.validated.jsonl` with `metadata` enriched (`replay_ok`, `public`, `bug_type`, `patch_size`,
`cluster_id`, `flaky`).

### 2.2 Splitting policy

* **Split by repository** (DESIGN.md §3): train / dev / test ≈ 80 / 10 / 10 of *repos*, stratified
  so dev and test cover the language, patch-size and bug-type mix.
* **Near-dup clusters never straddle splits.** Forks/vendored copies of a repo count as the same repo.
* **Frozen test set:** created once, hash-listed in the repo, never used for SFT, RL, verifier
  training, teacher-hint generation, retrieval index, or prompt tuning. Evaluated at milestones only.
* **Dev set** is for all iteration (ablations, mixing ratios, checkpoint selection).
* Optional **in-repo split** (a time-based split inside train repos: older fixes train, newer ones
  test) as a *secondary* metric, because the deployed agent will often run on repos it has seen.
  Report both; decide on the cross-repo number.

### 2.3 What each demo field is good for

| Field | Uses |
|---|---|
| `problem_statement` + repo@`base_commit` + tests | Environment for teacher rollouts (3b, 3c), RL (3g), eval (3j) |
| `trajectory` | Behavior cloning (3a), RL prefix fallback (3g), step-level preferences (3e) |
| `patch` | Privileged hint (3b), localization labels (3d), bug injection seed (3h), retrieval (3i), chosen side of preferences (3e) |
| `test_patch`, `FAIL_TO_PASS` | Grading, test-writing task labels (3d), hint (3b, train only) |
| `PASS_TO_PASS` | Regression guard; negative signal for verifier (3f) |

---

## 3. Proposals

### 3a. SFT / behavior cloning on re-executed engineer trajectories (P0, low weight)

**Idea.** Fine-tune the student on validated engineer trajectories rendered with `sft.render_messages`.

**Build.**
* Use only `replay_ok` demos. Observations are the **re-executed** `observation` strings, never raw logs.
* Loss masking: assistant turns only (`thought` + tool call); system, user (`build_task(problem_statement, test_cmd)`) and `tool` turns are context.
* Observation truncation: apply the *same* truncation the inference harness applies (head + tail, N chars) so training contexts match deployment exactly. One shared function, imported by both.
* Thought backfilling: where `thought` is empty or terse, have the teacher write a rationale given (issue, prior steps, *the next action fixed*). Constraint: it may not mention information not yet observed. Reject any backfill that references the gold `patch` content before the agent has viewed that code (string-match check).
* Format consistency: same `SYSTEM_PROMPT`, same tool schemas (`tools_as_functions()`), same chat template as serving. Add a unit test that renders one demo and one live inference prompt and diffs the prefix.
* Clean-up: collapse no-op steps (repeated `ls`, failed typo commands) only if the cleaned trajectory still replays and resolves.

**Fixes.** Tool-call format errors, patch-apply failures, "submit without running tests", navigation in unfamiliar repo layouts.

**Cost.** Low: conversion + re-execution compute + teacher backfill tokens (one call per step, short outputs).

**Risks.** Human traces are terse and skip reasoning; engineers use IDE features we mapped crudely to `str_replace`; overfitting to engineer style (e.g. never writing repro scripts). Backfilled thoughts may be post-hoc rationalizations that leak the answer.

**Measure.** Resolve rate on dev vs baseline; also format-error rate, patch-apply rate, % runs that execute tests before `submit`. Ablation: with vs without thought backfill; engineer-only vs teacher-only vs mixed.

### 3b. Teacher-hinted trajectory generation (P0)

**Idea.** For train instances the teacher can't solve unaided, give it privileged info, then strip it from the training record. This converts engineer *outcomes* into teacher-quality *trajectories*.

**Build.**
* Hint ladder (escalate only if the previous level fails k times): (1) `FAIL_TO_PASS` test names; (2) `files_changed` from `patch`; (3) full `test_patch`; (4) full gold `patch` "as a reference solution an engineer wrote; reproduce the reasoning, explore normally".
* The hint goes in a separate system block. The saved record keeps only the plain `build_task(problem_statement, test_cmd)` prompt; `metadata.hint_level` records what was used.
* Rejection-sample with `grade()`; keep only resolved.
* **Hint-leakage filter** (heuristics, all cheap):
  * The first edited file was never surfaced by a prior observation (grep/ls/view output) → drop.
  * Zero exploration steps before the first edit while `patch` touches >1 file or a non-obvious file → drop.
  * Thought text mentions test names, "the reference", "the hint", or quotes gold-patch lines before viewing them → drop.
  * Steps-to-first-edit far below the unhinted-teacher distribution for that patch-size bucket → down-weight or drop.
* Optionally re-write: ask the teacher to add a plausible exploration path, then re-execute; keep only if it replays and resolves.

**Fixes.** Extends coverage to the hard ~40% the teacher fails, where the student most needs examples.

**Cost.** Medium: up to 4 hint levels × k samples on the ~40% teacher-fail subset.

**Risks.** Hint leakage teaches the student to "guess" files (worse generalization); hinted trajectories are biased toward the engineer's fix style. Never run hints on dev/test.

**Measure.** Dev resolve rate on the *hard* bucket (instances base teacher fails) with vs without hinted data; leakage-filter ablation (filtered vs unfiltered) should show filtered ≥ unfiltered, otherwise the filter is mis-tuned.

### 3c. Rejection-sampled teacher distillation on demo issues (P0, biggest expected lever)

**Idea.** DESIGN.md stage 2: the teacher solves train instances in our exact harness, k samples each, keep resolved.

**Build.**
* k = 4 (from `models.yaml`); raise to 8 for instances with 0-1 successes if budget allows.
* Keep resolved trajectories; additionally require a test was run before `submit` (filters lucky guesses).
* Dedup near-identical trajectories per instance (same action sequence modulo args, or same final patch): keep at most 2 per instance, preferring shorter.
* Cap per-instance contribution so easy instances don't dominate (easy ones yield 4/4).
* Mix with engineer data: start teacher:engineer ≈ 3:1 by trajectory count, sweep {1:0, 3:1, 1:1} on dev.
* Teacher choice: sweep Opus vs Sonnet on 200 dev instances; pick by cost per *resolved* instance, subject to licensing (§5).
* Store failed teacher rollouts too: they feed 3e and 3f.

**Fixes.** Reasoning style, systematic exploration, repro-then-fix-then-verify loop, in the exact action space.

**Cost.** Highest token cost: ~8k instances × 4 samples × ~40 steps. Run a 200-instance pilot to get a real $/resolved figure before scaling.

**Risks.** Student imitates teacher behaviors it can't execute (very long contexts, huge edits); distribution limited to what the teacher can already solve; licensing.

**Measure.** Dev resolve rate; ablate data size (1k / 4k / all) to see if we're still on a slope; mixing-ratio sweep.

### 3d. Sub-skill auxiliary tasks mined from demos (P1)

**Idea.** Decompose the job (Agentless-style) into single-turn tasks with automatic labels, mixed into SFT at a small ratio.

| Task | Input | Target (from demo fields) | Verification |
|---|---|---|---|
| Fault localization | `problem_statement` + repo file tree (+ optional retrieved snippets) | files, then functions touched by `patch` (parse hunks + AST for enclosing def) | exact-match / recall@k |
| Patch generation given context | issue + the gold-touched functions' source | edit as `str_replace` calls | apply + `grade()` |
| Test writing | issue + relevant source | a test that fails on `base_commit` and passes with `patch` applied | execute both ways; F2P-like behavior required |
| Reproduction script | issue | a script exiting non-zero before `patch`, zero after | execute both ways |
| Patch critique | issue + candidate patch (gold or failed student patch) | resolved? + short reason | label from `grade()` |

**Fixes.** Localization is usually the dominant failure mode of weaker agents (verify in our error analysis first); test writing improves self-verification.

**Cost.** Low for localization/critique (pure data munging); medium for test/repro tasks (teacher generations + execution).

**Risks.** Single-turn formats differ from the agent loop; transfer to multi-turn isn't guaranteed. Keep the mix ≤ 20% of tokens.

**Measure.** Localization recall@k on dev as a diagnostic; the decision metric is still agent resolve rate with vs without aux mix. Also: does the agent's first-edited-file accuracy improve?

### 3e. Preference data: DPO / KTO (P2)

**Idea.** Use contrastive signal that SFT throws away.

**Build.**
* Trajectory-level pairs: resolved vs unresolved rollouts on the **same** instance (from 3c teacher and student rollouts). Prefer pairs from the student's own samples (on-policy).
* Patch-level pairs: engineer `patch` (chosen) vs failed student patch (rejected), in a single-turn "write the fix given context" format; do not splice the human patch into a student trajectory it didn't produce.
* Step-level: at the first divergence point between a resolved and unresolved rollout sharing a prefix, prefer the resolved branch's next action.
* KTO when pairs are scarce (only unpaired good/bad labels).

**Fixes.** Common failure behaviors: premature submit, editing the wrong file, breaking `PASS_TO_PASS`.

**Cost.** Low data cost (reuses rollouts); training cost moderate. **Risks.** Long-trajectory DPO is unstable, length bias (shorter = chosen). Likely subsumed by RL if 3g works.

**Measure.** Dev resolve rate SFT vs SFT+DPO; regression rate (P2P failures). Run only if RL infrastructure is delayed.

### 3f. Verifier / reward model (P1, cheap inference win)

**Idea.** Train a classifier: (issue, trajectory or final patch, test output) → P(resolved). Use for best-of-n at inference and as dense reward/critic in RL.

**Build.**
* Positives: resolved teacher/student rollouts + engineer demos. Negatives: unresolved rollouts, especially those that apply cleanly and pass the agent's own tests but fail hidden `FAIL_TO_PASS` or break `PASS_TO_PASS` (hard negatives).
* Label = `grade().resolved`; train on train-split repos only.
* Initialize from the student (or a smaller sibling) with a classification head, or as a generative judge ("yes/no" token).
* Inference: sample n = 4-8 student trajectories, rerank by verifier, optionally combined with "did the agent's own tests pass" and patch-agreement voting across samples.

**Fixes.** Converts pass@n into pass@1 at n× inference cost; no change to the policy.

**Cost.** Low data cost; n× serving cost per issue (the student is cheap, so likely still far below teacher cost).

**Risks.** Verifier learns superficial cues (patch length, presence of "tests passed" string); calibration drift as the policy changes. Retrain each policy iteration.

**Measure.** Best-of-n resolve rate vs pass@1 and vs oracle pass@n; the gap closed is the verifier's value. Report AUROC on dev rollouts.

### 3g. RL with execution reward (P1, stretch-goal enabler)

**Idea.** GRPO/RLOO with reward = `grade().resolved` (+ small step penalty, + format penalty), using validated train instances as the environment pool (SWE-Gym-style).

**Build.**
* Pool: train instances that passed the flakiness check, have deterministic test runtime under a cap.
* Curriculum: measure student pass@k (k = 8) per instance after SFT. Train first on 0 < pass@k < k (informative gradient), schedule harder ones later; drop always-solved ones.
* pass@k = 0 instances: **prefix-conditioned rollouts**: feed the first t steps of the engineer (or hinted teacher) trajectory, let the student continue; anneal t down as success rises. Mask loss on the given prefix.
* Reward hardening: the grader already drops edits under `tests/` and applies `test_patch` last; also penalize patches that modify test infra/config (`conftest.py`, `pytest.ini`, CI files), and log any reward from a patch touching those.
* Group size 8, rollouts in containers (DESIGN.md §2 production sandbox), strict timeouts.

**Fixes.** Student-specific failure modes imitation can't see (its own error recovery, its own context limits); the only path to beat the teacher on our distribution.

**Cost.** High infrastructure (parallel sandboxes, long rollouts); compute heavy. **Risks.** Reward hacking, flaky-test noise, sparse reward on hard instances, entropy collapse.

**Measure.** Dev resolve rate over RL steps; pass@1 vs pass@8 (RL should raise pass@1 without collapsing pass@8); median steps and cost per resolved.

### 3h. Data augmentation (P1 bug injection, P2 others)

**Bug injection (SWE-smith-style).** Take repo@`base_commit` with gold `patch` applied (a "known-good" state with passing tests). Generate bugs: AST mutations (flip comparison, off-by-one, drop a branch), teacher-written plausible bugs, or reverting historical fixes. Keep a mutation only if it makes ≥1 existing test fail (that test becomes F2P; the rest P2P). Have the teacher write a realistic issue from the failing test output *without* naming the test. Run validate. Yields instances for 3c and 3g at ~zero human cost; can multiply the pool many times over.
*Risk:* synthetic bugs are easier and stylistically different from real ones; cap at ≤50% of the RL pool and always evaluate on real demos only.

**Issue rephrasing (P2).** Teacher rewrites `problem_statement` at varying specificity (terse, user-perspective, stack-trace-only). Same tests. Robustness to bad issue writing. *Risk:* rephrasing can drop needed info; validate that teacher still solves ≥ baseline.

**Repo-context perturbation (P2).** Rename irrelevant files/dirs, add distractor modules, shuffle file order. Reduces memorization of repo layout. *Risk:* breaks imports; only use perturbations that keep validate passing.

**Measure.** Dev (real instances) resolve rate with vs without synthetic data in the SFT/RL pool, at matched compute.

### 3i. Retrieval at inference (P2)

**Idea.** Index train demos; at inference retrieve similar past fixes (by issue embedding, plus same-repo filter) and insert 1-2 compressed exemplars (issue + localized files + patch summary, not full trajectories) into the prompt.

**Build.** Embed `problem_statement`; store `files_changed`, a patch summary, bug type. Never index dev/test.
**Fixes.** Repo-specific conventions on repos we've seen; useful mostly for the in-repo deployment case.
**Cost.** Low. **Risks.** Context bloat, copying wrong fixes, little gain on unseen repos.
**Measure.** In-repo time-split metric with vs without retrieval; cross-repo metric must not regress.

### 3j. Evaluation set construction (P0)

**Build.**
* Frozen test: ~10% of repos (§2.2), target 500-1000 instances, stratified by language, patch size, bug type, public/private.
* Filter: validated 3x (non-flaky), issue judged sufficient, no test leakage in issue.
* Contamination check: report public vs private repos separately; probe student reproduction of gold patch from issue alone.
* **Human-reviewed slice:** 100-200 instances where an engineer confirms (a) the issue is solvable as written, (b) tests check the described behavior and not the specific gold implementation (would a different correct fix pass?). Report it as the "verified" headline number.
* Fixed eval harness version, seeds, `max_steps`, and n; report pass@1, best-of-n, cost per resolved, median steps, patch-apply rate, regression rate (P2P failure rate).

**Cost.** Low compute; ~2-4 engineer-days for the human slice. **Risk:** too small → noisy; with 500 instances a ±4 pt 95% CI near 50% is the floor, so treat sub-3-pt differences as noise.

---

## 4. Phased roadmap

Targets are hypotheses on the cross-repo dev split, measured with our harness. Re-baseline in
week 2; if the "30%/60%" doesn't reproduce in our scaffold, shift all targets accordingly.

| Phase | Weeks | Work | Data | Target (hypothesis) | Go / no-go |
|---|---|---|---|---|---|
| 0. Audit + baselines | 1-2 | Convert + validate 10k, dedup, split, freeze test; baseline student candidates + teacher; build 3j | 10k demos | Reproduce ~30 / ~60 | Go if ≥60% of demos validate and baselines are within ±5 pts of expected; else fix env builds first |
| 1. Pilot | 3-4 | 200-instance teacher rollouts (3c), engineer SFT smoke (3a); measure $/resolved per teacher | ~2k engineer + ~500 teacher trajectories | Student 33-38% | Go if SFT ≥ baseline + 3 pts and no format regressions |
| 2. Full distillation | 5-8 | 3c on all train, 3b on teacher-fail subset, mixing sweep, thought backfill | ~8k engineer + ~15-25k teacher trajectories | 40-48% | Go to RL if gain over phase 1 ≥ 5 pts; if flat, data quality is the bottleneck: re-audit before scaling |
| 3. Verifier + best-of-n | 7-9 (parallel) | 3f on all rollouts collected so far | rollouts from phases 1-2 | +3-8 pts best-of-4 over pass@1 | Ship to deployment if cost per resolved < 1/3 of teacher |
| 4. RL | 9-14 | 3g with curriculum + prefix fallback; 3h bug injection to widen pool | train pool + synthetic | 48-55% pass@1 | Continue if dev rises monotonically and hacking audits are clean |
| 5. Extras | 12+ | 3d, 3e, 3i as ablations | as above | each must add ≥ 2 pts on dev | Drop any that don't |
| Milestone eval | end of 2, 4, 5 | Frozen test + human slice | frozen test | — | Report only; never tune on it |

---

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Licensing of teacher outputs.** Both Anthropic's and OpenAI's commercial terms have historically contained restrictions on using outputs to develop competing models; whether fine-tuning an internal bug-fixing student is allowed depends on the exact agreement we have | Get a written legal read of our *current* Anthropic and OpenAI agreements (consumer ChatGPT terms differ from API/enterprise terms) before phase 1 rollouts. Fallback: open-weight teachers (Qwen3-Coder-480B, Kimi-K2, DeepSeek-V3.1); check each license too. Record `metadata.source = teacher:<model>` so data from any teacher can be removed later |
| **Student license** | Confirm the chosen candidate's license permits fine-tuning and commercial deployment (Apache-2.0 candidates in `models.yaml`) |
| **PII / secrets in demos** | Scan and redact before sending anything to a third-party API; zero-data-retention agreements with providers; keep raw demos in the secure environment |
| **Proprietary code sent to teacher APIs** | Confirm with security that our code may be sent to the provider; otherwise open-weight teacher on our infra |
| **Overfitting to engineer style** | Engineer data as minority of SFT mix; evaluate on unseen repos; RL corrects toward what works for the student |
| **Hint leakage** (3b) | Heuristic filters + ablation; hinted data capped as a fraction of the mix |
| **Reward hacking** (edit tests, special-case inputs, delete failing code paths) | Grader already strips `tests/` edits and applies `test_patch` last; extend the denylist to test config; periodic manual review of 50 random RL-resolved patches; P2P regression guard |
| **Flaky tests** | 3x validation, drop flaky instances; for RL, re-grade positives once before using them as reward |
| **Contamination** (public repos in pretraining) | Report public vs private metrics; trust private |
| **Eval overfitting** | Frozen test touched only at milestones; human-verified slice as headline |
| **Sandbox safety during rollouts** | Containers, no network, resource limits (DESIGN.md §2) before any large-scale rollout |

---

## 6. Open questions for the user

1. **Teacher licensing:** which agreements are in place (Anthropic API / enterprise, OpenAI API / ChatGPT Enterprise)? Has legal cleared training on outputs? This decides closed vs open-weight teacher.
2. **Can our code leave the building?** If not, the teacher must be open-weight, self-hosted.
3. **Demo format reality:** what fraction of the 10k have actual action logs vs only issue + final patch? (Affects how much 3a is possible; 3b/3c/3g only need the instance.)
4. **Tests:** do all 10k have hidden tests with F2P/P2P, or do some need tests written (3d test-writing could backfill)?
5. **Deployment target:** mostly repos in the demo set, or new repos? Decides cross-repo vs in-repo metric as the primary.
6. **Languages:** what is the language mix? The current harness and seed set are Python-only.
7. **Budget:** token budget for teacher rollouts and GPU budget for RL. The phase 1 pilot will give $/resolved to size this.
8. **Latency / cost envelope at inference:** is best-of-n (n× student calls) acceptable?
9. **Engineer time** for the 100-200 instance human-reviewed eval slice: who owns it?
