# Baseline: student vs teacher on the 100 seed demos

Run on 2026-11-22 with `repo-fix eval data/demos/seed.jsonl --model <m> --workers N --out runs/baseline/<m>.jsonl`.
Same agent loop, tools, 40-step budget and Docker sandbox for both; each patch graded by the hidden
FAIL_TO_PASS / PASS_TO_PASS tests. No errors (API or sandbox) in either run.

| | Student | Teacher |
|---|---|---|
| Model | `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` (Fireworks, 30B MoE / ~3B active, temp 0) | `claude-opus-5-5` (effort high, adaptive thinking) |
| **Resolved** | **85 / 100** | **98 / 100** |
| easy (32) | 25 (78%) | 31 (97%) |
| medium (47) | 42 (89%) | 46 (98%) |
| hard (21) | 18 (86%) | 21 (100%) |
| Mean steps | 16.6 | 5.5 |
| Submitted within budget | 97 | 100 |
| API usage | 1,638 calls; 521k input + 5.7M cached-read + 523k output tokens | 575 calls; 389k cache-write + 1.4M cache-read + 101k output tokens |
| Cost | Fireworks bill (not priced here) | about $4.24 (with prompt caching) |

Overlap: 84 solved by both; 14 by the teacher only; 1 by the student only (`seed-051`); 1 by neither (`seed-002`).

## What the student gets wrong (14 teacher-only instances)

* **Partial fixes (10):** one hidden test passes and another fails, e.g. `seed-067` (2/4),
  `seed-090` (1/3), `seed-096` (1/3). The student fixes the symptom shown in the issue and misses the
  rest of the contract.
* **Regressions (5):** the fix breaks a test that passed before (`seed-016`, `048`, `053`, `071`, `092`).
* **Budget:** `seed-016` used all 40 steps without submitting. The student needs 3x the teacher's
  steps on average.

## Caveats before using these numbers

1. **This set is too easy to separate the models the way production does.** The motivating numbers
   were roughly 30% (student) vs 60% (teacher); here they are 85% vs 98%. The seed issues are
   precise, with repro snippets, and the repos are small. Before the set is used for decisions,
   add harder, less-specified instances (multi-file, vague issues, larger repos), or evaluate on
   real engineer demos.
2. **`seed-002` is a dataset defect:** a hidden test (`test_caller_list_not_mutated`) checks
   behaviour the issue never mentions. Both models fail it the same way. Reword the issue or drop
   that test (the audit's issue-sufficiency judge, `--label-model`, is meant to catch this).
3. **Single sample per instance.** Opus runs at temperature 1, so expect ±2–3 points on a rerun.
   Rerunning the same command replays the cache; set `REPOFIX_LLM_CACHE=off` to resample.

Per-instance results: `runs/baseline/{student,teacher}.jsonl` (gitignored), with summaries next to them.

## 2026-11-23: student baseline v2 (new no-training scaffold, measured)

Same settings as v1: 100 seed instances, temperature 0, Docker sandbox, one sample each.
Run `20261123-075428-8b10cb` in `runs/repofix.sqlite`.

| | v1 (old scaffold) | v2 (reasoning replay + spec check + submit gate) |
|---|---|---|
| Resolved | 85 / 100 | **83 / 100** |
| easy / medium / hard | 78% / 89% / 86% | 81% / 89% / 71% |
| Mean steps (solved) | 16.2 | 19.5 |
| Hit the 40-step limit | 3 | 4 |
| Fireworks calls | 1,638 | 2,047 |

- **No measurable gain.**
  - 78 instances were solved by both versions, 5 only by v2, 7 only by v1, and 10 by neither.
  - Twelve flips out of 100 on one sample each is mostly noise, so 85 vs 83 is not a real
    difference.
  - The `seed-002` rewording worked: v2 solves it.
- **Reasoning replay works:** 100% of turns now carry reasoning (v1: 24%).
- **The spec check is mostly ignored:**
  - Only 37 of 100 episodes wrote a `scratch_*` repro script.
  - 11 of the 17 failures wrote none.
- **The submit gate never fired.** The student always runs the visible test suite after an edit,
  which the gate counts as a check. The failures, though, pass the visible tests and fail hidden
  edge cases, so the gate as designed can't catch them.
- **Steps:** the scaffold adds about 3 steps per solved episode. Two regressions (`seed-063`,
  `seed-070`) ran out of steps without submitting.
