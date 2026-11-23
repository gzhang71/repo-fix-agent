# Baseline: student vs teacher on the 100 seed demos

Run on 2026-11-22 with `repo-fix eval data/demos/seed.jsonl --model <m> --workers N`.
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

Per-instance results are in the run database (`runs/repofix.sqlite`, gitignored): `repo-fix db show "student baseline v1 (2026-11-22)"`.

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

## 2026-11-23: student baseline v3 (repro-only submit gate, measured)

The only change from v2: the submit gate counts only a run of a `scratch_*.py` repro script after
the last edit as a check. Running the existing suite no longer counts. Run `20261123-081218-7a82b0`.

| | v1 | v2 | v3 |
|---|---|---|---|
| Resolved | 85 | 83 | **88** |
| easy / medium / hard | 78 / 89 / 86% | 81 / 89 / 71% | 84 / 91 / 86% |
| Mean steps (all) | 16.6 | 20.4 | 24.0 |
| Episodes with a scratch script | - | 37 | 85 |
| Partial fixes among failures | 10 | 8 | 3 |

- **The gate:** fired in 60 episodes, and 55 of them were then solved.
- **v2 → v3:** 6 instances fixed (`seed-037`, `044`, `063`, `066`, `070`, `099`), 1 newly failed
  (`seed-023`).
- **Not fully paired (corrected 2026-11-23):** this was described as a paired comparison, with
  each episode replaying v2's cached responses until the gate fired. In fact tool output
  contained run-to-run noise (`ls -la` dates, pytest durations, memory addresses), so most
  episodes left the cache at an observation, not at the gate. Part of the +5 is sampling noise.
- **What's left:** "broke a passing test" (5) is now the largest failure mode, followed by
  running out of steps (3).

## 2026-11-23: student baseline v4 (steps-left notice, measured)

The only change from v3: "[N steps left]" on observations during the last 8 steps of the 40-step
budget. Run `20261123-084357-10f3f2`.

| | v3 | v4 |
|---|---|---|
| Resolved | 88 | 86 |
| easy / medium / hard | 84 / 91 / 86% | 84 / 89 / 81% |
| Ran out of steps | 3 | 4 |
| Episodes over 32 steps | 15 | 14 |

- **Not fully paired (corrected):** as with v3, noisy tool output let episodes diverge long
  before step 33, so the −2 is within noise.
- **Flips:** 2 fixed (`seed-023`, `seed-068`), 4 newly failed (`seed-037`, `051`, `063`, `094`).
- **Conclusion:** no sign the countdown helps; the student keeps working rather than wrapping up.
- **Response:** the notice is off, and the student's budget is raised from 40 to 50 steps
  (`student.serving.max_steps`).

## 2026-11-23: student baseline v5 (50-step budget, measured), and a correction

v5 changes one thing from v3: the student's step budget goes from 40 to 50 (the notice stays
off). Run `20261123-143114-69deb5`.

| | v3 | v5 |
|---|---|---|
| Resolved | 88 | 85 |
| easy / medium / hard | 84 / 91 / 86% | 81 / 91 / 76% |
| Ran out of steps | 3 | 2 |
| Solved after step 40 | – | 1 |

- **Flips:** 1 fixed (`seed-023`), 4 newly failed (`seed-025`, `037`, `094`, `095`).
- **Why most of those can't be the budget:** three of the four newly failed instances never
  reached step 40, so the budget can't have caused them.

**Correction: the scaffold comparisons weren't paired.** I expected v5 to replay v3 exactly up to
step 40, since requests are identical and temperature is 0. Comparing the stored episodes instead:
- **69 of 100** diverged at an *observation*: the same action produced different tool output.
- **The causes:** `ls -la` timestamps, pytest durations ("in 0.12s"), object addresses (`0xffff…`)
  and process counts.
- **The effect:** a different observation is a different prompt, so it misses the cache, gets a
  fresh response, and the episode goes its own way.

So v2→v3 (+5), v3→v4 (−2) and v3→v5 (−3) all include ordinary sampling noise. With one sample per
instance, differences of a few points can't be attributed to the change. With these masked, 40
episodes would have been identical and most of the rest would have diverged much later.

**Fix:** `RepoEnv.execute` now masks dates, timestamps, durations and memory addresses in tool
output before the model sees it (`env.normalize_observation`). Grading doesn't use this path.

**What still holds:**
- **The repro gate's behaviour change:** scratch-script use rose from 37 to 85 episodes, and
  partial fixes fell. That's a behaviour shift, not one noisy number.
- **The scores themselves:** every run sits in 83–88%.

**What to decide with:** a multi-sample run (`--samples 3` or more), or several paired runs now
that observations are deterministic.
