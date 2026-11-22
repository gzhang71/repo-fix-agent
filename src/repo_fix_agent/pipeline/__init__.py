"""Data pipeline implementing docs/DEMO_DATA_PROPOSALS.md.

Module -> proposal section:
  audit.py        §2.1 audit (validate x3/flakiness, replay, dedup, near-dup, leakage, secrets, sizes)
  split.py        §2.2 repo-level split, near-dup clusters, frozen test hash list; §3j eval set
  sft_data.py     §3a SFT mix (replay_ok filter, loss masks, thought backfill)
  distill.py      §3b hint ladder + leakage filter, §3c rejection-sampled teacher distillation
  aux_tasks.py    §3d sub-skill tasks (localization, patch-gen, test-writing, repro, critique)
  preferences.py  §3e DPO/KTO pairs
  verifier.py     §3f verifier dataset + best-of-n reranking
  rl.py           §3g RL environment, reward, curriculum, prefix-conditioned rollouts
  augment.py      §3h bug injection, issue rephrasing, repo perturbation
  retrieval.py    §3i exemplar retrieval index

Each module exposes `register(subparsers)` to add its `repo-fix <command>` CLI entry.
Model calls go through injected `Policy` (agent loop) or `llm.TextLLM` (single turn) objects so
everything is testable offline.
"""
