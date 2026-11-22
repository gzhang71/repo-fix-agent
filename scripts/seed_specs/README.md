# Seed instance specs

One module per batch (`batch_NN.py`), each defining `SPECS: list[dict]`. `batch_00.py` is the
reference. Spec keys:

| key | meaning |
|---|---|
| `id` | `seed-NNN-short-slug`; NNN unique, 3 digits |
| `bug_type`, `difficulty` | taxonomy label; `easy` / `medium` / `hard` |
| `issue` | issue text as a user would write it (never names the hidden tests) |
| `files` | `{relative path: source}` for the BUGGY repo, incl. existing passing tests under `tests/` |
| `hidden_test` | source of the hidden test file (`tests/test_issue_NNN.py`), top-level `def test_*` only |
| `p2p` | existing test ids that must keep passing |
| `steps` | `[(thought, Action), ...]` engineer trajectory built with `_helpers` (B, V, R, C, S, F, I, T, SUBMIT) |

Hidden tests that already pass on the buggy repo are moved to PASS_TO_PASS automatically; at
least one must fail. Build one batch: `python scripts/build_seed_dataset.py --batch batch_NN --out /tmp/x.jsonl`.
