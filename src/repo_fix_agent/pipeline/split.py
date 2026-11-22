"""§2.2 repo-level splitting + frozen test manifest, §3j evaluation-set construction.

* `split_demos`: train/dev/test by `metadata.repo_group` (fallback `demo.repo`). Groups that share a
  near-dup `metadata.cluster_id` are merged first, so clusters never straddle splits. dev/test are
  filled greedily to match the global bug_type / patch-size / difficulty / language / public mix.
* Frozen test manifest: per-instance sha256 over (instance_id, patch, test_patch, problem_statement)
  plus a content-only hash (catches renamed copies). Every training/hint/retrieval step should call
  `assert_disjoint(demos, manifest_or_ids, purpose)`.
* `build_eval_set` (§3j filters) and `sample_human_review` (markdown checklist for the verified slice).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..schema import Demo, load_demos, save_demos
from .audit import detect_leakage, patch_size

SPLITS = ("train", "dev", "test")
STRATA = ("bug_type", "patch_size", "difficulty", "language", "public")
MANIFEST_VERSION = 1


# ============================================================================================
# helpers
# ============================================================================================
def group_of(d: Demo) -> str:
    return d.metadata.get("repo_group") or d.repo


def strata_of(d: Demo) -> dict[str, str]:
    md = d.metadata
    ps = md.get("patch_size")
    bucket = ps.get("bucket") if isinstance(ps, dict) else patch_size(d.patch)["bucket"]
    return {"bug_type": str(md.get("bug_type") or "unknown"), "patch_size": bucket,
            "difficulty": str(md.get("difficulty") or "unknown"),
            "language": str(md.get("language") or "unknown"), "public": str(md.get("public"))}


def _units(demos: list[Demo]) -> list[list[Demo]]:
    """Repo groups merged when they share a near-dup cluster (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    cluster_first: dict[str, str] = {}
    for d in demos:
        g = "g:" + group_of(d)
        find(g)
        cid = d.metadata.get("cluster_id")
        if cid:
            if cid in cluster_first:
                union(g, cluster_first[cid])
            else:
                cluster_first[cid] = g
    units: dict[str, list[Demo]] = defaultdict(list)
    for d in demos:
        units[find("g:" + group_of(d))].append(d)
    return [units[k] for k in sorted(units)]


def _l1(counts: Counter, n: int, glob: dict[str, float]) -> float:
    keys = set(counts) | set(glob)
    return sum(abs((counts.get(k, 0) / n if n else 0.0) - glob.get(k, 0.0)) for k in keys)


# ============================================================================================
# split
# ============================================================================================
def split_demos(demos: list[Demo], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
                seed: int = 0, frozen_ids: Iterable[str] | None = None) -> dict[str, list[Demo]]:
    """Repo-level split. Deterministic given (demos, ratios, seed).

    `frozen_ids`: instances of an existing frozen test set; their units are forced into test."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios must be three non-negative numbers, got {ratios}")
    tot = sum(ratios)
    ratios = tuple(r / tot for r in ratios)
    demos = sorted(demos, key=lambda d: d.instance_id)
    n = len(demos)
    out: dict[str, list[Demo]] = {s: [] for s in SPLITS}
    if not n:
        return out
    units = _units(demos)
    rng = random.Random(seed)
    order = list(range(len(units)))
    rng.shuffle(order)
    rank = {u: i for i, u in enumerate(order)}

    glob: dict[str, dict[str, float]] = {}
    for f in STRATA:
        c = Counter(strata_of(d)[f] for d in demos)
        glob[f] = {k: v / n for k, v in c.items()}
    unit_strata = [{f: Counter(strata_of(d)[f] for d in u) for f in STRATA} for u in units]

    assigned: dict[int, str] = {}
    counts = {s: {f: Counter() for f in STRATA} for s in SPLITS}
    size = {s: 0 for s in SPLITS}

    def put(i: int, s: str) -> None:
        assigned[i] = s
        size[s] += len(units[i])
        for f in STRATA:
            counts[s][f].update(unit_strata[i][f])

    frozen = set(frozen_ids or ())
    for i, u in enumerate(units):
        if frozen and any(d.instance_id in frozen for d in u):
            put(i, "test")

    eval_splits = [s for s, r in zip(SPLITS, ratios) if s != "train" and r > 0]
    target = {s: (max(1, round(n * r)) if s in eval_splits else 0) for s, r in zip(SPLITS, ratios)}
    train_needed = ratios[0] > 0

    def score(i: int, s: str) -> float:
        m = size[s] + len(units[i])
        return sum(_l1(counts[s][f] + unit_strata[i][f], m, glob[f]) for f in STRATA)

    # alternate test/dev picks so both get a share of the rare strata
    active = [s for s in ("test", "dev") if s in eval_splits]
    while active:
        for s in list(active):
            free = [i for i in range(len(units)) if i not in assigned]
            if train_needed and len(free) <= 1:
                active = []
                break
            if size[s] >= target[s]:
                active.remove(s)
                continue
            fits = [i for i in free if size[s] + len(units[i]) <= target[s]]
            if not fits and size[s] == 0:  # always give a non-empty eval split one (smallest) unit
                smallest = min(len(units[i]) for i in free)
                fits = [i for i in free if len(units[i]) == smallest]
            if not fits:
                active.remove(s)
                continue
            best = min(fits, key=lambda i: (score(i, s), -len(units[i]), rank[i]))
            put(best, s)
    for i in range(len(units)):
        if i not in assigned:
            put(i, "train" if train_needed else min(eval_splits, key=lambda s: size[s] / max(1, target[s])))
    for i, u in enumerate(units):
        out[assigned[i]].extend(u)
    for s in SPLITS:
        out[s].sort(key=lambda d: d.instance_id)
    return out


def check_split_integrity(splits: dict[str, list[Demo]]) -> list[str]:
    """Violations: an instance in two splits, a repo group or a near-dup cluster straddling splits."""
    problems = []
    where: dict[str, dict[str, set[str]]] = {"instance": defaultdict(set), "repo_group": defaultdict(set),
                                             "cluster": defaultdict(set)}
    for s, ds in splits.items():
        for d in ds:
            where["instance"][d.instance_id].add(s)
            where["repo_group"][group_of(d)].add(s)
            if d.metadata.get("cluster_id"):
                where["cluster"][d.metadata["cluster_id"]].add(s)
    for kind, m in where.items():
        for key, ss in sorted(m.items()):
            if len(ss) > 1:
                problems.append(f"{kind} {key!r} appears in {sorted(ss)}")
    return problems


def time_split(demos: list[Demo], test_frac: float = 0.2) -> dict[str, list[Demo]]:
    """Secondary in-repo split: within each repo group, the newest `test_frac` of instances that
    have `metadata.created_at` (ISO string or number) go to in_repo_test. Groups with < 2 dated
    instances stay entirely in in_repo_train."""
    by_group: dict[str, list[Demo]] = defaultdict(list)
    for d in demos:
        by_group[group_of(d)].append(d)
    tr, te = [], []
    for g in sorted(by_group):
        ds = by_group[g]
        dated = sorted((d for d in ds if d.metadata.get("created_at") is not None),
                       key=lambda d: (str(d.metadata["created_at"]), d.instance_id))
        k = min(len(dated) - 1, max(1, round(len(dated) * test_frac))) if len(dated) >= 2 else 0
        newest = {d.instance_id for d in dated[len(dated) - k:]} if k else set()
        for d in ds:
            (te if d.instance_id in newest else tr).append(d)
    return {"in_repo_train": tr, "in_repo_test": te}


# ============================================================================================
# frozen test manifest
# ============================================================================================
class FrozenLeakError(ValueError):
    pass


def instance_hash(d: Demo) -> str:
    blob = json.dumps([d.instance_id, d.patch, d.test_patch, d.problem_statement], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_hash(d: Demo) -> str:
    blob = json.dumps([d.patch, d.test_patch, d.problem_statement], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _manifest(test_demos: list[Demo], created: str) -> dict[str, Any]:
    return {"version": MANIFEST_VERSION, "created": created, "count": len(test_demos),
            "hash": "sha256(json([instance_id, patch, test_patch, problem_statement]))",
            "instances": {d.instance_id: {"sha256": instance_hash(d), "content_sha256": content_hash(d)}
                          for d in sorted(test_demos, key=lambda d: d.instance_id)}}


def write_frozen_manifest(test_demos: list[Demo], path: str | Path, created: str | None = None,
                          overwrite: bool = False) -> dict[str, Any]:
    """Write the frozen test hash list. The frozen set is created once: re-writing an identical
    set keeps the original `created` date; a different set raises unless `overwrite=True`."""
    path = Path(path)
    new = _manifest(test_demos, created or _dt.date.today().isoformat())
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("instances") == new["instances"]:
            return old
        if not overwrite:
            raise FileExistsError(f"{path} already holds a different frozen test set; the frozen set is "
                                  "created once. Pass overwrite=True (--overwrite-frozen) to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    return new


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_frozen_ids(path: str | Path) -> set[str]:
    return set(load_manifest(path)["instances"])


@dataclass
class FrozenCheck:
    missing: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.modified)

    def __str__(self) -> str:
        if self.ok:
            return "frozen test set intact"
        return f"frozen test set changed: missing={self.missing} modified={self.modified}"


def verify_frozen(path: str | Path, demos: list[Demo]) -> FrozenCheck:
    """Check that every manifest instance is present in `demos` with an unchanged hash."""
    man = load_manifest(path)["instances"]
    by_id = {d.instance_id: d for d in demos}
    chk = FrozenCheck()
    for iid, h in sorted(man.items()):
        d = by_id.get(iid)
        if d is None:
            chk.missing.append(iid)
        elif instance_hash(d) != h["sha256"]:
            chk.modified.append(iid)
    return chk


def assert_disjoint(demos: Iterable[Demo | str], frozen: set[str] | Iterable[str] | str | Path,
                    purpose: str) -> None:
    """Raise FrozenLeakError if any frozen test instance is about to be used for `purpose`
    (SFT, RL, verifier training, teacher hints, retrieval index, prompt tuning...).

    `frozen` is a set of ids or a manifest path; with a manifest, renamed copies are also caught
    by content hash."""
    contents: set[str] = set()
    if isinstance(frozen, (str, Path)):
        man = load_manifest(frozen)["instances"]
        ids = set(man)
        contents = {h.get("content_sha256") for h in man.values() if h.get("content_sha256")}
    else:
        ids = set(frozen)
    hits = []
    for d in demos:
        if isinstance(d, str):
            if d in ids:
                hits.append(d)
        elif d.instance_id in ids:
            hits.append(d.instance_id)
        elif contents and content_hash(d) in contents:
            hits.append(f"{d.instance_id} (content of a frozen instance)")
    if hits:
        shown = ", ".join(hits[:10]) + (f", ... (+{len(hits) - 10} more)" if len(hits) > 10 else "")
        raise FrozenLeakError(
            f"{len(hits)} frozen test instance(s) would be used for {purpose}: {shown}. The frozen test "
            "set must never be used for SFT, RL, verifier training, teacher hints, retrieval, or prompt "
            "tuning; remove them (filter by load_frozen_ids) before continuing.")


# ============================================================================================
# §3j evaluation set
# ============================================================================================
def eval_exclusion_reason(d: Demo, require_non_flaky: bool = True, require_no_leakage: bool = True,
                          require_sufficient: bool = False) -> str | None:
    md = d.metadata
    if require_non_flaky:
        if md.get("flaky") is None:
            return "not validated 3x (metadata.flaky missing; run `repo-fix audit`)"
        if md["flaky"]:
            return "flaky"
    if require_no_leakage:
        leak = md.get("leakage")
        if leak is None:
            leak = detect_leakage(d)
        if any(f.startswith("issue ") for f in leak):  # trajectory leakage doesn't affect eval
            return "test leakage in issue"
    if require_sufficient and md.get("issue_sufficient") is not True:
        return ("issue judged insufficient" if md.get("issue_sufficient") is False
                else "issue sufficiency not judged")
    return None


def build_eval_set(demos: list[Demo], require_non_flaky: bool = True, require_no_leakage: bool = True,
                   require_sufficient: bool = False) -> list[Demo]:
    return [d for d in demos if eval_exclusion_reason(d, require_non_flaky, require_no_leakage,
                                                      require_sufficient) is None]


def _fence(text: str, lang: str = "") -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


def sample_human_review(demos: list[Demo], n: int, seed: int = 0) -> str:
    """Markdown checklist for the human-verified slice (§3j): a seeded sample, stratified by
    bug_type round-robin so the slice isn't dominated by the most common type."""
    by_type: dict[str, list[Demo]] = defaultdict(list)
    for d in sorted(demos, key=lambda d: d.instance_id):
        by_type[strata_of(d)["bug_type"]].append(d)
    rng = random.Random(seed)
    for k in sorted(by_type):
        rng.shuffle(by_type[k])
    picked: list[Demo] = []
    keys = sorted(by_type)
    rng.shuffle(keys)
    while len(picked) < min(n, len(demos)):
        for k in keys:
            if by_type[k] and len(picked) < n:
                picked.append(by_type[k].pop())
    L = ["# Human review: verified eval slice", "",
         f"{len(picked)} instance(s) sampled with seed {seed}. For each, tick the boxes that hold; an "
         "instance enters the verified slice only if all three hold.", ""]
    for i, d in enumerate(picked, 1):
        st = strata_of(d)
        L += [f"## {i}. `{d.instance_id}`", "",
              f"- repo: `{d.repo}` | bug_type: {st['bug_type']} | patch: {st['patch_size']} | "
              f"difficulty: {st['difficulty']}",
              f"- FAIL_TO_PASS: {', '.join(f'`{t}`' for t in d.FAIL_TO_PASS) or '(none)'}", "",
              "### Issue", "", _fence(d.problem_statement), "",
              "<details><summary>Gold patch</summary>", "", _fence(d.patch, "diff"), "", "</details>", "",
              "<details><summary>Hidden tests</summary>", "", _fence(d.test_patch, "diff"), "", "</details>", "",
              "### Checklist", "",
              "- [ ] The issue is solvable as written (it alone determines what the fix must do)",
              "- [ ] The tests check the described behavior, not the gold implementation's specifics",
              "- [ ] Another correct fix (different code, same behavior) would pass the tests",
              "", "Notes:", "", "Verdict: keep / exclude", "", "---", ""]
    return "\n".join(L)


# ============================================================================================
# report + CLI
# ============================================================================================
def split_report(splits: dict[str, list[Demo]], seed: int, ratios, manifest_path: str | None = None,
                 extra: dict[str, list[Demo]] | None = None) -> str:
    L = ["# Split report", "", f"seed={seed}, ratios={tuple(ratios)} (of instances, by repo group)", "",
         "| Split | Instances | Repo groups | Near-dup clusters |", "|---|---:|---:|---:|"]
    for s in SPLITS:
        ds = splits[s]
        L.append(f"| {s} | {len(ds)} | {len({group_of(d) for d in ds})} | "
                 f"{len({d.metadata.get('cluster_id') or d.instance_id for d in ds})} |")
    L.append("")
    probs = check_split_integrity(splits)
    L += ["## Integrity", "", "OK: no instance, repo group or near-dup cluster straddles splits." if not probs
          else "\n".join(f"- VIOLATION: {p}" for p in probs), ""]
    if manifest_path:
        L += [f"Frozen test manifest: `{manifest_path}` ({len(splits['test'])} instances). Never train, "
              "hint, retrieve or tune on these.", ""]
    L += ["## Strata coverage", ""]
    for f in STRATA:
        vals = sorted({strata_of(d)[f] for s in SPLITS for d in splits[s]})
        L += [f"**{f}**", "", "| Value | " + " | ".join(SPLITS) + " |", "|---|" + "---:|" * len(SPLITS)]
        for v in vals:
            L.append(f"| {v} | " + " | ".join(str(sum(strata_of(d)[f] == v for d in splits[s]))
                                               for s in SPLITS) + " |")
        L.append("")
    for s in ("dev", "test"):
        L += [f"## {s} repo groups", ""] + [f"- {g}" for g in sorted({group_of(d) for d in splits[s]})] + [""]
    if extra:
        L += ["## In-repo time split (secondary metric, within train)", ""]
        L += [f"- {k}: {len(v)}" for k, v in extra.items()] + [""]
    return "\n".join(L)


def _parse_ratios(s: str) -> tuple[float, float, float]:
    parts = tuple(float(x) for x in s.split(","))
    if len(parts) != 3:
        raise SystemExit(f"--ratios needs three comma-separated numbers, got {s!r}")
    return parts  # type: ignore[return-value]


def cmd_split(a) -> int:
    demos = load_demos(a.demos)
    ratios = _parse_ratios(a.ratios)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "frozen_test.manifest.json"
    frozen = load_frozen_ids(a.frozen) if a.frozen else None
    if a.frozen:
        chk = verify_frozen(a.frozen, demos)
        if not chk.ok:
            print(f"ERROR: {chk}")
            return 1
    splits = split_demos(demos, ratios, a.seed, frozen_ids=frozen)
    for s in SPLITS:
        save_demos(splits[s], out / f"{s}.jsonl")
    try:
        write_frozen_manifest(splits["test"], manifest, overwrite=a.overwrite_frozen)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        return 1
    extra = None
    if any(d.metadata.get("created_at") is not None for d in splits["train"]):
        extra = time_split(splits["train"])
        for k, v in extra.items():
            save_demos(v, out / f"{k}.jsonl")
    (out / "split_report.md").write_text(split_report(splits, a.seed, ratios, str(manifest), extra))
    print(" ".join(f"{s}={len(splits[s])}" for s in SPLITS) + f" -> {out}")
    return 1 if check_split_integrity(splits) else 0


def cmd_eval_set(a) -> int:
    demos = load_demos(a.demos)
    reasons = Counter()
    kept = []
    for d in demos:
        r = eval_exclusion_reason(d, not a.allow_flaky, not a.allow_leakage, a.require_sufficient)
        if r:
            reasons[r] += 1
        else:
            kept.append(d)
    save_demos(kept, a.out)
    print(f"eval set: kept {len(kept)}/{len(demos)} -> {a.out}")
    for r, c in reasons.most_common():
        print(f"  excluded {c}: {r}")
    if a.human_review:
        review_out = Path(a.review_out or Path(a.out).with_suffix(".review.md"))
        review_out.parent.mkdir(parents=True, exist_ok=True)
        review_out.write_text(sample_human_review(kept, a.human_review, a.seed))
        print(f"human review checklist ({min(a.human_review, len(kept))}) -> {review_out}")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("split", help="§2.2 repo-level train/dev/test split + frozen test manifest")
    p.add_argument("demos")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ratios", default="0.8,0.1,0.1", help="train,dev,test fractions")
    p.add_argument("--frozen", default=None, help="existing frozen manifest to honor (forced into test)")
    p.add_argument("--overwrite-frozen", action="store_true")
    p.set_defaults(fn=cmd_split)

    e = subparsers.add_parser("eval-set", help="§3j eval set filters + human-review sample")
    e.add_argument("demos")
    e.add_argument("--out", default="runs/eval/eval_set.jsonl")
    e.add_argument("--human-review", type=int, default=0, metavar="N")
    e.add_argument("--review-out", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--require-sufficient", action="store_true")
    e.add_argument("--allow-flaky", action="store_true")
    e.add_argument("--allow-leakage", action="store_true")
    e.set_defaults(fn=cmd_eval_set)
