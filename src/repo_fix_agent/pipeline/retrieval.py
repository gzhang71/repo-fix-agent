"""§3i Retrieval at inference: index train demos, retrieve similar past fixes, and append 1-2
compressed exemplars (issue gist + files changed + patch summary, never full trajectories) to the
task prompt.

Scoring is hybrid: alpha * dense-embedding cosine + (1 - alpha) * TF-IDF cosine (+ same-repo
boost). TF-IDF keeps exact symbol matches strong; the embedder (see embed.py: hash | st:MODEL |
voyage:MODEL | openai:MODEL) matches paraphrases. Without --embedder the index is TF-IDF only.

CLI:
  repo-fix retrieval-index IN.jsonl --out index.json [--embedder st:BAAI/bge-small-en-v1.5] [--exclude-ids FILE]
  repo-fix retrieve --index index.json --issue-file F [--repo R] [--k 2]
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..schema import Demo, load_demos
from .embed import Embedder, dot, make_embedder
from .verifier import ExcludedInstanceError, check_not_excluded, load_exclude_ids, parse_patch, record_patch

STOPWORDS = set("""a an the and or but if of to in on at by for with from into as is are was were be been being
it its this that these those i we you he she they them our your their my me us do does did not no yes so than
then there here when where which who what why how can could should would will just also only even still
instead some any all each every about after before while because via get gets got using use used""".split())

def _stem(w: str) -> str:
    """Tiny suffix stripper so page/pages/paging/paged and skip/skips/skipped collide."""
    if w.isdigit() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; identifiers are split on '_' and camelCase, light suffix stemming."""
    out = []
    for raw in re.findall(r"[A-Za-z0-9_]+", text or ""):
        parts = [p for chunk in raw.split("_") for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", chunk)]
        if len(parts) > 1:
            parts.append(raw)  # keep the whole identifier too: exact symbol matches are strong evidence
        for p in parts:
            w = p.lower()
            if w in STOPWORDS or len(w) < 2:
                continue
            out.append(_stem(w))
    return out


def summarize_patch(patch: str, max_lines: int = 8, max_line_chars: int = 100) -> str:
    out = []
    for line in (patch or "").splitlines():
        if line.startswith(("+++", "---")) or line[:1] not in "+-":
            continue
        body = " ".join(line[1:].split())
        if not body:
            continue
        out.append(f"{line[0]} {body[:max_line_chars]}")
        if len(out) >= max_lines:
            out.append("  ...")
            break
    return "\n".join(out)


def issue_gist(text: str, max_chars: int = 200) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= max_chars else t[: max_chars - 3].rstrip() + "..."


def _norm_repo(repo: str | None) -> str:
    r = (repo or "").strip().rstrip("/")
    r = r[:-4] if r.endswith(".git") else r
    return r.rsplit("/", 1)[-1].lower() if r else ""


@dataclass
class Hit:
    entry: dict[str, Any]
    score: float
    cosine: float
    same_repo: bool


class RetrievalIndex:
    def __init__(self, entries: list[dict[str, Any]] | None = None, embedder_spec: str | None = None,
                 alpha: float = 0.6, embedder: Embedder | None = None):
        self.entries = entries or []
        self.embedder_spec = embedder.spec if embedder is not None else embedder_spec
        self.alpha = alpha if self.embedder_spec else 0.0
        self._embedder = embedder
        self._reindex()

    @property
    def embedder(self) -> Embedder | None:
        if self._embedder is None and self.embedder_spec:
            self._embedder = make_embedder(self.embedder_spec)
        return self._embedder

    def _reindex(self) -> None:
        n = len(self.entries)
        df: Counter = Counter()
        for e in self.entries:
            df.update(set(e["tf"]))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vecs = [self._vector(e["tf"]) for e in self.entries]

    def _vector(self, tf: dict[str, int]) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf[t] for t, c in tf.items() if t in self.idf and c > 0}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    @classmethod
    def build(cls, demos: Iterable[Demo], exclude_ids: set[str] | None = None,
              embedder: Embedder | str | None = None, alpha: float = 0.6) -> "RetrievalIndex":
        demos = list(demos)
        check_not_excluded(demos, exclude_ids, "retrieval indexing")
        entries, seen = [], set()
        for d in demos:
            md = d.metadata or {}
            key = str(md.get("parent_id") or d.instance_id)
            if md.get("resolved") is False or key in seen:  # only index working fixes, one per instance
                continue
            seen.add(key)
            patch = record_patch(d)
            entries.append({
                "instance_id": key, "repo": d.repo,
                "files_changed": md.get("files_changed") or parse_patch(patch)["files"],
                "patch_summary": summarize_patch(patch), "bug_type": md.get("bug_type") or "unknown",
                "issue_gist": issue_gist(d.problem_statement), "tf": dict(Counter(tokenize(d.problem_statement))),
                "_text": d.problem_statement,
            })
        emb = make_embedder(embedder) if isinstance(embedder, str) else embedder
        texts = [e.pop("_text") for e in entries]
        if emb is not None and entries:
            for e, v in zip(entries, emb.embed(texts)):
                e["vec"] = [round(x, 6) for x in v]
        return cls(entries, alpha=alpha, embedder=emb)

    def query(self, issue: str, repo: str | None = None, k: int = 2, same_repo_boost: float = 0.2,
              exclude_instance: str | None = None, min_score: float = 0.0) -> list[Hit]:
        q = self._vector(dict(Counter(tokenize(issue))))
        qd = self.embedder.embed([issue])[0] if self.alpha > 0 and self.embedder is not None else None
        qrepo = _norm_repo(repo)
        hits = []
        for e, v in zip(self.entries, self._vecs):
            if exclude_instance and e["instance_id"] == exclude_instance:
                continue
            lex = sum(x * v.get(t, 0.0) for t, x in q.items())
            cos = lex if qd is None or "vec" not in e else \
                self.alpha * dot(qd, e["vec"]) + (1 - self.alpha) * lex
            same = bool(qrepo) and _norm_repo(e["repo"]) == qrepo
            score = cos + (same_repo_boost if same else 0.0)
            if score > min_score:
                hits.append(Hit(e, score, cos, same))
        hits.sort(key=lambda h: (-h.score, h.entry["instance_id"]))
        return hits[:k]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": 2, "embedder": self.embedder_spec, "alpha": self.alpha,
                                          "entries": self.entries}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "RetrievalIndex":
        data = json.loads(Path(path).read_text())
        return cls(data["entries"], data.get("embedder"), data.get("alpha", 0.6), embedder)

    def __len__(self) -> int:
        return len(self.entries)


EXEMPLAR_HEADER = ("Similar past fixes from this codebase family (for orientation only; they may not apply - "
                   "verify against the actual code):")


def _format_one(i: int, h: Hit) -> str:
    e = h.entry
    lines = [f"[{i}] Issue: {e['issue_gist']}",
             f"    Bug type: {e['bug_type']}; files changed: {', '.join(e['files_changed']) or '(unknown)'}"]
    if e.get("patch_summary"):
        lines.append("    Patch summary:")
        lines += [f"      {ln}" for ln in e["patch_summary"].splitlines()]
    return "\n".join(lines)


def format_exemplars(hits: Sequence[Hit], max_chars: int = 1500) -> str:
    if not hits:
        return ""
    out = EXEMPLAR_HEADER
    for i, h in enumerate(hits, 1):
        block = _format_one(i, h)
        if len(out) + 1 + len(block) > max_chars:
            if i == 1:  # always keep at least one, truncated
                out = (out + "\n" + block)[: max_chars - 3].rstrip() + "..."
            break
        out += "\n" + block
    return out


def augment_task(problem_statement: str, index: RetrievalIndex, repo: str | None = None, k: int = 2,
                 max_chars: int = 1500, min_score: float = 0.05) -> str:
    block = format_exemplars(index.query(problem_statement, repo, k, min_score=min_score), max_chars)
    return f"{problem_statement.rstrip()}\n\n{block}" if block else problem_statement


# ---- CLI ------------------------------------------------------------------------------------
def cmd_retrieval_index(a) -> int:
    try:
        idx = RetrievalIndex.build(load_demos(a.demos), load_exclude_ids(a.exclude_ids),
                                   embedder=a.embedder, alpha=a.alpha)
    except ExcludedInstanceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    idx.save(a.out)
    print(f"indexed {len(idx)} instances ({idx.embedder_spec or 'tf-idf only'}) -> {a.out}")
    return 0


def cmd_retrieve(a) -> int:
    idx = RetrievalIndex.load(a.index)
    issue = Path(a.issue_file).read_text() if a.issue_file else a.issue
    hits = idx.query(issue, a.repo, a.k)
    if a.json:
        print(json.dumps([{"instance_id": h.entry["instance_id"], "score": h.score, "cosine": h.cosine,
                           "same_repo": h.same_repo, "files_changed": h.entry["files_changed"]} for h in hits]))
    else:
        print(format_exemplars(hits, a.max_chars) or "(no similar fixes)")
    return 0


def register(subparsers) -> None:
    b = subparsers.add_parser("retrieval-index", help="§3i build a hybrid TF-IDF + embedding exemplar index")
    b.add_argument("demos")
    b.add_argument("--embedder", default=None,
                   help="hash[:DIM] | st:MODEL | voyage:MODEL | openai:MODEL (default: TF-IDF only)")
    b.add_argument("--alpha", type=float, default=0.6, help="weight of the embedding score in [0, 1]")
    b.add_argument("--out", default="runs/retrieval/index.json")
    b.add_argument("--exclude-ids", default=None)
    b.set_defaults(fn=cmd_retrieval_index)

    q = subparsers.add_parser("retrieve", help="§3i print exemplars for an issue")
    q.add_argument("--index", required=True)
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--issue")
    g.add_argument("--issue-file")
    q.add_argument("--repo", default=None)
    q.add_argument("--k", type=int, default=2)
    q.add_argument("--max-chars", type=int, default=1500)
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_retrieve)
