"""Record every paid LLM / embedding API call and replay it instead of paying twice.

Every request (model, messages, tools, parameters; never credentials) and its full response go
into a SQLite file, default `.cache/llm/llm_cache.sqlite` (gitignored). The key is a hash of the
provider + endpoint + canonical request JSON, plus an occurrence index:

  * Sampled requests (temperature unset or > 0, which includes every Claude request) use
    record/replay semantics: within one process the n-th identical request maps to the n-th
    recorded response. So k samples of the same prompt (best-of-n, RL groups, distillation) are
    still k different completions the first time, and rerunning the same command replays all k
    from the cache at zero cost.
  * Deterministic requests (temperature == 0) always map to the same entry (plain dedupe).

Responses are stored as JSON with their pydantic class and rebuilt into the same SDK objects, so
callers can't tell a hit from a live call.

  REPOFIX_LLM_CACHE=off              disable (always call the API)
  REPOFIX_LLM_CACHE_PATH=path        cache location
  repo-fix llm-cache stats | clear   inspect / wipe
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

DEFAULT_PATH = Path(".cache") / "llm" / "llm_cache.sqlite"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key TEXT PRIMARY KEY, provider TEXT, model TEXT, request TEXT, response TEXT, response_type TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency_s REAL, created REAL, hits INTEGER DEFAULT 0,
    last_hit REAL
)"""


def _jsonable(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")   # keep nulls: some SDK fields are required-but-nullable
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def canonical(request: dict) -> str:
    return json.dumps(request, sort_keys=True, default=_jsonable, separators=(",", ":"), ensure_ascii=False)


def usage_breakdown(resp: Any) -> dict[str, int]:
    """Billable token counts. Anthropic reports cache reads/writes separately from input_tokens."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    g = lambda *names: next((int(v) for n in names if (v := getattr(u, n, None)) is not None), 0)  # noqa: E731
    out = {"input": g("input_tokens", "prompt_tokens"), "output": g("output_tokens", "completion_tokens"),
           "cache_write": g("cache_creation_input_tokens"), "cache_read": g("cache_read_input_tokens")}
    details = getattr(u, "prompt_tokens_details", None)   # OpenAI-style cached prompt tokens
    if details is not None and getattr(details, "cached_tokens", None):
        out["cache_read"] = int(details.cached_tokens)
        out["input"] -= out["cache_read"]
    return out


def _usage(resp: Any) -> tuple[int | None, int | None]:
    u = getattr(resp, "usage", None)
    if u is None:
        return None, None
    inp = getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", None) or getattr(u, "total_tokens", None)
    out = getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", None)
    return inp, out


class LLMCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()
        self._lock = threading.Lock()
        self._occ: Counter = Counter()
        self.session = Counter()   # hits / misses in this process
        self.usage: dict[str, Counter] = {}   # model -> billable tokens of live (paid) calls this process
        self.saved: dict[str, Counter] = {}   # model -> tokens served from cache this process

    # -- core -----------------------------------------------------------------------------------
    def call(self, provider: str, request: dict, fn: Callable[[], Any]) -> Any:
        canon = canonical(request)
        base = hashlib.sha256(f"{provider}\n{canon}".encode()).hexdigest()
        with self._lock:
            if request.get("temperature") == 0:
                n = 0
            else:
                n = self._occ[base]
                self._occ[base] += 1
            key = f"{base}:{n}"
            row = self._db.execute("SELECT response, response_type FROM calls WHERE key=?", (key,)).fetchone()
            if row is not None:
                self._db.execute("UPDATE calls SET hits = hits + 1, last_hit = ? WHERE key=?", (time.time(), key))
                self._db.commit()
                self.session["hits"] += 1
        if row is not None:
            try:
                resp = _restore(json.loads(row[0]), row[1])
                with self._lock:
                    self.saved.setdefault(str(request.get("model", "")), Counter()).update(usage_breakdown(resp))
                return resp
            except Exception:  # noqa: BLE001 - e.g. SDK upgrade changed the schema: re-fetch
                self.session["stale"] += 1
        t0 = time.time()
        resp = fn()   # errors propagate and nothing is stored
        latency = time.time() - t0
        rtype = f"{type(resp).__module__}:{type(resp).__qualname__}" if hasattr(resp, "model_dump") else "json"
        inp, out = _usage(resp)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls (key, provider, model, request, response, response_type, input_tokens,"
                " output_tokens, latency_s, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (key, provider, str(request.get("model", "")), canon,
                 json.dumps(resp, default=_jsonable, ensure_ascii=False), rtype, inp, out, latency, time.time()))
            self._db.commit()
            self.session["misses"] += 1
            u = self.usage.setdefault(str(request.get("model", "")), Counter())
            u.update(usage_breakdown(resp))
            u["calls"] += 1
        return resp

    # -- maintenance ----------------------------------------------------------------------------
    def stats(self) -> dict:
        q = self._db.execute(
            "SELECT provider, model, COUNT(*), SUM(hits), SUM(COALESCE(input_tokens,0)),"
            " SUM(COALESCE(output_tokens,0)), SUM(hits * COALESCE(input_tokens,0)),"
            " SUM(hits * COALESCE(output_tokens,0)) FROM calls GROUP BY provider, model ORDER BY 3 DESC").fetchall()
        rows = [{"provider": p, "model": m, "entries": n, "hits": h or 0, "input_tokens": i, "output_tokens": o,
                 "saved_input_tokens": si or 0, "saved_output_tokens": so or 0} for p, m, n, h, i, o, si, so in q]
        return {"path": str(self.path), "rows": rows, "session": dict(self.session)}

    def clear(self, provider: str | None = None) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM calls" + (" WHERE provider LIKE ?" if provider else ""),
                                   (f"{provider}%",) if provider else ())
            self._db.commit()
            return cur.rowcount


def _restore(data: Any, rtype: str) -> Any:
    if rtype == "json" or not rtype:
        return data
    mod, _, qual = rtype.partition(":")
    obj = importlib.import_module(mod)
    for part in qual.split("."):
        obj = getattr(obj, part)
    try:
        return obj.model_validate(data)
    except Exception:  # noqa: BLE001
        # SDKs build API responses without strict validation (servers send fields/values newer
        # than the SDK's schema); Stainless models' construct() rebuilds nested models the same way.
        return obj.construct(**data)


# ---- process-wide cache -------------------------------------------------------------------------
_cache: LLMCache | None = None
_cache_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("REPOFIX_LLM_CACHE", "on").lower() not in ("0", "off", "false", "no")


def get_cache() -> LLMCache:
    global _cache
    with _cache_lock:
        path = Path(os.environ.get("REPOFIX_LLM_CACHE_PATH") or DEFAULT_PATH)
        if _cache is None or _cache.path != path:
            _cache = LLMCache(path)
        return _cache


def cached_call(provider: str, request: dict, fn: Callable[[], Any]) -> Any:
    """Return fn()'s result, served from / recorded to the cache. `request` must contain every
    input that affects the output (it is the cache key) and no secrets (it is stored)."""
    if not enabled():
        return fn()
    return get_cache().call(provider, request, fn)


# ---- CLI ------------------------------------------------------------------------------------------
def cmd_llm_cache(a) -> int:
    c = get_cache()
    if a.action == "clear":
        print(f"deleted {c.clear(a.provider)} cached call(s) from {c.path}")
        return 0
    s = c.stats()
    print(f"cache: {s['path']}  (REPOFIX_LLM_CACHE={'on' if enabled() else 'off'})")
    if not s["rows"]:
        print("empty")
    for r in s["rows"]:
        print(f"  {r['provider'][:40]:40} {r['model'][:40]:40} entries {r['entries']:5}  hits {r['hits']:5}  "
              f"tokens in/out {r['input_tokens']}/{r['output_tokens']}  saved {r['saved_input_tokens']}/"
              f"{r['saved_output_tokens']}")
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("llm-cache", help="inspect or clear the LLM request/response cache")
    p.add_argument("action", choices=["stats", "clear"])
    p.add_argument("--provider", default=None, help="clear only entries whose provider starts with this")
    p.set_defaults(fn=cmd_llm_cache)
