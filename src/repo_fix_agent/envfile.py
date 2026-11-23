"""Load KEY=VALUE pairs from a `.env` file into os.environ (no python-dotenv dependency).

The CLI calls load_env() at startup. Variables already set in the environment win, so CI and
shells can override the file. Lookup: $REPOFIX_ENV_FILE, else `.env` in the current directory or
the nearest parent that has one. Copy `.env.example` to `.env` to get started.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def find_env_file(start: Path | None = None) -> Path | None:
    if os.environ.get("REPOFIX_ENV_FILE"):
        p = Path(os.environ["REPOFIX_ENV_FILE"])
        return p if p.is_file() else None
    d = (start or Path.cwd()).resolve()
    for cand in (d, *d.parents):
        if (cand / ".env").is_file():
            return cand / ".env"
    return None


def parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.groups()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        else:
            val = re.sub(r"\s+#.*$", "", val)   # trailing comment on an unquoted value
        out[key] = val
    return out


def load_env(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Returns the names of the variables it set (never their values)."""
    p = Path(path) if path else find_env_file()
    if p is None or not p.is_file():
        return []
    loaded = []
    for k, v in parse(p.read_text()).items():
        if override or k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded
