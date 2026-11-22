"""Text embeddings for retrieval (§3i) and semantic near-duplicate detection (§1 audit).

Backends, selected by a spec string so indexes can record which one built them:

  hash[:DIM]            offline default, no dependencies. Signed feature hashing of identifier
                        sub-tokens, token bigrams and character 4-grams. Still lexical, but the
                        char-grams make "paginate"/"pagination"/"page" overlap and sub-tokens
                        make `parseDate` match "parse date". Use it for tests and air-gapped runs.
  st:MODEL              sentence-transformers, local, e.g. st:BAAI/bge-small-en-v1.5 or
                        st:jinaai/jina-embeddings-v2-base-code   (pip install sentence-transformers)
  voyage:MODEL          Voyage AI API, e.g. voyage:voyage-code-3  (pip install voyageai; VOYAGE_API_KEY)
  openai:MODEL          OpenAI API, e.g. openai:text-embedding-3-small  (pip install openai)

The dense backends are the ones that match *meaning* ("crashes on empty list" ~ "fails when no
items"). All embedders return L2-normalised vectors so a dot product is the cosine.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

Vector = list[float]


class Embedder(Protocol):
    spec: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def _normalise(v: Sequence[float]) -> Vector:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [float(x) / n for x in v]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---- offline hashing embedder ---------------------------------------------------------------
_WORD = re.compile(r"[A-Za-z0-9_]+")
_SUB = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def _subtokens(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall(text or ""):
        parts = [p.lower() for chunk in raw.split("_") for p in _SUB.findall(chunk)]
        out.extend(p for p in parts if len(p) > 1)
    return out


class HashingEmbedder:
    def __init__(self, dim: int = 1024, char_n: int = 4):
        self.dim, self.char_n = dim, char_n
        self.spec = f"hash:{dim}"

    def _features(self, text: str) -> dict[str, float]:
        toks = _subtokens(text)
        feats: dict[str, float] = {}

        def add(key: str, w: float) -> None:
            feats[key] = feats.get(key, 0.0) + w

        for t in toks:
            add("w:" + t, 1.0)
            padded = f"<{t}>"
            for i in range(max(1, len(padded) - self.char_n + 1)):
                add("c:" + padded[i:i + self.char_n], 0.35)
        for a, b in zip(toks, toks[1:]):
            add(f"b:{a}_{b}", 0.6)
        return feats

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for key, w in self._features(text).items():
                h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                v[h % self.dim] += sign * ((1.0 + math.log(w)) if w >= 1 else w)  # sublinear tf
            out.append(_normalise(v))
        return out


# ---- dense backends -------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("st:* embedders need `pip install sentence-transformers`") from e
        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.spec = f"st:{model}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-code-3", batch_size: int = 64, client=None):
        if client is None:
            try:
                import voyageai
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError("voyage:* embedders need `pip install voyageai` and VOYAGE_API_KEY") from e
            client = voyageai.Client()
        self.client, self.model, self.batch_size = client, model, batch_size
        self.spec = f"voyage:{model}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            r = self.client.embed(list(texts[i:i + self.batch_size]), model=self.model, input_type="document")
            out.extend(_normalise(v) for v in r.embeddings)
        return out


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128, client=None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError("openai:* embedders need `pip install openai`") from e
            client = OpenAI()
        self.client, self.model, self.batch_size = client, model, batch_size
        self.spec = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            r = self.client.embeddings.create(model=self.model, input=list(texts[i:i + self.batch_size]))
            out.extend(_normalise(d.embedding) for d in r.data)
        return out


def make_embedder(spec: str | None) -> Embedder:
    spec = (spec or "hash").strip()
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashingEmbedder(int(arg) if arg else 1024)
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "BAAI/bge-small-en-v1.5")
    if kind == "voyage":
        return VoyageEmbedder(arg or "voyage-code-3")
    if kind == "openai":
        return OpenAIEmbedder(arg or "text-embedding-3-small")
    raise ValueError(f"unknown embedder spec {spec!r} (hash[:DIM] | st:MODEL | voyage:MODEL | openai:MODEL)")


# ---- similarity search ----------------------------------------------------------------------
def similar_pairs(vecs: Sequence[Vector], threshold: float, block: int = 2048) -> list[tuple[int, int, float]]:
    """All (i, j, cos) with i < j and cos >= threshold. Uses numpy when available (blocked matrix
    products, fine for ~100k vectors); falls back to pure Python for small inputs."""
    n = len(vecs)
    if n < 2:
        return []
    try:
        import numpy as np
    except ImportError:
        if n > 3000:
            raise RuntimeError("semantic dedup over >3000 items needs numpy (pip install numpy)")
        return [(i, j, c) for i in range(n) for j in range(i + 1, n)
                if (c := dot(vecs[i], vecs[j])) >= threshold]
    m = np.asarray(vecs, dtype=np.float32)
    out = []
    for s in range(0, n, block):
        sims = m[s:s + block] @ m.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = s + r
            if c > i:
                out.append((i, c, float(sims[r, c])))
    return out
