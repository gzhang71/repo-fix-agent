"""pipeline/embed.py backends, hybrid retrieval, semantic dedup."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from repo_fix_agent.pipeline import audit as A
from repo_fix_agent.pipeline import embed as E
from repo_fix_agent.pipeline.retrieval import RetrievalIndex
from repo_fix_agent.schema import Demo, load_demos

ROOT = Path(__file__).resolve().parents[1]


class ConceptEmbedder:
    """Deterministic stand-in for a dense model: maps synonyms onto shared concept axes."""
    spec = "fake:concepts"
    AXES = [{"page", "pagination", "screen", "browsing"}, {"tax", "levy"}, {"coupon", "promo", "discount"},
            {"crash", "error", "exception", "fails"}, {"date", "day", "calendar"}]

    def embed(self, texts):
        out = []
        for t in texts:
            words = {w.strip(".,!?`'\"").lower() for w in t.split()}
            v = [float(len(words & ax)) for ax in self.AXES] + [0.01]
            out.append(E._normalise(v))
        return out


def _demo(iid, issue, patch="", repo="r"):
    return Demo(iid, repo, None, issue, "pytest", "", [], [], patch)


def test_hashing_embedder_normalised_and_deterministic():
    e = E.make_embedder("hash:256")
    a, b = e.embed(["parseDate fails on ISO weeks", "parseDate fails on ISO weeks"])
    assert a == b and len(a) == 256 and math.isclose(E.dot(a, a), 1.0, rel_tol=1e-6)
    near, far = e.embed(["pagination offset", "paginate offset"])[1], e.embed(["timezone rounding"])[0]
    assert E.dot(a, a) > E.dot(near, far)


def test_make_embedder_rejects_unknown():
    with pytest.raises(ValueError):
        E.make_embedder("word2vec:foo")


def test_api_embedders_with_fake_clients():
    voy = SimpleNamespace(embed=lambda texts, model, input_type: SimpleNamespace(embeddings=[[3, 4]] * len(texts)))
    v = E.VoyageEmbedder("voyage-code-3", batch_size=2, client=voy)
    assert v.embed(["a", "b", "c"]) == [[0.6, 0.8]] * 3 and v.spec == "voyage:voyage-code-3"
    oa = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=[0, 2])] * len(input))))
    assert E.OpenAIEmbedder(client=oa).embed(["x"]) == [[0.0, 1.0]]


def test_similar_pairs_numpy_and_pure_python(monkeypatch):
    vecs = [E._normalise(v) for v in ([1, 0], [0.99, 0.1], [0, 1])]
    want = [(0, 1)]
    assert [(i, j) for i, j, _ in E.similar_pairs(vecs, 0.95)] == want
    import builtins
    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__",
                        lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "numpy"
                        else real_import(name, *a, **k))
    assert [(i, j) for i, j, _ in E.similar_pairs(vecs, 0.95)] == want


def test_hybrid_retrieval_matches_paraphrase_tfidf_misses():
    demos = [_demo("pag", "Pagination skips the first page of results"),
             _demo("tax", "Tax computed on the full price when a coupon is used"),
             _demo("cal", "Calendar shows the wrong day for leap years")]
    q = "Promo applied but the levy ignores it"
    lexical = RetrievalIndex.build(demos)
    assert all(h.entry["instance_id"] != "tax" for h in lexical.query(q, k=1))
    hybrid = RetrievalIndex.build(demos, embedder=ConceptEmbedder(), alpha=0.7)
    assert hybrid.query(q, k=1)[0].entry["instance_id"] == "tax"


def test_index_roundtrip_records_embedder(tmp_path):
    demos = [_demo("a", "parse fails on empty input"), _demo("b", "rounding is off for money")]
    idx = RetrievalIndex.build(demos, embedder="hash:128", alpha=0.5)
    idx.save(tmp_path / "i.json")
    back = RetrievalIndex.load(tmp_path / "i.json")
    assert back.embedder_spec == "hash:128" and back.alpha == 0.5 and "vec" in back.entries[0]
    assert [h.entry["instance_id"] for h in back.query("empty input parse", k=1)] == ["a"]
    # v1 (tf-idf only) indexes still load
    plain = RetrievalIndex.build(demos)
    plain.save(tmp_path / "p.json")
    assert RetrievalIndex.load(tmp_path / "p.json").embedder_spec is None


def test_semantic_dup_pairs_catches_renamed_fix_and_paraphrase():
    seed = load_demos(ROOT / "data/demos/seed.jsonl")[:30]
    o = seed[5]
    twin = copy.deepcopy(o)
    twin.instance_id = "twin"
    twin.problem_statement = "Completely different words describing it."
    pairs = A.semantic_dup_pairs(seed + [twin], "hash:1024", threshold=0.99, patch_threshold=0.97)
    assert ("twin" in {p[0] for p in pairs} | {p[1] for p in pairs})
    assert not A.semantic_dup_pairs(seed, "hash:1024", threshold=0.99, patch_threshold=0.99)


def test_near_dup_clusters_union_semantic_pairs():
    ds = [_demo("a", "alpha beta gamma delta epsilon zeta"), _demo("b", "completely unrelated words here now")]
    assert len(set(A.near_dup_clusters(ds).values())) == 2
    merged = A.near_dup_clusters(ds, semantic_pairs=[("a", "b", 0.95)])
    assert merged["a"] == merged["b"] == "cluster:a"


def test_audit_records_semantic_dups():
    ds = [_demo("a", "Tax computed on full price with coupon", "+x = 1\n"),
          _demo("b", "Levy ignores the promo discount", "+y = 2\n"),
          _demo("c", "Calendar day wrong in leap years", "+z = 3\n")]
    kept, rep = A.audit_demos(ds, runs=0, replay_trajectories=False, embedder=ConceptEmbedder(),
                              semantic_threshold=0.9, patch_threshold=1.01)
    md = {d.instance_id: d.metadata for d in kept}
    assert md["a"]["cluster_id"] == md["b"]["cluster_id"] != md["c"]["cluster_id"]
    assert md["a"]["semantic_dups"][0][0] == "b"
