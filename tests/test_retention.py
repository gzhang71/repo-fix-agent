"""Anti-forgetting: retention records, replay, forgetting gate, adapter interpolation, observation
normalization, and the retention KL term. Offline."""
import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from repo_fix_agent import env as E
from repo_fix_agent.pipeline import retention as RT
from repo_fix_agent.pipeline import training as T
from repo_fix_agent.rundb import RunDB
from repo_fix_agent.schema import load_demos

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


def _ep(pag, i, ok, tool_err=0):
    traj = [dataclasses.asdict(s) for s in pag.trajectory]
    for s in traj[:tool_err]:
        s["observation"] = "Error: bad args"
    return {**dataclasses.asdict(pag), "instance_id": f"{pag.instance_id}#s{i}", "trajectory": traj,
            "metadata": {"parent_id": pag.instance_id, "resolved": ok, "sample_idx": i}}


def test_episodes_to_retain_keeps_resolved(pag):
    recs = RT.episodes_to_retain([_ep(pag, 0, True), _ep(pag, 1, False)])
    assert len(recs) == 1 and recs[0]["source"] == "retain:episode" and recs[0]["meta"]["retain"]
    assert recs[0]["meta"]["parent_id"] == pag.instance_id and any(recs[0]["loss_mask"])
    assert len(RT.episodes_to_retain([_ep(pag, 0, True), _ep(pag, 1, False)], resolved_only=False)) == 2


def test_general_prompts_and_records():
    prompts = RT.load_general_prompts()
    assert len(prompts) >= 30 and len({p["id"] for p in prompts}) == len(prompts)
    assert len({p["category"] for p in prompts}) >= 8                       # broad, not just Python
    seen = []
    recs = RT.general_records(prompts[:3], lambda m: seen.append(m) or "answer")
    assert len(recs) == 3 and recs[0]["messages"][-1] == {"role": "assistant", "content": "answer"}
    assert recs[0]["loss_mask"] == [False, True] and seen[0][0]["role"] == "user"
    assert RT.general_records(prompts[:2], lambda m: "  ") == []            # empty answers dropped


def test_replay_sample():
    prev = [{"id": f"p{i}", "weight": 2.0, "source": "teacher"} for i in range(100)]
    out = RT.replay_sample(prev, n_current=80, frac=0.2, weight=0.5)
    assert len(out) == 20                                                   # 20 / (80 + 20) = 20%
    assert all(r["weight"] == 1.0 and r["source"] == "replay:teacher" and r["id"].startswith("replay:") for r in out)
    assert RT.replay_sample(prev, 80, 0) == [] and RT.replay_sample([], 80, 0.2) == []
    assert prev[0]["id"] == "p0"                                            # input not mutated


def _res(iid, ok):
    return {"instance_id": iid, "sample": 0, "resolved": ok, "error": None}


def test_forgetting_report_and_gate(pag):
    before = [_res("a", True), _res("b", True), _res("c", False), _res("d", True)]
    after = [_res("a", True), _res("b", False), _res("c", True), _res("d", True)]
    rep = RT.forgetting_report(before, after, [_ep(pag, 0, True)], [_ep(pag, 0, True, tool_err=3)])
    assert rep["forgotten"] == ["b"] and rep["learned"] == ["c"] and rep["delta"] == 0
    assert rep["forgotten_rate"] == pytest.approx(1 / 3) and rep["tool_error_rate_after"] > rep["tool_error_rate_before"]
    why = RT.gate(rep)
    assert any("forgot 1/3" in w for w in why) and any("tool-error" in w for w in why)
    assert RT.gate(rep, max_forgotten_rate=0.5, max_tool_error_increase=1.0) == []
    assert RT.gate(RT.forgetting_report(before, [])) == ["the runs share no instances"]


def test_forgetting_check_cli_and_retain_set_from_run(pag, tmp_path, capsys):
    db = RunDB(tmp_path / "x.sqlite")
    b = db.start_run(model="m", kind="student", label="before")
    a = db.start_run(model="m", kind="student", label="after")
    for iid, x, y in (("i1", True, True), ("i2", True, False), ("i3", False, True)):
        db.add_result(b, _res(iid, x))
        db.add_result(a, _res(iid, y))
    from repo_fix_agent.schema import Demo
    db.add_episode(b, Demo.from_dict(_ep(pag, 0, True)))
    db.close()
    p = argparse.ArgumentParser()
    RT.register(p.add_subparsers(dest="cmd"))
    args = p.parse_args(["forgetting-check", "--db", str(tmp_path / "x.sqlite"), "--before", "before", "--after", "after"])
    assert args.fn(args) == 1 and "forgotten: i2" in capsys.readouterr().out
    args = p.parse_args(["forgetting-check", "--db", str(tmp_path / "x.sqlite"), "--before", "before", "--after",
                         "after", "--max-forgotten-rate", "0.6", "--json"])
    assert args.fn(args) == 0 and json.loads(capsys.readouterr().out)["pass"]
    args = p.parse_args(["retain-set", "--db", str(tmp_path / "x.sqlite"), "--from-run", "before",
                         "--config", "none", "--out", str(tmp_path / "r.jsonl")])
    assert args.fn(args) == 0 and len((tmp_path / "r.jsonl").read_text().splitlines()) == 1
    args = p.parse_args(["retain-set", "--general", "--out", str(tmp_path / "g.jsonl")])   # no --yes: no calls
    assert args.fn(args) == 1 and "dry run" in capsys.readouterr().err


def test_plan_has_anti_forgetting_stages(tmp_path):
    st = {s.name: s for s in T.plan(tmp_path / "r2", tmp_path / "s", prev_round_dir=tmp_path / "r1")}
    for name in ("dev-before", "retain-set", "retain-general", "forgetting-gate"):
        assert name in st
    for name in ("train-sft", "train-dpo", "train-rl"):
        assert "--retain-data" in st[name].external and "--retain-beta" in st[name].external
    assert "--replay" in st["sft-mix"].argv and str(tmp_path / "r1" / "sft.jsonl") in st["sft-mix"].argv
    gate = st["forgetting-gate"].argv
    assert gate[gate.index("--before") + 1] == "r2 dev before" and gate[gate.index("--after") + 1] == "r2 dev after"
    assert "forgetting-gate" in st["rl-rollouts"].needs
    assert "--replay" not in {s.name: s for s in T.plan(tmp_path / "r1", tmp_path / "s")}["sft-mix"].argv


def test_normalize_observation():
    ls = "drwxr-xr-x  5 u g 160 Nov 23 16:12 .\n-rw-r--r--  1 u g  42 Mar  4  2025 a.py"
    assert E.normalize_observation(ls) == "drwxr-xr-x  5 u g 160 Jan  1 00:00 .\n-rw-r--r--  1 u g  42 Jan  1 00:00 a.py"
    assert E.normalize_observation("3 passed, 1 failed in 0.37s") == "3 passed, 1 failed in 0.00s"
    assert E.normalize_observation("Ran 4 tests in 0.001s") == "Ran 4 tests in 0.00s"
    assert E.normalize_observation("<object at 0xffffa8800620>") == "<object at 0x0>"
    assert E.normalize_observation("logged 2026-11-23 16:12:05.123") == "logged 2000-01-01 00:00:00"
    keep = "assert paginate(items, 2, 3) == [3, 4, 5]\nretry in 3 attempts; 0x1F flag"
    assert E.normalize_observation(keep) == keep                            # real content untouched


# ---- adapter interpolation + retention KL (need torch) ---------------------------------------------
def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_interpolate_lora(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    src = tmp_path / "a"
    src.mkdir()
    st.save_file({"x.lora_A.weight": torch.ones(2, 3), "x.lora_B.weight": torch.full((3, 2), 4.0)},
                 str(src / "adapter_model.safetensors"))
    (src / "adapter_config.json").write_text("{}")
    il = _script("interpolate_lora")
    assert il.interpolate(src, 0.25, tmp_path / "b") == 1
    out = st.load_file(str(tmp_path / "b" / "adapter_model.safetensors"))
    assert torch.equal(out["x.lora_B.weight"], torch.full((3, 2), 1.0))   # scaled
    assert torch.equal(out["x.lora_A.weight"], torch.ones(2, 3))          # untouched
    assert (tmp_path / "b" / "adapter_config.json").exists()
    with pytest.raises(ValueError):
        il.interpolate(src, 0.5, src)


def test_retain_kl_zero_without_drift_and_positive_with():
    torch = pytest.importorskip("torch")
    tl = _script("train_lora")
    from contextlib import contextmanager
    from types import SimpleNamespace as NS

    class M(torch.nn.Module):   # "adapter" = an additive table that disable_adapter switches off
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.base = torch.nn.Embedding(9, 9)
            self.delta = torch.nn.Embedding(9, 9)
            torch.nn.init.zeros_(self.delta.weight)
            self.on, self.device = True, "cpu"

        def forward(self, input_ids):
            x = self.base(input_ids)
            return NS(logits=x + self.delta(input_ids) if self.on else x)

        @contextmanager
        def disable_adapter(self):
            self.on = False
            try:
                yield
            finally:
                self.on = True

    m = M()
    ids, w = [1, 2, 3, 4], torch.tensor([0., 1., 1.])
    assert tl.retain_kl(m, ids, w, torch).item() == pytest.approx(0.0, abs=1e-7)
    torch.nn.init.normal_(m.delta.weight, std=2.0)
    kl = tl.retain_kl(m, ids, w, torch)
    assert kl.item() > 0.01
    kl.backward()
    assert m.delta.weight.grad.abs().sum() > 0          # the anchor pulls the adapter back toward the base
    assert tl.retain_kl(m, ids, torch.zeros(3), torch) is None
