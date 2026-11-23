"""Distillation by teacher signal (pipeline/teacher_signal.py, chat_spans, onpolicy shaping). Offline."""
import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from repo_fix_agent import chat_spans as CS
from repo_fix_agent.pipeline import onpolicy as OP
from repo_fix_agent.pipeline import teacher_signal as TS
from repo_fix_agent.pipeline import training as T
from repo_fix_agent.policies import StudentServingConfig
from repo_fix_agent.schema import Action, load_demos

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


class CharTok:
    """One token per character; trivial chat template that renders reasoning before content."""
    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m.get('reasoning_content') or ''}{m.get('content') or ''}</>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False, **kw):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


# ---- mode selection --------------------------------------------------------------------------------
def test_mode_per_capability():
    stu = TS.ModelCaps("s", reasoning="full", tokenizer="fam/tok")
    plain = TS.ModelCaps("s2", reasoning="none", tokenizer="fam/tok")
    claude = TS.default_caps("claude-opus-5-5")
    assert claude.reasoning == "summarized" and claude.logprobs == "none"
    p = TS.select_mode(claude, stu)
    assert p.mode == "reasoning" and p.reasoning_target == "reasoning_content" and p.reasoning_weight == 0.5
    assert TS.select_mode(claude, plain).reasoning_target == "content"
    assert TS.select_mode(TS.default_caps("gpt-5"), stu).mode == "action"

    topk = TS.ModelCaps("big", reasoning="full", logprobs="topk", tokenizer="fam/tok", base_url="http://x")
    p = TS.select_mode(topk, stu)
    assert p.mode == "topk" and p.reasoning_weight == 1.0 and "tokenkl" in p.train_objectives[-1]

    other_tok = dataclasses.replace(topk, tokenizer="other/tok")
    p = TS.select_mode(other_tok, stu)
    assert p.mode == "reasoning" and any("different" in w for w in p.why)
    assert TS.select_mode(dataclasses.replace(topk, base_url=None), stu).mode == "reasoning"

    full = TS.ModelCaps("big", logprobs="full", tokenizer="fam/tok", weights="fam/big")
    assert TS.select_mode(full, stu).mode == "logits"
    assert TS.select_mode(dataclasses.replace(full, weights=None), stu).mode == "action"
    with pytest.raises(ValueError, match="not supported"):
        TS.select_mode(claude, stu, "topk")
    with pytest.raises(ValueError):
        TS.ModelCaps("x", reasoning="maybe")


def test_load_caps_from_repo_config():
    t, s = TS.load_caps(ROOT / "configs/models.yaml")
    assert t.name == "claude-opus-5-5" and t.reasoning == "summarized" and s.reasoning == "full"
    assert TS.select_mode(t, s).mode == "reasoning"


# ---- reasoning vs action records ------------------------------------------------------------------
def test_shape_target():
    action = TS.SignalPlan("action")
    assert TS.shape_target("think", "do it", action) == {"role": "assistant", "content": "do it"}
    rc = TS.SignalPlan("reasoning", reasoning_target="reasoning_content", reasoning_weight=0.5)
    m = TS.shape_target("think", "do it", rc)
    assert m["reasoning_content"] == "think" and m["content"] == "do it" and m["reasoning_weight"] == 0.5
    inline = TS.SignalPlan("reasoning", reasoning_target="content", reasoning_weight=1.0)
    m = TS.shape_target("think", "do it", inline)
    assert m["content"] == "think\n\ndo it" and m["reasoning_text"] == "think"


def test_onpolicy_uses_separate_channels(pag):
    class Teacher:
        def reset(self, system, task):
            pass

        def act(self, obs):
            self.last_reasoning, self.last_text = "deep reasoning", "running tests"
            return "deep reasoning\nrunning tests", Action("bash", {"command": "python -m pytest -q"})

    roll = dataclasses.replace(pag, metadata={"parent_id": pag.instance_id, "sample_idx": 0, "resolved": False})
    rel = OP.relabel_rollout(roll, Teacher, every=3, max_relabels=2)
    assert rel[0].teacher_reasoning == "deep reasoning" and rel[0].teacher_text == "running tests"
    cfg = StudentServingConfig()
    claude, stu = TS.default_caps("claude-opus-5-5"), TS.ModelCaps("s", reasoning="full")
    tgt = OP.to_sft_records(roll, rel, cfg, signal=TS.select_mode(claude, stu))[0]["messages"][-1]
    assert tgt["reasoning_content"] == "deep reasoning" and tgt["content"] == "running tests"
    assert tgt["reasoning_weight"] == 0.5
    tgt = OP.to_sft_records(roll, rel, cfg, signal=TS.select_mode(claude, stu, "action"))[0]["messages"][-1]
    assert "reasoning_content" not in tgt and tgt["content"] == "running tests"
    # export: chat-FT has one weight per message, so sub-message keys are dropped
    ex = T.to_chat_ft(OP.to_sft_records(roll, rel, cfg, signal=TS.select_mode(claude, stu))[0])
    assert all("reasoning_weight" not in m for m in ex["messages"])


def test_reasoning_subspan_weights():
    msgs = [{"role": "user", "content": "fix"},
            {"role": "assistant", "content": "AB", "reasoning_content": "rr", "reasoning_weight": 0.5}]
    text, spans = CS.weighted_spans(CharTok(), msgs, [0.0, 2.0], None)
    ids, tw, _ = CS.token_weights(CharTok(), text, spans)
    got = {c: w for c, w in zip(text, tw) if w}
    assert got["r"] == 1.0 and got["A"] == 2.0 and got["B"] == 2.0
    inline = [{"role": "user", "content": "fix"},
              {"role": "assistant", "content": "zz\n\nAB", "reasoning_text": "zz", "reasoning_weight": 0.0}]
    text, spans = CS.weighted_spans(CharTok(), inline, [0.0, 1.0], None)
    _, tw, _ = CS.token_weights(CharTok(), text, spans)
    assert {c for c, w in zip(text, tw) if w} >= {"A", "B"} and "z" not in {c for c, w in zip(text, tw) if w}
    assert "reasoning_text" not in CS.clean(inline[1])


# ---- token level -----------------------------------------------------------------------------------
class FakeCompletions:
    """Teacher endpoint: echoes the prompt one char per token with log-prob -0.1 * position."""
    def __init__(self, broken=False):
        self.broken, self.requests = broken, []

    def create(self, **req):
        self.requests.append(req)
        text = req["prompt"]
        toks = list(text) + ["!"]
        if self.broken:   # a different tokenizer: two-char tokens
            toks = [text[i:i + 2] for i in range(0, len(text), 2)] + ["!"]
        offs, o = [], 0
        for t in toks:
            offs.append(o)
            o += len(t)
        lps = [None] + [-0.1] * (len(toks) - 1)
        top = [None] + [{t: -0.1, "x": -2.5} for t in toks[1:]]
        return NS(choices=[NS(logprobs=NS(tokens=toks, text_offset=offs, token_logprobs=lps, top_logprobs=top))])


def test_token_record_aligned_and_mismatch(monkeypatch):
    monkeypatch.setenv("REPOFIX_LLM_CACHE", "off")
    caps = TS.ModelCaps("big", logprobs="topk", tokenizer="t", base_url="http://x", top_k=2)
    rec = {"id": "r1", "messages": [{"role": "user", "content": "fix"}, {"role": "assistant", "content": "AB"}],
           "loss_mask": [False, True], "meta": {"parent_id": "p"}}
    out = TS.token_record(rec, CharTok(), TS.CompletionsScorer(caps, NS(completions=FakeCompletions())))
    n = len(out["input_ids"])
    assert len(out["mask"]) == len(out["teacher_logprobs"]) == len(out["teacher_topk"]) == n
    assert out["teacher_logprobs"][0] is None and out["teacher_logprobs"][5] == -0.1
    trained = [i for i, m in enumerate(out["mask"]) if m]
    assert "".join(chr(out["input_ids"][i]) for i in trained).startswith("AB")
    assert sorted(t for t, _ in out["teacher_topk"][trained[0]]) == sorted([ord("A"), ord("x")])
    with pytest.raises(TS.TokenizerMismatch):
        TS.token_record(rec, CharTok(), TS.CompletionsScorer(caps, NS(completions=FakeCompletions(broken=True))))


def test_token_distill_cli(pag, tmp_path, monkeypatch, capsys):
    roll = dataclasses.replace(pag, instance_id="r#s0", metadata={"parent_id": pag.instance_id, "resolved": True})
    rp = tmp_path / "roll.jsonl"
    rp.write_text(json.dumps(dataclasses.asdict(roll)) + "\n")
    p = argparse.ArgumentParser()
    TS.register(p.add_subparsers(dest="cmd"))
    claude_cfg = ROOT / "configs/models.yaml"
    a = p.parse_args(["token-distill", "--rollouts", str(rp), "--out", str(tmp_path / "o.jsonl"),
                      "--config", str(claude_cfg)])
    assert a.fn(a) == 2 and "reasoning" in capsys.readouterr().err        # Claude: no token-level path
    a = p.parse_args(["token-distill", "--rollouts", str(rp), "--render-only", "--out", str(tmp_path / "r.jsonl"),
                      "--config", str(claude_cfg)])
    assert a.fn(a) == 0
    rec = json.loads((tmp_path / "r.jsonl").read_text().splitlines()[0])
    assert rec["meta"]["parent_id"] == pag.instance_id and rec["loss_mask"].count(True) == len(pag.trajectory)

    cfg = tmp_path / "models.yaml"
    cfg.write_text("distillation_signal:\n  teacher: {name: big, reasoning: full, logprobs: topk, tokenizer: t, "
                   "base_url: 'http://x'}\n  student: {name: s, reasoning: full, tokenizer: t}\n")
    monkeypatch.setattr(TS, "CompletionsScorer", lambda *a, **k: pytest.fail("dry run must not call the teacher"))
    a = p.parse_args(["token-distill", "--rollouts", str(rp), "--out", str(tmp_path / "o.jsonl"), "--config", str(cfg)])
    assert a.fn(a) == 0 and "dry run" in capsys.readouterr().out
    a = p.parse_args(["distill-signal", "--config", str(cfg), "--json"])
    assert a.fn(a) == 0 and json.loads(capsys.readouterr().out)["mode"] == "topk"


def test_plan_per_signal(tmp_path):
    names = lambda sig: [s.name for s in T.plan(tmp_path / "r", tmp_path / "s", signal=sig)]  # noqa: E731
    base = names("reasoning")
    assert "token-distill" not in base and "train-gkd" not in base
    on = next(s for s in T.plan(tmp_path / "r", tmp_path / "s", signal="action") if s.name == "onpolicy")
    assert on.argv[on.argv.index("--signal") + 1] == "action"
    tk = T.plan(tmp_path / "r", tmp_path / "s", signal="topk")
    n = [s.name for s in tk]
    assert n.index("train-dpo") < n.index("train-token") < n.index("deploy") and "token-distill" in n
    assert next(s for s in tk if s.name == "token-distill").paid
    assert next(s for s in tk if s.name == "deploy").needs == ["train-token"]
    lg = T.plan(tmp_path / "r", tmp_path / "s", signal="logits", teacher_weights="fam/big")
    gkd = next(s for s in lg if s.name == "train-gkd")
    assert "--teacher-base fam/big" in gkd.external and not next(s for s in lg if s.name == "render-rollouts").paid


# ---- training losses (need torch; skipped otherwise) ----------------------------------------------
def _train_lora():
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_lora", ROOT / "scripts/train_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_models(torch):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(11, 11)
            torch.nn.init.normal_(self.emb.weight, std=3.0)
            self.device = "cpu"

        def forward(self, input_ids):
            return NS(logits=self.emb(input_ids))
    torch.manual_seed(0)
    return M(), M()


def test_gkd_loss_matches_reference_kl():
    torch = pytest.importorskip("torch")
    tl = _train_lora()
    stu, tea = _toy_models(torch)
    ids, w = [1, 4, 2, 7, 3, 9], torch.tensor([0., 1., 1., 0., 1.])
    s, t = tl.all_logps(stu, ids, torch)[w != 0], tl.all_logps(tea, ids, torch)[w != 0]
    assert tl.gkd_loss(stu, tea, ids, w, 1.0, torch).item() == pytest.approx((s.exp() * (s - t)).sum(-1).mean().item(), 1e-5)
    assert tl.gkd_loss(stu, tea, ids, w, 0.0, torch).item() == pytest.approx((t.exp() * (t - s)).sum(-1).mean().item(), 1e-5)
    assert tl.gkd_loss(stu, stu, ids, w, 0.5, torch).item() == pytest.approx(0.0, abs=1e-6)
    opt = torch.optim.Adam(stu.parameters(), lr=0.1)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = tl.gkd_loss(stu, tea, ids, w, 1.0, torch)
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first / 2


def test_tokenkl_raises_teacher_preferred_tokens():
    torch = pytest.importorskip("torch")
    tl = _train_lora()
    stu, _ = _toy_models(torch)
    ids = [1, 4, 2, 7, 3, 9]
    rec = {"input_ids": ids, "mask": [0, 1, 1, 1, 1, 1], "teacher_logprobs": [None, -0.01, -0.01, -0.01, -0.01, -0.01],
           "teacher_topk": [None] + [[[t, -0.01], [0, -5.0]] for t in ids[1:]]}
    before = tl.token_logps(stu, ids, torch).sum().item()
    opt = torch.optim.Adam(stu.parameters(), lr=0.05)
    for _ in range(30):
        opt.zero_grad()
        tl.tokenkl_loss(stu, rec, 100, 5.0, 0.5, torch).backward()
        opt.step()
    assert tl.token_logps(stu, ids, torch).sum().item() > before   # moved toward tokens the teacher rates likely
    assert tl.tokenkl_loss(stu, {**rec, "teacher_logprobs": [None] * 6}, 100, 5.0, 0.5, torch) is None
