"""Student improvements: reasoning replay, spec-check scaffold + submit gate, multi-sample eval,
execution reranking, on-policy distillation, training export / plan / deploy, LoRA token weights."""
import argparse
import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from repo_fix_agent import agent as A
from repo_fix_agent import evaluation as EV
from repo_fix_agent import policies as P
from repo_fix_agent.env import RepoEnv
from repo_fix_agent.pipeline import onpolicy as OP
from repo_fix_agent.pipeline import rerank as RR
from repo_fix_agent.pipeline import sft_data as S
from repo_fix_agent.pipeline import training as T
from repo_fix_agent.policies import ScriptedPolicy, StudentServingConfig
from repo_fix_agent.schema import Action, Demo, Step, load_demos

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setenv("REPOFIX_SANDBOX", "local")


@pytest.fixture(scope="module")
def pag():
    return next(d for d in load_demos(ROOT / "data/demos/seed.jsonl") if d.instance_id == "seed-001-pagination-offset")


def script(d):
    return [(s.thought, s.action) for s in d.trajectory]


# ---- 1. reasoning replay -------------------------------------------------------------------------
class ReasoningClient:
    def __init__(self):
        self.requests = []
        self.base_url = "fake"
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(copy.deepcopy(kw))
        msg = NS(content="", reasoning_content=f"plan step {len(self.requests)}",
                 tool_calls=[NS(id=f"c{len(self.requests)}", function=NS(name="bash", arguments='{"command": "ls"}'))])
        return NS(model="m", choices=[NS(message=msg)])


@pytest.mark.parametrize("keep", [True, False])
def test_reasoning_kept_and_resent(keep):
    client = ReasoningClient()
    pol = P.LocalModelPolicy(StudentServingConfig(keep_reasoning=keep), client=client)
    pol.reset("sys", "task")
    thought, _ = pol.act(None)
    pol.act("out")
    prev = client.requests[1]["messages"][2]
    if keep:
        assert thought == "plan step 1" and prev["reasoning_content"] == "plan step 1"
    else:
        assert thought == "" and "reasoning_content" not in prev


def test_reasoning_of_variants():
    assert P.reasoning_of(NS(reasoning_content=" x ")) == "x"
    assert P.reasoning_of(NS(reasoning=None, model_extra={"reasoning": "y"})) == "y"
    assert P.reasoning_of(NS(content="z")) == ""


def test_student_temperature_override(monkeypatch):
    monkeypatch.setattr(P.StudentServingConfig, "load", classmethod(lambda cls, path="": cls()))
    monkeypatch.setattr(P.LocalModelPolicy, "__init__", lambda self, cfg=None, client=None: setattr(self, "cfg", cfg))
    assert P.make_policy("student", temperature=0.7).cfg.temperature == 0.7
    assert P.make_policy("student").cfg.temperature == 0.0


# ---- 2. spec check + submit gate ----------------------------------------------------------------
def test_options_prompt_and_student_defaults():
    assert A.AgentOptions().system_prompt() == A.SYSTEM_PROMPT
    assert "scratch_" in A.AgentOptions(spec_check=True).system_prompt()
    opts = A.student_options(str(ROOT / "configs/models.yaml"))
    assert opts.spec_check and opts.submit_gate


def test_submit_gate_bounces_once_after_unchecked_edit():
    g = A.SubmitGate(True)
    assert g.intercept("submit") is None                      # no edit yet
    g.observe("str_replace", {"path": "pkg/a.py"})
    assert g.intercept("submit") == A.SUBMIT_NUDGE
    assert g.intercept("submit") is None                      # only once
    g2 = A.SubmitGate(True)
    g2.observe("str_replace", {"path": "pkg/a.py"})
    g2.observe("bash", {"command": "python -m pytest -q"})
    g2.observe("run_tests", {})
    assert g2.intercept("submit") == A.SUBMIT_NUDGE           # the visible suite alone doesn't count
    for cmd in ("python scratch_repro.py", "cd /repo && python3 scratch_edge.py", "pytest -q scratch_t.py"):
        g = A.SubmitGate(True)
        g.observe("str_replace", {"path": "pkg/a.py"})
        g.observe("bash", {"command": cmd})
        assert g.intercept("submit") is None, cmd             # a repro script ran after the edit
    g4 = A.SubmitGate(True)
    g4.observe("bash", {"command": "python scratch_repro.py"})
    g4.observe("str_replace", {"path": "pkg/a.py"})           # edit after the repro: unchecked again
    assert g4.intercept("submit") == A.SUBMIT_NUDGE
    assert not A._is_check("bash", {"command": "cat scratch_repro.py"})
    g3 = A.SubmitGate(True)
    g3.observe("create_file", {"path": "scratch_repro.py"})   # scratch files aren't source edits
    assert g3.intercept("submit") is None
    assert A.SubmitGate(False).intercept("submit") is None


def test_run_agent_gate_and_scratch_excluded(pag):
    edit = next(s for s in pag.trajectory if s.action.tool == "str_replace")
    steps = [("repro", Action("create_file", {"path": "scratch_repro.py", "content": "print(1)\n"})),
             (edit.thought, edit.action), ("done", Action("submit", {})),
             ("verify", Action("bash", {"command": "python scratch_repro.py"})),
             ("done", Action("submit", {}))]
    with RepoEnv(ROOT / pag.repo) as env:
        res = A.run_agent(ScriptedPolicy(steps), env, pag.problem_statement, pag.test_cmd,
                          options=A.AgentOptions(spec_check=True, submit_gate=True))
    assert res.trajectory[2].observation == A.SUBMIT_NUDGE and res.submitted and len(res.trajectory) == 5
    assert "scratch_repro.py" not in res.patch and res.patch.strip()


def test_scripted_policy_has_no_gate(pag):
    with RepoEnv(ROOT / pag.repo) as env:
        res = A.run_agent(ScriptedPolicy(script(pag)), env, pag.problem_statement, pag.test_cmd)
    assert all(s.observation != A.SUBMIT_NUDGE for s in res.trajectory)


# ---- 3. multi-sample eval + rollouts ------------------------------------------------------------
def test_evaluate_samples_and_rollouts(pag, monkeypatch):
    monkeypatch.setattr(EV, "grade", lambda d, patch, root: NS(resolved=bool(patch.strip()), fail_to_pass={},
                                                               pass_to_pass={}, error=None))
    rolls = []
    recs = EV.evaluate([pag], lambda: ScriptedPolicy(script(pag)), ROOT, samples=3, source="student:x",
                       on_rollout=rolls.append, workers=3)
    assert [r["sample"] for r in recs] == [0, 1, 2] and all(r["resolved"] for r in recs)
    assert {r.metadata["sample_idx"] for r in rolls} == {0, 1, 2}
    assert rolls[0].metadata["source"] == "student:x" and rolls[0].metadata["parent_id"] == pag.instance_id
    s = EV.summarize("m", recs + [{**recs[0], "instance_id": "b", "resolved": False},
                                  {**recs[0], "instance_id": "b", "resolved": True, "sample": 1},
                                  {**recs[0], "instance_id": "b", "resolved": False, "sample": 2}])
    assert s["pass_at_k"] == 1.0 and s["solved_sometimes"] == 1 and abs(s["accuracy"] - 4 / 6) < 1e-9
    assert "pass@3" in EV.format_summary(s)


# ---- 4. execution reranking ---------------------------------------------------------------------
def test_rerank_prefers_patch_passing_pooled_checks(pag):
    good_patch = pag.patch
    bad_patch = good_patch.replace("+    start = (page - 1) * per_page", "+    start = page * per_page - 1")
    assert bad_patch != good_patch   # off-by-one in the other direction: applies, but wrong
    check = ("from webapp.pagination import paginate\n"
             "assert paginate(list(range(25)), 1, 10) == list(range(10))\n")
    mk = lambda i, patch, with_check: Demo(f"c{i}", pag.repo, None, pag.problem_statement, pag.test_cmd, "", [], [],  # noqa: E731
                                           patch, [Step("t", Action("create_file", {"path": "scratch_c.py",
                                                                                    "content": check}), "ok")]
                                           if with_check else [])
    cands = [mk(0, bad_patch, False), mk(1, good_patch, True), mk(2, "", False)]
    assert len(RR.pool_checks(cands)) == 1
    ranked = RR.rerank(cands, ROOT / pag.repo, None, pag.test_cmd, backend="local")
    assert ranked[0].index == 1 and ranked[0].checks_passed == 1
    assert ranked[-1].index == 2 and ranked[-1].score < 0


def test_extract_checks_follows_edits():
    d = Demo("x", "r", None, "", "", "", [], [], "", [
        Step("", Action("create_file", {"path": "scratch_a.py", "content": "assert f() == 1\n"}), ""),
        Step("", Action("str_replace", {"path": "scratch_a.py", "old_str": "== 1", "new_str": "== 2"}), ""),
        Step("", Action("create_file", {"path": "pkg/real.py", "content": "x"}), "")])
    (c,) = RR.extract_checks(d)
    assert c.content == "assert f() == 2\n"


# ---- 5. on-policy distillation --------------------------------------------------------------------
def test_onpolicy_relabel_records_and_pairs(pag):
    calls = []

    class Teacher:
        def reset(self, system, task):
            calls.append((system, task))

        def act(self, obs):
            return "teacher reason", Action("bash", {"command": "python -m pytest -q"})

    roll = dataclasses.replace(pag, metadata={"parent_id": pag.instance_id, "sample_idx": 1, "resolved": False})
    rel = OP.relabel_rollout(roll, Teacher, every=2, max_relabels=3, options=A.AgentOptions(spec_check=True))
    assert len(rel) == len(calls) <= 3 and calls[0][0].endswith(A.SPEC_CHECK)
    assert isinstance(calls[-1][1], list) and calls[-1][1][-1]["text"] == OP.RELABEL_INSTRUCTION
    cfg = StudentServingConfig()
    recs = OP.to_sft_records(roll, rel, cfg)
    r = recs[-1]
    assert r["loss_mask"][-1] and not any(r["loss_mask"][:-1])        # only the teacher's turn is trained
    assert r["messages"][-1]["reasoning_content"] == "teacher reason"
    assert r["messages"][0]["content"] == cfg.agent_options().system_prompt()
    pairs = OP.to_step_pairs(roll, rel, cfg)
    assert len(pairs) == sum(not x.agree for x in rel)
    assert pairs[0]["chosen"][0]["tool_calls"][0]["function"]["name"] == "bash"


def test_actions_agree_and_states():
    b = lambda c: Action("bash", {"command": c})  # noqa: E731
    assert OP.actions_agree(b("python  -m pytest -q"), b("python -m pytest -q"))
    assert not OP.actions_agree(b("ls"), b("pwd"))
    assert not OP.actions_agree(None, b("ls"))
    d = Demo("x", "r", None, "", "", "", [], [], "", [Step("", b("ls"), "")] * 10)
    assert OP.states_to_relabel(d, every=1, max_relabels=4) == [0, 3, 6, 9]


def test_onpolicy_cli_dry_run_makes_no_calls(pag, tmp_path, monkeypatch, capsys):
    roll = tmp_path / "r.jsonl"
    roll.write_text(json.dumps(dataclasses.replace(pag, metadata={"resolved": False}).to_dict()) + "\n")
    monkeypatch.setattr(P, "make_policy", lambda *a, **k: pytest.fail("dry run must not create a teacher"))
    p = argparse.ArgumentParser()
    OP.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["onpolicy-distill", "--rollouts", str(roll), "--dry-run"])
    assert a.fn(a) == 0
    err = capsys.readouterr().err
    assert '"teacher_calls"' in err and '"cost_upper_usd"' in err


# ---- 6. training: export, plan, deploy, sft-mix weighting ---------------------------------------
def test_chat_ft_export_weights_and_validation(pag, tmp_path):
    recs = S.build_sft_mix([pag, pag, pag], [], ratio=(1, 0))
    recs[0]["weight"] = 2.0
    n, problems = T.export_chat_ft(recs, tmp_path / "ft.jsonl")
    assert n == 3 and problems == []
    ex = json.loads((tmp_path / "ft.jsonl").read_text().splitlines()[0])
    roles_w = [(m["role"], m.get("weight")) for m in ex["messages"]]
    assert all(w in (1.0, 2.0) for r, w in roles_w if r == "assistant") and ex["tools"]
    assert all(w is None for r, w in roles_w if r != "assistant")
    assert "examples" in T.validate_chat_ft([])[0]
    bad = {"messages": [{"role": "assistant", "weight": 1, "tool_calls": [
        {"id": "a", "function": {"name": "x", "arguments": "{oops"}}]}]}
    assert any("not valid JSON" in p for p in T.validate_chat_ft([bad] * 3))


def test_edge_case_weight_and_extra_records(pag, tmp_path):
    multi = dataclasses.replace(pag, instance_id="m", metadata={**pag.metadata, "hidden_beyond_issue": 2})
    single = dataclasses.replace(pag, instance_id="s", metadata={**pag.metadata, "hidden_beyond_issue": 0})
    recs = {r["id"]: r for r in S.build_sft_mix([multi, single], [], ratio=(1, 0), edge_case_weight=2.0)}
    assert recs["m"]["weight"] == 2.0 and recs["s"]["weight"] == 1.0


def test_plan_paid_stages_need_yes(tmp_path, capsys):
    stages = T.plan(tmp_path / "r1", tmp_path / "splits")
    names = [s.name for s in stages]
    assert names[:2] == ["augment", "student-rollouts"] and names.index("weights") < names.index("distill")
    assert "onpolicy" in names
    assert names.index("rl-weights") < names.index("rl-rollouts") < names.index("train-rl")
    assert not next(s for s in stages if s.name == "weights").paid
    assert all(s.paid for s in stages if s.name in ("student-rollouts", "distill", "onpolicy", "deploy"))
    p = argparse.ArgumentParser()
    T.register(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["student-train", "--runs", str(tmp_path), "--splits", str(tmp_path), "--run", "distill"])
    assert a.fn(a) == 3 and "--yes" in capsys.readouterr().out
    a = p.parse_args(["student-train", "--runs", str(tmp_path), "--splits", str(tmp_path)])
    assert a.fn(a) == 0 and "PAID" in capsys.readouterr().out


def test_deploy_dry_run_and_request(monkeypatch, capsys):
    p = argparse.ArgumentParser()
    T.register(p.add_subparsers(dest="cmd"))
    monkeypatch.setattr(T, "create_deployment", lambda *a, **k: pytest.fail("dry run must not deploy"))
    a = p.parse_args(["fireworks-deploy", "--model", "accounts/me/models/ft1"])
    assert a.fn(a) == 0 and "dry run" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setenv("FIREWORKS_ACCOUNT_ID", "me")
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    seen = {}

    def opener(req, timeout=0):
        seen.update(url=req.full_url, body=json.loads(req.data))
        return NS(read=lambda: b'{"name": "accounts/me/deployments/d1"}', __enter__=lambda s: s,
                  __exit__=lambda s, *x: None)

    class Ctx:
        def __init__(self, r):
            self.r = r

        def __enter__(self):
            return self.r

        def __exit__(self, *x):
            return False

    out = T.create_deployment("accounts/me/models/ft1", opener=lambda req, timeout=0: Ctx(opener(req)))
    assert out["name"].endswith("d1") and seen["url"].endswith("/accounts/me/deployments")
    assert seen["body"] == {"baseModel": "accounts/me/models/ft1", "deploymentShape": "default",
                            "minReplicaCount": 0, "maxReplicaCount": 1}


# ---- 7. LoRA script: token weights land on assistant turns only ----------------------------------
def _load_train_lora():
    spec = importlib.util.spec_from_file_location("train_lora", ROOT / "scripts/train_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CharTok:
    """Fake tokenizer: one token per character, trivial chat template."""
    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<{m['role']}>{m.get('content') or ''}{m.get('reasoning_content') or ''}</>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=0):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_lora_encoder_weights():
    tl = _load_train_lora()
    msgs = [{"role": "user", "content": "fix"}, {"role": "assistant", "content": "AB", "weight": 2.0},
            {"role": "tool", "content": "out"}, {"role": "assistant", "content": "CD", "weight": 0}]
    w = tl.message_weights(msgs, "sft", {})
    ids, tw = tl.Encoder(CharTok(), 10_000).encode(msgs, w, [])
    text = "".join(map(chr, ids))
    weighted = "".join(c for c, x in zip(text, tw) if x)
    assert weighted == "AB</>" and set(x for x in tw if x) == {2.0}
    pg = tl.message_weights(msgs, "pg", {"advantage_grpo": -0.5, "loss_mask": [False, True, False, False]})
    assert pg == [0.0, -0.5, 0.0, 0.0]
