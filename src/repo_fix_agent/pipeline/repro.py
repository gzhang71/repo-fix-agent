"""Grounded offline issue writing for synthetic (bug-injected) instances.

The old template quoted pytest's assertion line, which is thin ("Observed: assert 31 == 30") and
leaks the hidden test. Instead we turn the failing test into a *reproduction*:

  1. extract_probes(): parse the hidden test, and for each assert / `pytest.raises` block build a
     probe = (imports, setup statements before it, the expression under test). Tests that need
     fixtures, parametrization or test-module helpers are skipped (we can't run them standalone).
  2. run_probes(): execute every probe in the *buggy* tree and in the *fixed* tree (one subprocess
     each, per-probe alarm). The first probe whose observable result differs is the symptom.
  3. compose_issue(): write an issue from what actually happened ("returns X, expected Y",
     "raises E", "accepts invalid input instead of raising E") with a runnable repro snippet,
     re-rendered through ast.unparse (so it is not a verbatim copy of test lines) plus the
     function's documented intent. Phrasing varies by a per-instance seed.

The expected value comes from running the fixed code, never from the test's literal.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import random
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MAX_REPR = 160
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")
_BUILTINS = set(dir(builtins))


@dataclass
class Probe:
    test_id: str
    imports: list[str]
    setup: list[str]
    expr: str
    expect_raises: str | None = None
    funcs: list[str] = field(default_factory=list)   # known source functions called in expr

    def script_body(self) -> str:
        return "\n".join([*self.imports, *self.setup])


@dataclass
class Outcome:
    kind: str          # value | raised | setup_error | timeout
    text: str          # repr(value) or "ExcName: message"

    @property
    def exc_name(self) -> str:
        return self.text.split(":", 1)[0] if self.kind == "raised" else ""


# ---- 1. probe extraction ----------------------------------------------------------------------
def _names_loaded(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _names_stored(node: ast.AST) -> set[str]:
    out = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        out.add(node.name)
    return out


def _free_names(node: ast.AST) -> set[str]:
    """Names a statement/expression reads from outside itself (comprehension variables, lambda
    and nested-def parameters and locals are bound inside)."""
    bound = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.Lambda, ast.AsyncFunctionDef)):
            a = n.args
            bound |= {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if x}
    loaded = _names_loaded(node)
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        loaded.discard(node.name)  # recursion
    return loaded - bound


def _called(node: ast.AST) -> list[str]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            out.append(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
    return [x for x in out if x]


def _is_pytest_raises(item: ast.withitem) -> str | None:
    c = item.context_expr
    if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "raises"
            and isinstance(c.func.value, ast.Name) and c.func.value.id == "pytest" and c.args):
        return ast.unparse(c.args[0])
    return None


def _subject(test: ast.expr, known: set[str]) -> ast.expr | None:
    """The side of an assertion that exercises library code."""
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        test = test.operand
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        sides = [test.left, test.comparators[0]]
        for s in sides:
            if set(_called(s)) & known:
                return s
        for s in sides:
            if _called(s) or isinstance(s, (ast.Name, ast.Attribute, ast.Subscript)):
                return s if not isinstance(s, ast.Constant) else None
        return None
    if isinstance(test, ast.Call) and getattr(test.func, "id", "") in ("isinstance", "len") and test.args:
        return test.args[0] if getattr(test.func, "id", "") == "isinstance" else test
    return test if not isinstance(test, ast.Constant) else None


def _find_test(tree: ast.Module, test_id: str) -> ast.FunctionDef | None:
    parts = test_id.split("::")[1:]
    if not parts:
        return None
    name = parts[-1].split("[")[0]
    scope: list[ast.stmt] = tree.body
    for cls in parts[:-1]:
        c = next((n for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if c is None:
            return None
        scope = c.body
    return next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == name), None)


def extract_probes(test_source: str, test_id: str, known_funcs: Iterable[str] = (),
                   max_probes: int = 8) -> list[Probe]:
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return []
    fn = _find_test(tree, test_id)
    if fn is None or isinstance(fn, ast.AsyncFunctionDef) or fn.decorator_list:
        return []
    params = [a.arg for a in fn.args.args if a.arg != "self"]
    if params or fn.args.vararg or fn.args.kwarg:
        return []  # fixtures / parametrization: not runnable standalone
    known = set(known_funcs)
    imports: list[ast.stmt] = []
    imported: set[str] = set()
    module_defs: set[str] = set()
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            if any((a.asname or a.name).split(".")[0] == "pytest" for a in n.names) or \
                    (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("pytest")):
                continue
            imports.append(n)
            imported |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, (ast.Assign, ast.AnnAssign)) and not (_free_names(n) - imported - _BUILTINS):
            imports.append(n)  # module constant built from imports only (e.g. UTC = timezone.utc)
            imported |= _names_stored(n)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            module_defs |= _names_stored(n)

    probes: list[Probe] = []
    setup: list[ast.stmt] = []
    defined: set[str] = set()
    literals: dict[str, ast.expr] = {}   # setup names bound to list/tuple literals (for loop unrolling)

    def resolvable(node: ast.AST) -> bool:
        return not (_free_names(node) - defined - imported - _BUILTINS)

    def render_imports(used: set[str]) -> list[str]:
        out = []
        for i in imports:
            if isinstance(i, ast.ImportFrom):
                names = [a for a in i.names if (a.asname or a.name) in used]
                if names:
                    out.append(ast.unparse(ast.ImportFrom(module=i.module, names=names, level=i.level)))
            elif isinstance(i, ast.Import):
                if {(a.asname or a.name).split(".")[0] for a in i.names} & used:
                    out.append(ast.unparse(i))
            elif _names_stored(i) & used:
                out.append(ast.unparse(i))
        # constants may need imports of their own
        extra = set().union(*(_free_names(i) for i in imports if not isinstance(i, (ast.Import, ast.ImportFrom))
                              and _names_stored(i) & used)) if out else set()
        if extra - used:
            return render_imports(used | extra)
        return out

    def add(expr: ast.expr, raises: str | None) -> None:
        if len(probes) >= max_probes or not resolvable(expr):
            return
        need = _free_names(expr)
        stmts, want = [], set(need)
        for st in reversed(setup):   # keep only setup statements the expression depends on
            stored = _names_stored(st)
            mutates = isinstance(st, ast.Expr) and bool(_free_names(st) & want)  # q.submit(...)
            if stored & want or mutates:
                stmts.insert(0, st)
                if isinstance(st, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef)):
                    want -= stored   # rebinding: earlier values of these names are irrelevant
                want |= _free_names(st)
        used = set().union(*(_free_names(s) for s in stmts), need)
        probes.append(Probe(test_id, render_imports(used), [ast.unparse(s) for s in stmts], ast.unparse(expr),
                            raises, sorted(set(_called(expr)) & known)))

    def visit(body: list[ast.stmt]) -> bool:
        """Process statements in order; False = stop (control flow we can't follow)."""
        for st in body:
            if isinstance(st, ast.Assert):
                subj = _subject(st.test, known)
                if subj is not None:
                    add(subj, None)
            elif isinstance(st, ast.With) and len(st.items) == 1 and _is_pytest_raises(st.items[0]):
                first = st.body[0] if st.body else None
                val = first.value if isinstance(first, (ast.Expr, ast.Assign)) else None
                if val is not None:
                    add(val, _is_pytest_raises(st.items[0]))
            elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.FunctionDef,
                                 ast.ClassDef)):
                if not resolvable(st):
                    return False
                setup.append(st)
                defined.update(_names_stored(st))
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    if isinstance(st.value, (ast.List, ast.Tuple)):
                        literals[st.targets[0].id] = st.value
                    else:
                        literals.pop(st.targets[0].id, None)
            elif isinstance(st, ast.For) and not st.orelse:
                it = st.iter
                if isinstance(it, ast.Name) and it.id in literals:
                    it = literals[it.id]
                if not isinstance(it, (ast.List, ast.Tuple)) or len(it.elts) > 8:
                    return False
                for elt in it.elts:   # unroll: `x = <element>` then the loop body
                    bind = ast.Assign(targets=[st.target], value=elt, lineno=0)
                    if not resolvable(bind):
                        return False
                    setup.append(bind)
                    defined.update(_names_stored(bind))
                    if not visit(st.body):
                        return False
            else:
                return False   # if / try / while / other with-blocks
        return True

    visit(fn.body)
    return probes


# ---- 2. execution -----------------------------------------------------------------------------
_RUNNER = textwrap.dedent('''
    import json, signal, sys
    sys.path.insert(0, ".")
    probes = json.loads(sys.stdin.read())
    def _alarm(*_):
        raise TimeoutError("probe timed out")
    signal.signal(signal.SIGALRM, _alarm)
    out = []
    for body, expr in probes:
        ns = {"__name__": "__repro__"}
        signal.alarm(3)
        try:
            try:
                exec(compile(body, "<repro>", "exec"), ns)
            except TimeoutError:
                out.append(["timeout", ""]); continue
            except BaseException as e:
                out.append(["setup_error", type(e).__name__ + ": " + str(e)]); continue
            try:
                v = eval(compile(expr, "<repro>", "eval"), ns)
            except TimeoutError:
                out.append(["timeout", ""]); continue
            except BaseException as e:
                out.append(["raised", type(e).__name__ + (": " + str(e) if str(e) else "")]); continue
            try:
                out.append(["value", repr(v)])
            except BaseException as e:
                out.append(["setup_error", "repr failed: " + type(e).__name__])
        finally:
            signal.alarm(0)
    print("__REPRO__" + json.dumps(out))
''')


def run_probes(probes: list[Probe], repo: str | Path, timeout: int = 30) -> list[Outcome] | None:
    if not probes:
        return []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run([sys.executable, "-c", _RUNNER], cwd=str(repo), capture_output=True, text=True,
                           input=json.dumps([[p.script_body(), p.expr] for p in probes]),
                           timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("__REPRO__")), None)
    if line is None:
        return None
    return [Outcome(k, t) for k, t in json.loads(line[len("__REPRO__"):])]


def _informative(o: Outcome) -> bool:
    return o.kind in ("value", "raised") and not _ADDR.search(o.text)


def pick_symptom(probes: list[Probe], buggy: list[Outcome] | None, fixed: list[Outcome] | None):
    """First probe whose buggy/fixed outcomes differ observably and match the test's intent."""
    if not buggy or not fixed:
        return None
    for p, b, f in zip(probes, buggy, fixed):
        if not (_informative(b) and _informative(f)) or (b.kind, b.text) == (f.kind, f.text):
            continue
        if p.expect_raises:
            want = p.expect_raises.split(".")[-1].strip("()")
            if f.kind != "raised" or (b.kind == "raised" and b.exc_name == f.exc_name):
                continue
            return {"probe": p, "buggy": b, "fixed": f, "expected_exc": want}
        if f.kind != "value":
            continue
        return {"probe": p, "buggy": b, "fixed": f, "expected_exc": None}
    return None


def find_symptom(test_sources: dict[str, str], failing: Iterable[str], known_funcs: Iterable[str],
                 buggy_repo: str | Path, fixed_repo: str | Path, max_tests: int = 4):
    """Try failing tests in order; return the first concrete symptom dict, or None."""
    known = list(known_funcs)
    for tid in sorted(failing)[:max_tests]:
        src = test_sources.get(tid.split("::")[0])
        probes = extract_probes(src or "", tid, known)
        if not probes:
            continue
        sym = pick_symptom(probes, run_probes(probes, buggy_repo), run_probes(probes, fixed_repo))
        if sym:
            return sym
    return None


# ---- 3. composition ---------------------------------------------------------------------------
def _short(text: str, n: int = MAX_REPR) -> str:
    """Truncate for display. Whitespace inside reprs is kept exactly (it can be the bug); only
    line breaks (exception messages) are flattened."""
    text = text.replace("\n", " ")
    if len(text) <= n:
        return text
    close = {"[": "]", "(": ")", "{": "}"}.get(text[:1], "")
    return text[: n - 4 - len(close)] + " ..." + close


def _func_name(sym) -> str | None:
    p: Probe = sym["probe"]
    return p.funcs[0] if p.funcs else None


def docstring_intent(source: str, func: str | None) -> str | None:
    if not func:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == func:
            doc = ast.get_docstring(n)
            if doc:
                first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
                return first if 10 <= len(first) <= 220 else None
    return None


def _repro_block(p: Probe, show: str, style: int) -> str:
    lines = [*p.imports, *p.setup]
    if style == 0:   # REPL transcript
        body = [f">>> {ln}" for ln in lines] + [f">>> {p.expr}", show]
    else:            # script
        body = lines + [f"print({p.expr})" if not show.startswith("Traceback") else p.expr]
    return "\n".join("    " + ln for ln in body)


def compose_issue(sym: dict, *, source_context: str = "", seed: str = "") -> str:
    rng = random.Random(seed)
    p: Probe = sym["probe"]
    b: Outcome = sym["buggy"]
    f: Outcome = sym["fixed"]
    func = _func_name(sym)
    fn = f"`{func}`" if func else f"`{_short(p.expr, 60)}`"
    style = rng.randrange(2)
    intent = docstring_intent(source_context, _func_name(sym))
    call = f"`{_short(p.expr, 90)}`"

    if sym["expected_exc"]:
        exc = sym["expected_exc"]
        got = f"returns `{_short(b.text)}`" if b.kind == "value" else f"raises `{_short(b.text)}`"
        title = rng.choice([f"{fn} accepts invalid input instead of raising {exc}",
                            f"{fn} no longer raises {exc}",
                            f"Missing {exc} from {fn}"])
        summary = rng.choice([
            f"{call} should be rejected with `{exc}`, but it {got}.",
            f"I expected {call} to raise `{exc}` (as it did for this kind of input), but it {got}.",
        ])
        shown = (_short(b.text) if b.kind == "value" else f"Traceback (most recent call last):\n  ...\n{b.text}")
        expected = f"Expected: `{f.text.split(':', 1)[0]}` is raised."
    elif b.kind == "raised":
        title = rng.choice([f"{fn} crashes with {b.exc_name}",
                            f"{b.exc_name} from {fn} on valid input",
                            f"{fn} raises {b.exc_name} unexpectedly"])
        summary = rng.choice([f"{call} raises `{_short(b.text)}`.",
                              f"Calling {call} blows up with `{_short(b.text)}`."])
        shown = f"Traceback (most recent call last):\n  ...\n{_short(b.text)}"
        expected = f"Expected it to return `{_short(f.text)}`."
    else:
        short_b, short_f = _short(b.text, 40), _short(f.text, 40)
        verb = "returns" if func else "is"
        title = rng.choice(
            ([f"{fn} {verb} {short_b} instead of {short_f}"] if len(b.text) <= 40 and len(f.text) <= 40 else [])
            + ([f"{fn} returns the wrong result", f"Incorrect output from {fn}", f"Wrong value from {fn}"]
               if func else [f"Wrong value for {fn}", f"{fn} has the wrong value"]))
        summary = rng.choice([f"{call} returns `{_short(b.text)}`.",
                              f"Got `{_short(b.text)}` from {call}.",
                              f"{call} gives `{_short(b.text)}`, which looks wrong."] if func else
                             [f"{call} is `{_short(b.text)}`.", f"{call} evaluates to `{_short(b.text)}`."])
        shown = _short(b.text)
        expected = rng.choice([f"Expected `{_short(f.text)}`.", f"It should be `{_short(f.text)}`."])

    parts = [title, ""]
    if intent and rng.random() < 0.8:
        parts += [f"{fn} is documented as: \"{intent}\"", ""]
    parts += [summary, "", "To reproduce:" if style else "Reproduction:", "",
              _repro_block(p, shown, style), "", expected]
    if style == 1:
        parts += ["", f"Actual output: `{_short(b.text)}`" if b.kind == "value" else f"Actual: `{_short(b.text)}`"]
    return "\n".join(parts).rstrip() + "\n"


def symptom_summary(sym: dict) -> str:
    """Plain-text ground truth for an LLM issue writer."""
    p: Probe = sym["probe"]
    lines = ["Verified reproduction (runs standalone):", *("    " + x for x in [*p.imports, *p.setup, p.expr]),
             f"Buggy code: {sym['buggy'].kind} {sym['buggy'].text[:400]}",
             f"Correct code: {sym['fixed'].kind} {sym['fixed'].text[:400]}"]
    if sym["expected_exc"]:
        lines.append(f"Correct behaviour is to raise {sym['expected_exc']}.")
    return "\n".join(lines)
