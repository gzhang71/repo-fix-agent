"""pipeline/repro.py: executable reproductions -> grounded offline issues."""
import textwrap
from pathlib import Path

import pytest

from repo_fix_agent.pipeline import repro as R
from repo_fix_agent.pipeline.audit import detect_leakage
from repo_fix_agent.schema import Demo

TEST_SRC = textwrap.dedent('''
    import pytest
    from shop.cart import Cart, total, parse_qty

    HELPER = 3

    def make_cart():
        return Cart([4])

    def test_total():
        cart = Cart([2, 3])
        extra = 1
        assert total(cart) == 5

    def test_rejects_negative():
        with pytest.raises(ValueError):
            parse_qty("-1")

    def test_fixture(tmp_path):
        assert total(Cart([1])) == 1

    def test_uses_module_constant():
        assert total(Cart([HELPER])) == 3

    def test_uses_module_helper():
        assert total(make_cart()) == 4

    def test_loop_and_mutation():
        cart = Cart([])
        for x in [1, 2]:
            cart.items.append(x)
        assert total(cart) == 3
        xs = [5, 6]
        assert [total(Cart([x])) for x in xs] == [5, 6]

    def test_nested_def():
        def double(c):
            return 2 * total(c)
        assert double(Cart([1])) == 2

    class TestCart:
        def test_in_class(self):
            assert total(Cart([])) == 0
''')

FIXED = textwrap.dedent('''
    class Cart:
        def __init__(self, items):
            self.items = list(items)


    def total(cart):
        """Sum of all item prices in the cart."""
        return sum(cart.items)


    def parse_qty(s):
        n = int(s)
        if n < 0:
            raise ValueError("quantity must be >= 0")
        return n
''')
BUGGY = FIXED.replace("return sum(cart.items)", "return sum(cart.items[1:])").replace(
    "if n < 0:", "if n < -5:")


def _repo(tmp_path, name, src):
    root = tmp_path / name
    (root / "shop").mkdir(parents=True)
    (root / "shop" / "__init__.py").write_text("")
    (root / "shop" / "cart.py").write_text(src)
    return root


def test_extract_probes_keeps_only_needed_setup():
    (p,) = R.extract_probes(TEST_SRC, "tests/test_cart.py::test_total", ["Cart", "total", "parse_qty"])
    assert p.expr == "total(cart)"
    assert p.setup == ["cart = Cart([2, 3])"]          # `extra = 1` is irrelevant
    assert p.imports == ["from shop.cart import Cart, total"]   # pruned to used names
    assert p.funcs == ["total"]
    assert "pytest" not in p.script_body()


def test_extract_probes_pytest_raises():
    (p,) = R.extract_probes(TEST_SRC, "tests/test_cart.py::test_rejects_negative", ["parse_qty"])
    assert p.expr == "parse_qty('-1')" and p.expect_raises == "ValueError"


@pytest.mark.parametrize("tid", ["test_fixture", "test_uses_module_helper", "missing"])
def test_extract_probes_skips_non_standalone(tid):
    assert R.extract_probes(TEST_SRC, f"tests/test_cart.py::{tid}", ["total", "Cart"]) == []


def test_extract_probes_module_constant_loop_mutation_nested_def():
    known = ["Cart", "total"]
    (c,) = R.extract_probes(TEST_SRC, "tests/test_cart.py::test_uses_module_constant", known)
    assert c.imports == ["from shop.cart import Cart, total", "HELPER = 3"]
    first, comp = R.extract_probes(TEST_SRC, "tests/test_cart.py::test_loop_and_mutation", known)
    assert first.setup == ["cart = Cart([])", "x = 1", "cart.items.append(x)", "x = 2", "cart.items.append(x)"]
    assert comp.expr == "[total(Cart([x])) for x in xs]" and comp.setup == ["xs = [5, 6]"]
    (n,) = R.extract_probes(TEST_SRC, "tests/test_cart.py::test_nested_def", known)
    assert n.setup[0].startswith("def double(c):") and n.expr == "double(Cart([1]))"


def test_extract_probes_class_method():
    (p,) = R.extract_probes(TEST_SRC, "tests/test_cart.py::TestCart::test_in_class", ["total", "Cart"])
    assert p.expr == "total(Cart([]))"


def test_find_symptom_value_and_missing_raise(tmp_path):
    fixed, buggy = _repo(tmp_path, "fixed", FIXED), _repo(tmp_path, "buggy", BUGGY)
    src = {"tests/test_cart.py": TEST_SRC}
    known = ["Cart", "total", "parse_qty"]
    sym = R.find_symptom(src, ["tests/test_cart.py::test_total"], known, buggy, fixed)
    assert (sym["buggy"].kind, sym["buggy"].text, sym["fixed"].text) == ("value", "3", "5")
    sym2 = R.find_symptom(src, ["tests/test_cart.py::test_rejects_negative"], known, buggy, fixed)
    assert sym2["expected_exc"] == "ValueError" and sym2["buggy"].text == "-1"


def test_run_probes_handles_hang_and_setup_error(tmp_path):
    root = _repo(tmp_path, "r", "def spin():\n    while True:\n        pass\n")
    probes = [R.Probe("t", ["from shop.cart import spin"], [], "spin()"),
              R.Probe("t", ["from shop.cart import nope"], [], "nope()")]
    out = R.run_probes(probes, root)
    assert [o.kind for o in out] == ["timeout", "setup_error"]


def test_compose_issue_is_grounded_and_does_not_leak(tmp_path):
    fixed, buggy = _repo(tmp_path, "fixed", FIXED), _repo(tmp_path, "buggy", BUGGY)
    sym = R.find_symptom({"tests/test_cart.py": TEST_SRC}, ["tests/test_cart.py::test_total"],
                         ["Cart", "total", "parse_qty"], buggy, fixed)
    texts = {R.compose_issue(sym, source_context=FIXED, seed=str(i)) for i in range(12)}
    assert len(texts) > 3                                  # phrasing varies by seed
    test_patch = "".join(f"+{ln}\n" for ln in TEST_SRC.splitlines())
    for text in texts:
        assert "`3`" in text and "5" in text and "total(cart)" in text
        assert "test_" not in text and "assert" not in text
        d = Demo("x", "", None, text, "", test_patch, ["tests/test_cart.py::test_total"], [], "")
        assert detect_leakage(d) == []


def test_compose_issue_uses_docstring_sometimes(tmp_path):
    fixed, buggy = _repo(tmp_path, "fixed", FIXED), _repo(tmp_path, "buggy", BUGGY)
    sym = R.find_symptom({"tests/test_cart.py": TEST_SRC}, ["tests/test_cart.py::test_total"],
                         ["Cart", "total"], buggy, fixed)
    assert any("Sum of all item prices" in R.compose_issue(sym, source_context=FIXED, seed=str(i))
               for i in range(10))


def test_symptom_summary_for_llm(tmp_path):
    fixed, buggy = _repo(tmp_path, "fixed", FIXED), _repo(tmp_path, "buggy", BUGGY)
    sym = R.find_symptom({"tests/test_cart.py": TEST_SRC}, ["tests/test_cart.py::test_rejects_negative"],
                         ["parse_qty"], buggy, fixed)
    s = R.symptom_summary(sym)
    assert "parse_qty('-1')" in s and "raise ValueError" in s
