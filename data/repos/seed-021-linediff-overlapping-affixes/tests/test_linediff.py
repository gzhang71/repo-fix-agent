from linediff import Change, apply_change, diff_lines, summary, unified


def test_replace_middle():
    assert diff_lines(["a", "b", "c"], ["a", "x", "c"]) == Change(1, ["b"], 1, ["x"])


def test_equal_inputs():
    assert diff_lines(["a"], ["a"]) is None
    assert summary(["a", "b"], ["a", "b"]) == "no changes"


def test_unified_output():
    assert unified(["a", "b"], ["a", "c", "d"]) == "--- a\n+++ b\n@@ -2,1 +2,2 @@\n-b\n+c\n+d\n"


def test_round_trip_simple():
    pairs = [(["a", "b", "c"], ["a", "c"]), (["a"], ["a", "b"]), ([], ["x"]), (["p", "q"], ["r"])]
    for a, b in pairs:
        assert apply_change(a, diff_lines(a, b)) == b
