from fuzzy.distance import levenshtein, similarity
from fuzzy.suggest import closest


def test_empty_strings():
    assert levenshtein("", "abc") == 3
    assert similarity("", "") == 1.0


def test_single_char():
    assert levenshtein("a", "b") == 1


def test_no_candidate_within_range():
    assert closest("zzzzzz", ["pull", "push"]) is None
