from search.bisect_utils import find_index


def test_finds_middle():
    assert find_index([1, 3, 5], 3) == 1


def test_absent():
    assert find_index([1, 3, 5], 4) == -1
    assert find_index([], 1) == -1
