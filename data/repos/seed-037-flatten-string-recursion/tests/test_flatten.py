from nested.flatten import flatten


def test_nested_numbers():
    assert flatten([1, [2, [3, (4, 5)]], []]) == [1, 2, 3, 4, 5]


def test_scalar():
    assert flatten(7) == [7]


def test_empty():
    assert flatten([]) == []
