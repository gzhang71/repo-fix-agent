from contacts.iterutils import chunked, dedupe


def test_dedupe_plain_values():
    assert list(dedupe([3, 1, 3, 2, 1])) == [3, 1, 2]


def test_chunked():
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
