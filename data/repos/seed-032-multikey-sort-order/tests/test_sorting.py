from records.sorting import sort_by

ROWS = [
    {"name": "Ana", "dept": "eng", "salary": 120},
    {"name": "Bo", "dept": "ops", "salary": 90},
    {"name": "Cy", "dept": "eng", "salary": 100},
]


def test_single_field_ascending():
    assert [r["name"] for r in sort_by(ROWS, "salary")] == ["Bo", "Cy", "Ana"]


def test_single_field_descending():
    assert [r["name"] for r in sort_by(ROWS, "-salary")] == ["Ana", "Cy", "Bo"]


def test_stable_for_equal_keys():
    assert [r["name"] for r in sort_by(ROWS, "dept")] == ["Ana", "Cy", "Bo"]


def test_input_not_mutated():
    rows = list(ROWS)
    sort_by(rows, "salary")
    assert rows == ROWS
