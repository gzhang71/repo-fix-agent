import pytest

from contacts import load_contacts
from contacts.csvline import split_line


def test_plain_fields():
    assert split_line("a,b,,c") == ["a", "b", "", "c"]


def test_quoted_separator():
    assert split_line('1,"Smith, John",x') == ["1", "Smith, John", "x"]


def test_load_contacts_header():
    rows = load_contacts("ID,Name\n1,Ann\n2,Ben\n")
    assert rows == [{"id": "1", "name": "Ann"}, {"id": "2", "name": "Ben"}]


def test_field_count_mismatch():
    with pytest.raises(ValueError, match="line 2"):
        load_contacts("id,name\n1,Ann,extra\n")
