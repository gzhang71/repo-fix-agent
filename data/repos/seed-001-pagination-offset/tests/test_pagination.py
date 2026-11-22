import pytest
from webapp.pagination import paginate, total_pages


def test_total_pages():
    assert total_pages(25, 10) == 3
    assert total_pages(0, 10) == 0


def test_rejects_page_zero():
    with pytest.raises(ValueError):
        paginate([1, 2, 3], 0, 2)
