import pytest

from catalog.api import list_products, parse_params
from catalog.fixtures import PRODUCTS
from catalog.query import UnknownFilter
from catalog.repository import ProductRepository


@pytest.fixture
def repo():
    return ProductRepository(PRODUCTS)


def test_in_stock_true(repo):
    assert list_products(repo, {"in_stock": "true"})["ids"] == [1, 3, 5, 7, 8]


def test_category_sorted_by_price(repo):
    out = list_products(repo, {"category": "office", "sort": "-price"})
    assert out["ids"] == [6, 3, 4, 7]


def test_paging(repo):
    out = list_products(repo, {"sort": "price", "per_page": "3", "page": "2"})
    assert out == {"total": 8, "page": 2, "ids": [3, 2, 1]}


def test_blank_params_ignored(repo):
    assert list_products(repo, {"category": " ", "min_rating": "5"})["ids"] == [3, 5]


def test_parse_params_types():
    assert parse_params({"in_stock": "no", "max_price": "100"}) == (
        {"in_stock": False, "max_price": 100}, {})


def test_unknown_filter(repo):
    with pytest.raises(UnknownFilter):
        list_products(repo, {"colour": "red"})
