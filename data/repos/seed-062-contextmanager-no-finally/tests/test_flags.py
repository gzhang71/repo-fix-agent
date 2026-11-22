from flags.checkout import checkout_url
from flags.registry import is_enabled, snapshot
from flags.testing import override_flags


def test_override_and_restore():
    with override_flags(new_checkout=True):
        assert checkout_url(3) == "/v2/checkout/3"
    assert checkout_url(3) == "/checkout/3"


def test_new_flag_removed_afterwards():
    before = snapshot()
    with override_flags(beta_search=True):
        assert is_enabled("beta_search")
    assert snapshot() == before
