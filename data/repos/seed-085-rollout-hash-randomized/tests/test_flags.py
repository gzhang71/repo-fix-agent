from flags import FlagRegistry
from flags.rollout import bucket


def reg(**cfg):
    return FlagRegistry({"f": {"enabled": True, **cfg}})


def test_disabled_flag_off():
    assert not FlagRegistry({"f": {"enabled": False, "rollout": 100}}).is_enabled("f", "u1")
    assert not FlagRegistry({}).is_enabled("f", "u1")


def test_full_and_zero_rollout():
    users = [f"user-{i}" for i in range(200)]
    assert all(reg(rollout=100).is_enabled("f", u) for u in users)
    assert not any(reg(rollout=0).is_enabled("f", u) for u in users)


def test_allow_list():
    assert reg(rollout=0, allow=["vip"]).is_enabled("f", "vip")


def test_bucket_stable_and_in_range():
    assert bucket("f", "u1") == bucket("f", "u1")
    assert all(0 <= bucket("f", f"u{i}") < 100 for i in range(500))
