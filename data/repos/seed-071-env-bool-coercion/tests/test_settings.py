from appcfg import load_settings


def test_defaults_when_env_empty():
    assert load_settings({}) == {"debug": False, "workers": 4,
                                 "cache_enabled": True, "region": "local"}


def test_int_and_str_overrides():
    s = load_settings({"APP_WORKERS": " 8 ", "APP_REGION": "eu-west-1"})
    assert s["workers"] == 8 and s["region"] == "eu-west-1"


def test_debug_enabled():
    assert load_settings({"APP_DEBUG": "1"})["debug"] is True
