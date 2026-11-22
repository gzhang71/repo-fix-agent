from svcconf import build_settings


def test_defaults():
    assert build_settings() == {"log_level": "warning", "port": "8080", "region": "local"}


def test_file_overrides_defaults():
    s = build_settings("log_level = info  # baked in\nport=9000\n")
    assert s["log_level"] == "info" and s["port"] == "9000"


def test_env_overrides_defaults():
    assert build_settings(environ={"SVC_REGION": "eu", "HOME": "/x"})["region"] == "eu"


def test_overrides_win():
    s = build_settings("port=9000", {"SVC_PORT": "9100"}, {"port": "1"})
    assert s["port"] == "1"
