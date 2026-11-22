import json

from deployctl.main import load_settings


def reader(files):
    return lambda path: json.dumps(files[path])


def test_builtin_defaults():
    assert load_settings([]) == {"host": "127.0.0.1", "port": 8000, "workers": 2,
                                 "debug": False, "region": "local", "log_level": "info"}


def test_cli_flag():
    assert load_settings(["--port", "9001", "--debug"])["port"] == 9001


def test_file_only_keys():
    s = load_settings(["--config", "d.json"], reader({"d.json": {"region": "eu"}}))
    assert s["region"] == "eu"
