import os

from buildcache.builder import Builder
from buildcache.statefile import load_state, save_state


def upper(text):
    return text.upper()


def write(p, text, ns):
    p.write_text(text)
    os.utime(p, ns=(ns, ns))


def test_unchanged_file_not_recompiled(tmp_path):
    src = tmp_path / "a.src"
    write(src, "x = 1", 1_700_000_000_000_000_000)
    b = Builder(upper)
    assert b.build(src) == "X = 1"
    assert b.build(src) == "X = 1"
    assert b.compiled == 1


def test_change_seconds_later_recompiles(tmp_path):
    src = tmp_path / "a.src"
    write(src, "x = 1", 1_700_000_000_000_000_000)
    b = Builder(upper)
    b.build(src)
    write(src, "x = 2", 1_700_000_005_000_000_000)
    assert b.build(src) == "X = 2"


def test_state_survives_restart(tmp_path):
    src, state_file = tmp_path / "a.src", tmp_path / "state.json"
    write(src, "y", 1_700_000_000_250_000_000)
    b = Builder(upper)
    b.build(src)
    save_state(b.state, state_file)
    b2 = Builder(upper, load_state(state_file))
    assert b2.build(src) == "Y"
    assert b2.compiled == 0
