"""Seed instances 071-085: configuration, CLI, serialization, paths & caching."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V

SPECS = [
    # 71 ---------------------------------------------------------------------------------------
    dict(
        id="seed-071-env-bool-coercion",
        bug_type="type-coercion", difficulty="easy",
        issue="""\
            APP_CACHE_ENABLED=false does not disable the cache

            We set `APP_CACHE_ENABLED=false` in the staging env file but `load_settings()` still
            reports `cache_enabled: True`. Same thing with `APP_DEBUG=0`, which turns debug *on*.
            Integers like `APP_WORKERS=8` are picked up fine.
            """,
        files={
            "appcfg/__init__.py": "from .settings import load_settings\n",
            "appcfg/coerce.py": '''\
                def to_bool(raw):
                    """Parse an environment-variable string as a boolean."""
                    return bool(raw.strip())


                def to_int(raw):
                    return int(raw.strip())
                ''',
            "appcfg/settings.py": '''\
                from .coerce import to_bool, to_int

                DEFAULTS = {"debug": False, "workers": 4, "cache_enabled": True, "region": "local"}
                PREFIX = "APP_"


                def load_settings(environ):
                    """Build settings from DEFAULTS, overridden by APP_* variables in `environ`."""
                    settings = dict(DEFAULTS)
                    for key, default in DEFAULTS.items():
                        raw = environ.get(PREFIX + key.upper())
                        if raw is None:
                            continue
                        if isinstance(default, bool):
                            settings[key] = to_bool(raw)
                        elif isinstance(default, int):
                            settings[key] = to_int(raw)
                        else:
                            settings[key] = raw
                    return settings
                ''',
            "tests/test_settings.py": '''\
                from appcfg import load_settings


                def test_defaults_when_env_empty():
                    assert load_settings({}) == {"debug": False, "workers": 4,
                                                 "cache_enabled": True, "region": "local"}


                def test_int_and_str_overrides():
                    s = load_settings({"APP_WORKERS": " 8 ", "APP_REGION": "eu-west-1"})
                    assert s["workers"] == 8 and s["region"] == "eu-west-1"


                def test_debug_enabled():
                    assert load_settings({"APP_DEBUG": "1"})["debug"] is True
                ''',
        },
        hidden_test='''\
            from appcfg import load_settings
            from appcfg.coerce import to_bool


            def test_false_strings_are_false():
                for raw in ("false", "False", "0", "no", "off", " FALSE "):
                    assert to_bool(raw) is False, raw


            def test_true_strings_are_true():
                for raw in ("true", "TRUE", "1", "yes", "on"):
                    assert to_bool(raw) is True, raw


            def test_env_can_disable_cache():
                s = load_settings({"APP_CACHE_ENABLED": "false", "APP_DEBUG": "0"})
                assert s["cache_enabled"] is False
                assert s["debug"] is False
            ''',
        p2p=["tests/test_settings.py::test_defaults_when_env_empty",
             "tests/test_settings.py::test_int_and_str_overrides",
             "tests/test_settings.py::test_debug_enabled"],
        steps=[
            ("Find where APP_* variables are read.", S("PREFIX|environ", "appcfg")),
            ("Booleans go through to_bool; read the coercion helpers.", V("appcfg/coerce.py")),
            ("`bool(raw.strip())` is True for any non-empty string, including 'false' and '0'. Confirm.",
             B("python -c 'from appcfg import load_settings; print(load_settings({\"APP_CACHE_ENABLED\": \"false\", \"APP_DEBUG\": \"0\"}))'")),
            ("Parse the usual truthy/falsy spellings explicitly and reject anything else instead of guessing.",
             R("appcfg/coerce.py",
               'def to_bool(raw):\n    """Parse an environment-variable string as a boolean."""\n    return bool(raw.strip())\n',
               '_TRUE = {"1", "true", "yes", "on"}\n_FALSE = {"0", "false", "no", "off", ""}\n\n\n'
               'def to_bool(raw):\n    """Parse an environment-variable string as a boolean."""\n'
               '    value = raw.strip().lower()\n    if value in _TRUE:\n        return True\n'
               '    if value in _FALSE:\n        return False\n'
               '    raise ValueError(f"invalid boolean value: {raw!r}")\n')),
            ("Re-run the repro.",
             B("python -c 'from appcfg import load_settings; print(load_settings({\"APP_CACHE_ENABLED\": \"false\", \"APP_DEBUG\": \"0\"}))'")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 72 ---------------------------------------------------------------------------------------
    dict(
        id="seed-072-json-datetime-as-date",
        bug_type="isinstance-order", difficulty="easy",
        issue="""\
            Loading saved events crashes: Invalid isoformat string

            Events with a `datetime` timestamp can be dumped but not loaded back:

                >>> from datetime import datetime
                >>> from eventlog.serialize import dumps, loads
                >>> loads(dumps({"at": datetime(2024, 3, 1, 10, 30)}))
                Traceback (most recent call last):
                  ...
                  File "eventlog/serialize.py", line 23, in _hook
                    return date.fromisoformat(obj["__date__"])
                ValueError: Invalid isoformat string: '2024-03-01T10:30:00'

            Plain dates and sets round-trip fine.
            """,
        files={
            "eventlog/__init__.py": "",
            "eventlog/serialize.py": '''\
                import json
                from datetime import date, datetime


                class EventEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, date):
                            return {"__date__": obj.isoformat()}
                        if isinstance(obj, datetime):
                            return {"__datetime__": obj.isoformat()}
                        if isinstance(obj, (set, frozenset)):
                            return {"__set__": sorted(obj)}
                        return super().default(obj)


                def dumps(event):
                    return json.dumps(event, cls=EventEncoder, sort_keys=True)


                def _hook(obj):
                    if "__datetime__" in obj:
                        return datetime.fromisoformat(obj["__datetime__"])
                    if "__date__" in obj:
                        return date.fromisoformat(obj["__date__"])
                    if "__set__" in obj:
                        return set(obj["__set__"])
                    return obj


                def loads(text):
                    return json.loads(text, object_hook=_hook)
                ''',
            "eventlog/store.py": '''\
                from .serialize import dumps, loads


                def append(stream, event):
                    stream.write(dumps(event) + "\\n")


                def read_all(stream):
                    stream.seek(0)
                    return [loads(line) for line in stream if line.strip()]
                ''',
            "tests/test_serialize.py": '''\
                import io
                from datetime import date

                from eventlog.serialize import dumps, loads
                from eventlog.store import append, read_all


                def test_date_round_trip():
                    ev = {"day": date(2024, 2, 29), "kind": "close"}
                    assert loads(dumps(ev)) == ev


                def test_set_round_trip():
                    ev = {"tags": {"b", "a"}}
                    assert loads(dumps(ev)) == ev
                    assert '"__set__": ["a", "b"]' in dumps(ev)


                def test_store_lines():
                    buf = io.StringIO()
                    append(buf, {"n": 1})
                    append(buf, {"n": 2})
                    assert read_all(buf) == [{"n": 1}, {"n": 2}]
                ''',
        },
        hidden_test='''\
            import io
            from datetime import date, datetime, timezone

            from eventlog.serialize import dumps, loads
            from eventlog.store import append, read_all


            def test_datetime_round_trip():
                ev = {"at": datetime(2024, 3, 1, 10, 30, 15)}
                out = loads(dumps(ev))
                assert out == ev
                assert type(out["at"]) is datetime


            def test_aware_datetime_round_trip_via_store():
                buf = io.StringIO()
                ev = {"at": datetime(2024, 3, 1, 23, 59, tzinfo=timezone.utc), "day": date(2024, 3, 1)}
                append(buf, ev)
                assert read_all(buf) == [ev]


            def test_date_stays_date():
                out = loads(dumps({"day": date(2024, 1, 5)}))
                assert type(out["day"]) is date
            ''',
        p2p=["tests/test_serialize.py::test_date_round_trip",
             "tests/test_serialize.py::test_set_round_trip",
             "tests/test_serialize.py::test_store_lines"],
        steps=[
            ("Open the serializer named in the traceback.", V("eventlog/serialize.py")),
            ("Check what the encoder emits for a datetime.",
             B("python -c 'from datetime import datetime; from eventlog.serialize import dumps; print(dumps({\"at\": datetime(2024, 3, 1, 10, 30)}))'")),
            ("datetime is a subclass of date, so the `date` branch catches it first and tags it `__date__`. Check the more specific type first.",
             R("eventlog/serialize.py",
               '        if isinstance(obj, date):\n            return {"__date__": obj.isoformat()}\n'
               '        if isinstance(obj, datetime):\n            return {"__datetime__": obj.isoformat()}\n',
               '        if isinstance(obj, datetime):  # must precede date: datetime subclasses date\n'
               '            return {"__datetime__": obj.isoformat()}\n'
               '        if isinstance(obj, date):\n            return {"__date__": obj.isoformat()}\n')),
            ("Re-run the round trip from the issue.",
             B("python -c 'from datetime import datetime, date; from eventlog.serialize import dumps, loads; print(loads(dumps({\"at\": datetime(2024, 3, 1, 10, 30), \"d\": date(2024, 3, 1)})))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 73 ---------------------------------------------------------------------------------------
    dict(
        id="seed-073-csv-export-unquoted",
        bug_type="missing-escaping", difficulty="easy",
        issue="""\
            Contacts CSV export has shifted columns

            When a contact's name contains a comma (we store some as "Doe, Jane") the exported
            file has an extra column on that row and the email ends up under "notes" in Excel.
            Notes containing quotes or line breaks also mangle the file.
            """,
        files={
            "exporter/__init__.py": "",
            "exporter/csv_export.py": '''\
                HEADER = ["id", "name", "email", "notes"]


                def rows_to_csv(rows, out):
                    """Write contact dicts to the text stream `out` as CSV with a header row."""
                    out.write(",".join(HEADER) + "\\n")
                    for row in rows:
                        out.write(",".join(str(row.get(col, "")) for col in HEADER) + "\\n")
                ''',
            "exporter/contacts.py": '''\
                import io

                from .csv_export import rows_to_csv


                def export_contacts(contacts):
                    """Return the CSV text for a list of contact dicts, sorted by id."""
                    buf = io.StringIO()
                    rows_to_csv(sorted(contacts, key=lambda c: c["id"]), buf)
                    return buf.getvalue()
                ''',
            "tests/test_export.py": '''\
                from exporter.contacts import export_contacts


                def test_simple_export():
                    text = export_contacts([
                        {"id": 2, "name": "Bob", "email": "bob@x.io", "notes": "vip"},
                        {"id": 1, "name": "Ann", "email": "ann@x.io"},
                    ])
                    assert text == "id,name,email,notes\\n1,Ann,ann@x.io,\\n2,Bob,bob@x.io,vip\\n"
                ''',
        },
        hidden_test='''\
            import csv
            import io

            from exporter.contacts import export_contacts


            def _parse(text):
                return list(csv.reader(io.StringIO(text)))


            def test_comma_in_field_round_trips():
                rows = _parse(export_contacts([{"id": 1, "name": "Doe, Jane", "email": "j@x.io", "notes": ""}]))
                assert rows[1] == ["1", "Doe, Jane", "j@x.io", ""]


            def test_quotes_and_newlines_round_trip():
                note = 'said "call me"\\nafter 5pm'
                rows = _parse(export_contacts([{"id": 7, "name": "Li", "email": "li@x.io", "notes": note}]))
                assert len(rows) == 2
                assert rows[1][3] == note


            def test_plain_rows_unchanged():
                text = export_contacts([{"id": 1, "name": "Ann", "email": "ann@x.io", "notes": "a"}])
                assert text == "id,name,email,notes\\n1,Ann,ann@x.io,a\\n"
            ''',
        p2p=["tests/test_export.py::test_simple_export"],
        steps=[
            ("Locate the CSV writer.", S("def rows_to_csv")),
            ("Read it.", V("exporter/csv_export.py")),
            ("It joins fields with ',' by hand, so nothing is quoted. Reproduce with a comma in the name.",
             B("python - <<'EOF'\nimport csv, io\nfrom exporter.contacts import export_contacts\n"
               "text = export_contacts([{\"id\": 1, \"name\": \"Doe, Jane\", \"email\": \"j@x.io\"}])\n"
               "print(list(csv.reader(io.StringIO(text))))\nEOF")),
            ("Use the csv module, which quotes fields containing delimiters, quotes or newlines. Add the import first.",
             I("exporter/csv_export.py", 0, "import csv\n\n")),
            ("Replace the manual join with csv.writer, keeping '\\n' line endings so plain output is unchanged.",
             R("exporter/csv_export.py",
               '    out.write(",".join(HEADER) + "\\n")\n    for row in rows:\n'
               '        out.write(",".join(str(row.get(col, "")) for col in HEADER) + "\\n")\n',
               '    writer = csv.writer(out, lineterminator="\\n")\n    writer.writerow(HEADER)\n'
               '    for row in rows:\n        writer.writerow([row.get(col, "") for col in HEADER])\n')),
            ("Re-run the repro.",
             B("python - <<'EOF'\nimport csv, io\nfrom exporter.contacts import export_contacts\n"
               "text = export_contacts([{\"id\": 1, \"name\": \"Doe, Jane\", \"email\": \"j@x.io\"}])\n"
               "print(list(csv.reader(io.StringIO(text))))\nEOF")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 74 ---------------------------------------------------------------------------------------
    dict(
        id="seed-074-output-name-first-dot",
        bug_type="path-extension-handling", difficulty="easy",
        issue="""\
            Image conversion fails for files in versioned folders

                >>> plan_conversions(["releases/v1.2/hero.png", "releases/v1.3/hero.png"], ".webp")
                Traceback (most recent call last):
                  File "assets/pipeline.py", line 12, in plan_conversions
                    raise ValueError(f"two inputs map to the same output {dst!r}")
                ValueError: two inputs map to the same output 'releases/v1.webp'

            The outputs should be `releases/v1.2/hero.webp` and `releases/v1.3/hero.webp`.
            Something similar happens with names like `logo.dark.png`.
            """,
        files={
            "assets/__init__.py": "",
            "assets/naming.py": '''\
                def output_name(src, new_ext):
                    """Replace the extension of `src`: ('img/cat.png', '.webp') -> 'img/cat.webp'."""
                    base = src.split(".")[0]
                    return base + new_ext
                ''',
            "assets/pipeline.py": '''\
                from .naming import output_name


                def plan_conversions(sources, new_ext):
                    """Map each source path to its converted output path."""
                    plan = {}
                    seen = set()
                    for src in sources:
                        dst = output_name(src, new_ext)
                        if dst in seen:
                            raise ValueError(f"two inputs map to the same output {dst!r}")
                        seen.add(dst)
                        plan[src] = dst
                    return plan
                ''',
            "tests/test_pipeline.py": '''\
                import pytest

                from assets.naming import output_name
                from assets.pipeline import plan_conversions


                def test_output_name_simple():
                    assert output_name("img/cat.png", ".webp") == "img/cat.webp"


                def test_plan():
                    assert plan_conversions(["a.png", "b.jpg"], ".webp") == {"a.png": "a.webp", "b.jpg": "b.webp"}


                def test_plan_detects_collision():
                    with pytest.raises(ValueError):
                        plan_conversions(["a.png", "a.jpg"], ".webp")
                ''',
        },
        hidden_test='''\
            from assets.naming import output_name
            from assets.pipeline import plan_conversions


            def test_dotted_directories():
                assert output_name("releases/v1.2/hero.png", ".webp") == "releases/v1.2/hero.webp"


            def test_multi_dot_filename():
                assert output_name("img/logo.dark.png", ".webp") == "img/logo.dark.webp"


            def test_plan_versioned_dirs():
                plan = plan_conversions(["releases/v1.2/hero.png", "releases/v1.3/hero.png"], ".webp")
                assert plan == {"releases/v1.2/hero.png": "releases/v1.2/hero.webp",
                                "releases/v1.3/hero.png": "releases/v1.3/hero.webp"}


            def test_no_extension():
                assert output_name("bin/README", ".txt") == "bin/README.txt"
            ''',
        p2p=["tests/test_pipeline.py::test_output_name_simple", "tests/test_pipeline.py::test_plan",
             "tests/test_pipeline.py::test_plan_detects_collision"],
        steps=[
            ("The error is raised in the pipeline; look at how it computes outputs.", V("assets/pipeline.py")),
            ("Find output_name.", S("def output_name", "assets")),
            ("Read it.", V("assets/naming.py")),
            ("`split('.')[0]` cuts at the first dot anywhere in the path, including directory names. Confirm.",
             B("python -c 'from assets.naming import output_name as o; print(o(\"releases/v1.2/hero.png\", \".webp\"), o(\"img/logo.dark.png\", \".webp\"))'")),
            ("Use os.path.splitext, which only strips the final extension of the last path component.",
             R("assets/naming.py",
               "def output_name(src, new_ext):",
               "import os\n\n\ndef output_name(src, new_ext):")),
            ("Replace the split.",
             R("assets/naming.py",
               '    base = src.split(".")[0]\n',
               "    base, _ext = os.path.splitext(src)\n")),
            ("Re-run the examples, plus a file with no extension.",
             B("python -c 'from assets.naming import output_name as o; print(o(\"releases/v1.2/hero.png\", \".webp\"), o(\"img/logo.dark.png\", \".webp\"), o(\"bin/README\", \".txt\"))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 75 ---------------------------------------------------------------------------------------
    dict(
        id="seed-075-version-string-compare",
        bug_type="lexicographic-comparison", difficulty="easy",
        issue="""\
            All plugins skipped after upgrading host to 1.10.0

            After bumping the host to 1.10.0, `load_plugins` puts every plugin that declares
            `requires_host: "1.9.0"` (or "1.2") into `skipped` with reason "requires host >= 1.9.0".
            On 1.9.4 they loaded fine.
            """,
        files={
            "pluginhost/__init__.py": "",
            "pluginhost/versions.py": '''\
                def normalize(version):
                    return version.strip().lstrip("v")


                def satisfies(installed, minimum):
                    """True if version `installed` is >= `minimum` (dotted numeric versions)."""
                    return normalize(installed) >= normalize(minimum)
                ''',
            "pluginhost/loader.py": '''\
                from .versions import satisfies


                def load_plugins(manifests, host_version):
                    """Split plugin manifests into (loaded names, {name: reason} for skipped)."""
                    loaded, skipped = [], {}
                    for m in manifests:
                        need = m.get("requires_host", "0")
                        if satisfies(host_version, need):
                            loaded.append(m["name"])
                        else:
                            skipped[m["name"]] = f"requires host >= {need}"
                    return loaded, skipped
                ''',
            "tests/test_loader.py": '''\
                from pluginhost.loader import load_plugins
                from pluginhost.versions import satisfies


                def test_satisfies_basic():
                    assert satisfies("1.2.0", "1.1.0")
                    assert satisfies("v2.0.0", "2.0.0")
                    assert not satisfies("1.0.0", "1.2.0")


                def test_load_plugins():
                    loaded, skipped = load_plugins(
                        [{"name": "a", "requires_host": "1.1.0"}, {"name": "b", "requires_host": "2.0.0"}, {"name": "c"}],
                        "1.5.0")
                    assert loaded == ["a", "c"]
                    assert skipped == {"b": "requires host >= 2.0.0"}
                ''',
        },
        hidden_test='''\
            from pluginhost.loader import load_plugins
            from pluginhost.versions import satisfies


            def test_multi_digit_components():
                assert satisfies("1.10.0", "1.9.0")
                assert not satisfies("1.9.9", "1.10")
                assert satisfies("10.0", "9.5.1")


            def test_missing_components_are_zero():
                assert satisfies("2.0", "2.0.0")
                assert satisfies("2.0.0", "2")
                assert not satisfies("2", "2.0.1")


            def test_loader_on_new_host():
                loaded, skipped = load_plugins([{"name": "x", "requires_host": "1.9.0"},
                                                {"name": "y", "requires_host": "1.2"}], "1.10.0")
                assert loaded == ["x", "y"] and skipped == {}
            ''',
        p2p=["tests/test_loader.py::test_satisfies_basic", "tests/test_loader.py::test_load_plugins"],
        steps=[
            ("Start at the loader that produces the skip reason.", V("pluginhost/loader.py")),
            ("The decision comes from satisfies(); read it.", V("pluginhost/versions.py")),
            ("It compares versions as strings, so '1.10.0' < '1.9.0'. Confirm.",
             B("python -c 'from pluginhost.versions import satisfies; print(satisfies(\"1.10.0\", \"1.9.0\"), satisfies(\"2.0\", \"2.0.0\"))'")),
            ("Compare numeric tuples instead, padding missing components with zeros so '2.0' == '2.0.0'.",
             R("pluginhost/versions.py",
               '    return normalize(installed) >= normalize(minimum)\n',
               '    a, b = _parts(installed), _parts(minimum)\n'
               '    width = max(len(a), len(b))\n'
               '    return a + (0,) * (width - len(a)) >= b + (0,) * (width - len(b))\n')),
            ("Add the parsing helper.",
             R("pluginhost/versions.py",
               '\n\ndef satisfies(installed, minimum):',
               '\n\ndef _parts(version):\n    return tuple(int(p) for p in normalize(version).split("."))\n\n\ndef satisfies(installed, minimum):')),
            ("Re-check the reported cases.",
             B("python -c 'from pluginhost.versions import satisfies as s; print(s(\"1.10.0\", \"1.9.0\"), s(\"2.0\", \"2.0.0\"), s(\"1.9.9\", \"1.10\"))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 76 ---------------------------------------------------------------------------------------
    dict(
        id="seed-076-env-vs-file-precedence",
        bug_type="precedence-order", difficulty="medium",
        issue="""\
            SVC_LOG_LEVEL is ignored when the config file also sets log_level

            Our image ships a `service.conf` with `log_level = info`. In the debugging
            deployment we set `SVC_LOG_LEVEL=debug`, but the service keeps logging at info.
            Removing the line from the conf file makes the env var work. The README says
            environment variables take precedence over the config file.
            """,
        files={
            "svcconf/__init__.py": "from .layers import build_settings\n",
            "svcconf/sources.py": '''\
                PREFIX = "SVC_"


                def from_env(environ):
                    """SVC_FOO=bar -> {"foo": "bar"}."""
                    return {k[len(PREFIX):].lower(): v for k, v in environ.items() if k.startswith(PREFIX)}


                def from_file_text(text):
                    """Parse `key = value` lines; '#' starts a comment."""
                    out = {}
                    for line in text.splitlines():
                        line = line.split("#", 1)[0].strip()
                        if not line:
                            continue
                        key, _, value = line.partition("=")
                        out[key.strip()] = value.strip()
                    return out
                ''',
            "svcconf/layers.py": '''\
                from collections import ChainMap

                from .sources import from_env, from_file_text

                DEFAULTS = {"log_level": "warning", "port": "8080", "region": "local"}


                def build_settings(file_text="", environ=None, overrides=None):
                    """Resolve settings. Precedence, highest first:
                    explicit overrides > environment (SVC_*) > config file > DEFAULTS.
                    """
                    chain = ChainMap(overrides or {}, from_file_text(file_text),
                                     from_env(environ or {}), DEFAULTS)
                    return dict(chain)
                ''',
            "tests/test_layers.py": '''\
                from svcconf import build_settings


                def test_defaults():
                    assert build_settings() == {"log_level": "warning", "port": "8080", "region": "local"}


                def test_file_overrides_defaults():
                    s = build_settings("log_level = info  # baked in\\nport=9000\\n")
                    assert s["log_level"] == "info" and s["port"] == "9000"


                def test_env_overrides_defaults():
                    assert build_settings(environ={"SVC_REGION": "eu", "HOME": "/x"})["region"] == "eu"


                def test_overrides_win():
                    s = build_settings("port=9000", {"SVC_PORT": "9100"}, {"port": "1"})
                    assert s["port"] == "1"
                ''',
        },
        hidden_test='''\
            from svcconf import build_settings


            def test_env_beats_file():
                s = build_settings("log_level = info\\n", {"SVC_LOG_LEVEL": "debug"})
                assert s["log_level"] == "debug"


            def test_file_still_used_for_keys_not_in_env():
                s = build_settings("log_level = info\\nport = 9000\\n", {"SVC_LOG_LEVEL": "debug"})
                assert s == {"log_level": "debug", "port": "9000", "region": "local"}


            def test_override_beats_env_and_file():
                s = build_settings("region = us", {"SVC_REGION": "eu"}, {"region": "ap"})
                assert s["region"] == "ap"
            ''',
        p2p=["tests/test_layers.py::test_defaults", "tests/test_layers.py::test_file_overrides_defaults",
             "tests/test_layers.py::test_env_overrides_defaults", "tests/test_layers.py::test_overrides_win"],
        steps=[
            ("List the package files.", F("svcconf/*.py")),
            ("Check how SVC_* variables are parsed first.", V("svcconf/sources.py")),
            ("Parsing looks right ('SVC_LOG_LEVEL' -> 'log_level'). Look at how the layers are combined.",
             V("svcconf/layers.py")),
            ("Reproduce with both the file and the env var setting log_level.",
             B("python -c 'from svcconf import build_settings; print(build_settings(\"log_level = info\", {\"SVC_LOG_LEVEL\": \"debug\"}))'")),
            ("ChainMap looks keys up left to right, and the file mapping sits before the env mapping, contradicting the docstring. Swap them.",
             R("svcconf/layers.py",
               "    chain = ChainMap(overrides or {}, from_file_text(file_text),\n"
               "                     from_env(environ or {}), DEFAULTS)\n",
               "    chain = ChainMap(overrides or {}, from_env(environ or {}),\n"
               "                     from_file_text(file_text), DEFAULTS)\n")),
            ("Re-run the repro.",
             B("python -c 'from svcconf import build_settings; print(build_settings(\"log_level = info\", {\"SVC_LOG_LEVEL\": \"debug\"}))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 77 ---------------------------------------------------------------------------------------
    dict(
        id="seed-077-json-decimal-float",
        bug_type="serialization-type-loss", difficulty="medium",
        issue="""\
            balance() crashes on ledgers with cents

                Traceback (most recent call last):
                  File "ledger/report.py", line 9, in balance
                    total += entry["amount"]
                TypeError: unsupported operand type(s) for +=: 'decimal.Decimal' and 'float'

            Happens for any saved ledger where an amount has a fractional part, e.g. 19.99.
            Ledgers with only whole amounts work. We use Decimal everywhere for money.
            """,
        files={
            "ledger/__init__.py": "",
            "ledger/codec.py": '''\
                import json
                from decimal import Decimal


                class _Encoder(json.JSONEncoder):
                    def default(self, o):
                        if isinstance(o, Decimal):
                            return float(o)
                        return super().default(o)


                def dump_entries(entries):
                    return json.dumps({"entries": entries}, cls=_Encoder)


                def load_entries(text):
                    return json.loads(text)["entries"]
                ''',
            "ledger/report.py": '''\
                from decimal import ROUND_HALF_UP, Decimal

                from .codec import load_entries


                def balance(text, opening=Decimal("0.00")):
                    total = opening
                    for entry in load_entries(text):
                        total += entry["amount"]
                    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ''',
            "tests/test_ledger.py": '''\
                from decimal import Decimal

                from ledger.codec import dump_entries, load_entries
                from ledger.report import balance


                def test_dump_writes_numbers():
                    text = dump_entries([{"memo": "coffee", "amount": Decimal("3.50")}])
                    assert text == '{"entries": [{"memo": "coffee", "amount": 3.5}]}'


                def test_balance_whole_amounts():
                    text = dump_entries([{"amount": 10}, {"amount": -3}])
                    assert balance(text) == Decimal("7.00")


                def test_load_keeps_other_fields():
                    assert load_entries('{"entries": [{"memo": "x", "amount": 1}]}') == [{"memo": "x", "amount": 1}]
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from ledger.codec import dump_entries, load_entries
            from ledger.report import balance


            def test_fractional_amounts_load_as_decimal():
                entries = load_entries(dump_entries([{"amount": Decimal("19.99")}]))
                assert entries[0]["amount"] == Decimal("19.99")
                assert isinstance(entries[0]["amount"], Decimal)


            def test_balance_with_cents_is_exact():
                text = dump_entries([{"amount": Decimal("0.10")}, {"amount": Decimal("0.20")}])
                assert balance(text, Decimal("100.00")) == Decimal("100.30")


            def test_integers_stay_integers():
                assert load_entries('{"entries": [{"amount": 5}]}')[0]["amount"] == 5
            ''',
        p2p=["tests/test_ledger.py::test_dump_writes_numbers", "tests/test_ledger.py::test_balance_whole_amounts",
             "tests/test_ledger.py::test_load_keeps_other_fields"],
        steps=[
            ("Look at the function from the traceback.", V("ledger/report.py")),
            ("The amounts come from load_entries; check the codec.", V("ledger/codec.py")),
            ("json.loads turns every fractional number into a float. Confirm the loaded type.",
             B("python -c 'from decimal import Decimal; from ledger.codec import dump_entries, load_entries; print(load_entries(dump_entries([{\"amount\": Decimal(\"19.99\")}])))'")),
            ("Parse fractional JSON numbers as Decimal on load so money stays exact; integers are unaffected.",
             R("ledger/codec.py",
               '    return json.loads(text)["entries"]\n',
               '    return json.loads(text, parse_float=Decimal)["entries"]\n')),
            ("Re-run the failing balance scenario.",
             B("python -c 'from decimal import Decimal as D; from ledger.codec import dump_entries; from ledger.report import balance; print(balance(dump_entries([{\"amount\": D(\"0.10\")}, {\"amount\": D(\"0.20\")}])))'")),
            ("Make sure nothing else calls json.loads on ledger data directly.", S("json\\.loads", "ledger")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 78 ---------------------------------------------------------------------------------------
    dict(
        id="seed-078-path-prefix-sibling",
        bug_type="path-containment-check", difficulty="medium",
        issue="""\
            plan_upload crashes when a sibling folder shares the project prefix

                >>> plan_upload("/data/proj", ["/data/proj/a.txt", "/data/proj-old/b.txt"])
                Traceback (most recent call last):
                  File "syncer/plan.py", line 6, in plan_upload
                    ...
                ValueError: '/data/proj-old/b.txt' is not in the subpath of '/data/proj'

            Files outside the root are supposed to be ignored, and that works for e.g.
            `/data/other/x.txt`.
            """,
        files={
            "syncer/__init__.py": "",
            "syncer/paths.py": '''\
                from pathlib import PurePosixPath


                def is_inside(root, path):
                    """True if `path` is `root` itself or somewhere beneath it."""
                    return str(PurePosixPath(path)).startswith(str(PurePosixPath(root)))


                def relative_to_root(root, path):
                    return str(PurePosixPath(path).relative_to(root))
                ''',
            "syncer/plan.py": '''\
                from .paths import is_inside, relative_to_root


                def plan_upload(root, changed_paths):
                    """Root-relative paths of the changed files that live under `root`, sorted."""
                    return sorted(relative_to_root(root, p) for p in changed_paths if is_inside(root, p))
                ''',
            "tests/test_plan.py": '''\
                from syncer.paths import is_inside
                from syncer.plan import plan_upload


                def test_ignores_unrelated_dirs():
                    assert plan_upload("/data/proj", ["/data/proj/b/c.txt", "/data/other/x", "/data/proj/a.txt"]) == [
                        "a.txt", "b/c.txt"]


                def test_is_inside_nested():
                    assert is_inside("/data/proj", "/data/proj/deep/er/file")
                    assert not is_inside("/data/proj", "/etc/passwd")
                ''',
        },
        hidden_test='''\
            from syncer.paths import is_inside
            from syncer.plan import plan_upload


            def test_sibling_with_common_prefix_is_outside():
                assert not is_inside("/data/proj", "/data/proj-old/b.txt")
                assert not is_inside("/data/proj", "/data/project/x")


            def test_plan_skips_prefix_sibling():
                assert plan_upload("/data/proj", ["/data/proj/a.txt", "/data/proj-old/b.txt"]) == ["a.txt"]


            def test_root_and_trailing_slash():
                assert is_inside("/data/proj", "/data/proj")
                assert is_inside("/data/proj/", "/data/proj/a.txt")
            ''',
        p2p=["tests/test_plan.py::test_ignores_unrelated_dirs", "tests/test_plan.py::test_is_inside_nested"],
        steps=[
            ("Find the path helpers used by plan_upload.", S("is_inside|relative_to", "syncer")),
            ("Read them.", V("syncer/paths.py")),
            ("The relative_to failure is a symptom: the filter let the sibling through because it is a string-prefix check. Confirm.",
             B("python -c 'from syncer.paths import is_inside; print(is_inside(\"/data/proj\", \"/data/proj-old/b.txt\"), is_inside(\"/data/proj\", \"/data/proj/a.txt\"))'")),
            ("Compare path components instead of characters: inside means equal to root or root is one of its parents.",
             R("syncer/paths.py",
               "    return str(PurePosixPath(path)).startswith(str(PurePosixPath(root)))\n",
               "    root, path = PurePosixPath(root), PurePosixPath(path)\n"
               "    return path == root or root in path.parents\n")),
            ("Re-run the issue example.",
             B("python -c 'from syncer.plan import plan_upload; print(plan_upload(\"/data/proj\", [\"/data/proj/a.txt\", \"/data/proj-old/b.txt\"]))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 79 ---------------------------------------------------------------------------------------
    dict(
        id="seed-079-glob-regex-unescaped",
        bug_type="regex-escaping", difficulty="medium",
        issue="""\
            --include '**/*.py' picks up files that aren't Python

            `select(paths, include=["**/*.py"])` returns things like `scripts/deploy_py` and
            `docs/happy`. Separately, excluding `build+tmp/*` doesn't exclude anything in our
            `build+tmp/` folder.
            """,
        files={
            "srcfind/__init__.py": "",
            "srcfind/patterns.py": '''\
                import re


                def glob_to_regex(pattern):
                    """Translate a glob into an anchored regex over '/'-separated paths.

                    '**/' matches zero or more directories, '*' anything except '/', '?' one character.
                    """
                    out, i = [], 0
                    while i < len(pattern):
                        if pattern.startswith("**/", i):
                            out.append("(?:.*/)?")
                            i += 3
                        elif pattern[i] == "*":
                            out.append("[^/]*")
                            i += 1
                        elif pattern[i] == "?":
                            out.append("[^/]")
                            i += 1
                        else:
                            out.append(pattern[i])
                            i += 1
                    return re.compile("^" + "".join(out) + "$")
                ''',
            "srcfind/finder.py": '''\
                from .patterns import glob_to_regex


                def select(paths, include, exclude=()):
                    """Paths matching any include glob and no exclude glob, sorted."""
                    inc = [glob_to_regex(p) for p in include]
                    exc = [glob_to_regex(p) for p in exclude]
                    return sorted(p for p in paths
                                  if any(r.match(p) for r in inc) and not any(r.match(p) for r in exc))
                ''',
            "tests/test_finder.py": '''\
                from srcfind.finder import select
                from srcfind.patterns import glob_to_regex


                def test_double_star():
                    rx = glob_to_regex("**/*.py")
                    assert rx.match("a/b/c.py") and rx.match("c.py")
                    assert not rx.match("c.pyc")


                def test_question_mark():
                    assert glob_to_regex("v?/x").match("v1/x")
                    assert not glob_to_regex("v?/x").match("v10/x")


                def test_select_with_exclude():
                    paths = ["src/a.py", "src/b.py", "tests/t.py", "README.md"]
                    assert select(paths, ["**/*.py"], ["tests/*"]) == ["src/a.py", "src/b.py"]
                ''',
        },
        hidden_test='''\
            from srcfind.finder import select
            from srcfind.patterns import glob_to_regex


            def test_dot_is_literal():
                rx = glob_to_regex("**/*.py")
                assert not rx.match("docs/happy")
                assert not rx.match("scripts/deploy_py")


            def test_regex_metachars_are_literal():
                assert glob_to_regex("build+tmp/*").match("build+tmp/out.o")
                assert glob_to_regex("notes(1).txt").match("notes(1).txt")
                assert not glob_to_regex("a.b").match("axb")


            def test_select_end_to_end():
                paths = ["docs/happy", "src/app.py", "build+tmp/gen.py", "scripts/deploy_py"]
                assert select(paths, ["**/*.py"], ["build+tmp/*"]) == ["src/app.py"]
            ''',
        p2p=["tests/test_finder.py::test_double_star", "tests/test_finder.py::test_question_mark",
             "tests/test_finder.py::test_select_with_exclude"],
        steps=[
            ("select() delegates matching to glob_to_regex; find it.", S("def glob_to_regex|glob_to_regex\\(")),
            ("Read the translator.", V("srcfind/patterns.py")),
            ("Literal characters are appended raw, so '.' and '+' act as regex operators. Print the generated regexes.",
             B("python -c 'from srcfind.patterns import glob_to_regex as g; r=g(\"**/*.py\"); print(r.pattern, bool(r.match(\"docs/happy\")), g(\"build+tmp/*\").pattern)'")),
            ("Escape every literal character.",
             R("srcfind/patterns.py",
               "            out.append(pattern[i])\n            i += 1\n",
               "            out.append(re.escape(pattern[i]))\n            i += 1\n")),
            ("Re-run, including the exclude case from the issue.",
             B("python -c 'from srcfind.finder import select; print(select([\"docs/happy\", \"src/app.py\", \"build+tmp/gen.py\", \"scripts/deploy_py\"], [\"**/*.py\"], [\"build+tmp/*\"]))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 80 ---------------------------------------------------------------------------------------
    dict(
        id="seed-080-cache-key-param-order",
        bug_type="cache-key-normalization", difficulty="medium",
        issue="""\
            Response cache misses for identical requests

            The on-disk cache barely ever hits. Two calls that should be the same request,

                client.get("/search", q="shoes", page=2)
                client.get("/search", page=2, q="shoes")

            both reach the backend and leave two different files in the cache directory. Some of
            our callers build the params from dicts assembled in different orders.
            """,
        files={
            "fetchcache/__init__.py": "",
            "fetchcache/keys.py": '''\
                import hashlib
                import json


                def cache_key(endpoint, params):
                    payload = json.dumps({"endpoint": endpoint, "params": params})
                    return hashlib.sha256(payload.encode()).hexdigest()
                ''',
            "fetchcache/store.py": '''\
                import json
                from pathlib import Path


                class DiskCache:
                    def __init__(self, directory):
                        self.dir = Path(directory)
                        self.dir.mkdir(parents=True, exist_ok=True)

                    def _path(self, key):
                        return self.dir / f"{key}.json"

                    def get(self, key):
                        p = self._path(key)
                        return json.loads(p.read_text()) if p.exists() else None

                    def set(self, key, value):
                        self._path(key).write_text(json.dumps(value))
                ''',
            "fetchcache/client.py": '''\
                from .keys import cache_key


                class CachedClient:
                    def __init__(self, fetch, cache):
                        self._fetch = fetch
                        self._cache = cache

                    def get(self, endpoint, **params):
                        key = cache_key(endpoint, params)
                        hit = self._cache.get(key)
                        if hit is not None:
                            return hit
                        value = self._fetch(endpoint, params)
                        self._cache.set(key, value)
                        return value
                ''',
            "tests/test_client.py": '''\
                from fetchcache.client import CachedClient
                from fetchcache.store import DiskCache


                def make(tmp_path):
                    calls = []

                    def fetch(endpoint, params):
                        calls.append((endpoint, dict(params)))
                        return {"endpoint": endpoint, "n": len(calls)}

                    return CachedClient(fetch, DiskCache(tmp_path / "cache")), calls


                def test_second_identical_call_hits(tmp_path):
                    client, calls = make(tmp_path)
                    assert client.get("/a", x=1) == client.get("/a", x=1)
                    assert len(calls) == 1


                def test_different_params_miss(tmp_path):
                    client, calls = make(tmp_path)
                    client.get("/a", x=1)
                    client.get("/a", x=2)
                    assert len(calls) == 2
                ''',
        },
        hidden_test='''\
            from fetchcache.client import CachedClient
            from fetchcache.keys import cache_key
            from fetchcache.store import DiskCache


            def test_key_ignores_param_order():
                assert cache_key("/s", {"q": "x", "page": 2}) == cache_key("/s", {"page": 2, "q": "x"})


            def test_key_ignores_nested_order():
                a = cache_key("/s", {"filter": {"color": "red", "size": 9}})
                b = cache_key("/s", {"filter": {"size": 9, "color": "red"}})
                assert a == b


            def test_key_distinguishes_endpoint_and_values():
                assert cache_key("/s", {"q": "x"}) != cache_key("/t", {"q": "x"})
                assert cache_key("/s", {"q": "x"}) != cache_key("/s", {"q": "y"})


            def test_client_hits_regardless_of_kwarg_order(tmp_path):
                calls = []
                client = CachedClient(lambda e, p: calls.append(1) or {"ok": True}, DiskCache(tmp_path))
                client.get("/search", q="shoes", page=2)
                client.get("/search", page=2, q="shoes")
                assert len(calls) == 1
                assert len(list(tmp_path.glob("*.json"))) == 1
            ''',
        p2p=["tests/test_client.py::test_second_identical_call_hits",
             "tests/test_client.py::test_different_params_miss"],
        steps=[
            ("List the package.", F("fetchcache/*.py")),
            ("See how the client derives the cache key.", V("fetchcache/client.py")),
            ("Read cache_key.", V("fetchcache/keys.py")),
            ("json.dumps preserves dict insertion order, so the same params in a different order hash differently. Confirm.",
             B("python -c 'from fetchcache.keys import cache_key as k; print(k(\"/s\", {\"q\": \"x\", \"page\": 2}) == k(\"/s\", {\"page\": 2, \"q\": \"x\"}))'")),
            ("Serialize canonically: sorted keys (recursively, which sort_keys does) and fixed separators.",
             R("fetchcache/keys.py",
               '    payload = json.dumps({"endpoint": endpoint, "params": params})\n',
               '    payload = json.dumps({"endpoint": endpoint, "params": params},\n'
               '                         sort_keys=True, separators=(",", ":"))\n')),
            ("Re-check, including nested dicts.",
             B("python -c 'from fetchcache.keys import cache_key as k; print(k(\"/s\", {\"q\": \"x\", \"page\": 2}) == k(\"/s\", {\"page\": 2, \"q\": \"x\"}), k(\"/s\", {\"f\": {\"a\": 1, \"b\": 2}}) == k(\"/s\", {\"f\": {\"b\": 2, \"a\": 1}}))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 81 ---------------------------------------------------------------------------------------
    dict(
        id="seed-081-memoize-ignores-kwargs",
        bug_type="cache-key-collision", difficulty="medium",
        issue="""\
            EU quotes come back with US tax

            In the checkout service, if a cart is quoted for the US first, quoting the same cart
            with `region="eu"` returns exactly the US total. Restarting the process "fixes" it
            until the next US quote. Passing a different `discount=` is also ignored.
            """,
        files={
            "quotes/__init__.py": "",
            "quotes/memo.py": '''\
                import functools


                def _freeze(value):
                    """Turn lists/dicts/sets into hashable equivalents so they can be cache keys."""
                    if isinstance(value, dict):
                        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
                    if isinstance(value, (list, tuple)):
                        return tuple(_freeze(v) for v in value)
                    if isinstance(value, (set, frozenset)):
                        return frozenset(_freeze(v) for v in value)
                    return value


                def memoize(fn):
                    cache = {}

                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        key = _freeze(args)
                        if key not in cache:
                            cache[key] = fn(*args, **kwargs)
                        return cache[key]

                    wrapper.cache = cache
                    wrapper.cache_clear = cache.clear
                    return wrapper
                ''',
            "quotes/pricing.py": '''\
                from .memo import memoize

                TAX_PERCENT = {"us": 7, "eu": 20, "uk": 20}


                @memoize
                def quote(items, region="us", discount=None):
                    """items: list of (sku, qty, unit_cents). discount: {"percent": n} or None. Returns cents."""
                    subtotal = sum(qty * unit for _, qty, unit in items)
                    if discount:
                        subtotal -= subtotal * discount["percent"] // 100
                    return subtotal + subtotal * TAX_PERCENT[region] // 100
                ''',
            "tests/test_pricing.py": '''\
                from quotes.pricing import quote


                CART = [["sku-1", 2, 500], ["sku-2", 1, 1000]]


                def test_quote_us():
                    quote.cache_clear()
                    assert quote(CART) == 2140


                def test_list_args_are_cached():
                    quote.cache_clear()
                    quote(CART)
                    quote([list(row) for row in CART])
                    assert len(quote.cache) == 1
                ''',
        },
        hidden_test='''\
            from quotes.pricing import quote

            CART = [["sku-1", 2, 500], ["sku-2", 1, 1000]]


            def test_region_kwarg_changes_result():
                quote.cache_clear()
                assert quote(CART, region="us") == 2140
                assert quote(CART, region="eu") == 2400


            def test_discount_kwarg_changes_result():
                quote.cache_clear()
                assert quote(CART, discount={"percent": 10}) == 1926
                assert quote(CART, discount={"percent": 50}) == 1070
                assert quote(CART) == 2140


            def test_repeated_kwargs_still_cached():
                quote.cache_clear()
                quote(CART, region="uk", discount={"percent": 10})
                quote(CART, discount={"percent": 10}, region="uk")
                assert len(quote.cache) == 1
            ''',
        p2p=["tests/test_pricing.py::test_quote_us", "tests/test_pricing.py::test_list_args_are_cached"],
        steps=[
            ("Read the pricing function.", V("quotes/pricing.py")),
            ("The tax math looks right, and quote is memoized. Reproduce US-then-EU.",
             B("python -c 'from quotes.pricing import quote; c=[[\"a\", 2, 500], [\"b\", 1, 1000]]; print(quote(c, region=\"us\"), quote(c, region=\"eu\"))'")),
            ("Check the memoize decorator.", S("def memoize", "quotes")),
            ("Read it.", V("quotes/memo.py")),
            ("The key is built from positional args only, so calls differing only in kwargs collide. Include frozen kwargs (dict freezing sorts items, so kwarg order doesn't matter).",
             R("quotes/memo.py",
               "        key = _freeze(args)\n",
               "        key = (_freeze(args), _freeze(kwargs))\n")),
            ("Re-run the repro.",
             B("python -c 'from quotes.pricing import quote; c=[[\"a\", 2, 500], [\"b\", 1, 1000]]; print(quote(c, region=\"us\"), quote(c, region=\"eu\"), quote(c, discount={\"percent\": 10}))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 82 ---------------------------------------------------------------------------------------
    dict(
        id="seed-082-logging-duplicate-handlers",
        bug_type="duplicate-handlers", difficulty="medium",
        issue="""\
            Log lines duplicated after config reload

            After a config reload every worker log line shows up twice on stderr; after a
            second reload, three times. Also, lowering verbosity on reload (log_level: warning)
            still leaves INFO lines coming through once.
            """,
        files={
            "applog/__init__.py": "",
            "applog/setup.py": '''\
                import logging

                FORMAT = "%(levelname)s %(name)s: %(message)s"


                def configure(level="INFO", stream=None):
                    """Configure the 'applog' logger. Safe to call again, e.g. on config reload."""
                    logger = logging.getLogger("applog")
                    handler = logging.StreamHandler(stream)
                    handler.setLevel(level.upper())
                    handler.setFormatter(logging.Formatter(FORMAT))
                    logger.addHandler(handler)
                    logger.setLevel(level.upper())
                    logger.propagate = False
                    return logger
                ''',
            "applog/reload.py": '''\
                from .setup import configure


                def apply_config(cfg, stream=None):
                    """Called at startup and again whenever the config file changes."""
                    return configure(cfg.get("log_level", "INFO"), stream)
                ''',
            "applog/worker.py": '''\
                import logging

                log = logging.getLogger("applog.worker")


                def process(jobs):
                    for job in jobs:
                        log.info("processing %s", job)
                    log.warning("done: %d jobs", len(jobs))
                ''',
            "tests/test_logging.py": '''\
                import io

                from applog.reload import apply_config
                from applog.worker import process


                def test_worker_logs_info():
                    buf = io.StringIO()
                    apply_config({"log_level": "info"}, buf)
                    process(["a"])
                    assert "INFO applog.worker: processing a" in buf.getvalue()


                def test_warning_level_hides_info():
                    buf = io.StringIO()
                    apply_config({"log_level": "warning"}, buf)
                    process(["a"])
                    assert "processing" not in buf.getvalue()
                    assert "WARNING applog.worker: done: 1 jobs" in buf.getvalue()
                ''',
        },
        hidden_test='''\
            import io

            from applog.reload import apply_config
            from applog.worker import process


            def test_reload_does_not_duplicate_lines():
                buf = io.StringIO()
                apply_config({"log_level": "info"}, buf)
                apply_config({"log_level": "info"}, buf)
                apply_config({"log_level": "info"}, buf)
                process(["a"])
                assert buf.getvalue().count("processing a") == 1


            def test_reload_lowers_verbosity():
                buf = io.StringIO()
                apply_config({"log_level": "debug"}, buf)
                apply_config({"log_level": "warning"}, buf)
                process(["a"])
                assert "processing" not in buf.getvalue()
                assert buf.getvalue().count("done: 1 jobs") == 1
            ''',
        p2p=["tests/test_logging.py::test_worker_logs_info", "tests/test_logging.py::test_warning_level_hides_info"],
        steps=[
            ("Find where handlers are attached.", S("addHandler|getLogger", "applog")),
            ("Read the setup function.", V("applog/setup.py")),
            ("Every call adds a new StreamHandler without removing the previous one. Reproduce three reloads on one stream.",
             B("python - <<'EOF'\nimport io\nfrom applog.reload import apply_config\nfrom applog.worker import process\n"
               "buf = io.StringIO()\nfor level in (\"debug\", \"info\", \"warning\"):\n    apply_config({\"log_level\": level}, buf)\n"
               "process([\"a\"])\nprint(buf.getvalue())\nEOF")),
            ("Remove handlers left over from a previous configure() before adding the new one; the stale DEBUG handler also explains INFO leaking after lowering the level.",
             R("applog/setup.py",
               '    logger = logging.getLogger("applog")\n    handler = logging.StreamHandler(stream)\n',
               '    logger = logging.getLogger("applog")\n'
               '    for old in list(logger.handlers):\n'
               '        logger.removeHandler(old)\n'
               '        old.close()\n'
               '    handler = logging.StreamHandler(stream)\n')),
            ("Re-run the repro: expect only the WARNING line once.",
             B("python - <<'EOF'\nimport io\nfrom applog.reload import apply_config\nfrom applog.worker import process\n"
               "buf = io.StringIO()\nfor level in (\"debug\", \"info\", \"warning\"):\n    apply_config({\"log_level\": level}, buf)\n"
               "process([\"a\"])\nprint(buf.getvalue())\nEOF")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 83 ---------------------------------------------------------------------------------------
    dict(
        id="seed-083-cli-defaults-clobber-config",
        bug_type="argparse-default-precedence", difficulty="hard",
        issue="""\
            Config file values ignored for port/workers/debug

            deploy.json:

                {"port": 9000, "workers": 8, "debug": true, "region": "eu"}

            `deployctl --config deploy.json` starts with port 8000, 2 workers and debug off.
            `region` from the same file *is* applied. Passing `--port 9000` explicitly works,
            but the whole point of the file is not to have to.
            """,
        files={
            "deployctl/__init__.py": "",
            "deployctl/cli.py": '''\
                import argparse


                def build_parser():
                    p = argparse.ArgumentParser(prog="deployctl")
                    p.add_argument("--config", help="path to a JSON config file")
                    p.add_argument("--host", default="127.0.0.1")
                    p.add_argument("--port", type=int, default=8000)
                    p.add_argument("--workers", type=int, default=2)
                    p.add_argument("--debug", action="store_true")
                    return p
                ''',
            "deployctl/settings.py": '''\
                DEFAULTS = {"region": "local", "log_level": "info"}


                def resolve(file_cfg, cli_opts):
                    """Precedence: DEFAULTS < config file < command-line flags.

                    Options the user did not pass on the command line are None and are ignored.
                    """
                    merged = dict(DEFAULTS)
                    merged.update(file_cfg or {})
                    merged.update({k: v for k, v in cli_opts.items() if v is not None})
                    return merged
                ''',
            "deployctl/main.py": '''\
                import json
                from pathlib import Path

                from .cli import build_parser
                from .settings import resolve


                def load_settings(argv, read_file=None):
                    read_file = read_file or (lambda path: Path(path).read_text())
                    opts = vars(build_parser().parse_args(argv))
                    path = opts.pop("config")
                    file_cfg = json.loads(read_file(path)) if path else {}
                    return resolve(file_cfg, opts)
                ''',
            "tests/test_main.py": '''\
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
                ''',
        },
        hidden_test='''\
            import json

            from deployctl.main import load_settings


            def reader(files):
                return lambda path: json.dumps(files[path])


            FILE = {"d.json": {"port": 9000, "workers": 8, "debug": True, "host": "0.0.0.0"}}


            def test_file_values_used_when_flags_absent():
                s = load_settings(["--config", "d.json"], reader(FILE))
                assert (s["port"], s["workers"], s["debug"], s["host"]) == (9000, 8, True, "0.0.0.0")


            def test_flags_beat_file():
                s = load_settings(["--config", "d.json", "--port", "9100", "--workers", "3"], reader(FILE))
                assert (s["port"], s["workers"], s["debug"]) == (9100, 3, True)


            def test_debug_flag_without_file():
                s = load_settings(["--debug"])
                assert s["debug"] is True and s["port"] == 8000


            def test_defaults_without_file_or_flags():
                assert load_settings([]) == {"host": "127.0.0.1", "port": 8000, "workers": 2,
                                             "debug": False, "region": "local", "log_level": "info"}
            ''',
        p2p=["tests/test_main.py::test_builtin_defaults", "tests/test_main.py::test_cli_flag",
             "tests/test_main.py::test_file_only_keys"],
        steps=[
            ("List the package.", F("deployctl/*.py")),
            ("Start at the entry point that combines the config file and the flags.", V("deployctl/main.py")),
            ("Check the merge order.", V("deployctl/settings.py")),
            ("resolve() only skips None CLI values. Look at the parser defaults.", V("deployctl/cli.py")),
            ("argparse fills in 8000/2/'127.0.0.1'/False when a flag is absent, so they are never None and always beat the file. Reproduce.",
             B("python - <<'EOF'\nimport json\nfrom deployctl.cli import build_parser\nfrom deployctl.main import load_settings\n"
               "print(vars(build_parser().parse_args([\"--config\", \"d.json\"])))\n"
               "print(load_settings([\"--config\", \"d.json\"], lambda p: json.dumps({\"port\": 9000, \"workers\": 8, \"debug\": True})))\nEOF")),
            ("Make unspecified flags None (including the store_true flag) so resolve() can tell 'not passed' from a value.",
             R("deployctl/cli.py",
               '    p.add_argument("--host", default="127.0.0.1")\n'
               '    p.add_argument("--port", type=int, default=8000)\n'
               '    p.add_argument("--workers", type=int, default=2)\n'
               '    p.add_argument("--debug", action="store_true")\n',
               '    # Defaults live in settings.DEFAULTS; None here means "not given on the command line".\n'
               '    p.add_argument("--host", default=None)\n'
               '    p.add_argument("--port", type=int, default=None)\n'
               '    p.add_argument("--workers", type=int, default=None)\n'
               '    p.add_argument("--debug", action="store_true", default=None)\n')),
            ("Move the built-in defaults into the lowest-precedence layer.",
             R("deployctl/settings.py",
               'DEFAULTS = {"region": "local", "log_level": "info"}\n',
               'DEFAULTS = {"host": "127.0.0.1", "port": 8000, "workers": 2, "debug": False,\n'
               '            "region": "local", "log_level": "info"}\n')),
            ("Re-run: file values should apply, explicit flags should still win.",
             B("python - <<'EOF'\nimport json\nfrom deployctl.main import load_settings\n"
               "r = lambda p: json.dumps({\"port\": 9000, \"workers\": 8, \"debug\": True})\n"
               "print(load_settings([\"--config\", \"d.json\"], r))\n"
               "print(load_settings([\"--config\", \"d.json\", \"--port\", \"9100\"], r))\nprint(load_settings([]))\nEOF")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 84 ---------------------------------------------------------------------------------------
    dict(
        id="seed-084-mtime-fingerprint-truncated",
        bug_type="stale-cache-invalidation", difficulty="hard",
        issue="""\
            Watch mode sometimes serves stale build output

            With `Builder` in watch mode, if I fix a typo and save quickly (e.g. change
            `x = 1` to `x = 2` right after the previous save) the rebuilt output is still the
            old one. Touching the file again a bit later fixes it. Feels like a race but it
            reproduces reliably with fast saves where the file length doesn't change.
            """,
        files={
            "buildcache/__init__.py": "",
            "buildcache/fingerprint.py": '''\
                import os


                def fingerprint(path):
                    """Cheap change detector for a source file, stored in the state file as a string."""
                    st = os.stat(path)
                    return f"{int(st.st_mtime)}:{st.st_size}"
                ''',
            "buildcache/builder.py": '''\
                from pathlib import Path

                from .fingerprint import fingerprint


                class Builder:
                    def __init__(self, compile_fn, state=None):
                        self.compile_fn = compile_fn
                        self.state = state if state is not None else {}  # path -> [fingerprint, output]
                        self.compiled = 0

                    def build(self, path):
                        fp = fingerprint(path)
                        entry = self.state.get(str(path))
                        if entry and entry[0] == fp:
                            return entry[1]
                        out = self.compile_fn(Path(path).read_text())
                        self.compiled += 1
                        self.state[str(path)] = [fp, out]
                        return out
                ''',
            "buildcache/statefile.py": '''\
                import json
                from pathlib import Path


                def save_state(state, path):
                    Path(path).write_text(json.dumps(state, sort_keys=True))


                def load_state(path):
                    p = Path(path)
                    return json.loads(p.read_text()) if p.exists() else {}
                ''',
            "tests/test_builder.py": '''\
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
                ''',
        },
        hidden_test='''\
            import os

            from buildcache.builder import Builder
            from buildcache.statefile import load_state, save_state


            def write(p, text, ns):
                p.write_text(text)
                os.utime(p, ns=(ns, ns))


            def test_same_second_same_size_edit_recompiles(tmp_path):
                src = tmp_path / "a.src"
                write(src, "x = 1", 1_700_000_000_100_000_000)
                b = Builder(str.upper)
                assert b.build(src) == "X = 1"
                write(src, "x = 2", 1_700_000_000_600_000_000)
                assert b.build(src) == "X = 2"
                assert b.compiled == 2


            def test_reloaded_state_still_hits_after_subsecond_mtime(tmp_path):
                src, state_file = tmp_path / "a.src", tmp_path / "state.json"
                write(src, "z = 3", 1_700_000_000_123_456_789)
                b = Builder(str.upper)
                b.build(src)
                save_state(b.state, state_file)
                b2 = Builder(str.upper, load_state(state_file))
                assert b2.build(src) == "Z = 3"
                assert b2.compiled == 0
            ''',
        p2p=["tests/test_builder.py::test_unchanged_file_not_recompiled",
             "tests/test_builder.py::test_change_seconds_later_recompiles",
             "tests/test_builder.py::test_state_survives_restart"],
        steps=[
            ("Look at how the builder decides whether to recompile.", V("buildcache/builder.py")),
            ("It compares fingerprints; find the implementation.", S("def fingerprint")),
            ("Read it.", V("buildcache/fingerprint.py")),
            ("`int(st.st_mtime)` drops sub-second precision, so two saves within the same second with equal size look identical. Reproduce with explicit nanosecond mtimes in a scratch dir inside the repo.",
             B("mkdir -p tests/_scratch && python - <<'EOF'\nimport os\nfrom pathlib import Path\nfrom buildcache.builder import Builder\n"
               "p = Path('tests/_scratch/a.src')\np.write_text('x = 1'); os.utime(p, ns=(1_700_000_000_100_000_000,) * 2)\n"
               "b = Builder(str.upper); print(b.build(p))\n"
               "p.write_text('x = 2'); os.utime(p, ns=(1_700_000_000_600_000_000,) * 2)\nprint(b.build(p), 'compiled', b.compiled)\nEOF\nrm -rf tests/_scratch")),
            ("The state is persisted as JSON, so the fingerprint must stay a plain string (a tuple would come back as a list and never match). Check the state file code.",
             V("buildcache/statefile.py")),
            ("Use the nanosecond mtime, keeping the string format.",
             R("buildcache/fingerprint.py",
               '    return f"{int(st.st_mtime)}:{st.st_size}"\n',
               '    return f"{st.st_mtime_ns}:{st.st_size}"\n')),
            ("Re-run the repro.",
             B("mkdir -p tests/_scratch && python - <<'EOF'\nimport os\nfrom pathlib import Path\nfrom buildcache.builder import Builder\n"
               "p = Path('tests/_scratch/a.src')\np.write_text('x = 1'); os.utime(p, ns=(1_700_000_000_100_000_000,) * 2)\n"
               "b = Builder(str.upper); print(b.build(p))\n"
               "p.write_text('x = 2'); os.utime(p, ns=(1_700_000_000_600_000_000,) * 2)\nprint(b.build(p), 'compiled', b.compiled)\nEOF\nrm -rf tests/_scratch")),
            ("Run the tests, including the restart/persistence one.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 85 ---------------------------------------------------------------------------------------
    dict(
        id="seed-085-rollout-hash-randomized",
        bug_type="nondeterministic-hash", difficulty="hard",
        issue="""\
            new_checkout flag flips on and off for the same user

            We rolled `new_checkout` out to 25%. Support has several reports of customers who
            got the new checkout, refreshed, and got the old one. We run 4 worker processes
            behind the load balancer. Allow-listed users are always fine. Unit tests pass and
            the flag looks stable when I call it repeatedly in a REPL.
            """,
        files={
            "flags/__init__.py": "from .registry import FlagRegistry\n",
            "flags/rollout.py": '''\
                def bucket(flag_name, user_id, buckets=100):
                    """Deterministically map (flag, user) to a bucket in [0, buckets)."""
                    return hash(f"{flag_name}:{user_id}") % buckets
                ''',
            "flags/registry.py": '''\
                from .rollout import bucket


                class FlagRegistry:
                    """config: {flag: {"enabled": bool, "rollout": percent, "allow": [user ids]}}"""

                    def __init__(self, config):
                        self.config = config

                    def is_enabled(self, flag, user_id):
                        cfg = self.config.get(flag)
                        if not cfg or not cfg.get("enabled", False):
                            return False
                        if user_id in cfg.get("allow", ()):
                            return True
                        return bucket(flag, user_id) < cfg.get("rollout", 0)
                ''',
            "tests/test_flags.py": '''\
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
                ''',
        },
        hidden_test='''\
            import os
            import subprocess
            import sys
            from pathlib import Path

            from flags import FlagRegistry

            REPO = Path(__file__).resolve().parents[1]
            SNIPPET = "from flags.rollout import bucket; print([bucket('new_checkout', f'user-{i}') for i in range(40)])"


            def _buckets_in_fresh_process(seed):
                env = {**os.environ, "PYTHONHASHSEED": seed}
                r = subprocess.run([sys.executable, "-c", SNIPPET], cwd=REPO, env=env,
                                   capture_output=True, text=True, check=True)
                return r.stdout.strip()


            def test_buckets_identical_across_processes():
                assert _buckets_in_fresh_process("0") == _buckets_in_fresh_process("12345")


            def test_rollout_roughly_matches_percentage():
                reg = FlagRegistry({"f": {"enabled": True, "rollout": 25}})
                on = sum(reg.is_enabled("f", f"user-{i}") for i in range(2000))
                assert 350 < on < 650
            ''',
        p2p=["tests/test_flags.py::test_disabled_flag_off", "tests/test_flags.py::test_full_and_zero_rollout",
             "tests/test_flags.py::test_allow_list", "tests/test_flags.py::test_bucket_stable_and_in_range"],
        steps=[
            ("Read the flag evaluation.", V("flags/registry.py")),
            ("Allow-listed users are fine, so look at the percentage bucketing.", V("flags/rollout.py")),
            ("Built-in hash() of str is salted per process (PYTHONHASHSEED), so each worker buckets users differently. Compare a few fresh processes.",
             B("for s in 1 2 3; do PYTHONHASHSEED=$s python -c 'from flags.rollout import bucket; print([bucket(\"new_checkout\", f\"user-{i}\") for i in range(8)])'; done")),
            ("Use a stable digest instead. Add the import.", I("flags/rollout.py", 0, "import hashlib\n\n\n")),
            ("Bucket from the first 8 bytes of a SHA-256 of the key.",
             R("flags/rollout.py",
               '    return hash(f"{flag_name}:{user_id}") % buckets\n',
               '    digest = hashlib.sha256(f"{flag_name}:{user_id}".encode()).digest()\n'
               '    return int.from_bytes(digest[:8], "big") % buckets\n')),
            ("Re-run across seeds; the lists should now be identical.",
             B("for s in 1 2 3; do PYTHONHASHSEED=$s python -c 'from flags.rollout import bucket; print([bucket(\"new_checkout\", f\"user-{i}\") for i in range(8)])'; done")),
            ("Sanity-check the share of users enabled at 25%.",
             B("python -c 'from flags import FlagRegistry; r=FlagRegistry({\"f\": {\"enabled\": True, \"rollout\": 25}}); print(sum(r.is_enabled(\"f\", f\"user-{i}\") for i in range(2000)))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
]
