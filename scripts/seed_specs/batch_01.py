"""Seed instances 011-025: string/text processing & parsing."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V, d

SPECS = [
    # 11 ---------------------------------------------------------------------------------------
    dict(
        id="seed-011-csv-doubled-quotes",
        bug_type="escaping", difficulty="medium",
        issue="""\
            Contact import drops quote characters inside quoted fields

            We export contacts from the CRM as CSV, and names with nicknames come back mangled
            after `load_contacts`. This row:

                3,"Bob ""The Builder"" Smith",bob@example.com

            is imported with name `Bob The Builder Smith`; it should be `Bob "The Builder" Smith`
            (doubled quotes inside a quoted field are the standard CSV escape for a literal quote).
            Commas inside quotes work fine.
            """,
        files={
            "contacts/__init__.py": "from .importer import load_contacts\n",
            "contacts/csvline.py": '''\
                def split_line(line, sep=","):
                    """Split one CSV record into fields.

                    Fields may be wrapped in double quotes so they can contain `sep`; a literal
                    quote inside a quoted field is written as two quotes ("").
                    """
                    fields, buf, in_quotes = [], [], False
                    i = 0
                    while i < len(line):
                        ch = line[i]
                        if ch == '"':
                            in_quotes = not in_quotes
                        elif ch == sep and not in_quotes:
                            fields.append("".join(buf))
                            buf = []
                        else:
                            buf.append(ch)
                        i += 1
                    fields.append("".join(buf))
                    return fields
                ''',
            "contacts/importer.py": '''\
                from .csvline import split_line


                def load_contacts(text):
                    """Parse CSV text with a header row into a list of dicts."""
                    lines = [l.rstrip("\\r") for l in text.splitlines() if l.strip()]
                    if not lines:
                        return []
                    header = [h.strip().lower() for h in split_line(lines[0])]
                    rows = []
                    for n, line in enumerate(lines[1:], start=2):
                        values = split_line(line)
                        if len(values) != len(header):
                            raise ValueError(f"line {n}: expected {len(header)} fields, got {len(values)}")
                        rows.append(dict(zip(header, values)))
                    return rows
                ''',
            "tests/test_csvline.py": '''\
                import pytest

                from contacts import load_contacts
                from contacts.csvline import split_line


                def test_plain_fields():
                    assert split_line("a,b,,c") == ["a", "b", "", "c"]


                def test_quoted_separator():
                    assert split_line('1,"Smith, John",x') == ["1", "Smith, John", "x"]


                def test_load_contacts_header():
                    rows = load_contacts("ID,Name\\n1,Ann\\n2,Ben\\n")
                    assert rows == [{"id": "1", "name": "Ann"}, {"id": "2", "name": "Ben"}]


                def test_field_count_mismatch():
                    with pytest.raises(ValueError, match="line 2"):
                        load_contacts("id,name\\n1,Ann,extra\\n")
                ''',
        },
        hidden_test='''\
            from contacts import load_contacts
            from contacts.csvline import split_line


            def test_doubled_quote_is_literal():
                assert split_line('3,"Bob ""The Builder"" Smith",b@x.io') == ["3", 'Bob "The Builder" Smith', "b@x.io"]
                assert split_line('""""') == ['"']


            def test_empty_quoted_field():
                assert split_line('a,"",b') == ["a", "", "b"]


            def test_import_keeps_quotes():
                rows = load_contacts('id,name\\n1,"Say ""hi"", Ann"\\n')
                assert rows == [{"id": "1", "name": 'Say "hi", Ann'}]
            ''',
        p2p=["tests/test_csvline.py::test_plain_fields", "tests/test_csvline.py::test_quoted_separator",
             "tests/test_csvline.py::test_load_contacts_header", "tests/test_csvline.py::test_field_count_mismatch"],
        steps=[
            ("See what the package contains.", F("contacts/*.py")),
            ("load_contacts delegates field splitting to split_line; read it.", V("contacts/csvline.py")),
            ("Reproduce with the row from the report.",
             B(d('''\
                 python - <<'EOF'
                 from contacts.csvline import split_line
                 print(split_line('3,"Bob ""The Builder"" Smith",bob@example.com'))
                 EOF'''))),
            ("Every quote just toggles in_quotes, so a doubled quote toggles out and back in and is "
             "dropped. Inside a quoted field, `\"\"` must emit one literal quote and skip the second.",
             R("contacts/csvline.py",
               "        if ch == '\"':\n            in_quotes = not in_quotes\n",
               "        if ch == '\"':\n"
               "            if in_quotes and line[i + 1:i + 2] == '\"':\n"
               "                buf.append('\"')\n"
               "                i += 1\n"
               "            else:\n"
               "                in_quotes = not in_quotes\n")),
            ("Re-run the repro plus an empty quoted field and a field that is a single literal quote.",
             B(d('''\
                 python - <<'EOF'
                 from contacts.csvline import split_line
                 print(split_line('3,"Bob ""The Builder"" Smith",bob@example.com'))
                 print(split_line('a,"",b'), split_line('""""'))
                 EOF'''))),
            ("Run the test suite.", T()),
            ("Doubled quotes are now decoded correctly.", SUBMIT),
        ],
    ),
    # 12 ---------------------------------------------------------------------------------------
    dict(
        id="seed-012-querystring-plus-decoding",
        bug_type="decoding-order", difficulty="easy",
        issue="""\
            Searching for "C++" searches for "C"

            Typing `C++` in the search box sends `/search?q=C%2B%2B` (checked in the network tab),
            but the backend searches for `c` only. Looks like the encoded plus signs are being turned
            into spaces somewhere in our query-string handling. Same for `a+b=c` style queries.
            """,
        files={
            "webkit/__init__.py": "",
            "webkit/querystring.py": '''\
                from urllib.parse import unquote


                def parse_qs(qs):
                    """Parse an application/x-www-form-urlencoded query string into {key: [values]}."""
                    if qs.startswith("?"):
                        qs = qs[1:]
                    out = {}
                    for pair in qs.split("&"):
                        if not pair:
                            continue
                        key, _, value = pair.partition("=")
                        key = unquote(key).replace("+", " ")
                        value = unquote(value).replace("+", " ")
                        out.setdefault(key, []).append(value)
                    return out


                def get_first(qs, key, default=None):
                    return parse_qs(qs).get(key, [default])[0]
                ''',
            "webkit/search.py": '''\
                from .querystring import get_first


                def search_terms(url):
                    """Lower-cased search terms from the `q` parameter of a request URL."""
                    _, _, qs = url.partition("?")
                    return get_first(qs, "q", "").lower().split()
                ''',
            "tests/test_querystring.py": '''\
                from webkit.querystring import get_first, parse_qs
                from webkit.search import search_terms


                def test_repeated_keys():
                    assert parse_qs("?tag=a&tag=b&x=1") == {"tag": ["a", "b"], "x": ["1"]}


                def test_percent_escapes():
                    assert get_first("name=J%C3%BCrgen", "name") == "J\\u00fcrgen"


                def test_search_terms_split_on_plus():
                    assert search_terms("/search?q=Hello+World") == ["hello", "world"]
                ''',
        },
        hidden_test='''\
            from webkit.querystring import parse_qs
            from webkit.search import search_terms


            def test_encoded_plus_is_literal():
                assert parse_qs("q=C%2B%2B") == {"q": ["C++"]}
                assert parse_qs("expr=a%2Bb%3Dc") == {"expr": ["a+b=c"]}


            def test_search_cpp():
                assert search_terms("/search?q=C%2B%2B+tutorial") == ["c++", "tutorial"]


            def test_plus_still_means_space():
                assert parse_qs("a+b=c+d") == {"a b": ["c d"]}
            ''',
        p2p=["tests/test_querystring.py::test_repeated_keys", "tests/test_querystring.py::test_percent_escapes",
             "tests/test_querystring.py::test_search_terms_split_on_plus"],
        steps=[
            ("Find where query strings get decoded.", S("unquote|replace\\(\"\\+\"", "webkit")),
            ("Read the parser.", V("webkit/querystring.py")),
            ("Reproduce with the encoded query from the report.",
             B("python -c 'from webkit.querystring import parse_qs; print(parse_qs(\"q=C%2B%2B\"))'")),
            ("Percent-decoding runs first, so `%2B` becomes `+` and is then turned into a space. "
             "`+` -> space has to happen before percent-decoding; unquote_plus does exactly that.",
             R("webkit/querystring.py",
               "        key = unquote(key).replace(\"+\", \" \")\n        value = unquote(value).replace(\"+\", \" \")\n",
               "        key = unquote_plus(key)\n        value = unquote_plus(value)\n")),
            ("Update the import accordingly.",
             R("webkit/querystring.py", "from urllib.parse import unquote\n", "from urllib.parse import unquote_plus\n")),
            ("Verify encoded plus, literal plus-as-space, and the search helper.",
             B("python -c 'from webkit.querystring import parse_qs; from webkit.search import search_terms; "
               "print(parse_qs(\"q=C%2B%2B\"), parse_qs(\"a+b=c+d\"), search_terms(\"/s?q=C%2B%2B+tutorial\"))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 13 ---------------------------------------------------------------------------------------
    dict(
        id="seed-013-wrap-leading-long-word",
        bug_type="edge-case", difficulty="easy",
        issue="""\
            fill() emits a blank first line when a paragraph starts with a long URL

                >>> print(fill("https://example.com/docs/getting-started is the guide", 20, indent="> "))
                >
                > https://example.com/docs/getting-started
                > is the guide

            The first line is just the indent. Long words in the middle of a paragraph are fine.
            """,
        files={
            "textfmt/__init__.py": "from .fill import fill\nfrom .wrap import wrap\n",
            "textfmt/wrap.py": '''\
                def wrap(text, width):
                    """Greedy word wrap into lines of at most `width` characters.

                    Words longer than `width` are never split; they go on a line of their own.
                    """
                    if width < 1:
                        raise ValueError("width must be >= 1")
                    lines, current = [], ""
                    for word in text.split():
                        needed = len(word) + (1 if current else 0)
                        if len(current) + needed > width:
                            lines.append(current)
                            current = word
                        else:
                            current = f"{current} {word}" if current else word
                    if current:
                        lines.append(current)
                    return lines
                ''',
            "textfmt/fill.py": '''\
                from .wrap import wrap


                def fill(text, width=72, indent=""):
                    """Wrap text to `width` columns, prefixing every line with `indent`."""
                    inner = width - len(indent)
                    if inner < 1:
                        raise ValueError("indent is wider than width")
                    return "\\n".join(indent + line for line in wrap(text, inner))
                ''',
            "tests/test_wrap.py": '''\
                import pytest

                from textfmt import fill, wrap


                def test_basic_wrap():
                    assert wrap("the quick brown fox", 10) == ["the quick", "brown fox"]


                def test_long_word_in_middle():
                    assert wrap("a bbbbbbbbbbbb c", 5) == ["a", "bbbbbbbbbbbb", "c"]


                def test_empty():
                    assert wrap("   ", 10) == []


                def test_fill_indent():
                    assert fill("one two three", 9, indent="# ") == "# one two\\n# three"
                    with pytest.raises(ValueError):
                        fill("x", 2, indent="-->")
                ''',
        },
        hidden_test='''\
            from textfmt import fill, wrap


            def test_leading_long_word_no_blank_line():
                assert wrap("supercalifragilistic is long", 10) == ["supercalifragilistic", "is long"]
                assert wrap("abcdefghijk", 5) == ["abcdefghijk"]


            def test_fill_leading_url():
                out = fill("https://example.com/docs/getting-started is the guide", 20, indent="> ")
                assert out.splitlines() == ["> https://example.com/docs/getting-started", "> is the guide"]


            def test_exact_width_fits():
                assert wrap("abcde fghij", 5) == ["abcde", "fghij"]
                assert wrap("ab cd", 5) == ["ab cd"]
            ''',
        p2p=["tests/test_wrap.py::test_basic_wrap", "tests/test_wrap.py::test_long_word_in_middle",
             "tests/test_wrap.py::test_empty", "tests/test_wrap.py::test_fill_indent"],
        steps=[
            ("fill just indents the output of wrap; look at wrap.", V("textfmt/wrap.py")),
            ("Reproduce at the wrap level.",
             B("python -c 'from textfmt import wrap; print(wrap(\"supercalifragilistic is long\", 10))'")),
            ("With an empty `current` the first word can still exceed width, which flushes an empty line. "
             "Only flush when there is something on the current line.",
             R("textfmt/wrap.py", "        if len(current) + needed > width:",
               "        if current and len(current) + needed > width:")),
            ("Re-run, plus the fill example from the report.",
             B("python -c 'from textfmt import wrap, fill; print(wrap(\"supercalifragilistic is long\", 10)); "
               "print(fill(\"https://example.com/docs/getting-started is the guide\", 20, indent=\"> \"))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 14 ---------------------------------------------------------------------------------------
    dict(
        id="seed-014-camel-to-snake-acronyms",
        bug_type="regex-logic", difficulty="medium",
        issue="""\
            KeyError: 'user_id' when loading users from the v2 API

            ```
            Traceback (most recent call last):
              File "sync.py", line 12, in <module>
                users = [User.from_api(u) for u in payload["users"]]
              File "apiclient/models.py", line 15, in from_api
                return cls(user_id=data["user_id"], display_name=data["display_name"],
            KeyError: 'user_id'
            ```
            The v2 API returns keys like `userID`, `displayName` and `avatarURL`. `displayName`
            works, the ones ending in capitals don't.
            """,
        files={
            "apiclient/__init__.py": "",
            "apiclient/casing.py": '''\
                import re

                _BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


                def camel_to_snake(name):
                    """userName -> user_name"""
                    return _BOUNDARY.sub("_", name).lower()


                def snake_to_camel(name):
                    head, *rest = name.split("_")
                    return head + "".join(part.title() for part in rest)
                ''',
            "apiclient/payload.py": '''\
                from .casing import camel_to_snake


                def normalize_keys(obj):
                    """Recursively convert camelCase dict keys from the API to snake_case."""
                    if isinstance(obj, dict):
                        return {camel_to_snake(k): normalize_keys(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [normalize_keys(v) for v in obj]
                    return obj
                ''',
            "apiclient/models.py": '''\
                from dataclasses import dataclass

                from .payload import normalize_keys


                @dataclass
                class User:
                    user_id: int
                    display_name: str
                    avatar_url: str | None = None

                    @classmethod
                    def from_api(cls, raw):
                        data = normalize_keys(raw)
                        return cls(user_id=data["user_id"], display_name=data["display_name"],
                                   avatar_url=data.get("avatar_url"))
                ''',
            "tests/test_casing.py": '''\
                from apiclient.casing import camel_to_snake, snake_to_camel
                from apiclient.payload import normalize_keys


                def test_simple_camel():
                    assert camel_to_snake("displayName") == "display_name"
                    assert camel_to_snake("Simple") == "simple"
                    assert camel_to_snake("already_snake") == "already_snake"


                def test_snake_to_camel():
                    assert snake_to_camel("display_name") == "displayName"


                def test_normalize_nested():
                    raw = {"pageInfo": {"nextCursor": "x"}, "items": [{"itemName": "a"}]}
                    assert normalize_keys(raw) == {"page_info": {"next_cursor": "x"}, "items": [{"item_name": "a"}]}
                ''',
        },
        hidden_test='''\
            from apiclient.casing import camel_to_snake
            from apiclient.models import User


            def test_trailing_acronym():
                assert camel_to_snake("userID") == "user_id"
                assert camel_to_snake("avatarURL") == "avatar_url"


            def test_acronym_in_middle_and_start():
                assert camel_to_snake("getHTTPResponse") == "get_http_response"
                assert camel_to_snake("HTTPServerError") == "http_server_error"


            def test_digits_and_simple_words():
                assert camel_to_snake("version2Beta") == "version2_beta"
                assert camel_to_snake("displayName") == "display_name"


            def test_user_from_api():
                u = User.from_api({"userID": 7, "displayName": "Ann", "avatarURL": "https://x/a.png"})
                assert u == User(7, "Ann", "https://x/a.png")
            ''',
        p2p=["tests/test_casing.py::test_simple_camel", "tests/test_casing.py::test_snake_to_camel",
             "tests/test_casing.py::test_normalize_nested"],
        steps=[
            ("Start where the KeyError is raised.", V("apiclient/models.py")),
            ("Keys come from normalize_keys; find the key conversion it uses.", S("def (normalize_keys|camel_to_snake)", "apiclient")),
            ("Check what the v2 keys turn into.",
             B("python -c 'from apiclient.payload import normalize_keys; print(normalize_keys({\"userID\": 7, \"displayName\": \"Ann\", \"avatarURL\": \"u\"}))'")),
            ("Read the converter.", V("apiclient/casing.py")),
            ("The regex inserts `_` before every capital, so acronyms become `i_d`/`u_r_l`. Split only at "
             "lower/digit->Upper boundaries and at the end of an acronym that is followed by a word.",
             R("apiclient/casing.py",
               '_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")\n\n\ndef camel_to_snake(name):\n    """userName -> user_name"""\n    return _BOUNDARY.sub("_", name).lower()\n',
               '_ACRONYM_END = re.compile(r"([A-Z]+)([A-Z][a-z])")\n_WORD_START = re.compile(r"([a-z0-9])([A-Z])")\n\n\n'
               'def camel_to_snake(name):\n    """userName -> user_name, userID -> user_id, HTTPServer -> http_server"""\n'
               '    name = _ACRONYM_END.sub(r"\\1_\\2", name)\n    return _WORD_START.sub(r"\\1_\\2", name).lower()\n')),
            ("Check a spread of inputs, including acronyms at the start/middle/end and digits.",
             B("python -c 'from apiclient.casing import camel_to_snake as c; "
               "print([c(x) for x in (\"userID\", \"avatarURL\", \"getHTTPResponse\", \"HTTPServerError\", \"version2Beta\", \"Simple\", \"already_snake\")])'")),
            ("Re-run the failing path end to end.",
             B("python -c 'from apiclient.models import User; print(User.from_api({\"userID\": 7, \"displayName\": \"Ann\", \"avatarURL\": \"u\"}))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 15 ---------------------------------------------------------------------------------------
    dict(
        id="seed-015-html-escape-double-encoding",
        bug_type="escaping-order", difficulty="medium",
        issue="""\
            Rendered comments show literal "&lt;" entities

            A user posted a comment containing `<script>` and the page displays
            `&amp;lt;script&amp;gt;` instead of `<script>` as text. Plain ampersands
            (e.g. "Tom & Jerry") look fine. We render comments with `render("{{ body }}", ...)`.
            """,
        files={
            "minitpl/__init__.py": "from .template import render\n",
            "minitpl/escape.py": '''\
                _REPLACEMENTS = [
                    ("<", "&lt;"),
                    (">", "&gt;"),
                    ('"', "&quot;"),
                    ("'", "&#x27;"),
                    ("&", "&amp;"),
                ]


                def escape_html(value):
                    """Escape a value for safe inclusion in HTML text or attribute values."""
                    s = str(value)
                    for old, new in _REPLACEMENTS:
                        s = s.replace(old, new)
                    return s
                ''',
            "minitpl/template.py": '''\
                import re

                from .escape import escape_html

                _VAR = re.compile(r"\\{\\{\\s*(\\w+)(\\|safe)?\\s*\\}\\}")


                def render(template, context):
                    """Substitute {{ name }} (HTML-escaped) and {{ name|safe }} (raw)."""
                    def sub(m):
                        name, safe = m.group(1), m.group(2)
                        if name not in context:
                            raise KeyError(f"template variable {name!r} is not defined")
                        value = context[name]
                        return str(value) if safe else escape_html(value)
                    return _VAR.sub(sub, template)
                ''',
            "tests/test_template.py": '''\
                import pytest

                from minitpl import render
                from minitpl.escape import escape_html


                def test_plain_substitution():
                    assert render("Hi {{ name }}!", {"name": "Ann"}) == "Hi Ann!"


                def test_ampersand():
                    assert escape_html("Tom & Jerry") == "Tom &amp; Jerry"


                def test_safe_filter_is_raw():
                    assert render("{{ html|safe }}", {"html": "<b>x</b>"}) == "<b>x</b>"


                def test_missing_variable():
                    with pytest.raises(KeyError):
                        render("{{ nope }}", {})
                ''',
        },
        hidden_test='''\
            from minitpl import render
            from minitpl.escape import escape_html


            def test_tags_escaped_once():
                assert escape_html("<script>") == "&lt;script&gt;"
                assert escape_html('a "b" \\'c\\'') == "a &quot;b&quot; &#x27;c&#x27;"


            def test_render_comment():
                out = render("<p>{{ body }}</p>", {"body": "<script>alert(1)</script> & more"})
                assert out == "<p>&lt;script&gt;alert(1)&lt;/script&gt; &amp; more</p>"


            def test_existing_entity_text_escaped():
                assert escape_html("&lt;") == "&amp;lt;"
            ''',
        p2p=["tests/test_template.py::test_plain_substitution", "tests/test_template.py::test_ampersand",
             "tests/test_template.py::test_safe_filter_is_raw", "tests/test_template.py::test_missing_variable"],
        steps=[
            ("Look at how render escapes values.", V("minitpl/template.py")),
            ("Escaping is delegated; find escape_html.", S("def escape_html", ".", "*.py")),
            ("Reproduce with the comment from the report.",
             B("python -c 'from minitpl import render; print(render(\"{{ body }}\", {\"body\": \"<script>\"}))'")),
            ("Read the escaper.", V("minitpl/escape.py")),
            ("`&` is replaced last, so the `&` in the freshly inserted `&lt;`/`&gt;` gets escaped again. "
             "Ampersands must be escaped first.",
             R("minitpl/escape.py",
               '_REPLACEMENTS = [\n    ("<", "&lt;"),\n',
               '_REPLACEMENTS = [\n    ("&", "&amp;"),  # must come first, the other entities contain "&"\n    ("<", "&lt;"),\n')),
            ("Remove the old trailing `&` entry.",
             R("minitpl/escape.py", '    ("\'", "&#x27;"),\n    ("&", "&amp;"),\n]', '    ("\'", "&#x27;"),\n]')),
            ("Verify tags, ampersands and quotes.",
             B("python -c 'from minitpl import render; print(render(\"<p>{{ body }}</p>\", {\"body\": \"<script> & \\\"q\\\"\"}))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 16 ---------------------------------------------------------------------------------------
    dict(
        id="seed-016-ini-inline-comment-hash",
        bug_type="parsing-edge-case", difficulty="medium",
        issue="""\
            Theme colours from settings.ini fail to load

            With this in settings.ini:

                [theme]
                accent = #ff8800
                link   = #0066cc ; brand blue

            `Settings(text).color("accent")` raises `ValueError: invalid colour ''`. The docs say
            inline comments need whitespace before the `#`/`;`, so a value that *starts* with `#`
            should be fine. The `homepage = https://example.com/#/start` setting is also cut short.
            """,
        files={
            "iniparse/__init__.py": "from .parser import ParseError, parse_ini\n",
            "iniparse/parser.py": '''\
                class ParseError(ValueError):
                    pass


                def _strip_comment(line):
                    """Drop a trailing inline comment.

                    Inside a value, an inline comment starts at a '#' or ';' preceded by whitespace,
                    so values such as `#ff8800` or `a;b` are kept intact.
                    """
                    for marker in ("#", ";"):
                        idx = line.find(marker)
                        if idx != -1:
                            line = line[:idx]
                    return line


                def parse_ini(text):
                    data, section = {}, None
                    for lineno, raw in enumerate(text.splitlines(), 1):
                        line = _strip_comment(raw).strip()
                        if not line:
                            continue
                        if line.startswith("[") and line.endswith("]"):
                            section = line[1:-1].strip()
                            data.setdefault(section, {})
                            continue
                        if "=" not in line:
                            raise ParseError(f"line {lineno}: expected 'key = value'")
                        if section is None:
                            raise ParseError(f"line {lineno}: key outside of a section")
                        key, _, value = line.partition("=")
                        data[section][key.strip().lower()] = value.strip()
                    return data
                ''',
            "iniparse/settings.py": '''\
                import re

                from .parser import parse_ini

                _HEX = re.compile(r"#[0-9a-fA-F]{6}")


                class Settings:
                    def __init__(self, text):
                        self._data = parse_ini(text)

                    def get(self, key, section="general", default=None):
                        return self._data.get(section, {}).get(key, default)

                    def color(self, key):
                        value = self.get(key, section="theme", default="")
                        if not _HEX.fullmatch(value):
                            raise ValueError(f"invalid colour {value!r}")
                        return value
                ''',
            "tests/test_ini.py": '''\
                import pytest

                from iniparse import ParseError, parse_ini
                from iniparse.settings import Settings


                def test_sections_and_keys():
                    data = parse_ini("[general]\\nName = demo\\n[db]\\nport=5432\\n")
                    assert data == {"general": {"name": "demo"}, "db": {"port": "5432"}}


                def test_comment_lines_ignored():
                    assert parse_ini("# top\\n[a]\\n; note\\nx = 1\\n") == {"a": {"x": "1"}}


                def test_inline_comment_after_value():
                    assert parse_ini("[a]\\nx = 1  # one\\n") == {"a": {"x": "1"}}


                def test_errors():
                    with pytest.raises(ParseError, match="line 1"):
                        parse_ini("x = 1\\n")
                    with pytest.raises(ParseError, match="line 2"):
                        parse_ini("[a]\\njunk\\n")


                def test_settings_get_default():
                    assert Settings("[general]\\n").get("missing", default=3) == 3
                ''',
        },
        hidden_test='''\
            from iniparse import parse_ini
            from iniparse.settings import Settings


            def test_value_starting_with_hash():
                text = "[theme]\\naccent = #ff8800\\nlink   = #0066cc ; brand blue\\n"
                s = Settings(text)
                assert s.color("accent") == "#ff8800"
                assert s.color("link") == "#0066cc"


            def test_markers_inside_values_kept():
                data = parse_ini("[general]\\nhomepage = https://example.com/#/start\\nquery = a;b\\n")
                assert data["general"] == {"homepage": "https://example.com/#/start", "query": "a;b"}


            def test_indented_comment_line():
                assert parse_ini("[a]\\n   # indented note\\nx = 1 ; c\\n") == {"a": {"x": "1"}}
            ''',
        p2p=["tests/test_ini.py::test_sections_and_keys", "tests/test_ini.py::test_comment_lines_ignored",
             "tests/test_ini.py::test_inline_comment_after_value", "tests/test_ini.py::test_errors",
             "tests/test_ini.py::test_settings_get_default"],
        steps=[
            ("Find where the colour error comes from and where values are parsed.", S("invalid colour|def parse_ini|def _strip_comment", "iniparse")),
            ("Reproduce at the parser level.",
             B("python -c 'from iniparse import parse_ini; print(parse_ini(\"[theme]\\naccent = #ff8800\\nhome = https://x/#/s\\n\"))'")),
            ("Read the parser and its comment stripping.", V("iniparse/parser.py")),
            ("It cuts the raw line at the first '#' or ';' anywhere, contrary to its docstring. And since the "
             "rule is 'whitespace before the marker inside a value', it can't run on the whole line ('= #ff8800' "
             "has a space before '#'). Make _strip_comment work on a value, skipping a marker at position 0.",
             R("iniparse/parser.py",
               '    for marker in ("#", ";"):\n        idx = line.find(marker)\n        if idx != -1:\n            line = line[:idx]\n    return line\n',
               '    for i in range(1, len(line)):\n        if line[i] in "#;" and line[i - 1].isspace():\n            return line[:i]\n'
               '    return line\n')),
            ("Handle full-line comments in parse_ini directly instead of stripping the raw line.",
             R("iniparse/parser.py",
               "        line = _strip_comment(raw).strip()\n        if not line:\n",
               "        line = raw.strip()\n        if not line or line[0] in \"#;\":\n")),
            ("Strip inline comments from the value only, after the key/value split.",
             R("iniparse/parser.py",
               "        data[section][key.strip().lower()] = value.strip()",
               "        data[section][key.strip().lower()] = _strip_comment(value.strip()).strip()")),
            ("Re-run the repro, plus the inline-comment case from the report.",
             B("python -c 'from iniparse import parse_ini; from iniparse.settings import Settings; "
               "print(parse_ini(\"[theme]\\naccent = #ff8800\\nhome = https://x/#/s\\n\")); "
               "print(Settings(\"[theme]\\nlink   = #0066cc ; brand blue\\n\").color(\"link\"))'")),
            ("Run tests.", T("tests")),
            ("Done.", SUBMIT),
        ],
    ),
    # 17 ---------------------------------------------------------------------------------------
    dict(
        id="seed-017-calc-subtraction-associativity",
        bug_type="associativity", difficulty="hard",
        issue="""\
            Formula results are wrong for chained subtraction

            In the formula cell `=100 - 20 - 5` the result shown is 85; it should be 75.
            `1 - 2 + 3` also gives -4 instead of 2. Multiplication/division chains and
            parentheses look right. These go through `calc.calculate`.
            """,
        files={
            "calc/__init__.py": '''\
                from .evaluator import evaluate
                from .parser import Parser
                from .tokenizer import tokenize


                def calculate(text):
                    return evaluate(Parser(tokenize(text)).parse())
                ''',
            "calc/tokenizer.py": '''\
                import re
                from dataclasses import dataclass


                @dataclass(frozen=True)
                class Token:
                    kind: str
                    value: str


                _TOKEN = re.compile(r"\\s*(?:(?P<NUM>\\d+(?:\\.\\d+)?)|(?P<OP>[-+*/])|(?P<LPAREN>\\()|(?P<RPAREN>\\)))")


                def tokenize(text):
                    tokens, pos = [], 0
                    text = text.rstrip()
                    while pos < len(text):
                        m = _TOKEN.match(text, pos)
                        if not m:
                            raise SyntaxError(f"unexpected character {text[pos]!r} at {pos}")
                        kind = m.lastgroup
                        tokens.append(Token(kind, m.group(kind)))
                        pos = m.end()
                    return tokens
                ''',
            "calc/parser.py": '''\
                """Recursive-descent parser producing tuple ASTs.

                    expr   := term (('+' | '-') term)*
                    term   := factor (('*' | '/') factor)*
                    factor := NUM | '-' factor | '(' expr ')'
                """


                class Parser:
                    def __init__(self, tokens):
                        self.tokens = tokens
                        self.pos = 0

                    def peek(self):
                        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

                    def advance(self):
                        tok = self.peek()
                        self.pos += 1
                        return tok

                    def parse(self):
                        node = self.expr()
                        if self.peek() is not None:
                            raise SyntaxError(f"unexpected token {self.peek().value!r}")
                        return node

                    def expr(self):
                        left = self.term()
                        tok = self.peek()
                        if tok is not None and tok.kind == "OP" and tok.value in ("+", "-"):
                            self.advance()
                            return (tok.value, left, self.expr())
                        return left

                    def term(self):
                        left = self.factor()
                        while (tok := self.peek()) is not None and tok.kind == "OP" and tok.value in ("*", "/"):
                            self.advance()
                            left = (tok.value, left, self.factor())
                        return left

                    def factor(self):
                        tok = self.advance()
                        if tok is None:
                            raise SyntaxError("unexpected end of expression")
                        if tok.kind == "NUM":
                            return ("num", float(tok.value) if "." in tok.value else int(tok.value))
                        if tok.kind == "OP" and tok.value == "-":
                            return ("neg", self.factor())
                        if tok.kind == "LPAREN":
                            node = self.expr()
                            closing = self.advance()
                            if closing is None or closing.kind != "RPAREN":
                                raise SyntaxError("expected ')'")
                            return node
                        raise SyntaxError(f"unexpected token {tok.value!r}")
                ''',
            "calc/evaluator.py": '''\
                import operator

                _BINARY = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


                def evaluate(node):
                    kind = node[0]
                    if kind == "num":
                        return node[1]
                    if kind == "neg":
                        return -evaluate(node[1])
                    left, right = evaluate(node[1]), evaluate(node[2])
                    if kind == "/" and right == 0:
                        raise ZeroDivisionError("division by zero in expression")
                    return _BINARY[kind](left, right)
                ''',
            "tests/test_calc.py": '''\
                import pytest

                from calc import calculate


                def test_precedence():
                    assert calculate("2 + 3 * 4") == 14


                def test_parentheses_and_unary():
                    assert calculate("(2 + 3) * -4") == -20
                    assert calculate("-(1.5 + 0.5)") == -2.0


                def test_division_chain():
                    assert calculate("8 / 4 / 2") == 1


                def test_errors():
                    with pytest.raises(ZeroDivisionError):
                        calculate("1 / (2 - 2)")
                    with pytest.raises(SyntaxError):
                        calculate("2 +")
                    with pytest.raises(SyntaxError):
                        calculate("2 $ 3")
                ''',
        },
        hidden_test='''\
            from calc import calculate
            from calc.parser import Parser
            from calc.tokenizer import tokenize


            def test_chained_subtraction():
                assert calculate("100 - 20 - 5") == 75
                assert calculate("10 - 4 - 3") == 3


            def test_mixed_additive_left_to_right():
                assert calculate("1 - 2 + 3") == 2
                assert calculate("2 * 3 - 4 - 1") == 1


            def test_ast_is_left_associative():
                assert Parser(tokenize("1 - 2 - 3")).parse() == ("-", ("-", ("num", 1), ("num", 2)), ("num", 3))


            def test_precedence_still_holds():
                assert calculate("2 + 3 * 4 - 1") == 13
                assert calculate("-(2 - 5) - 1") == 2
            ''',
        p2p=["tests/test_calc.py::test_precedence", "tests/test_calc.py::test_parentheses_and_unary",
             "tests/test_calc.py::test_division_chain", "tests/test_calc.py::test_errors"],
        steps=[
            ("Get an overview of the calculator modules.", F("calc/*.py")),
            ("Reproduce both examples from the report and compare with a multiplicative chain.",
             B("python -c 'from calc import calculate; print(calculate(\"100 - 20 - 5\"), calculate(\"1 - 2 + 3\"), calculate(\"8 / 4 / 2\"))'")),
            ("Check whether the evaluator or the parse tree is at fault.",
             B("python -c 'from calc.parser import Parser; from calc.tokenizer import tokenize; print(Parser(tokenize(\"100 - 20 - 5\")).parse())'")),
            ("The tree is `100 - (20 - 5)`, so the evaluator is fine and the parser groups to the right. Read it.",
             V("calc/parser.py", 27, 40)),
            ("expr recurses into itself for the right operand, making + and - right-associative; term uses a "
             "loop and is correct. Rewrite expr as a loop, matching the grammar in the module docstring.",
             R("calc/parser.py",
               "        left = self.term()\n        tok = self.peek()\n"
               "        if tok is not None and tok.kind == \"OP\" and tok.value in (\"+\", \"-\"):\n"
               "            self.advance()\n            return (tok.value, left, self.expr())\n        return left\n",
               "        left = self.term()\n"
               "        while (tok := self.peek()) is not None and tok.kind == \"OP\" and tok.value in (\"+\", \"-\"):\n"
               "            self.advance()\n            left = (tok.value, left, self.term())\n        return left\n")),
            ("Re-check the tree and the results.",
             B("python -c 'from calc import calculate; from calc.parser import Parser; from calc.tokenizer import tokenize; "
               "print(Parser(tokenize(\"100 - 20 - 5\")).parse()); print(calculate(\"100 - 20 - 5\"), calculate(\"1 - 2 + 3\"), calculate(\"2 + 3 * 4 - 1\"))'")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 18 ---------------------------------------------------------------------------------------
    dict(
        id="seed-018-username-unicode-normalization",
        bug_type="unicode-normalization", difficulty="medium",
        issue="""\
            Duplicate usernames that look identical

            Support found two accounts both named "José". One signed up from the iOS app, the
            other from the web. In the DB the names are `'Jos\\u00e9'` and `'Jose\\u0301'`
            (precomposed é vs e + combining accent). `Registry.register` is supposed to reject
            names that are already taken (case-insensitively), but it let the second one through.
            """,
        files={
            "accounts/__init__.py": "from .registry import Registry, UsernameTaken\n",
            "accounts/normalize.py": '''\
                def canonical_username(name):
                    """Canonical form used for uniqueness checks: trimmed and case-insensitive."""
                    return name.strip().lower()
                ''',
            "accounts/registry.py": '''\
                from .normalize import canonical_username


                class UsernameTaken(ValueError):
                    pass


                class Registry:
                    def __init__(self):
                        self._users = {}

                    def register(self, name):
                        key = canonical_username(name)
                        if not key:
                            raise ValueError("username must not be blank")
                        if key in self._users:
                            raise UsernameTaken(name)
                        self._users[key] = name
                        return key

                    def is_taken(self, name):
                        return canonical_username(name) in self._users
                ''',
            "tests/test_registry.py": '''\
                import pytest

                from accounts import Registry, UsernameTaken


                def test_case_insensitive():
                    r = Registry()
                    r.register("Alice")
                    with pytest.raises(UsernameTaken):
                        r.register("  alice ")


                def test_blank_rejected():
                    with pytest.raises(ValueError):
                        Registry().register("   ")


                def test_is_taken():
                    r = Registry()
                    r.register("bob")
                    assert r.is_taken("BOB") and not r.is_taken("rob")
                ''',
        },
        hidden_test='''\
            import pytest

            from accounts import Registry, UsernameTaken
            from accounts.normalize import canonical_username


            def test_composed_and_decomposed_collide():
                r = Registry()
                r.register("Jos\\u00e9")
                with pytest.raises(UsernameTaken):
                    r.register("Jose\\u0301")
                assert r.is_taken("JOSE\\u0301")


            def test_canonical_form_is_composed():
                assert canonical_username(" Jose\\u0301 ") == canonical_username("jos\\u00e9")


            def test_accents_still_significant():
                r = Registry()
                r.register("jose")
                r.register("jos\\u00e9")
                assert r.is_taken("JOSE") and r.is_taken("Jos\\u00e9")
            ''',
        p2p=["tests/test_registry.py::test_case_insensitive", "tests/test_registry.py::test_blank_rejected",
             "tests/test_registry.py::test_is_taken"],
        steps=[
            ("Locate the registry code.", F("accounts/*.py")),
            ("See how register decides a name is taken.", V("accounts/registry.py")),
            ("Uniqueness is keyed on canonical_username; check where else it's used.", S("canonical_username", "accounts")),
            ("Reproduce with the two byte sequences from the report.",
             B("python -c 'from accounts.normalize import canonical_username as c; a, b = c(\"Jos\\u00e9\"), c(\"Jose\\u0301\"); print(ascii(a), ascii(b), a == b)'")),
            ("Only lower() is applied, so canonically-equivalent sequences stay different. Normalize to NFC "
             "first; this keeps accents significant (jose != josé) unlike stripping them.",
             I("accounts/normalize.py", 0, "import unicodedata\n\n\n")),
            ("Apply the normalization in canonical_username.",
             R("accounts/normalize.py", "    return name.strip().lower()",
               "    return unicodedata.normalize(\"NFC\", name).strip().lower()")),
            ("Re-run the repro and confirm accents still distinguish names.",
             B("python -c 'from accounts.normalize import canonical_username as c; print(c(\"Jos\\u00e9\") == c(\"Jose\\u0301\"), c(\"jose\") == c(\"jos\\u00e9\"))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 19 ---------------------------------------------------------------------------------------
    dict(
        id="seed-019-truncate-ellipsis-length",
        bug_type="length-calculation", difficulty="easy",
        issue="""\
            Saving long notification titles fails with "value too long"

            ```
            psycopg.errors.StringDataRightTruncation: value too long for type character varying(40)
            ```
            We pass titles through `card_title()` precisely so they fit the 40-char column, but
            long titles come out 41 characters long.
            """,
        files={
            "textutil/__init__.py": "",
            "textutil/truncate.py": '''\
                ELLIPSIS = "\\u2026"


                def truncate(text, max_len, ellipsis=ELLIPSIS):
                    """Shorten text to at most `max_len` characters, marking the cut with `ellipsis`."""
                    if max_len < len(ellipsis):
                        raise ValueError("max_len is shorter than the ellipsis")
                    if len(text) <= max_len:
                        return text
                    return text[:max_len] + ellipsis
                ''',
            "textutil/cards.py": '''\
                from .truncate import truncate

                TITLE_COLUMN_WIDTH = 40  # notifications.title is VARCHAR(40)


                def card_title(title):
                    return truncate(" ".join(title.split()), TITLE_COLUMN_WIDTH)


                def preview(body, width=80):
                    return truncate(body.replace("\\n", " "), width, ellipsis="...")
                ''',
            "tests/test_truncate.py": '''\
                import pytest

                from textutil.cards import card_title
                from textutil.truncate import truncate


                def test_short_text_unchanged():
                    assert truncate("hello", 10) == "hello"
                    assert truncate("exactly10!", 10) == "exactly10!"


                def test_rejects_tiny_limit():
                    with pytest.raises(ValueError):
                        truncate("abc", 2, ellipsis="...")


                def test_card_title_collapses_whitespace():
                    assert card_title("  Build   failed \\n on main ") == "Build failed on main"
                ''',
        },
        hidden_test='''\
            from textutil.cards import card_title, preview
            from textutil.truncate import truncate


            def test_result_fits_limit():
                assert truncate("abcdefghijkl", 10) == "abcdefghi\\u2026"
                assert truncate("abcdefghijkl", 10, ellipsis="...") == "abcdefg..."


            def test_card_title_fits_column():
                t = card_title("Deployment of service payments-api to production finished with warnings")
                assert len(t) == 40 and t.endswith("\\u2026")


            def test_preview_width():
                assert len(preview("x" * 200, width=80)) == 80


            def test_exact_limit_not_truncated():
                assert truncate("a" * 40, 40) == "a" * 40
            ''',
        p2p=["tests/test_truncate.py::test_short_text_unchanged", "tests/test_truncate.py::test_rejects_tiny_limit",
             "tests/test_truncate.py::test_card_title_collapses_whitespace"],
        steps=[
            ("Check how card_title shortens titles.", V("textutil/cards.py")),
            ("It relies on truncate; read it.", V("textutil/truncate.py")),
            ("Measure the output length for a long title.",
             B("python -c 'from textutil.cards import card_title; t = card_title(\"Deployment of service payments-api to production finished\"); print(repr(t), len(t))'")),
            ("The ellipsis is appended after keeping max_len characters, so results are len(ellipsis) too long. "
             "Reserve room for it.",
             R("textutil/truncate.py", "    return text[:max_len] + ellipsis",
               "    return text[:max_len - len(ellipsis)] + ellipsis")),
            ("Verify lengths with the default and a 3-char ellipsis.",
             B("python -c 'from textutil.cards import card_title, preview; t = card_title(\"Deployment of service payments-api to production finished\"); print(repr(t), len(t), len(preview(\"x\" * 200)))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 20 ---------------------------------------------------------------------------------------
    dict(
        id="seed-020-markdown-link-greedy",
        bug_type="regex-greedy", difficulty="easy",
        issue="""\
            Two links in one paragraph render as one broken link

            Input:

                See [docs](https://a.io/docs) and [FAQ](https://a.io/faq).

            The output is a single `<a href="https://a.io/faq">` whose text is
            `docs](https://a.io/docs) and [FAQ`. One link per line works.
            """,
        files={
            "mdlite/__init__.py": "from .block import render\n",
            "mdlite/inline.py": '''\
                import re
                from html import escape

                _CODE = re.compile(r"`([^`]+)`")
                _BOLD = re.compile(r"\\*\\*(.+?)\\*\\*")
                _LINK = re.compile(r"\\[(.+)\\]\\((.+)\\)")


                def render_inline(text):
                    text = escape(text, quote=False)
                    text = _CODE.sub(r"<code>\\1</code>", text)
                    text = _BOLD.sub(r"<strong>\\1</strong>", text)
                    text = _LINK.sub(r'<a href="\\2">\\1</a>', text)
                    return text
                ''',
            "mdlite/block.py": '''\
                from .inline import render_inline


                def render(source):
                    """Render headings (#..###) and paragraphs separated by blank lines."""
                    out = []
                    for block in source.strip().split("\\n\\n"):
                        block = " ".join(line.strip() for line in block.splitlines())
                        if not block:
                            continue
                        level = len(block) - len(block.lstrip("#"))
                        if 1 <= level <= 3 and block[level:level + 1] == " ":
                            out.append(f"<h{level}>{render_inline(block[level + 1:])}</h{level}>")
                        else:
                            out.append(f"<p>{render_inline(block)}</p>")
                    return "\\n".join(out)
                ''',
            "tests/test_mdlite.py": '''\
                from mdlite import render
                from mdlite.inline import render_inline


                def test_single_link():
                    assert render_inline("[docs](https://a.io)") == '<a href="https://a.io">docs</a>'


                def test_bold_and_code():
                    assert render_inline("**hi** `x<y`") == "<strong>hi</strong> <code>x&lt;y</code>"


                def test_blocks():
                    assert render("# Title\\n\\nline one\\nline two") == "<h1>Title</h1>\\n<p>line one line two</p>"
                ''',
        },
        hidden_test='''\
            from mdlite import render
            from mdlite.inline import render_inline


            def test_two_links_one_line():
                out = render_inline("See [docs](https://a.io/docs) and [FAQ](https://a.io/faq).")
                assert out == 'See <a href="https://a.io/docs">docs</a> and <a href="https://a.io/faq">FAQ</a>.'


            def test_link_followed_by_parenthetical():
                out = render_inline("[home](https://a.io) (main site)")
                assert out == '<a href="https://a.io">home</a> (main site)'


            def test_links_in_paragraph():
                assert render("[a](/a) [b](/b)") == '<p><a href="/a">a</a> <a href="/b">b</a></p>'


            def test_bold_link_text():
                assert render_inline("[**x**](/u)") == '<a href="/u"><strong>x</strong></a>'
            ''',
        p2p=["tests/test_mdlite.py::test_single_link", "tests/test_mdlite.py::test_bold_and_code",
             "tests/test_mdlite.py::test_blocks"],
        steps=[
            ("Find the link pattern.", S("_LINK", "mdlite")),
            ("Reproduce with the paragraph from the report.",
             B("python -c 'from mdlite.inline import render_inline; print(render_inline(\"See [docs](https://a.io/docs) and [FAQ](https://a.io/faq).\"))'")),
            ("`(.+)` is greedy, so the first `[` pairs with the last `](...)` on the line. Constrain the "
             "text to non-`]` and the URL to non-`)`/whitespace characters.",
             R("mdlite/inline.py", '_LINK = re.compile(r"\\[(.+)\\]\\((.+)\\)")',
               '_LINK = re.compile(r"\\[([^\\]]+)\\]\\(([^)\\s]+)\\)")')),
            ("Re-run the repro and a link followed by a parenthetical.",
             B("python -c 'from mdlite.inline import render_inline as r; print(r(\"See [docs](https://a.io/docs) and [FAQ](https://a.io/faq).\")); print(r(\"[home](https://a.io) (main site)\"))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 21 ---------------------------------------------------------------------------------------
    dict(
        id="seed-021-linediff-overlapping-affixes",
        bug_type="overlapping-ranges", difficulty="hard",
        issue="""\
            linediff says "no changes" for files that differ

            I removed a duplicated blank line from a config file and `summary(old, new)` reports
            `no changes` (and `unified()` prints nothing). On another file where I inserted a block
            next to a similar line, the change produced by `diff_lines` doesn't round-trip: applying
            it to the old lines gives something that is neither old nor new. Simple edits work.

                >>> summary(["x", "x"], ["x"])
                'no changes'
            """,
        files={
            "linediff/__init__.py": '''\
                from .core import Change, diff_lines
                from .patch import apply_change
                from .unified import summary, unified
                ''',
            "linediff/core.py": '''\
                from dataclasses import dataclass


                @dataclass
                class Change:
                    """Replace a[start_a:start_a + len(removed)] with `added` (inserted at b[start_b])."""
                    start_a: int
                    removed: list
                    start_b: int
                    added: list


                def _common_affixes(a, b):
                    prefix = 0
                    while prefix < len(a) and prefix < len(b) and a[prefix] == b[prefix]:
                        prefix += 1
                    suffix = 0
                    while suffix < len(a) and suffix < len(b) and a[-1 - suffix] == b[-1 - suffix]:
                        suffix += 1
                    return prefix, suffix


                def diff_lines(a, b):
                    """Single change spanning everything between the common prefix and suffix, or None."""
                    prefix, suffix = _common_affixes(a, b)
                    removed = a[prefix:len(a) - suffix]
                    added = b[prefix:len(b) - suffix]
                    if not removed and not added:
                        return None
                    return Change(prefix, removed, prefix, added)
                ''',
            "linediff/patch.py": '''\
                def apply_change(a, change):
                    """Apply a Change from diff_lines(a, b) to `a`, returning b."""
                    if change is None:
                        return list(a)
                    end = change.start_a + len(change.removed)
                    if a[change.start_a:end] != change.removed:
                        raise ValueError("change does not apply: context mismatch")
                    return a[:change.start_a] + change.added + a[end:]
                ''',
            "linediff/unified.py": '''\
                from .core import diff_lines


                def unified(a, b, name_a="a", name_b="b"):
                    change = diff_lines(a, b)
                    if change is None:
                        return ""
                    out = [f"--- {name_a}", f"+++ {name_b}",
                           f"@@ -{change.start_a + 1},{len(change.removed)} +{change.start_b + 1},{len(change.added)} @@"]
                    out += ["-" + line for line in change.removed]
                    out += ["+" + line for line in change.added]
                    return "\\n".join(out) + "\\n"


                def summary(a, b):
                    change = diff_lines(a, b)
                    if change is None:
                        return "no changes"
                    return f"{len(change.added)} added, {len(change.removed)} removed"
                ''',
            "tests/test_linediff.py": '''\
                from linediff import Change, apply_change, diff_lines, summary, unified


                def test_replace_middle():
                    assert diff_lines(["a", "b", "c"], ["a", "x", "c"]) == Change(1, ["b"], 1, ["x"])


                def test_equal_inputs():
                    assert diff_lines(["a"], ["a"]) is None
                    assert summary(["a", "b"], ["a", "b"]) == "no changes"


                def test_unified_output():
                    assert unified(["a", "b"], ["a", "c", "d"]) == "--- a\\n+++ b\\n@@ -2,1 +2,2 @@\\n-b\\n+c\\n+d\\n"


                def test_round_trip_simple():
                    pairs = [(["a", "b", "c"], ["a", "c"]), (["a"], ["a", "b"]), ([], ["x"]), (["p", "q"], ["r"])]
                    for a, b in pairs:
                        assert apply_change(a, diff_lines(a, b)) == b
                ''',
        },
        hidden_test='''\
            from linediff import apply_change, diff_lines, summary


            def test_removed_duplicate_line_detected():
                assert summary(["x", "x"], ["x"]) == "0 added, 1 removed"
                assert summary(["a", "a", "a"], ["a"]) == "0 added, 2 removed"


            def test_insert_repeating_neighbour():
                a, b = ["a", "b", "c"], ["a", "b", "x", "b", "c"]
                ch = diff_lines(a, b)
                assert ch.start_a == 2 and ch.removed == [] and ch.added == ["x", "b"]


            def test_round_trip_with_repeats():
                pairs = [
                    (["x", "x"], ["x"]),
                    (["x"], ["x", "x"]),
                    (["a", "b", "c"], ["a", "b", "x", "b", "c"]),
                    (["", "k=1", "", ""], ["", "k=1", ""]),
                    (["a", "b"], ["a", "b", "a", "b"]),
                ]
                for a, b in pairs:
                    assert apply_change(a, diff_lines(a, b)) == b


            def test_simple_cases_unchanged():
                assert summary(["a", "b", "c"], ["a", "x", "c"]) == "1 added, 1 removed"
            ''',
        p2p=["tests/test_linediff.py::test_replace_middle", "tests/test_linediff.py::test_equal_inputs",
             "tests/test_linediff.py::test_unified_output", "tests/test_linediff.py::test_round_trip_simple"],
        steps=[
            ("Look at the package layout.", F("linediff/*.py")),
            ("summary just reports what diff_lines returns; reproduce both symptoms at that level.",
             B(d("""\
                 python - <<'EOF'
                 from linediff import diff_lines, apply_change
                 print(diff_lines(["x", "x"], ["x"]))
                 a, b = ["a", "b", "c"], ["a", "b", "x", "b", "c"]
                 ch = diff_lines(a, b)
                 print(ch, apply_change(a, ch))
                 EOF"""))),
            ("Read the core diff logic.", V("linediff/core.py", 13, 30)),
            ("The suffix scan ignores what the prefix already consumed, so with repeated lines the prefix "
             "and suffix overlap (prefix + suffix > len(a)) and the middle slices come out empty or short. "
             "Bound the suffix by the lines left after the prefix.",
             R("linediff/core.py",
               "    while suffix < len(a) and suffix < len(b) and a[-1 - suffix] == b[-1 - suffix]:\n",
               "    limit = min(len(a), len(b)) - prefix  # prefix and suffix must not overlap\n"
               "    while suffix < limit and a[-1 - suffix] == b[-1 - suffix]:\n")),
            ("Re-run the repro.",
             B(d("""\
                 python - <<'EOF'
                 from linediff import diff_lines, apply_change, summary
                 print(diff_lines(["x", "x"], ["x"]), summary(["x", "x"], ["x"]))
                 a, b = ["a", "b", "c"], ["a", "b", "x", "b", "c"]
                 ch = diff_lines(a, b)
                 print(ch, apply_change(a, ch) == b)
                 EOF"""))),
            ("Brute-force round-trip over all short sequences of two symbols to be sure.",
             B(d("""\
                 python - <<'EOF'
                 from itertools import product
                 from linediff import diff_lines, apply_change
                 seqs = [list(p) for n in range(5) for p in product("ab", repeat=n)]
                 bad = [(a, b) for a in seqs for b in seqs if apply_change(a, diff_lines(a, b)) != b]
                 print(len(seqs) ** 2, "pairs,", len(bad), "failures")
                 EOF"""))),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 22 ---------------------------------------------------------------------------------------
    dict(
        id="seed-022-amount-regex-alternation",
        bug_type="regex-alternation-order", difficulty="medium",
        issue="""\
            Receipt totals are off for items over 1,000

            Scanning a receipt with these lines:

                Coffee beans       12.50
                Espresso machine   1,249.00
                2 x Filter         3.00

            `receipt_total(lines)` returns 264.5 instead of 1264.5. Receipts where every item is
            under 1,000 add up correctly.
            """,
        files={
            "receipts/__init__.py": "",
            "receipts/extract.py": '''\
                import re

                # 1234.50, 1,234.50, -3, 12
                _AMOUNT = re.compile(r"-?(?:\\d+|\\d{1,3}(?:,\\d{3})+)(?:\\.\\d{2})?")


                def find_amounts(text):
                    """All monetary amounts in `text`, in order, as floats."""
                    return [float(m.group().replace(",", "")) for m in _AMOUNT.finditer(text)]
                ''',
            "receipts/receipt.py": '''\
                from .extract import find_amounts


                def line_amount(line):
                    """The price of a receipt line is the last amount on it (quantities come first)."""
                    amounts = find_amounts(line)
                    return amounts[-1] if amounts else None


                def receipt_total(lines):
                    total = 0.0
                    for line in lines:
                        amount = line_amount(line)
                        if amount is not None:
                            total += amount
                    return round(total, 2)
                ''',
            "tests/test_receipt.py": '''\
                from receipts.extract import find_amounts
                from receipts.receipt import line_amount, receipt_total


                def test_find_simple_amounts():
                    assert find_amounts("2 x Bagel 3.00, refund -1.50") == [2.0, 3.0, -1.5]


                def test_line_amount_uses_last_number():
                    assert line_amount("2 x Filter   3.00") == 3.0
                    assert line_amount("THANK YOU") is None


                def test_total():
                    assert receipt_total(["Tea  2.20", "Cake 3.80", "VISA ****"]) == 6.0
                ''',
        },
        hidden_test='''\
            from receipts.extract import find_amounts
            from receipts.receipt import receipt_total


            def test_thousands_separator():
                assert find_amounts("Espresso machine   1,249.00") == [1249.0]
                assert find_amounts("Total 12,345,678.90 due") == [12345678.9]


            def test_receipt_total_with_large_item():
                lines = ["Coffee beans       12.50", "Espresso machine   1,249.00", "2 x Filter         3.00"]
                assert receipt_total(lines) == 1264.5


            def test_plain_long_numbers():
                assert find_amounts("ref 12345.67 and -1,000") == [12345.67, -1000.0]


            def test_small_amounts_unchanged():
                assert find_amounts("Tea 2.20 x 3") == [2.2, 3.0]
            ''',
        p2p=["tests/test_receipt.py::test_find_simple_amounts", "tests/test_receipt.py::test_line_amount_uses_last_number",
             "tests/test_receipt.py::test_total"],
        steps=[
            ("Find how receipt lines are priced.", S("def (receipt_total|line_amount|find_amounts)", "receipts")),
            ("Reproduce per line to see which one is wrong.",
             B("python -c 'from receipts.receipt import line_amount; from receipts.extract import find_amounts; "
               "print(line_amount(\"Espresso machine   1,249.00\"), find_amounts(\"Espresso machine   1,249.00\"))'")),
            ("The extractor splits `1,249.00` into 1 and 249.00. Read the pattern.", V("receipts/extract.py")),
            ("Regex alternation is ordered: `\\d+` matches `1` before the thousands-separated branch is "
             "tried. Put the comma-grouped alternative first; plain digits still fall through to `\\d+`.",
             R("receipts/extract.py", "(?:\\d+|\\d{1,3}(?:,\\d{3})+)",
               "(?:\\d{1,3}(?:,\\d{3})+|\\d+)")),
            ("Re-run with grouped, plain and negative amounts plus the receipt from the report.",
             B("python -c 'from receipts.extract import find_amounts as f; from receipts.receipt import receipt_total as t; "
               "print(f(\"1,249.00\"), f(\"ref 12345.67 and -1,000\"), t([\"Coffee beans 12.50\", \"Espresso machine 1,249.00\", \"2 x Filter 3.00\"]))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 23 ---------------------------------------------------------------------------------------
    dict(
        id="seed-023-lexer-line-counter-reset",
        bug_type="state-reset", difficulty="medium",
        issue="""\
            Template linter reports errors on lines that don't exist

            Running `check_all` over our templates, only the first file gets correct line numbers.
            For the others the reported line keeps growing, e.g.

                footer.tpl: unexpected '@' at line 9

            but footer.tpl only has 2 lines, and the `@` is on line 2. Linting footer.tpl on its own
            reports line 2 correctly. Token line numbers are off the same way.
            """,
        files={
            "tpllint/__init__.py": "",
            "tpllint/lexer.py": '''\
                import re
                from dataclasses import dataclass


                @dataclass(frozen=True)
                class Token:
                    kind: str
                    value: str
                    line: int


                class LexError(Exception):
                    pass


                _SPEC = [
                    ("NAME", r"[A-Za-z_]\\w*"),
                    ("NUMBER", r"\\d+"),
                    ("STRING", r'"[^"\\n]*"'),
                    ("OP", r"[=+\\-*/(),{}|]"),
                    ("NEWLINE", r"\\n"),
                    ("SKIP", r"[ \\t]+"),
                ]
                _RX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _SPEC))


                class Lexer:
                    def __init__(self):
                        self.line = 1

                    def tokenize(self, text):
                        tokens = []
                        pos = 0
                        while pos < len(text):
                            m = _RX.match(text, pos)
                            if m is None:
                                raise LexError(f"unexpected {text[pos]!r} at line {self.line}")
                            kind = m.lastgroup
                            if kind == "NEWLINE":
                                self.line += 1
                            elif kind != "SKIP":
                                tokens.append(Token(kind, m.group(), self.line))
                            pos = m.end()
                        return tokens
                ''',
            "tpllint/api.py": '''\
                from .lexer import Lexer, LexError

                _lexer = Lexer()


                def tokenize(text):
                    return _lexer.tokenize(text)


                def check(text):
                    """Return a list of error messages (empty if the template lexes cleanly)."""
                    try:
                        tokenize(text)
                    except LexError as e:
                        return [str(e)]
                    return []


                def check_all(files):
                    """files: {name: text}. Returns ["name: message", ...]."""
                    return [f"{name}: {msg}" for name, text in files.items() for msg in check(text)]
                ''',
            "tests/test_lexer.py": '''\
                from tpllint.api import check
                from tpllint.lexer import Lexer, Token


                def test_tokens_and_lines():
                    toks = Lexer().tokenize('x = 1\\ny = "a"')
                    assert toks[-1] == Token("STRING", '"a"', 2)
                    assert [t.kind for t in toks] == ["NAME", "OP", "NUMBER", "NAME", "OP", "STRING"]


                def test_check_reports_line():
                    assert check("a = 1\\nb = @") == ["unexpected '@' at line 2"]


                def test_check_clean():
                    assert check("{ name | upper }") == []
                ''',
        },
        hidden_test='''\
            from tpllint.api import check, check_all, tokenize
            from tpllint.lexer import Lexer


            def test_lexer_reusable():
                lx = Lexer()
                lx.tokenize("a\\nb\\nc\\n")
                assert lx.tokenize("x")[0].line == 1


            def test_module_tokenize_repeated_calls():
                tokenize("a\\nb\\n")
                assert [t.line for t in tokenize("p\\nq")] == [1, 2]


            def test_check_all_line_numbers():
                files = {"header.tpl": "a\\nb\\nc\\nd\\ne\\nf\\ng", "footer.tpl": "x = 1\\ny = @"}
                assert check_all(files) == ["footer.tpl: unexpected '@' at line 2"]


            def test_check_after_error_resets():
                check("a\\n@")
                assert check("@") == ["unexpected '@' at line 1"]


            def test_fresh_lexer_counts_blank_lines():
                assert [t.line for t in Lexer().tokenize("a\\n\\nb")] == [1, 3]
            ''',
        p2p=["tests/test_lexer.py::test_tokens_and_lines", "tests/test_lexer.py::test_check_reports_line",
             "tests/test_lexer.py::test_check_clean"],
        steps=[
            ("Start from check_all.", V("tpllint/api.py")),
            ("Reproduce: lint two files in one process.",
             B("python -c 'from tpllint.api import check_all; print(check_all({\"header.tpl\": \"a\\nb\\nc\\nd\\ne\\nf\\ng\", \"footer.tpl\": \"x = 1\\ny = @\"}))'")),
            ("All calls share one module-level Lexer. Look at how it tracks lines.", V("tpllint/lexer.py", 26, 44)),
            ("`self.line` is only initialised in __init__, so every tokenize() continues counting from the "
             "previous text. Reset it at the start of each call.",
             I("tpllint/lexer.py", 31, "        self.line = 1\n")),
            ("Check the edited method reads correctly.", V("tpllint/lexer.py", 30, 36)),
            ("Re-run the repro.",
             B("python -c 'from tpllint.api import check_all; print(check_all({\"header.tpl\": \"a\\nb\\nc\\nd\\ne\\nf\\ng\", \"footer.tpl\": \"x = 1\\ny = @\"}))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 24 ---------------------------------------------------------------------------------------
    dict(
        id="seed-024-version-regex-anchoring",
        bug_type="regex-anchoring", difficulty="easy",
        issue="""\
            Release script accepted the tag "1.4.2rc1" as a valid version

            We only publish plain MAJOR.MINOR.PATCH versions, and `is_valid_version` is supposed to
            guard that. It returns True for `1.4.2rc1`, `1.4.2.1` and `2.0.0-beta`, and
            `parse_version("1.4.2.1")` silently returns `(1, 4, 2)`.
            """,
        files={
            "pkgmeta/__init__.py": "",
            "pkgmeta/version.py": '''\
                import re

                _VERSION = re.compile(r"(\\d+)\\.(\\d+)\\.(\\d+)")


                class InvalidVersion(ValueError):
                    pass


                def parse_version(text):
                    """Parse 'MAJOR.MINOR.PATCH' into a tuple of ints."""
                    m = _VERSION.match(text.strip())
                    if not m:
                        raise InvalidVersion(text)
                    return tuple(int(g) for g in m.groups())


                def is_valid_version(text):
                    try:
                        parse_version(text)
                    except InvalidVersion:
                        return False
                    return True
                ''',
            "pkgmeta/release.py": '''\
                from .version import parse_version


                def next_version(current, part="patch"):
                    major, minor, patch = parse_version(current)
                    if part == "major":
                        return f"{major + 1}.0.0"
                    if part == "minor":
                        return f"{major}.{minor + 1}.0"
                    return f"{major}.{minor}.{patch + 1}"
                ''',
            "tests/test_version.py": '''\
                import pytest

                from pkgmeta.release import next_version
                from pkgmeta.version import InvalidVersion, is_valid_version, parse_version


                def test_parse():
                    assert parse_version(" 1.10.3\\n") == (1, 10, 3)


                def test_invalid():
                    assert not is_valid_version("1.2")
                    with pytest.raises(InvalidVersion):
                        parse_version("v1.2.3")


                def test_next_version():
                    assert next_version("1.4.2") == "1.4.3"
                    assert next_version("1.4.2", "minor") == "1.5.0"
                ''',
        },
        hidden_test='''\
            import pytest

            from pkgmeta.release import next_version
            from pkgmeta.version import InvalidVersion, is_valid_version, parse_version


            def test_trailing_garbage_rejected():
                for bad in ("1.4.2rc1", "1.4.2.1", "2.0.0-beta", "1.2.3 extra"):
                    assert not is_valid_version(bad), bad


            def test_parse_rejects_four_parts():
                with pytest.raises(InvalidVersion):
                    parse_version("1.4.2.1")
                with pytest.raises(InvalidVersion):
                    next_version("1.4.2rc1")


            def test_whitespace_still_allowed():
                assert parse_version("  0.0.1  ") == (0, 0, 1)
            ''',
        p2p=["tests/test_version.py::test_parse", "tests/test_version.py::test_invalid",
             "tests/test_version.py::test_next_version"],
        steps=[
            ("Find the version validation.", B("grep -rn 'def is_valid_version\\|_VERSION' --include=*.py pkgmeta")),
            ("Read it.", V("pkgmeta/version.py", 1, 16)),
            ("Reproduce with the strings from the report.",
             B("python -c 'from pkgmeta.version import is_valid_version as v, parse_version as p; print([v(s) for s in (\"1.4.2rc1\", \"1.4.2.1\", \"2.0.0-beta\")], p(\"1.4.2.1\"))'")),
            ("re.match only anchors at the start, so any suffix is accepted. The whole (stripped) string "
             "must match: use fullmatch.",
             R("pkgmeta/version.py", "    m = _VERSION.match(text.strip())", "    m = _VERSION.fullmatch(text.strip())")),
            ("Re-run, including a valid version with surrounding whitespace.",
             B("python -c 'from pkgmeta.version import is_valid_version as v; print([v(s) for s in (\"1.4.2rc1\", \"1.4.2.1\", \"2.0.0-beta\", \" 1.4.2 \")])'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 25 ---------------------------------------------------------------------------------------
    dict(
        id="seed-025-properties-escaped-backslash-continuation",
        bug_type="escape-parity", difficulty="hard",
        issue=r"""
            Property after a Windows path disappears from build.properties

            ```
            Traceback (most recent call last):
              File "build.py", line 31, in main
                artifact = cfg.require("output.name")
              File "propfile/config.py", line 15, in require
                raise KeyError(f"missing required property {key!r}") from None
            KeyError: "missing required property 'output.name'"
            ```

            build.properties:

                output.dir = C:\\builds\\
                output.name = app.zip

            `cfg.get("output.dir")` comes back as `C:\buildsoutput.name = app.zip`. Our other
            multi-line properties (ending a line with a single `\`) still need to work.
            """,
        files={
            "propfile/__init__.py": "from .config import Config\nfrom .reader import load\n",
            "propfile/escapes.py": '''\
                _ESCAPES = {"n": "\\n", "t": "\\t", "r": "\\r"}


                def unescape(s):
                    """Decode backslash escapes: \\\\n, \\\\t, \\\\r; any other escaped char stands for itself."""
                    out, i = [], 0
                    while i < len(s):
                        ch = s[i]
                        if ch == "\\\\" and i + 1 < len(s):
                            nxt = s[i + 1]
                            out.append(_ESCAPES.get(nxt, nxt))
                            i += 2
                        else:
                            out.append(ch)
                            i += 1
                    return "".join(out)
                ''',
            "propfile/reader.py": '''\
                from .escapes import unescape


                def _logical_lines(text):
                    """Join physical lines ending in a backslash with the following line."""
                    buf = None
                    for raw in text.splitlines():
                        line = raw.lstrip() if buf is not None else raw
                        if line.endswith("\\\\"):
                            buf = (buf or "") + line[:-1]
                            continue
                        yield (buf or "") + line
                        buf = None
                    if buf is not None:
                        yield buf


                def _split_key_value(line):
                    """Split at the first unescaped '=' or ':'."""
                    i = 0
                    while i < len(line):
                        ch = line[i]
                        if ch == "\\\\":
                            i += 2
                            continue
                        if ch in "=:":
                            return line[:i].rstrip(), line[i + 1:].lstrip()
                        i += 1
                    return line, ""


                def load(text):
                    props = {}
                    for line in _logical_lines(text):
                        stripped = line.strip()
                        if not stripped or stripped[0] in "#!":
                            continue
                        key, value = _split_key_value(stripped)
                        props[unescape(key)] = unescape(value)
                    return props
                ''',
            "propfile/config.py": '''\
                from .reader import load


                class Config:
                    def __init__(self, text):
                        self._props = load(text)

                    def get(self, key, default=None):
                        return self._props.get(key, default)

                    def require(self, key):
                        try:
                            return self._props[key]
                        except KeyError:
                            raise KeyError(f"missing required property {key!r}") from None

                    def get_list(self, key, sep=","):
                        value = self.get(key, "")
                        return [item.strip() for item in value.split(sep) if item.strip()]
                ''',
            "tests/test_properties.py": r'''
                import pytest

                from propfile import Config, load


                def test_basic_and_comments():
                    text = "# comment\n! also comment\nname = demo\nport: 8080\n\n"
                    assert load(text) == {"name": "demo", "port": "8080"}


                def test_continuation_lines():
                    text = "colors = red, \\\n    green, \\\n    blue\nnext = 1\n"
                    cfg = Config(text)
                    assert cfg.get_list("colors") == ["red", "green", "blue"]
                    assert cfg.get("next") == "1"


                def test_escapes():
                    assert load(r"msg = a\tb\=c" + "\n" + r"k\:x = 1") == {"msg": "a\tb=c", "k:x": "1"}


                def test_require_missing():
                    with pytest.raises(KeyError, match="output.name"):
                        Config("a = 1\n").require("output.name")
                ''',
        },
        hidden_test=r'''
            from propfile import Config, load


            def test_escaped_trailing_backslash_is_not_continuation():
                text = "output.dir = C:\\\\builds\\\\\noutput.name = app.zip\n"
                cfg = Config(text)
                assert cfg.require("output.dir") == "C:\\builds\\"
                assert cfg.require("output.name") == "app.zip"


            def test_escaped_backslash_then_continuation():
                text = "path = a\\\\\\\n    b\nz = 2\n"
                assert load(text) == {"path": "a\\b", "z": "2"}


            def test_plain_continuation_still_works():
                assert load("k = one \\\n  two\n") == {"k": "one two"}
            ''',
        p2p=["tests/test_properties.py::test_basic_and_comments", "tests/test_properties.py::test_continuation_lines",
             "tests/test_properties.py::test_escapes", "tests/test_properties.py::test_require_missing"],
        steps=[
            ("Find the reader and escape handling.", F("propfile/*.py")),
            ("Reproduce with the two lines from the report.",
             B(d(r"""
                 python - <<'EOF'
                 from propfile import load
                 print(load("output.dir = C:\\\\builds\\\\\noutput.name = app.zip\n"))
                 EOF"""))),
            ("output.name got glued onto output.dir. Look at how physical lines are joined.", V("propfile/reader.py", 1, 16)),
            ("Escape decoding happens later in unescape; confirm `\\\\` is treated as an escaped backslash there.",
             S("_ESCAPES|def unescape", "propfile")),
            ("unescape treats `\\\\` as a literal backslash, but _logical_lines treats any trailing backslash "
             "as a continuation. A line only continues if it ends in an odd number of backslashes; add a helper.",
             R("propfile/reader.py",
               "from .escapes import unescape\n\n\n",
               "from .escapes import unescape\n\n\n"
               "def _is_continued(line):\n"
               "    \"\"\"True if the line ends in an odd number of backslashes (an even run is escaped backslashes).\"\"\"\n"
               "    return (len(line) - len(line.rstrip(\"\\\\\"))) % 2 == 1\n\n\n")),
            ("Use it in _logical_lines.",
             R("propfile/reader.py", "        if line.endswith(\"\\\\\"):\n", "        if _is_continued(line):\n")),
            ("Re-run the repro, plus an escaped backslash followed by a real continuation.",
             B(d(r"""
                 python - <<'EOF'
                 from propfile import load
                 print(load("output.dir = C:\\\\builds\\\\\noutput.name = app.zip\n"))
                 print(load("path = a\\\\\\\n    b\nz = 2\n"))
                 EOF"""))),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
]
