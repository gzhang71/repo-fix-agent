"""Build the 10-instance seed dataset.

For each instance we:
  1. materialize a small buggy repo under data/repos/<id>/ (source + existing passing tests),
  2. replay the engineer's action script through the real RepoEnv tools, recording the
     true observations (so trajectories are consistent with what an agent would see),
  3. take the resulting `git diff` as the gold patch,
  4. store the hidden failing test as `test_patch` + FAIL_TO_PASS ids.

This mirrors how the 10k engineer demos should be converted: keep their actions, re-execute
them in the sandbox to regenerate observations, and drop any demo whose patch doesn't validate.

Usage:  python scripts/build_seed_dataset.py && repo-fix validate data/demos/seed.jsonl
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from repo_fix_agent.agent import run_agent  # noqa: E402
from repo_fix_agent.env import RepoEnv  # noqa: E402
from repo_fix_agent.grader import _run_tests  # noqa: E402
from repo_fix_agent.policies import ScriptedPolicy  # noqa: E402
from repo_fix_agent.schema import Action, Demo, save_demos  # noqa: E402

TEST_CMD = "python -m pytest -q -p no:cacheprovider"


def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def B(cmd):  # bash
    return Action("bash", {"command": cmd})


def V(path, s=None, e=None):
    args = {"path": path}
    if s is not None:
        args |= {"start_line": s, "end_line": e}
    return Action("view_file", args)


def R(path, old, new):
    return Action("str_replace", {"path": path, "old_str": old, "new_str": new})


SUBMIT = Action("submit", {})

# --------------------------------------------------------------------------------------------
# Instance specs
# --------------------------------------------------------------------------------------------
SPECS = [
    # 1 ----------------------------------------------------------------------------------------
    dict(
        id="seed-001-pagination-offset",
        bug_type="off-by-one", difficulty="easy",
        issue="""\
            Pagination skips the first page of results

            `paginate(items, page=1, per_page=10)` returns items 10-19 instead of 0-9, and the
            last page is sometimes empty even though `total_pages()` says it exists. Pages are
            documented as 1-indexed.
            """,
        files={
            "webapp/__init__.py": "",
            "webapp/pagination.py": '''\
                def paginate(items, page, per_page):
                    """Return the items on a 1-indexed page."""
                    if page < 1 or per_page < 1:
                        raise ValueError("page and per_page must be >= 1")
                    start = page * per_page
                    return items[start:start + per_page]


                def total_pages(n_items, per_page):
                    return (n_items + per_page - 1) // per_page
                ''',
            "webapp/views.py": '''\
                from .pagination import paginate, total_pages


                def list_view(items, page=1, per_page=10):
                    return {"items": paginate(items, page, per_page),
                            "pages": total_pages(len(items), per_page)}
                ''',
            "tests/test_pagination.py": '''\
                import pytest
                from webapp.pagination import paginate, total_pages


                def test_total_pages():
                    assert total_pages(25, 10) == 3
                    assert total_pages(0, 10) == 0


                def test_rejects_page_zero():
                    with pytest.raises(ValueError):
                        paginate([1, 2, 3], 0, 2)
                ''',
        },
        hidden_test='''\
            from webapp.pagination import paginate


            def test_first_page_starts_at_zero():
                assert paginate(list(range(25)), 1, 10) == list(range(10))


            def test_last_partial_page():
                assert paginate(list(range(25)), 3, 10) == [20, 21, 22, 23, 24]
            ''',
        p2p=["tests/test_pagination.py::test_total_pages", "tests/test_pagination.py::test_rejects_page_zero"],
        steps=[
            ("Find where paginate is defined.", B("grep -rn 'def paginate' --include=*.py .")),
            ("Read the implementation.", V("webapp/pagination.py")),
            ("Reproduce: page 1 of 25 items should be 0..9.",
             B("python -c 'from webapp.pagination import paginate; print(paginate(list(range(25)), 1, 10))'")),
            ("The offset treats page as 0-indexed. For 1-indexed pages the start is (page - 1) * per_page.",
             R("webapp/pagination.py", "start = page * per_page", "start = (page - 1) * per_page")),
            ("Verify first and last pages.",
             B("python -c 'from webapp.pagination import paginate as p; r=list(range(25)); print(p(r,1,10), p(r,3,10))'")),
            ("Run the existing tests for regressions.", B(TEST_CMD)),
            ("Fix verified.", SUBMIT),
        ],
    ),
    # 2 ----------------------------------------------------------------------------------------
    dict(
        id="seed-002-mutable-default-tags",
        bug_type="mutable-default-argument", difficulty="easy",
        issue="""\
            Tags leak between inventory items

            Creating two `Item`s without passing `tags` and calling `add_tag` on one of them
            makes the tag appear on the other item too.

                a = Item("apple"); b = Item("banana")
                a.add_tag("fruit")
                b.tags  # ['fruit']  <- expected []
            """,
        files={
            "inventory/__init__.py": "from .item import Item\n",
            "inventory/item.py": '''\
                class Item:
                    def __init__(self, name, tags=[]):
                        self.name = name
                        self.tags = tags

                    def add_tag(self, tag):
                        if tag not in self.tags:
                            self.tags.append(tag)

                    def __repr__(self):
                        return f"Item({self.name!r}, tags={self.tags!r})"
                ''',
            "tests/test_item.py": '''\
                from inventory import Item


                def test_add_tag_dedups():
                    it = Item("pear", tags=["fruit"])
                    it.add_tag("fruit")
                    it.add_tag("green")
                    assert it.tags == ["fruit", "green"]
                ''',
        },
        hidden_test='''\
            from inventory import Item


            def test_default_tags_not_shared():
                a, b = Item("apple"), Item("banana")
                a.add_tag("fruit")
                assert b.tags == []


            def test_caller_list_not_mutated():
                tags = ["x"]
                Item("kiwi", tags=tags).add_tag("y")
                assert tags == ["x"]
            ''',
        p2p=["tests/test_item.py::test_add_tag_dedups"],
        steps=[
            ("Look at the Item class.", V("inventory/item.py")),
            ("`tags=[]` is a mutable default shared by every instance. Confirm with a repro.",
             B("python -c 'from inventory import Item; a=Item(\"a\"); b=Item(\"b\"); a.add_tag(\"x\"); print(b.tags)'")),
            ("Use None as the default and copy the input so the caller's list is not mutated either.",
             R("inventory/item.py",
               "    def __init__(self, name, tags=[]):\n        self.name = name\n        self.tags = tags\n",
               "    def __init__(self, name, tags=None):\n        self.name = name\n        self.tags = list(tags) if tags else []\n")),
            ("Re-run the repro.",
             B("python -c 'from inventory import Item; a=Item(\"a\"); b=Item(\"b\"); a.add_tag(\"x\"); print(b.tags)'")),
            ("Run the test suite.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 3 ----------------------------------------------------------------------------------------
    dict(
        id="seed-003-leap-year-century",
        bug_type="logic-error", difficulty="easy",
        issue="""\
            February 2000 reported as having 28 days

            `days_in_month(2000, 2)` returns 28. 2000 was a leap year (divisible by 400).
            This breaks our billing period calculation for historical data.
            """,
        files={
            "calendar_utils/__init__.py": "",
            "calendar_utils/dates.py": '''\
                def is_leap_year(year):
                    return year % 4 == 0 and year % 100 != 0


                def days_in_month(year, month):
                    if not 1 <= month <= 12:
                        raise ValueError("month must be in 1..12")
                    if month == 2:
                        return 29 if is_leap_year(year) else 28
                    return 30 if month in (4, 6, 9, 11) else 31
                ''',
            "calendar_utils/billing.py": '''\
                from .dates import days_in_month


                def prorate(amount, year, month, days_used):
                    return round(amount * days_used / days_in_month(year, month), 2)
                ''',
            "tests/test_dates.py": '''\
                from calendar_utils.dates import days_in_month, is_leap_year


                def test_common_years():
                    assert days_in_month(2023, 2) == 28
                    assert days_in_month(1900, 2) == 28


                def test_regular_leap_year():
                    assert is_leap_year(2024)
                    assert days_in_month(2024, 4) == 30
                ''',
        },
        hidden_test='''\
            from calendar_utils.dates import days_in_month, is_leap_year


            def test_divisible_by_400_is_leap():
                assert is_leap_year(2000)
                assert days_in_month(2000, 2) == 29
                assert not is_leap_year(2100)
            ''',
        p2p=["tests/test_dates.py::test_common_years", "tests/test_dates.py::test_regular_leap_year"],
        steps=[
            ("Find the leap-year logic.", B("grep -rn 'leap' --include=*.py .")),
            ("Read dates.py.", V("calendar_utils/dates.py", 1, 3)),
            ("The Gregorian rule is missing the 'divisible by 400' exception. Add it.",
             R("calendar_utils/dates.py",
               "    return year % 4 == 0 and year % 100 != 0",
               "    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0")),
            ("Check 2000, 1900, 2024, 2100.",
             B("python -c 'from calendar_utils.dates import is_leap_year as f; print([f(y) for y in (2000,1900,2024,2100)])'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 4 ----------------------------------------------------------------------------------------
    dict(
        id="seed-004-mean-integer-division",
        bug_type="wrong-operator", difficulty="easy",
        issue="""\
            mean() truncates results

            `stats.metrics.mean([1, 2])` returns `1` instead of `1.5`. The dashboard shows
            whole-number averages for latency, which is misleading.
            """,
        files={
            "stats/__init__.py": "",
            "stats/metrics.py": '''\
                def mean(values):
                    if not values:
                        raise ValueError("mean() of empty sequence")
                    return sum(values) // len(values)


                def median(values):
                    s = sorted(values)
                    n = len(s)
                    if n == 0:
                        raise ValueError("median() of empty sequence")
                    mid = n // 2
                    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
                ''',
            "tests/test_metrics.py": '''\
                import pytest
                from stats.metrics import mean, median


                def test_mean_of_evenly_divisible():
                    assert mean([2, 4, 6]) == 4


                def test_median():
                    assert median([3, 1, 2]) == 2
                    assert median([1, 2, 3, 4]) == 2.5


                def test_mean_empty():
                    with pytest.raises(ValueError):
                        mean([])
                ''',
        },
        hidden_test='''\
            import pytest

            from stats.metrics import mean


            def test_mean_is_not_truncated():
                assert mean([1, 2]) == 1.5
                assert mean([0.1, 0.2, 0.3]) == pytest.approx(0.2)
            ''',
        p2p=["tests/test_metrics.py::test_mean_of_evenly_divisible", "tests/test_metrics.py::test_median",
             "tests/test_metrics.py::test_mean_empty"],
        steps=[
            ("Open the metrics module.", V("stats/metrics.py")),
            ("`//` is floor division; mean needs true division.",
             R("stats/metrics.py", "return sum(values) // len(values)", "return sum(values) / len(values)")),
            ("Verify.", B("python -c 'from stats.metrics import mean; print(mean([1,2]), mean([2,4,6]))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 5 ----------------------------------------------------------------------------------------
    dict(
        id="seed-005-config-partial-override",
        bug_type="missing-default-merge", difficulty="medium",
        issue="""\
            KeyError: 'retries' when overriding only the timeout

            ```
            >>> from app.client import Client
            >>> Client({"timeout": 5}).fetch("x")
            KeyError: 'retries'
            ```
            Passing any override seems to drop all the other default settings.
            """,
        files={
            "app/__init__.py": "",
            "app/config.py": '''\
                DEFAULTS = {"timeout": 30, "retries": 3, "base_url": "https://api.example.com"}


                def load_config(overrides=None):
                    cfg = overrides or dict(DEFAULTS)
                    return cfg
                ''',
            "app/client.py": '''\
                from .config import load_config


                class Client:
                    def __init__(self, overrides=None):
                        self.cfg = load_config(overrides)

                    def fetch(self, path):
                        url = f"{self.cfg['base_url']}/{path}"
                        return {"url": url, "timeout": self.cfg["timeout"], "retries": self.cfg["retries"]}
                ''',
            "tests/test_config.py": '''\
                from app.config import DEFAULTS, load_config


                def test_no_overrides_returns_defaults():
                    assert load_config() == DEFAULTS


                def test_defaults_not_mutated():
                    load_config()["timeout"] = 1
                    assert DEFAULTS["timeout"] == 30
                ''',
        },
        hidden_test='''\
            from app.client import Client
            from app.config import DEFAULTS, load_config


            def test_partial_override_keeps_defaults():
                cfg = load_config({"timeout": 5})
                assert cfg == {**DEFAULTS, "timeout": 5}


            def test_client_with_override():
                assert Client({"timeout": 5}).fetch("x")["retries"] == 3


            def test_overrides_dict_not_mutated():
                o = {"timeout": 5}
                load_config(o)
                assert o == {"timeout": 5}
            ''',
        p2p=["tests/test_config.py::test_no_overrides_returns_defaults", "tests/test_config.py::test_defaults_not_mutated"],
        steps=[
            ("Trace where the config dict is built.", B("grep -rn 'load_config\\|DEFAULTS' --include=*.py app")),
            ("Read load_config.", V("app/config.py")),
            ("`overrides or dict(DEFAULTS)` replaces defaults entirely when overrides is non-empty. Reproduce.",
             B("python -c 'from app.config import load_config; print(load_config({\"timeout\": 5}))'")),
            ("Merge: start from a copy of DEFAULTS and update with overrides (without mutating either).",
             R("app/config.py",
               "    cfg = overrides or dict(DEFAULTS)\n",
               "    cfg = dict(DEFAULTS)\n    cfg.update(overrides or {})\n")),
            ("Re-run the original failing call.",
             B("python -c 'from app.client import Client; print(Client({\"timeout\": 5}).fetch(\"x\"))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 6 ----------------------------------------------------------------------------------------
    dict(
        id="seed-006-slugify-case",
        bug_type="normalization", difficulty="easy",
        issue="""\
            Slugs are case-sensitive, causing duplicate URLs

            `slugify("Hello World")` returns `Hello-World`, while `slugify("hello world")`
            returns `hello-world`. Both posts end up with different URLs for the same title.
            Slugs should always be lowercase.
            """,
        files={
            "blog/__init__.py": "",
            "blog/slug.py": '''\
                import re


                def slugify(title):
                    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title)
                    return slug.strip("-")
                ''',
            "blog/posts.py": '''\
                from .slug import slugify


                def make_url(title, post_id):
                    return f"/posts/{post_id}/{slugify(title)}"
                ''',
            "tests/test_slug.py": '''\
                from blog.slug import slugify


                def test_strips_punctuation():
                    assert slugify("hello, world!") == "hello-world"


                def test_collapses_separators():
                    assert slugify("a  --  b") == "a-b"
                ''',
        },
        hidden_test='''\
            from blog.posts import make_url
            from blog.slug import slugify


            def test_slug_is_lowercase():
                assert slugify("Hello World") == "hello-world"
                assert make_url("Python 3.13 Released", 7) == "/posts/7/python-3-13-released"
            ''',
        p2p=["tests/test_slug.py::test_strips_punctuation", "tests/test_slug.py::test_collapses_separators"],
        steps=[
            ("Open slugify.", V("blog/slug.py")),
            ("No case normalization happens. Lowercase before replacing.",
             R("blog/slug.py", 'slug = re.sub(r"[^a-zA-Z0-9]+", "-", title)',
               'slug = re.sub(r"[^a-z0-9]+", "-", title.lower())')),
            ("Verify.", B("python -c 'from blog.slug import slugify; print(slugify(\"Hello World\"))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 7 ----------------------------------------------------------------------------------------
    dict(
        id="seed-007-binary-search-bounds",
        bug_type="loop-bound", difficulty="medium",
        issue="""\
            find_index misses the last element

            `find_index([1, 3, 5], 5)` returns -1, and `find_index([5], 5)` also returns -1.
            Looks like elements at the end of the search window are never checked.
            """,
        files={
            "search/__init__.py": "",
            "search/bisect_utils.py": '''\
                def find_index(sorted_items, target):
                    """Return the index of target in sorted_items, or -1 if absent."""
                    lo, hi = 0, len(sorted_items) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if sorted_items[mid] == target:
                            return mid
                        if sorted_items[mid] < target:
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    return -1
                ''',
            "tests/test_bisect.py": '''\
                from search.bisect_utils import find_index


                def test_finds_middle():
                    assert find_index([1, 3, 5], 3) == 1


                def test_absent():
                    assert find_index([1, 3, 5], 4) == -1
                    assert find_index([], 1) == -1
                ''',
        },
        hidden_test='''\
            from search.bisect_utils import find_index


            def test_finds_edges():
                xs = [1, 3, 5, 7, 9]
                assert [find_index(xs, x) for x in xs] == [0, 1, 2, 3, 4]
                assert find_index([5], 5) == 0
            ''',
        p2p=["tests/test_bisect.py::test_finds_middle", "tests/test_bisect.py::test_absent"],
        steps=[
            ("Read the search function.", V("search/bisect_utils.py")),
            ("Check every position of a 5-element list to see which ones are missed.",
             B("python -c 'from search.bisect_utils import find_index as f; xs=[1,3,5,7,9]; print([f(xs,x) for x in xs], f([5],5))'")),
            ("With an inclusive `hi`, the loop must run while lo <= hi; otherwise the final candidate is skipped.",
             R("search/bisect_utils.py", "    while lo < hi:", "    while lo <= hi:")),
            ("Re-check, including absent values and empty list.",
             B("python -c 'from search.bisect_utils import find_index as f; xs=[1,3,5,7,9]; print([f(xs,x) for x in xs], f([5],5), f(xs,4), f([],1))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 8 ----------------------------------------------------------------------------------------
    dict(
        id="seed-008-report-date-range-inclusive",
        bug_type="off-by-one", difficulty="medium",
        issue="""\
            Daily sales report is missing the last day

            Running the report for 2024-03-01..2024-03-03 only returns rows for the 1st and
            2nd. The end date should be included (the CLI help says "inclusive").
            """,
        files={
            "reports/__init__.py": "",
            "reports/ranges.py": '''\
                from datetime import timedelta


                def date_range(start, end):
                    """Yield each date from start to end, inclusive."""
                    d = start
                    while d < end:
                        yield d
                        d += timedelta(days=1)
                ''',
            "reports/daily.py": '''\
                from collections import defaultdict

                from .ranges import date_range


                def daily_totals(records, start, end):
                    """records: iterable of (date, amount). Returns {date: total} for every day in range."""
                    totals = defaultdict(float)
                    for day, amount in records:
                        totals[day] += amount
                    return {day: totals.get(day, 0.0) for day in date_range(start, end)}
                ''',
            "tests/test_daily.py": '''\
                from datetime import date

                from reports.daily import daily_totals


                def test_sums_per_day():
                    recs = [(date(2024, 3, 1), 5.0), (date(2024, 3, 1), 2.5)]
                    out = daily_totals(recs, date(2024, 3, 1), date(2024, 3, 2))
                    assert out[date(2024, 3, 1)] == 7.5
                ''',
        },
        hidden_test='''\
            from datetime import date

            from reports.daily import daily_totals
            from reports.ranges import date_range


            def test_range_includes_end():
                assert list(date_range(date(2024, 3, 1), date(2024, 3, 3)))[-1] == date(2024, 3, 3)
                assert list(date_range(date(2024, 3, 1), date(2024, 3, 1))) == [date(2024, 3, 1)]


            def test_report_includes_last_day():
                recs = [(date(2024, 3, 3), 4.0)]
                out = daily_totals(recs, date(2024, 3, 1), date(2024, 3, 3))
                assert out == {date(2024, 3, 1): 0.0, date(2024, 3, 2): 0.0, date(2024, 3, 3): 4.0}
            ''',
        p2p=["tests/test_daily.py::test_sums_per_day"],
        steps=[
            ("Start from the report entry point.", V("reports/daily.py")),
            ("The day list comes from date_range; check it.", V("reports/ranges.py")),
            ("Docstring says inclusive but the loop uses `<`. Reproduce.",
             B("python -c 'from datetime import date; from reports.ranges import date_range; print(list(date_range(date(2024,3,1), date(2024,3,3))))'")),
            ("Make the bound inclusive to match the documented contract.",
             R("reports/ranges.py", "    while d < end:", "    while d <= end:")),
            ("Re-run.",
             B("python -c 'from datetime import date; from reports.ranges import date_range; print(list(date_range(date(2024,3,1), date(2024,3,3))))'")),
            ("Check other callers of date_range rely on inclusive semantics too.", B("grep -rn 'date_range' --include=*.py .")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 9 ----------------------------------------------------------------------------------------
    dict(
        id="seed-009-retry-swallows-exception",
        bug_type="error-handling", difficulty="medium",
        issue="""\
            @retry returns None instead of raising after the last attempt

            When every attempt fails, functions decorated with `@retry(times=3)` silently
            return `None`. Callers then crash later with `TypeError: 'NoneType' object is not
            subscriptable`, hiding the real error. The original exception should propagate.
            """,
        files={
            "net/__init__.py": "",
            "net/retry.py": '''\
                import functools
                import time


                def retry(times=3, delay=0.0, exceptions=(Exception,)):
                    def deco(fn):
                        @functools.wraps(fn)
                        def wrapper(*args, **kwargs):
                            for attempt in range(times):
                                try:
                                    return fn(*args, **kwargs)
                                except exceptions:
                                    if attempt == times - 1:
                                        break
                                    time.sleep(delay)
                        return wrapper
                    return deco
                ''',
            "tests/test_retry.py": '''\
                from net.retry import retry


                def test_succeeds_after_transient_failures():
                    calls = []

                    @retry(times=3)
                    def flaky():
                        calls.append(1)
                        if len(calls) < 3:
                            raise ConnectionError("boom")
                        return "ok"

                    assert flaky() == "ok"
                    assert len(calls) == 3
                ''',
        },
        hidden_test='''\
            import pytest

            from net.retry import retry


            def test_reraises_after_last_attempt():
                calls = []

                @retry(times=2)
                def always_fails():
                    calls.append(1)
                    raise ConnectionError("down")

                with pytest.raises(ConnectionError, match="down"):
                    always_fails()
                assert len(calls) == 2


            def test_unlisted_exception_not_retried():
                calls = []

                @retry(times=3, exceptions=(ConnectionError,))
                def bad():
                    calls.append(1)
                    raise KeyError("k")

                with pytest.raises(KeyError):
                    bad()
                assert len(calls) == 1
            ''',
        p2p=["tests/test_retry.py::test_succeeds_after_transient_failures"],
        steps=[
            ("Read the decorator.", V("net/retry.py")),
            ("On the final attempt it `break`s out of the loop and falls through to an implicit `return None`. Reproduce.",
             B("python -c '\nfrom net.retry import retry\n@retry(times=2)\ndef f(): raise ConnectionError(\"down\")\nprint(repr(f()))'")),
            ("Re-raise the caught exception on the last attempt so the original traceback propagates.",
             R("net/retry.py",
               "                    if attempt == times - 1:\n                        break\n",
               "                    if attempt == times - 1:\n                        raise\n")),
            ("Re-run the repro; expect ConnectionError.",
             B("python -c '\nfrom net.retry import retry\n@retry(times=2)\ndef f(): raise ConnectionError(\"down\")\nprint(repr(f()))' 2>&1 | tail -1")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 10 ---------------------------------------------------------------------------------------
    dict(
        id="seed-010-lru-get-does-not-refresh",
        bug_type="state-update", difficulty="medium",
        issue="""\
            LRUCache evicts keys that were just read

            With capacity 2:
                c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
            `c.get("a")` now returns None, but "b" was the least recently used key and should
            have been evicted instead.
            """,
        files={
            "cache/__init__.py": "from .lru import LRUCache\n",
            "cache/lru.py": '''\
                from collections import OrderedDict


                class LRUCache:
                    def __init__(self, capacity):
                        if capacity < 1:
                            raise ValueError("capacity must be >= 1")
                        self.capacity = capacity
                        self._data = OrderedDict()

                    def get(self, key, default=None):
                        if key not in self._data:
                            return default
                        return self._data[key]

                    def put(self, key, value):
                        if key in self._data:
                            self._data.move_to_end(key)
                        self._data[key] = value
                        if len(self._data) > self.capacity:
                            self._data.popitem(last=False)

                    def __len__(self):
                        return len(self._data)
                ''',
            "tests/test_lru.py": '''\
                from cache import LRUCache


                def test_evicts_oldest_insert():
                    c = LRUCache(2)
                    c.put("a", 1); c.put("b", 2); c.put("c", 3)
                    assert c.get("a") is None and c.get("c") == 3
                    assert len(c) == 2


                def test_put_refreshes():
                    c = LRUCache(2)
                    c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3)
                    assert c.get("a") == 10 and c.get("b") is None
                ''',
        },
        hidden_test='''\
            from cache import LRUCache


            def test_get_refreshes_recency():
                c = LRUCache(2)
                c.put("a", 1); c.put("b", 2)
                assert c.get("a") == 1
                c.put("c", 3)
                assert c.get("a") == 1
                assert c.get("b") is None


            def test_get_missing_does_not_insert():
                c = LRUCache(1)
                assert c.get("zzz", 0) == 0
                assert len(c) == 0
            ''',
        p2p=["tests/test_lru.py::test_evicts_oldest_insert", "tests/test_lru.py::test_put_refreshes"],
        steps=[
            ("Open the cache implementation.", V("cache/lru.py")),
            ("Reproduce the scenario from the issue.",
             B("python -c 'from cache import LRUCache; c=LRUCache(2); c.put(\"a\",1); c.put(\"b\",2); c.get(\"a\"); c.put(\"c\",3); print(c.get(\"a\"), c.get(\"b\"))'")),
            ("`get` never updates recency, so reads don't protect a key from eviction. Move the key to the end on a hit.",
             R("cache/lru.py",
               "            return default\n        return self._data[key]\n",
               "            return default\n        self._data.move_to_end(key)\n        return self._data[key]\n")),
            ("Re-run the scenario.",
             B("python -c 'from cache import LRUCache; c=LRUCache(2); c.put(\"a\",1); c.put(\"b\",2); c.get(\"a\"); c.put(\"c\",3); print(c.get(\"a\"), c.get(\"b\"))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
]


# --------------------------------------------------------------------------------------------
def write_repo(repo_dir: Path, files: dict[str, str]) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    for rel, content in files.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d(content) if content else "")


def make_test_patch(repo_dir: Path, rel: str, content: str) -> str:
    """Diff that adds the hidden test file, relative to the repo root."""
    with tempfile.TemporaryDirectory() as tmp:
        new = Path(tmp) / rel
        new.parent.mkdir(parents=True)
        new.write_text(content)
        r = subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", rel],
                           cwd=tmp, capture_output=True, text=True)
        assert f"+++ b/{rel}" in r.stdout, r.stdout[:200]
        return r.stdout


def hidden_test_ids(rel: str, content: str) -> list[str]:
    return [f"{rel}::{line.split('(')[0][4:]}" for line in content.splitlines() if line.startswith("def test_")]


def build() -> list[Demo]:
    demos = []
    for spec in SPECS:
        repo_rel = f"data/repos/{spec['id']}"
        repo_dir = ROOT / repo_rel
        write_repo(repo_dir, spec["files"])
        hidden_rel = f"tests/test_issue_{spec['id'].split('-')[1]}.py"
        hidden = d(spec["hidden_test"])
        issue = d(spec["issue"])

        with RepoEnv(repo_dir) as env:
            res = run_agent(ScriptedPolicy(spec["steps"]), env, issue, TEST_CMD)
        assert res.submitted, spec["id"]
        for s in res.trajectory:
            if s.observation.startswith("Error:") or "command not found" in s.observation:
                raise RuntimeError(f"{spec['id']}: step failed: {s.action} -> {s.observation}")

        # Split hidden tests the SWE-bench way: tests that already pass on the buggy repo are
        # regression guards (PASS_TO_PASS), the rest must flip (FAIL_TO_PASS).
        test_patch = make_test_patch(repo_dir, hidden_rel, hidden)
        with RepoEnv(repo_dir) as env:
            env.apply_patch(test_patch)
            before = _run_tests(env, TEST_CMD, hidden_test_ids(hidden_rel, hidden))
        f2p = [t for t, ok in before.items() if not ok]
        p2p = spec["p2p"] + [t for t, ok in before.items() if ok]
        assert f2p, f"{spec['id']}: no hidden test fails on the buggy repo"

        files_changed = sorted({l[6:] for l in res.patch.splitlines() if l.startswith("+++ b/")})
        demos.append(Demo(
            instance_id=spec["id"],
            repo=repo_rel,
            base_commit=None,
            problem_statement=issue,
            test_cmd=TEST_CMD,
            test_patch=test_patch,
            FAIL_TO_PASS=f2p,
            PASS_TO_PASS=p2p,
            patch=res.patch,
            trajectory=res.trajectory,
            metadata={
                "source": "engineer_seed",
                "bug_type": spec["bug_type"],
                "difficulty": spec["difficulty"],
                "files_changed": files_changed,
                "num_steps": len(res.trajectory),
                "language": "python",
            },
        ))
        print(f"built {spec['id']}: {len(res.trajectory)} steps, files={files_changed}")
    return demos


if __name__ == "__main__":
    out = ROOT / "data/demos/seed.jsonl"
    save_demos(build(), out)
    print(f"wrote {out.relative_to(ROOT)}")
