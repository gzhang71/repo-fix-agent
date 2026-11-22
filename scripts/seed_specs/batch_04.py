"""Seed instances 056-070: Python object model & language semantics."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V

SPECS = [
    # 56 ---------------------------------------------------------------------------------------
    dict(
        id="seed-056-tag-hash-ignores-case",
        bug_type="eq-hash-inconsistency", difficulty="easy",
        issue="""\
            Duplicate tags when the same tag is typed with different casing

            Tags are supposed to be case-insensitive (the `Tag` docstring says so, and
            `Tag("Python") == Tag("python")` is True), but `parse_tags("Python, python, PYTHON")`
            returns three tags, and `tag_counts()` reports `Python` and `python` as separate
            entries on the dashboard. We expect one tag, keeping the first spelling.
            """,
        files={
            "tags/__init__.py": "from .model import Tag\n",
            "tags/model.py": '''\
                class Tag:
                    """A label attached to articles. Tags are case-insensitive: "Python" == "python"."""

                    def __init__(self, name):
                        name = name.strip()
                        if not name:
                            raise ValueError("tag name must not be empty")
                        self.name = name

                    def __eq__(self, other):
                        if not isinstance(other, Tag):
                            return NotImplemented
                        return self.name.lower() == other.name.lower()

                    def __hash__(self):
                        return hash(self.name)

                    def __repr__(self):
                        return f"Tag({self.name!r})"
                ''',
            "tags/collection.py": '''\
                from .model import Tag


                def parse_tags(raw):
                    """Parse a comma-separated tag string, keeping the first spelling of each tag."""
                    seen = set()
                    out = []
                    for part in raw.split(","):
                        if not part.strip():
                            continue
                        tag = Tag(part)
                        if tag not in seen:
                            seen.add(tag)
                            out.append(tag)
                    return out


                def tag_counts(articles):
                    """articles: iterable of lists of Tag. Returns {Tag: count}."""
                    counts = {}
                    for tags in articles:
                        for tag in tags:
                            counts[tag] = counts.get(tag, 0) + 1
                    return counts
                ''',
            "tests/test_tags.py": '''\
                import pytest

                from tags import Tag
                from tags.collection import parse_tags


                def test_equality_ignores_case():
                    assert Tag("Python") == Tag("python")
                    assert Tag("go") != Tag("rust")


                def test_parse_skips_blanks_and_exact_dupes():
                    assert [t.name for t in parse_tags("a, b,,a")] == ["a", "b"]


                def test_empty_name_rejected():
                    with pytest.raises(ValueError):
                        Tag("   ")
                ''',
        },
        hidden_test='''\
            from tags import Tag
            from tags.collection import parse_tags, tag_counts


            def test_parse_dedups_case_insensitively():
                assert [t.name for t in parse_tags("Python, python, PYTHON, rust")] == ["Python", "rust"]


            def test_hash_consistent_with_eq():
                assert hash(Tag("Go")) == hash(Tag("go"))
                assert len({Tag("Go"), Tag("GO"), Tag("go")}) == 1


            def test_counts_merge_casings():
                counts = tag_counts([[Tag("Python")], [Tag("python"), Tag("rust")]])
                assert counts == {Tag("python"): 2, Tag("rust"): 1}


            def test_distinct_tags_stay_distinct():
                assert len({Tag("go"), Tag("rust")}) == 2
            ''',
        p2p=["tests/test_tags.py::test_equality_ignores_case",
             "tests/test_tags.py::test_parse_skips_blanks_and_exact_dupes",
             "tests/test_tags.py::test_empty_name_rejected"],
        steps=[
            ("Find where tags are parsed and deduplicated.", S("def (parse_tags|tag_counts)", "tags")),
            ("Dedup relies on a set of Tag objects, so look at how Tag defines equality and hashing.",
             V("tags/model.py")),
            ("__eq__ lowercases but __hash__ hashes the raw name, so equal tags land in different hash buckets. Confirm.",
             B("python -c 'from tags import Tag; from tags.collection import parse_tags; "
               "print(Tag(\"Go\") == Tag(\"go\"), hash(Tag(\"Go\")) == hash(Tag(\"go\")), parse_tags(\"Python, python\"))'")),
            ("Hash the same normalized key that __eq__ compares.",
             R("tags/model.py", "        return hash(self.name)", "        return hash(self.name.lower())")),
            ("Re-run the repro.",
             B("python -c 'from tags import Tag; from tags.collection import parse_tags; "
               "print(hash(Tag(\"Go\")) == hash(Tag(\"go\")), parse_tags(\"Python, python, PYTHON, rust\"))'")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 57 ---------------------------------------------------------------------------------------
    dict(
        id="seed-057-mixin-bypasses-mro",
        bug_type="super-mro-bypass", difficulty="hard",
        issue="""\
            Orders never show up in the audit log

            After `Customer(store, id=1).save()` there's a `("customers", 1)` entry in
            `store["audit_log"]`, as expected. But `Order(store, id=7, customer_id=1).save()`
            writes the order to `store["orders"]` and nothing to the audit log, even though
            `Order` lists `AuditMixin` in its bases. Compliance needs every order save audited.
            """,
        files={
            "store/__init__.py": "",
            "store/base.py": '''\
                class Model:
                    """Minimal persistence base. `save()` writes the record to the backing store."""

                    table = None

                    def __init__(self, store, **fields):
                        self.store = store
                        self.fields = dict(fields)

                    def save(self):
                        self.store.setdefault(self.table, []).append(dict(self.fields))
                        return self
                ''',
            "store/mixins.py": '''\
                from .base import Model


                class AuditMixin:
                    """Records every save in store["audit_log"] as (table, id)."""

                    def save(self):
                        result = super().save()
                        self.store.setdefault("audit_log", []).append((self.table, self.fields.get("id")))
                        return result


                class ValidationMixin:
                    """Checks that every field named in `required` is present before saving."""

                    required = ()

                    def save(self):
                        missing = [f for f in self.required if self.fields.get(f) in (None, "")]
                        if missing:
                            raise ValueError(f"{type(self).__name__} missing required fields: {', '.join(missing)}")
                        return Model.save(self)
                ''',
            "store/models.py": '''\
                from .base import Model
                from .mixins import AuditMixin, ValidationMixin


                class Customer(AuditMixin, Model):
                    table = "customers"


                class Order(ValidationMixin, AuditMixin, Model):
                    table = "orders"
                    required = ("id", "customer_id")


                class Note(ValidationMixin, Model):
                    table = "notes"
                    required = ("text",)
                ''',
            "tests/test_models.py": '''\
                import pytest

                from store.models import Customer, Note, Order


                def test_customer_save_is_audited():
                    store = {}
                    Customer(store, id=1, name="Ann").save()
                    assert store["customers"] == [{"id": 1, "name": "Ann"}]
                    assert store["audit_log"] == [("customers", 1)]


                def test_order_requires_customer():
                    with pytest.raises(ValueError, match="customer_id"):
                        Order({}, id=7).save()


                def test_note_saved():
                    store = {}
                    Note(store, text="hi").save()
                    assert store == {"notes": [{"text": "hi"}]}
                ''',
        },
        hidden_test='''\
            import pytest

            from store.models import Customer, Note, Order


            def test_order_save_is_audited():
                store = {}
                Order(store, id=7, customer_id=1).save()
                assert store["orders"] == [{"id": 7, "customer_id": 1}]
                assert store["audit_log"] == [("orders", 7)]


            def test_invalid_order_not_saved_or_audited():
                store = {}
                with pytest.raises(ValueError):
                    Order(store, id=8).save()
                assert store == {}


            def test_unaudited_model_still_unaudited():
                store = {}
                Note(store, text="x").save()
                assert "audit_log" not in store


            def test_save_returns_instance():
                o = Order({}, id=1, customer_id=2)
                assert o.save() is o
                c = Customer({}, id=3)
                assert c.save() is c
            ''',
        p2p=["tests/test_models.py::test_customer_save_is_audited",
             "tests/test_models.py::test_order_requires_customer",
             "tests/test_models.py::test_note_saved"],
        steps=[
            ("Look at how the models compose their mixins.", V("store/models.py")),
            ("Order's MRO should be Order -> ValidationMixin -> AuditMixin -> Model. Check what each save() calls.",
             V("store/mixins.py")),
            ("Reproduce and print the MRO to confirm AuditMixin sits between ValidationMixin and Model.",
             B("python - <<'EOF'\n"
               "from store.models import Order\n"
               "print([c.__name__ for c in Order.__mro__])\n"
               "store = {}\n"
               "Order(store, id=7, customer_id=1).save()\n"
               "print(store)\n"
               "EOF")),
            ("ValidationMixin calls Model.save(self) directly, which jumps over AuditMixin in the MRO. "
             "Use cooperative super() so the next class in the chain runs.",
             R("store/mixins.py", "        return Model.save(self)", "        return super().save()")),
            ("The direct Model import is now unused; drop it.",
             R("store/mixins.py", "from .base import Model\n\n\n", "")),
            ("Re-run the repro.",
             B("python - <<'EOF'\n"
               "from store.models import Order, Note\n"
               "store = {}\n"
               "Order(store, id=7, customer_id=1).save()\n"
               "Note(store, text='x').save()\n"
               "print(store)\n"
               "EOF")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 58 ---------------------------------------------------------------------------------------
    dict(
        id="seed-058-dataclass-order-field-sequence",
        bug_type="dataclass-field-ordering", difficulty="medium",
        issue="""\
            JobQueue ignores priority and runs jobs alphabetically

                q = JobQueue()
                q.submit("zeta-backup", priority=1)
                q.submit("alpha-report", priority=9)
                q.pop().name   # 'alpha-report', expected 'zeta-backup'

            Lower priority numbers are supposed to run first, and jobs with the same priority
            should run in the order they were submitted. In production our nightly backup keeps
            getting starved by anything whose name sorts earlier.
            """,
        files={
            "jobs/__init__.py": "",
            "jobs/model.py": '''\
                from dataclasses import dataclass


                @dataclass(order=True)
                class Job:
                    """A unit of work. Lower `priority` runs first; ties run in submission order (`seq`)."""

                    name: str
                    priority: int
                    seq: int = 0
                ''',
            "jobs/queue.py": '''\
                import heapq
                import itertools

                from .model import Job


                class JobQueue:
                    def __init__(self):
                        self._heap = []
                        self._counter = itertools.count()

                    def submit(self, name, priority=5):
                        job = Job(name, priority, next(self._counter))
                        heapq.heappush(self._heap, job)
                        return job

                    def pop(self):
                        if not self._heap:
                            raise IndexError("pop from empty JobQueue")
                        return heapq.heappop(self._heap)

                    def drain(self):
                        out = []
                        while self._heap:
                            out.append(self.pop().name)
                        return out

                    def __len__(self):
                        return len(self._heap)
                ''',
            "tests/test_queue.py": '''\
                import pytest

                from jobs.queue import JobQueue


                def test_submit_assigns_sequence():
                    q = JobQueue()
                    a = q.submit("a")
                    b = q.submit("b")
                    assert (a.seq, b.seq) == (0, 1)
                    assert len(q) == 2


                def test_pop_empty_raises():
                    with pytest.raises(IndexError):
                        JobQueue().pop()


                def test_drain_empties_queue():
                    q = JobQueue()
                    q.submit("only", priority=3)
                    assert q.drain() == ["only"]
                    assert len(q) == 0
                ''',
        },
        hidden_test='''\
            from jobs.model import Job
            from jobs.queue import JobQueue


            def test_lower_priority_number_runs_first():
                q = JobQueue()
                q.submit("zeta-backup", priority=1)
                q.submit("alpha-report", priority=9)
                q.submit("mid", priority=5)
                assert q.drain() == ["zeta-backup", "mid", "alpha-report"]


            def test_ties_run_in_submission_order():
                q = JobQueue()
                for name in ["c", "b", "a"]:
                    q.submit(name, priority=2)
                assert q.drain() == ["c", "b", "a"]


            def test_job_repr_and_fields():
                j = Job("x", 3, 4)
                assert (j.name, j.priority, j.seq) == ("x", 3, 4)
                assert "x" in repr(j)
            ''',
        p2p=["tests/test_queue.py::test_submit_assigns_sequence", "tests/test_queue.py::test_pop_empty_raises",
             "tests/test_queue.py::test_drain_empties_queue"],
        steps=[
            ("Start at the queue implementation.", V("jobs/queue.py")),
            ("Ordering comes entirely from Job's comparison methods, since heapq compares the objects. Check Job.",
             V("jobs/model.py")),
            ("order=True compares fields as a tuple in declaration order, so `name` is compared before `priority`. Reproduce.",
             B("python -c 'from jobs.queue import JobQueue; q=JobQueue(); q.submit(\"zeta-backup\", 1); "
               "q.submit(\"alpha-report\", 9); print(q.drain())'")),
            ("Exclude `name` from comparisons so ordering is (priority, seq). This keeps the positional constructor unchanged.",
             R("jobs/model.py",
               "from dataclasses import dataclass\n",
               "from dataclasses import dataclass, field\n")),
            ("Apply the field change.",
             R("jobs/model.py", "    name: str\n", "    name: str = field(compare=False)\n")),
            ("Re-run the repro plus a tie case.",
             B("python -c 'from jobs.queue import JobQueue; q=JobQueue(); q.submit(\"zeta-backup\", 1); "
               "q.submit(\"alpha-report\", 9); [q.submit(n, 2) for n in \"cba\"]; print(q.drain())'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 59 ---------------------------------------------------------------------------------------
    dict(
        id="seed-059-init-bypasses-property-setter",
        bug_type="property-setter-bypass", difficulty="easy",
        issue="""\
            Thermostat accepts out-of-range targets from saved profiles

            `t.target = 95` correctly clamps to 30.0, but loading a profile with
            `from_profile({"target": 95})` gives a thermostat whose `target` is 95.0, and it
            then heats forever. Same with negative values. The supported range is 10-30.
            """,
        files={
            "climate/__init__.py": "",
            "climate/thermostat.py": '''\
                MIN_TEMP = 10.0
                MAX_TEMP = 30.0


                class Thermostat:
                    def __init__(self, target=20.0, mode="heat"):
                        self._target = float(target)
                        self.mode = mode

                    @property
                    def target(self):
                        return self._target

                    @target.setter
                    def target(self, value):
                        """Set the target temperature, clamped to the supported range."""
                        self._target = min(MAX_TEMP, max(MIN_TEMP, float(value)))

                    def needs_heating(self, current):
                        return self.mode == "heat" and current < self.target - 0.5
                ''',
            "climate/profiles.py": '''\
                from .thermostat import Thermostat


                def from_profile(profile):
                    """Build a thermostat from a saved profile dict."""
                    return Thermostat(target=profile.get("target", 20.0), mode=profile.get("mode", "heat"))
                ''',
            "tests/test_thermostat.py": '''\
                from climate.profiles import from_profile
                from climate.thermostat import Thermostat


                def test_setter_clamps():
                    t = Thermostat()
                    t.target = 95
                    assert t.target == 30.0
                    t.target = -4
                    assert t.target == 10.0


                def test_needs_heating():
                    t = Thermostat(target=21)
                    assert t.needs_heating(19.0)
                    assert not t.needs_heating(20.8)


                def test_profile_defaults():
                    t = from_profile({})
                    assert t.target == 20.0 and t.mode == "heat"
                ''',
        },
        hidden_test='''\
            from climate.profiles import from_profile
            from climate.thermostat import Thermostat


            def test_constructor_clamps():
                assert Thermostat(target=95).target == 30.0
                assert Thermostat(target=-5).target == 10.0


            def test_profile_clamps():
                assert from_profile({"target": 95, "mode": "cool"}).target == 30.0


            def test_in_range_value_kept_as_float():
                t = Thermostat(target=21)
                assert t.target == 21.0 and isinstance(t.target, float)
            ''',
        p2p=["tests/test_thermostat.py::test_setter_clamps", "tests/test_thermostat.py::test_needs_heating",
             "tests/test_thermostat.py::test_profile_defaults"],
        steps=[
            ("Find where the clamping happens.", S("MAX_TEMP|_target", "climate")),
            ("Read the class.", V("climate/thermostat.py")),
            ("__init__ writes `_target` directly, skipping the clamping setter. Reproduce via the profile loader.",
             B("python -c 'from climate.profiles import from_profile; print(from_profile({\"target\": 95}).target)'")),
            ("Route the constructor through the property so the same validation applies.",
             R("climate/thermostat.py", "        self._target = float(target)\n", "        self.target = target\n")),
            ("Verify both bounds.",
             B("python -c 'from climate.thermostat import Thermostat as T; print(T(95).target, T(-5).target, T(21).target)'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 60 ---------------------------------------------------------------------------------------
    dict(
        id="seed-060-class-level-handler-registry",
        bug_type="class-vs-instance-attribute", difficulty="medium",
        issue="""\
            Billing handlers fire on order events

            We create one `EventBus` per subsystem in `build_buses()` precisely so handlers
            don't cross-talk, but after `wire()` publishing `"created"` on the orders bus calls
            the billing handler too (and vice versa). `publish` also returns 2 instead of 1.
            """,
        files={
            "events/__init__.py": "",
            "events/bus.py": '''\
                class EventBus:
                    """Simple synchronous publish/subscribe bus."""

                    _handlers = {}

                    def subscribe(self, event, handler):
                        self._handlers.setdefault(event, []).append(handler)

                    def publish(self, event, payload=None):
                        handlers = self._handlers.get(event, [])
                        for handler in handlers:
                            handler(payload)
                        return len(handlers)
                ''',
            "events/app.py": '''\
                from .bus import EventBus


                def build_buses():
                    """Each subsystem gets its own bus so handlers don't cross-talk."""
                    return {"orders": EventBus(), "billing": EventBus()}


                def wire(buses, log):
                    buses["orders"].subscribe("created", lambda p: log.append(("orders", p)))
                    buses["billing"].subscribe("created", lambda p: log.append(("billing", p)))
                ''',
            "tests/test_bus.py": '''\
                from events.bus import EventBus


                def test_publish_calls_handlers_in_order():
                    bus, seen = EventBus(), []
                    bus.subscribe("ping", lambda p: seen.append(("first", p)))
                    bus.subscribe("ping", lambda p: seen.append(("second", p)))
                    assert bus.publish("ping", 1) == 2
                    assert seen == [("first", 1), ("second", 1)]


                def test_publish_without_handlers():
                    assert EventBus().publish("nobody-listens") == 0
                ''',
        },
        hidden_test='''\
            from events.app import build_buses, wire
            from events.bus import EventBus


            def test_buses_are_isolated():
                a, b = EventBus(), EventBus()
                a.subscribe("evt", lambda p: None)
                assert b.publish("evt") == 0
                assert a.publish("evt") == 1


            def test_wired_subsystems_do_not_cross_talk():
                log = []
                buses = build_buses()
                wire(buses, log)
                assert buses["orders"].publish("created", 42) == 1
                assert log == [("orders", 42)]
            ''',
        p2p=["tests/test_bus.py::test_publish_calls_handlers_in_order",
             "tests/test_bus.py::test_publish_without_handlers"],
        steps=[
            ("See how the buses are created and wired.", V("events/app.py")),
            ("Two separate instances still share handlers, so look for shared state in EventBus.", V("events/bus.py")),
            ("`_handlers` is a class attribute; setdefault mutates the one dict shared by every bus. Reproduce.",
             B("python -c 'from events.bus import EventBus; a=EventBus(); b=EventBus(); a.subscribe(\"x\", print); "
               "print(b.publish(\"x\", \"leaked\"), a._handlers is b._handlers)'")),
            ("Give each bus its own registry in __init__.",
             R("events/bus.py", "    _handlers = {}\n", "    def __init__(self):\n        self._handlers = {}\n")),
            ("Check nothing subclasses EventBus with its own __init__ that would skip this.",
             S("EventBus", ".", "*.py")),
            ("Re-run the repro.",
             B("python -c 'from events.bus import EventBus; a=EventBus(); b=EventBus(); a.subscribe(\"x\", print); "
               "print(b.publish(\"x\", \"leaked\"), a._handlers is b._handlers)'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 61 ---------------------------------------------------------------------------------------
    dict(
        id="seed-061-generator-consumed-twice",
        bug_type="iterator-exhaustion", difficulty="medium",
        issue="""\
            Latency report crashes on real log files

            ```
            >>> from metrics.report import report
            >>> report(open("api.log"))
            Traceback (most recent call last):
              File "metrics/report.py", line 6, in report
                s = summarize(read_latencies(lines))
              File "metrics/summary.py", line 7, in summarize
                slowest = max(samples, key=lambda s: s[1])
            ValueError: max() iterable argument is empty
            ```
            The log definitely has samples (`count` is right if I print it). `summarize()` on a
            plain list works fine in the unit tests.
            """,
        files={
            "metrics/__init__.py": "",
            "metrics/reader.py": '''\
                def read_latencies(lines):
                    """Parse 'endpoint,ms' log lines lazily, skipping blanks and comments."""
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        endpoint, ms = line.split(",")
                        yield endpoint.strip(), float(ms)
                ''',
            "metrics/summary.py": '''\
                def summarize(samples):
                    """samples: iterable of (endpoint, ms). Returns count, mean, max and the slowest endpoint."""
                    count = sum(1 for _ in samples)
                    if count == 0:
                        return {"count": 0, "mean": 0.0, "max": 0.0, "slowest": None}
                    total = sum(ms for _, ms in samples)
                    slowest = max(samples, key=lambda s: s[1])
                    return {"count": count, "mean": total / count, "max": slowest[1], "slowest": slowest[0]}
                ''',
            "metrics/report.py": '''\
                from .reader import read_latencies
                from .summary import summarize


                def report(lines):
                    s = summarize(read_latencies(lines))
                    if s["count"] == 0:
                        return "no samples"
                    return f"{s['count']} samples, mean {s['mean']:.1f}ms, max {s['max']:.1f}ms ({s['slowest']})"
                ''',
            "tests/test_metrics.py": '''\
                from metrics.reader import read_latencies
                from metrics.report import report
                from metrics.summary import summarize


                def test_reader_skips_comments():
                    rows = list(read_latencies(["# header", "", "/a, 12.5", "/b,3"]))
                    assert rows == [("/a", 12.5), ("/b", 3.0)]


                def test_summarize_list():
                    s = summarize([("/a", 10.0), ("/b", 30.0)])
                    assert s == {"count": 2, "mean": 20.0, "max": 30.0, "slowest": "/b"}


                def test_report_empty():
                    assert report(["# nothing yet"]) == "no samples"
                ''',
        },
        hidden_test='''\
            from metrics.report import report
            from metrics.summary import summarize


            def test_report_from_lines():
                lines = ["/a,10", "/b,30", "/c,20"]
                assert report(lines) == "3 samples, mean 20.0ms, max 30.0ms (/b)"


            def test_summarize_accepts_iterator():
                s = summarize(iter([("/x", 4.0), ("/y", 2.0)]))
                assert s == {"count": 2, "mean": 3.0, "max": 4.0, "slowest": "/x"}


            def test_summarize_empty_iterator():
                assert summarize(iter([]))["count"] == 0
            ''',
        p2p=["tests/test_metrics.py::test_reader_skips_comments", "tests/test_metrics.py::test_summarize_list",
             "tests/test_metrics.py::test_report_empty"],
        steps=[
            ("Follow the call from the traceback: report -> read_latencies -> summarize.", V("metrics/report.py")),
            ("read_latencies is a generator, so it can only be iterated once.", V("metrics/reader.py")),
            ("summarize iterates `samples` three times (count, total, max).", V("metrics/summary.py")),
            ("Reproduce with an iterator instead of a list.",
             B("python -c 'from metrics.report import report; print(report([\"/a,10\", \"/b,30\"]))' 2>&1 | tail -3")),
            ("Materialize the samples once at the top of summarize so any iterable works.",
             I("metrics/summary.py", 2, "    samples = list(samples)")),
            ("Re-run the repro.",
             B("python -c 'from metrics.report import report; print(report([\"/a,10\", \"/b,30\", \"/c,20\"]))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 62 ---------------------------------------------------------------------------------------
    dict(
        id="seed-062-contextmanager-no-finally",
        bug_type="context-manager-cleanup", difficulty="medium",
        issue="""\
            override_flags() leaks flag values when the block raises

            Our test suite became order-dependent. If an assertion fails inside
            `with override_flags(new_checkout=True): ...`, every test that runs afterwards sees
            `new_checkout` enabled. Overriding a flag that didn't exist before also leaves it
            behind in that case. The happy path restores everything correctly.
            """,
        files={
            "flags/__init__.py": "",
            "flags/registry.py": '''\
                _FLAGS = {"new_checkout": False, "dark_mode": False}
                _MISSING = object()


                def is_enabled(name):
                    return bool(_FLAGS.get(name, False))


                def set_flag(name, value):
                    _FLAGS[name] = bool(value)


                def snapshot():
                    return dict(_FLAGS)
                ''',
            "flags/testing.py": '''\
                from contextlib import contextmanager

                from . import registry


                @contextmanager
                def override_flags(**overrides):
                    """Temporarily set feature flags, restoring the previous values afterwards."""
                    previous = {name: registry._FLAGS.get(name, registry._MISSING) for name in overrides}
                    for name, value in overrides.items():
                        registry.set_flag(name, value)
                    yield
                    for name, old in previous.items():
                        if old is registry._MISSING:
                            del registry._FLAGS[name]
                        else:
                            registry._FLAGS[name] = old
                ''',
            "flags/checkout.py": '''\
                from .registry import is_enabled


                def checkout_url(cart_id):
                    if is_enabled("new_checkout"):
                        return f"/v2/checkout/{cart_id}"
                    return f"/checkout/{cart_id}"
                ''',
            "tests/test_flags.py": '''\
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
                ''',
        },
        hidden_test='''\
            import pytest

            from flags.registry import is_enabled, snapshot
            from flags.testing import override_flags


            def test_restored_when_block_raises():
                before = snapshot()
                with pytest.raises(AssertionError):
                    with override_flags(new_checkout=True, beta_search=True):
                        assert False, "boom"
                assert snapshot() == before
                assert not is_enabled("new_checkout")


            def test_exception_propagates_unchanged():
                with pytest.raises(KeyError, match="missing"):
                    with override_flags(dark_mode=True):
                        raise KeyError("missing")
                assert not is_enabled("dark_mode")


            def test_nested_overrides_restore_in_order():
                with override_flags(dark_mode=True):
                    with override_flags(dark_mode=False):
                        assert not is_enabled("dark_mode")
                    assert is_enabled("dark_mode")
                assert not is_enabled("dark_mode")
            ''',
        p2p=["tests/test_flags.py::test_override_and_restore", "tests/test_flags.py::test_new_flag_removed_afterwards"],
        steps=[
            ("Locate the context manager.", F("**/testing.py")),
            ("Read it.", V("flags/testing.py")),
            ("The restore code sits after a bare `yield`; an exception raised in the with-block is thrown in at the "
             "yield and skips it. Reproduce.",
             B("python - <<'EOF'\n"
               "from flags.registry import snapshot\n"
               "from flags.testing import override_flags\n"
               "try:\n"
               "    with override_flags(new_checkout=True, beta_search=True):\n"
               "        raise RuntimeError('boom')\n"
               "except RuntimeError:\n"
               "    pass\n"
               "print(snapshot())\n"
               "EOF")),
            ("Wrap the yield in try/finally so restoration always runs and the exception still propagates.",
             R("flags/testing.py",
               "    yield\n"
               "    for name, old in previous.items():\n"
               "        if old is registry._MISSING:\n"
               "            del registry._FLAGS[name]\n"
               "        else:\n"
               "            registry._FLAGS[name] = old\n",
               "    try:\n"
               "        yield\n"
               "    finally:\n"
               "        for name, old in previous.items():\n"
               "            if old is registry._MISSING:\n"
               "                del registry._FLAGS[name]\n"
               "            else:\n"
               "                registry._FLAGS[name] = old\n")),
            ("Re-run the repro.",
             B("python - <<'EOF'\n"
               "from flags.registry import snapshot\n"
               "from flags.testing import override_flags\n"
               "try:\n"
               "    with override_flags(new_checkout=True, beta_search=True):\n"
               "        raise RuntimeError('boom')\n"
               "except RuntimeError as e:\n"
               "    print('propagated:', e)\n"
               "print(snapshot())\n"
               "EOF")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 63 ---------------------------------------------------------------------------------------
    dict(
        id="seed-063-descriptor-stores-on-class",
        bug_type="descriptor-shared-state", difficulty="hard",
        issue="""\
            Cart total is wrong once it has more than one product

                mug = Product("MUG", price=5, stock=10)
                shirt = Product("SHIRT", price=12, stock=3)
                cart = Cart(); cart.add(mug, 1); cart.add(shirt, 1)
                cart.total()   # 24, expected 17

            Also, adding 5 mugs fails with "only 3 of MUG in stock" even though we created it
            with stock=10. Single-product carts look fine.
            """,
        files={
            "catalog/__init__.py": "",
            "catalog/fields.py": '''\
                class PositiveNumber:
                    """Data descriptor for numeric attributes that must not be negative."""

                    def __init__(self, default=0):
                        self.default = default
                        self.value = default

                    def __set_name__(self, owner, name):
                        self.name = name

                    def __get__(self, obj, objtype=None):
                        if obj is None:
                            return self
                        return self.value

                    def __set__(self, obj, value):
                        if value < 0:
                            raise ValueError(f"{self.name} must be >= 0, got {value!r}")
                        self.value = value
                ''',
            "catalog/product.py": '''\
                from .fields import PositiveNumber


                class Product:
                    price = PositiveNumber()
                    stock = PositiveNumber()

                    def __init__(self, sku, price, stock=0):
                        self.sku = sku
                        self.price = price
                        self.stock = stock

                    def __repr__(self):
                        return f"Product({self.sku!r}, price={self.price!r}, stock={self.stock!r})"
                ''',
            "catalog/cart.py": '''\
                class Cart:
                    def __init__(self):
                        self.lines = []

                    def add(self, product, qty=1):
                        if qty > product.stock:
                            raise ValueError(f"only {product.stock} of {product.sku} in stock")
                        self.lines.append((product, qty))

                    def total(self):
                        return sum(p.price * qty for p, qty in self.lines)
                ''',
            "tests/test_catalog.py": '''\
                import pytest

                from catalog.cart import Cart
                from catalog.product import Product


                def test_negative_price_rejected():
                    with pytest.raises(ValueError, match="price"):
                        Product("X", price=-1)


                def test_single_product_cart():
                    p = Product("MUG", price=5, stock=10)
                    cart = Cart()
                    cart.add(p, 3)
                    assert cart.total() == 15


                def test_out_of_stock():
                    p = Product("MUG", price=5, stock=1)
                    with pytest.raises(ValueError, match="only 1"):
                        Cart().add(p, 2)
                ''',
        },
        hidden_test='''\
            from catalog.cart import Cart
            from catalog.fields import PositiveNumber
            from catalog.product import Product


            def test_instances_do_not_share_values():
                mug = Product("MUG", price=5, stock=10)
                shirt = Product("SHIRT", price=12, stock=3)
                assert (mug.price, mug.stock) == (5, 10)
                assert (shirt.price, shirt.stock) == (12, 3)


            def test_multi_product_cart_total():
                mug = Product("MUG", price=5, stock=10)
                shirt = Product("SHIRT", price=12, stock=3)
                cart = Cart()
                cart.add(mug, 5)
                cart.add(shirt, 1)
                assert cart.total() == 37


            def test_class_access_returns_descriptor():
                assert isinstance(Product.price, PositiveNumber)
                assert Product.price is not Product.stock
            ''',
        p2p=["tests/test_catalog.py::test_negative_price_rejected", "tests/test_catalog.py::test_single_product_cart",
             "tests/test_catalog.py::test_out_of_stock"],
        steps=[
            ("Start from the symptom: the cart total.", V("catalog/cart.py")),
            ("Cart just multiplies product.price by qty, which looks right. Check how Product stores price/stock.",
             V("catalog/product.py")),
            ("They are PositiveNumber descriptors; read the descriptor.", V("catalog/fields.py")),
            ("__set__ stores the value on the descriptor itself, which lives on the class, so every Product shares "
             "one price and one stock. Confirm.",
             B("python -c 'from catalog.product import Product; a=Product(\"MUG\", 5, 10); b=Product(\"SHIRT\", 12, 3); print(a, b)'")),
            ("Store per-instance values in obj.__dict__ under the attribute name (safe, since a data descriptor "
             "takes precedence over the instance dict) and fall back to the default.",
             R("catalog/fields.py",
               "        return self.value\n",
               "        return obj.__dict__.get(self.name, self.default)\n")),
            ("Update __set__ the same way.",
             R("catalog/fields.py",
               "        self.value = value\n",
               "        obj.__dict__[self.name] = value\n")),
            ("Remove the now-unused shared `value` slot from __init__.",
             R("catalog/fields.py", "        self.default = default\n        self.value = default\n",
               "        self.default = default\n")),
            ("Re-run the issue's scenario.",
             B("python - <<'EOF'\n"
               "from catalog.cart import Cart\n"
               "from catalog.product import Product\n"
               "mug, shirt = Product('MUG', price=5, stock=10), Product('SHIRT', price=12, stock=3)\n"
               "cart = Cart(); cart.add(mug, 5); cart.add(shirt, 1)\n"
               "print(mug, shirt, cart.total())\n"
               "EOF")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 64 ---------------------------------------------------------------------------------------
    dict(
        id="seed-064-repr-recurses-via-str",
        bug_type="repr-recursion", difficulty="easy",
        issue="""\
            RecursionError when logging a user

            Anything that formats a `User` blows up:

            ```
            >>> print(describe_change(u, "email", "a@x.io", "b@x.io"))
              File "accounts/user.py", line 19, in __repr__
                return f"<User {self} id={self.id}>"
              File "accounts/user.py", line 19, in __repr__
                return f"<User {self} id={self.id}>"
              [Previous line repeated 996 more times]
            RecursionError: maximum recursion depth exceeded
            ```
            """,
        files={
            "accounts/__init__.py": "",
            "accounts/user.py": '''\
                class User:
                    """An account holder.

                    str(user)  -> the display name, e.g. "Ada Lovelace"
                    repr(user) -> "<User Ada Lovelace id=7>"
                    """

                    def __init__(self, id, first, last, email):
                        self.id = id
                        self.first = first
                        self.last = last
                        self.email = email

                    @property
                    def display_name(self):
                        return f"{self.first} {self.last}".strip()

                    def __repr__(self):
                        return f"<User {self} id={self.id}>"
                ''',
            "accounts/audit.py": '''\
                def describe_change(user, field, old, new):
                    return f"{user}: {field} changed from {old!r} to {new!r}"
                ''',
            "tests/test_user.py": '''\
                from accounts.user import User


                def test_display_name():
                    assert User(1, "Ada", "Lovelace", "ada@x.io").display_name == "Ada Lovelace"


                def test_display_name_without_last():
                    assert User(2, "Cher", "", "cher@x.io").display_name == "Cher"
                ''',
        },
        hidden_test='''\
            from accounts.audit import describe_change
            from accounts.user import User


            def test_str_is_display_name():
                assert str(User(7, "Ada", "Lovelace", "a@x.io")) == "Ada Lovelace"


            def test_repr_format():
                assert repr(User(7, "Ada", "Lovelace", "a@x.io")) == "<User Ada Lovelace id=7>"


            def test_describe_change():
                u = User(7, "Ada", "Lovelace", "a@x.io")
                assert describe_change(u, "email", "a@x.io", "b@x.io") == \\
                    "Ada Lovelace: email changed from 'a@x.io' to 'b@x.io'"
            ''',
        p2p=["tests/test_user.py::test_display_name", "tests/test_user.py::test_display_name_without_last"],
        steps=[
            ("Open the User class.", V("accounts/user.py")),
            ("`{self}` calls str(), and with no __str__ defined object.__str__ falls back to __repr__, which "
             "recurses. Reproduce.",
             B("python -c 'from accounts.user import User; print(repr(User(7, \"Ada\", \"Lovelace\", \"a@x.io\")))' 2>&1 | tail -1")),
            ("The docstring says str() should be the display name; add __str__ after the display_name property.",
             I("accounts/user.py", 17, "    def __str__(self):\n        return self.display_name\n\n")),
            ("Check the result reads well.", V("accounts/user.py", 14, 22)),
            ("Verify str, repr and the audit message.",
             B("python -c 'from accounts.user import User; from accounts.audit import describe_change as d; "
               "u=User(7, \"Ada\", \"Lovelace\", \"a@x.io\"); print(str(u), repr(u)); print(d(u, \"email\", \"a\", \"b\"))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 65 ---------------------------------------------------------------------------------------
    dict(
        id="seed-065-clone-shallow-copy",
        bug_type="shallow-copy-aliasing", difficulty="medium",
        issue="""\
            Every generated invoice gets more expensive

                invoice_for("acme", [("Support", 50)]).total()   # 150, fine
                invoice_for("globex").total()                     # 150, expected 100

            The Globex invoice contains Acme's "Support" line, and the shared monthly template
            itself now has three lines after a few calls. `clone()` is documented to return an
            independent copy.
            """,
        files={
            "invoicing/__init__.py": "",
            "invoicing/template.py": '''\
                import copy


                class InvoiceTemplate:
                    def __init__(self, name, lines=None, meta=None):
                        self.name = name
                        self.lines = list(lines or [])  # [{"desc": str, "amount": int}]
                        self.meta = dict(meta or {})

                    def clone(self, **changes):
                        """Return an independent copy, optionally overriding attributes."""
                        new = copy.copy(self)
                        for key, value in changes.items():
                            setattr(new, key, value)
                        return new

                    def total(self):
                        return sum(line["amount"] for line in self.lines)
                ''',
            "invoicing/builder.py": '''\
                from .template import InvoiceTemplate

                MONTHLY = InvoiceTemplate(
                    "monthly",
                    lines=[{"desc": "Hosting", "amount": 100}],
                    meta={"terms": "net30", "tags": ["recurring"]},
                )


                def invoice_for(customer, extras=()):
                    inv = MONTHLY.clone(name=f"monthly-{customer}")
                    inv.meta["customer"] = customer
                    for desc, amount in extras:
                        inv.lines.append({"desc": desc, "amount": amount})
                    return inv
                ''',
            "tests/test_template.py": '''\
                from invoicing.template import InvoiceTemplate


                def test_total():
                    t = InvoiceTemplate("t", lines=[{"desc": "a", "amount": 3}, {"desc": "b", "amount": 4}])
                    assert t.total() == 7


                def test_clone_overrides_attributes():
                    t = InvoiceTemplate("t", lines=[{"desc": "a", "amount": 3}])
                    c = t.clone(name="copy")
                    assert (c.name, t.name) == ("copy", "t")
                    assert c.total() == 3
                ''',
        },
        hidden_test='''\
            from invoicing.builder import MONTHLY, invoice_for
            from invoicing.template import InvoiceTemplate


            def test_invoices_are_independent():
                a = invoice_for("acme", [("Support", 50)])
                b = invoice_for("globex")
                assert a.total() == 150
                assert b.total() == 100
                assert b.meta["customer"] == "globex"


            def test_template_untouched():
                invoice_for("initech", [("Setup", 10)])
                assert MONTHLY.total() == 100
                assert "customer" not in MONTHLY.meta


            def test_nested_values_not_shared():
                t = InvoiceTemplate("t", lines=[{"desc": "a", "amount": 1}], meta={"tags": ["x"]})
                c = t.clone()
                c.lines[0]["amount"] = 99
                c.meta["tags"].append("y")
                assert t.lines[0]["amount"] == 1
                assert t.meta["tags"] == ["x"]
            ''',
        p2p=["tests/test_template.py::test_total", "tests/test_template.py::test_clone_overrides_attributes"],
        steps=[
            ("Start where invoices are built.", V("invoicing/builder.py")),
            ("invoice_for mutates the clone's lines and meta, so clone() must not share them. Find clone().",
             S("def clone", "invoicing")),
            ("Read the template.", V("invoicing/template.py")),
            ("copy.copy only copies the top-level object; `lines` and `meta` are the same list/dict as the template's. Confirm.",
             B("python -c 'from invoicing.builder import MONTHLY, invoice_for; a=invoice_for(\"acme\", [(\"Support\", 50)]); "
               "print(a.lines is MONTHLY.lines, MONTHLY.total(), MONTHLY.meta)'")),
            ("Use a deep copy so nested line dicts and meta lists are independent too.",
             R("invoicing/template.py", "        new = copy.copy(self)\n", "        new = copy.deepcopy(self)\n")),
            ("Re-run the issue scenario.",
             B("python -c 'from invoicing.builder import MONTHLY, invoice_for; a=invoice_for(\"acme\", [(\"Support\", 50)]); "
               "b=invoice_for(\"globex\"); print(a.total(), b.total(), MONTHLY.total(), MONTHLY.meta)'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 66 ---------------------------------------------------------------------------------------
    dict(
        id="seed-066-lambda-late-binding",
        bug_type="closure-late-binding", difficulty="medium",
        issue="""\
            Signup form errors name the wrong field

            Submitting the signup form without a username returns
            `{"username": "display_name is required"}`. Also a 30-character username is
            accepted, although the limit is 20; it looks like every field is being validated
            with display_name's limit (50).
            """,
        files={
            "forms/__init__.py": "",
            "forms/validators.py": '''\
                def required(field, value):
                    if value in (None, ""):
                        return f"{field} is required"
                    return None


                def max_length(limit):
                    def check(field, value):
                        if value is not None and len(value) > limit:
                            return f"{field} must be at most {limit} characters"
                        return None
                    return check
                ''',
            "forms/schema.py": '''\
                from .validators import max_length, required

                FIELDS = {
                    "username": 20,
                    "email": 254,
                    "display_name": 50,
                }


                def build_rules(fields=None):
                    """Return {field: [callable(value) -> error message or None]}."""
                    rules = {}
                    for name, limit in (fields or FIELDS).items():
                        rules[name] = [
                            lambda value: required(name, value),
                            lambda value: max_length(limit)(name, value),
                        ]
                    return rules


                def validate(data, rules=None):
                    rules = rules or build_rules()
                    errors = {}
                    for name, checks in rules.items():
                        for check in checks:
                            msg = check(data.get(name))
                            if msg:
                                errors[name] = msg
                                break
                    return errors
                ''',
            "tests/test_validators.py": '''\
                from forms.schema import validate
                from forms.validators import max_length, required

                GOOD = {"username": "ada", "email": "ada@x.io", "display_name": "Ada"}


                def test_valid_data():
                    assert validate(GOOD) == {}


                def test_missing_display_name():
                    assert validate({**GOOD, "display_name": ""}) == {"display_name": "display_name is required"}


                def test_validator_functions():
                    assert required("x", None) == "x is required"
                    assert max_length(2)("x", "abc") == "x must be at most 2 characters"
                ''',
        },
        hidden_test='''\
            from forms.schema import build_rules, validate

            GOOD = {"username": "ada", "email": "ada@x.io", "display_name": "Ada"}


            def test_missing_username_message():
                assert validate({**GOOD, "username": ""}) == {"username": "username is required"}


            def test_username_length_limit():
                assert validate({**GOOD, "username": "u" * 21}) == {
                    "username": "username must be at most 20 characters"}
                assert validate({**GOOD, "username": "u" * 20}) == {}


            def test_custom_fields():
                rules = build_rules({"a": 1, "b": 3})
                assert validate({"a": "xx", "b": "yyy"}, rules) == {"a": "a must be at most 1 characters"}
            ''',
        p2p=["tests/test_validators.py::test_valid_data", "tests/test_validators.py::test_missing_display_name",
             "tests/test_validators.py::test_validator_functions"],
        steps=[
            ("Find where the per-field rules are assembled.", S("required\\(|max_length\\(", "forms")),
            ("Read build_rules.", V("forms/schema.py", 10, 19)),
            ("The lambdas close over the loop variables `name` and `limit`, which are looked up at call time, so "
             "they all see the last field. Reproduce.",
             B("python -c 'from forms.schema import validate; print(validate({\"username\": \"\", \"email\": \"a@x.io\", "
               "\"display_name\": \"Ada\"})); print(validate({\"username\": \"u\"*30, \"email\": \"a@x.io\", \"display_name\": \"Ada\"}))'")),
            ("Bind the current values as default arguments so each lambda captures its own field and limit.",
             R("forms/schema.py",
               "            lambda value: required(name, value),\n"
               "            lambda value: max_length(limit)(name, value),\n",
               "            lambda value, name=name: required(name, value),\n"
               "            lambda value, name=name, limit=limit: max_length(limit)(name, value),\n")),
            ("Re-run the repro.",
             B("python -c 'from forms.schema import validate; print(validate({\"username\": \"\", \"email\": \"a@x.io\", "
               "\"display_name\": \"Ada\"})); print(validate({\"username\": \"u\"*30, \"email\": \"a@x.io\", \"display_name\": \"Ada\"}))'")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 67 ---------------------------------------------------------------------------------------
    dict(
        id="seed-067-decorator-loses-name",
        bug_type="decorator-metadata", difficulty="easy",
        issue="""\
            `deploy` command missing from CLI

            ```
            >>> from cli import commands, registry
            >>> registry.run("deploy", "prod")
            SystemExit: unknown command: deploy
            ```
            `greet` works. `help_text()` lists a command called `wrapper` with no
            description instead of `deploy` and `rollback`. This started when we added call
            counting to the deploy commands.
            """,
        files={
            "cli/__init__.py": "",
            "cli/decorators.py": '''\
                def counted(fn):
                    """Count how many times a command has been invoked (exposed as `.calls`)."""
                    def wrapper(*args, **kwargs):
                        wrapper.calls += 1
                        return fn(*args, **kwargs)
                    wrapper.calls = 0
                    return wrapper
                ''',
            "cli/registry.py": '''\
                COMMANDS = {}


                def command(fn):
                    """Register fn under its function name; its docstring becomes the help text."""
                    COMMANDS[fn.__name__] = fn
                    return fn


                def help_text():
                    return "\\n".join(f"{name}: {(fn.__doc__ or '').strip()}" for name, fn in sorted(COMMANDS.items()))


                def run(name, *args):
                    try:
                        fn = COMMANDS[name]
                    except KeyError:
                        raise SystemExit(f"unknown command: {name}") from None
                    return fn(*args)
                ''',
            "cli/commands.py": '''\
                from .decorators import counted
                from .registry import command


                @command
                def greet(name):
                    """Say hello."""
                    return f"hello {name}"


                @command
                @counted
                def deploy(env):
                    """Deploy the current build."""
                    return f"deployed to {env}"


                @command
                @counted
                def rollback(env):
                    """Roll back the last deploy."""
                    return f"rolled back {env}"
                ''',
            "tests/test_cli.py": '''\
                from cli import commands  # noqa: F401  (registers commands)
                from cli.decorators import counted
                from cli.registry import run


                def test_greet():
                    assert run("greet", "bob") == "hello bob"


                def test_counted_counts_calls():
                    @counted
                    def f(x):
                        return x * 2

                    assert f(2) == 4 and f(3) == 6
                    assert f.calls == 2
                ''',
        },
        hidden_test='''\
            from cli import commands
            from cli.decorators import counted
            from cli.registry import COMMANDS, help_text, run


            def test_counted_commands_registered_by_name():
                assert run("deploy", "prod") == "deployed to prod"
                assert run("rollback", "prod") == "rolled back prod"
                assert "wrapper" not in COMMANDS


            def test_help_lists_docstrings():
                text = help_text()
                assert "deploy: Deploy the current build." in text
                assert "rollback: Roll back the last deploy." in text


            def test_counted_preserves_metadata():
                @counted
                def sample():
                    """Doc."""
                    return 1

                assert sample.__name__ == "sample"
                assert sample.__doc__ == "Doc."
                assert sample() == 1 and sample.calls == 1


            def test_registered_command_still_counts():
                before = commands.deploy.calls
                run("deploy", "staging")
                assert commands.deploy.calls == before + 1
            ''',
        p2p=["tests/test_cli.py::test_greet", "tests/test_cli.py::test_counted_counts_calls"],
        steps=[
            ("See how commands are registered.", V("cli/registry.py")),
            ("Registration keys on fn.__name__; check what the counting decorator returns.", V("cli/decorators.py")),
            ("`wrapper` doesn't copy the wrapped function's metadata, so deploy and rollback both register as "
             "'wrapper'. Confirm.",
             B("python -c 'from cli import commands; from cli.registry import COMMANDS; print(sorted(COMMANDS))'")),
            ("Use functools.wraps to carry over __name__, __doc__ and friends.",
             R("cli/decorators.py",
               "def counted(fn):\n"
               "    \"\"\"Count how many times a command has been invoked (exposed as `.calls`).\"\"\"\n"
               "    def wrapper(*args, **kwargs):\n",
               "import functools\n\n\n"
               "def counted(fn):\n"
               "    \"\"\"Count how many times a command has been invoked (exposed as `.calls`).\"\"\"\n"
               "    @functools.wraps(fn)\n"
               "    def wrapper(*args, **kwargs):\n")),
            ("Check for other decorators in the package with the same problem.", S("def wrapper", "cli")),
            ("Re-run the failing command and the help output.",
             B("python -c 'from cli import commands; from cli.registry import run, help_text; print(run(\"deploy\", \"prod\")); print(help_text())'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 68 ---------------------------------------------------------------------------------------
    dict(
        id="seed-068-eq-not-notimplemented",
        bug_type="comparison-protocol", difficulty="easy",
        issue="""\
            Changelog.position() crashes with AttributeError

            ```
            >>> cl = Changelog(); cl.add_release("1.0"); cl.add_release("1.1")
            >>> cl.position("1.0")
            Traceback (most recent call last):
              ...
              File "releases/version.py", line 11, in __eq__
                return self.parts == other.parts
            AttributeError: 'str' object has no attribute 'parts'
            ```
            The sections list starts with the "Unreleased" heading, which is a plain string.
            """,
        files={
            "releases/__init__.py": "",
            "releases/version.py": '''\
                from functools import total_ordering


                @total_ordering
                class Version:
                    def __init__(self, text):
                        self.text = text
                        self.parts = tuple(int(p) for p in text.split("."))

                    def __eq__(self, other):
                        return self.parts == other.parts

                    def __lt__(self, other):
                        return self.parts < other.parts

                    def __hash__(self):
                        return hash(self.parts)

                    def __repr__(self):
                        return f"Version({self.text!r})"
                ''',
            "releases/changelog.py": '''\
                from .version import Version

                UNRELEASED = "Unreleased"


                class Changelog:
                    def __init__(self):
                        self.sections = [UNRELEASED]
                        self.notes = {UNRELEASED: []}

                    def add_release(self, text):
                        """Promote the Unreleased notes to a new version section (newest first)."""
                        v = Version(text)
                        self.sections.insert(1, v)
                        self.notes[v] = self.notes.pop(UNRELEASED)
                        self.notes[UNRELEASED] = []
                        return v

                    def note(self, message, version=UNRELEASED):
                        key = version if version == UNRELEASED else Version(version)
                        self.notes[key].append(message)

                    def position(self, version):
                        """Index of a version's section; 0 is Unreleased."""
                        return self.sections.index(Version(version))
                ''',
            "tests/test_releases.py": '''\
                from releases.changelog import Changelog
                from releases.version import Version


                def test_ordering():
                    assert Version("1.2") < Version("1.10")
                    assert Version("2.0") >= Version("1.9")
                    assert sorted([Version("1.10"), Version("1.2")]) == [Version("1.2"), Version("1.10")]


                def test_release_moves_notes():
                    cl = Changelog()
                    cl.note("fix crash")
                    cl.add_release("1.0")
                    assert cl.notes[Version("1.0")] == ["fix crash"]
                    assert cl.notes["Unreleased"] == []
                ''',
        },
        hidden_test='''\
            import pytest

            from releases.changelog import Changelog
            from releases.version import Version


            def test_position():
                cl = Changelog()
                cl.add_release("1.0")
                cl.add_release("1.1")
                assert cl.position("1.1") == 1
                assert cl.position("1.0") == 2


            def test_compare_with_other_types():
                assert Version("1.0") != "1.0"
                assert not (Version("1.0") == None)  # noqa: E711
                with pytest.raises(TypeError):
                    Version("1.0") < "2.0"


            def test_ordering_still_works():
                assert Version("1.9") < Version("1.10") <= Version("1.10")
                assert Version("3.0") == Version("3.0")
            ''',
        p2p=["tests/test_releases.py::test_ordering", "tests/test_releases.py::test_release_moves_notes"],
        steps=[
            ("Read Version's comparison methods.", V("releases/version.py")),
            ("list.index compares the str heading with a Version; str returns NotImplemented, Python tries the "
             "reflected Version.__eq__, and that assumes `other` has .parts. Reproduce.",
             B("python -c 'from releases.version import Version; print(\"Unreleased\" == Version(\"1.0\"))' 2>&1 | tail -1")),
            ("Return NotImplemented for foreign types so Python falls back to its default (identity / TypeError).",
             R("releases/version.py",
               "    def __eq__(self, other):\n        return self.parts == other.parts\n",
               "    def __eq__(self, other):\n        if not isinstance(other, Version):\n            return NotImplemented\n"
               "        return self.parts == other.parts\n")),
            ("Same for __lt__, which total_ordering builds the other operators from.",
             R("releases/version.py",
               "    def __lt__(self, other):\n        return self.parts < other.parts\n",
               "    def __lt__(self, other):\n        if not isinstance(other, Version):\n            return NotImplemented\n"
               "        return self.parts < other.parts\n")),
            ("Verify the changelog call and mixed-type comparisons.",
             B("python - <<'EOF'\n"
               "from releases.changelog import Changelog\n"
               "from releases.version import Version\n"
               "cl = Changelog(); cl.add_release('1.0'); cl.add_release('1.1')\n"
               "print(cl.position('1.0'), Version('1.0') == '1.0')\n"
               "try:\n"
               "    Version('1.0') < '2.0'\n"
               "except TypeError as e:\n"
               "    print('TypeError:', e)\n"
               "EOF")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 69 ---------------------------------------------------------------------------------------
    dict(
        id="seed-069-enum-name-vs-value",
        bug_type="enum-serialization", difficulty="medium",
        issue="""\
            Can't load orders we just saved

            ```
            >>> from orders import store
            >>> from orders.order import Order
            >>> store.loads(store.dumps([Order(1)]))
            Traceback (most recent call last):
              ...
            ValueError: 'PENDING' is not a valid OrderStatus
            ```
            The API docs say statuses are lowercase on the wire (`"pending"`, `"shipped"`, ...),
            and loading payloads from clients in that format works fine.
            """,
        files={
            "orders/__init__.py": "",
            "orders/status.py": '''\
                from enum import Enum


                class OrderStatus(Enum):
                    PENDING = "pending"
                    PAID = "paid"
                    SHIPPED = "shipped"
                    CANCELLED = "cancelled"

                    @property
                    def is_final(self):
                        return self in (OrderStatus.SHIPPED, OrderStatus.CANCELLED)
                ''',
            "orders/order.py": '''\
                from dataclasses import dataclass

                from .status import OrderStatus


                @dataclass
                class Order:
                    id: int
                    status: OrderStatus = OrderStatus.PENDING

                    def advance(self, new_status):
                        if self.status.is_final:
                            raise ValueError(f"order {self.id} is already {self.status.value}")
                        self.status = OrderStatus(new_status)
                ''',
            "orders/serialize.py": '''\
                from .order import Order
                from .status import OrderStatus


                def to_dict(order):
                    return {"id": order.id, "status": order.status.name}


                def from_dict(data):
                    return Order(id=int(data["id"]), status=OrderStatus(data["status"]))
                ''',
            "orders/store.py": '''\
                import json

                from .serialize import from_dict, to_dict


                def dumps(orders):
                    return json.dumps([to_dict(o) for o in orders])


                def loads(text):
                    return [from_dict(d) for d in json.loads(text)]
                ''',
            "tests/test_orders.py": '''\
                import pytest

                from orders.order import Order
                from orders.serialize import from_dict
                from orders.status import OrderStatus


                def test_from_dict_lowercase():
                    assert from_dict({"id": "3", "status": "shipped"}) == Order(3, OrderStatus.SHIPPED)


                def test_advance():
                    o = Order(1)
                    o.advance("paid")
                    assert o.status is OrderStatus.PAID


                def test_cannot_advance_final():
                    o = Order(1, OrderStatus.CANCELLED)
                    with pytest.raises(ValueError, match="already cancelled"):
                        o.advance("paid")
                ''',
        },
        hidden_test='''\
            import json

            from orders import store
            from orders.order import Order
            from orders.serialize import to_dict
            from orders.status import OrderStatus


            def test_to_dict_uses_wire_format():
                assert to_dict(Order(5, OrderStatus.SHIPPED)) == {"id": 5, "status": "shipped"}


            def test_round_trip():
                orders = [Order(1), Order(2, OrderStatus.PAID), Order(3, OrderStatus.CANCELLED)]
                assert store.loads(store.dumps(orders)) == orders


            def test_loads_client_payload():
                assert store.loads(json.dumps([{"id": 9, "status": "paid"}])) == [Order(9, OrderStatus.PAID)]
            ''',
        p2p=["tests/test_orders.py::test_from_dict_lowercase", "tests/test_orders.py::test_advance",
             "tests/test_orders.py::test_cannot_advance_final"],
        steps=[
            ("Find the save/load path.", F("orders/*.py")),
            ("store just JSON-encodes to_dict / from_dict output; read those.", V("orders/serialize.py")),
            ("Check the enum definition: names are uppercase, values lowercase.", V("orders/status.py")),
            ("to_dict writes `.name` ('PENDING') but from_dict parses by value, so the round trip breaks. Reproduce.",
             B("python -c 'from orders import store; from orders.order import Order; s=store.dumps([Order(1)]); "
               "print(s); store.loads(s)' 2>&1 | tail -2")),
            ("Serialize the enum's value, which is the documented wire format and what from_dict expects.",
             R("orders/serialize.py", '"status": order.status.name}', '"status": order.status.value}')),
            ("Re-run the round trip.",
             B("python -c 'from orders import store; from orders.order import Order; from orders.status import OrderStatus as S; "
               "s=store.dumps([Order(1), Order(2, S.SHIPPED)]); print(s); print(store.loads(s))'")),
            ("Run tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 70 ---------------------------------------------------------------------------------------
    dict(
        id="seed-070-getattr-recursion-on-copy",
        bug_type="getattr-recursion", difficulty="hard",
        issue="""\
            per_request() dies with RecursionError

            Since we switched handlers to `settings = per_request()`, every request fails:

            ```
              File "settings/runtime.py", line 10, in per_request
                return copy.deepcopy(settings)
              ...
              File "settings/proxy.py", line 13, in __getattr__
                if name in self._data:
              File "settings/proxy.py", line 13, in __getattr__
                if name in self._data:
              [Previous line repeated 990 more times]
            RecursionError: maximum recursion depth exceeded
            ```
            Reading settings off `BASE` directly works fine. Pickling a SettingsProxy (we tried it
            for the worker pool) fails the same way.
            """,
        files={
            "settings/__init__.py": "",
            "settings/proxy.py": '''\
                class SettingsProxy:
                    """Read-only view over a settings dict; attribute access maps to keys.

                    Keys missing from `data` fall back to `defaults`.
                    """

                    def __init__(self, data, defaults=None):
                        self._data = dict(data)
                        self._defaults = dict(defaults or {})

                    def __getattr__(self, name):
                        # Only called when normal attribute lookup fails.
                        if name in self._data:
                            return self._data[name]
                        if name in self._defaults:
                            return self._defaults[name]
                        raise AttributeError(f"no setting named {name!r}")

                    def with_overrides(self, **overrides):
                        return SettingsProxy({**self._data, **overrides}, self._defaults)
                ''',
            "settings/runtime.py": '''\
                import copy

                from .proxy import SettingsProxy

                BASE = SettingsProxy({"debug": False, "workers": 4}, defaults={"timeout": 30})


                def per_request(settings=BASE):
                    """Give each request its own copy so handlers can't affect each other."""
                    return copy.deepcopy(settings)
                ''',
            "tests/test_proxy.py": '''\
                import pytest

                from settings.proxy import SettingsProxy
                from settings.runtime import BASE


                def test_attribute_access_and_defaults():
                    assert BASE.workers == 4
                    assert BASE.timeout == 30


                def test_unknown_setting():
                    with pytest.raises(AttributeError, match="nope"):
                        BASE.nope


                def test_with_overrides():
                    s = SettingsProxy({"a": 1}, {"b": 2}).with_overrides(a=5)
                    assert (s.a, s.b) == (5, 2)
                ''',
        },
        hidden_test='''\
            import copy
            import pickle

            from settings.proxy import SettingsProxy
            from settings.runtime import BASE, per_request


            def test_per_request_returns_independent_copy():
                s = per_request()
                assert s is not BASE
                assert (s.debug, s.workers, s.timeout) == (False, 4, 30)
                s._data["workers"] = 99
                assert BASE.workers == 4


            def test_shallow_copy_and_pickle():
                assert copy.copy(BASE).workers == 4
                restored = pickle.loads(pickle.dumps(SettingsProxy({"x": 1}, {"y": 2})))
                assert (restored.x, restored.y) == (1, 2)


            def test_hasattr_on_missing_setting():
                assert not hasattr(BASE, "missing")
                assert hasattr(BASE, "debug")
            ''',
        p2p=["tests/test_proxy.py::test_attribute_access_and_defaults", "tests/test_proxy.py::test_unknown_setting",
             "tests/test_proxy.py::test_with_overrides"],
        steps=[
            ("Look at what per_request does.", V("settings/runtime.py")),
            ("Read the proxy's __getattr__.", V("settings/proxy.py")),
            ("Reproduce with a minimal copy, keeping only the tail of the traceback.",
             B("python -c 'import copy; from settings.runtime import BASE; copy.copy(BASE)' 2>&1 | tail -4")),
            ("copy/pickle build the new object with __new__ (no __init__) and then probe it for hooks like "
             "__setstate__. On that bare object `self._data` is missing, so __getattr__('_data') calls itself forever. "
             "Check which names are being looked up.",
             B("python - <<'EOF'\n"
               "from settings.proxy import SettingsProxy\n"
               "bare = SettingsProxy.__new__(SettingsProxy)\n"
               "try:\n"
               "    getattr(bare, '__setstate__', None)\n"
               "except RecursionError as e:\n"
               "    print('RecursionError:', e)\n"
               "EOF")),
            ("Settings keys are public names, so private/dunder lookups that reach __getattr__ should fail "
             "immediately with AttributeError instead of consulting _data.",
             R("settings/proxy.py",
               "        # Only called when normal attribute lookup fails.\n",
               "        # Only called when normal attribute lookup fails.\n"
               "        if name.startswith(\"_\"):\n"
               "            # Internal/dunder names (e.g. _data before __init__ ran, __setstate__ probed by\n"
               "            # copy/pickle) are never settings; don't recurse into self._data for them.\n"
               "            raise AttributeError(name)\n")),
            ("Re-run the deepcopy path and a pickle round trip.",
             B("python - <<'EOF'\n"
               "import pickle\n"
               "from settings.runtime import BASE, per_request\n"
               "s = per_request()\n"
               "print(s is BASE, s.workers, s.timeout)\n"
               "r = pickle.loads(pickle.dumps(BASE))\n"
               "print(r.debug, r.timeout, hasattr(r, 'missing'))\n"
               "EOF")),
            ("Run tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
]
