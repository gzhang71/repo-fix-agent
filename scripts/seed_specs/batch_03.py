"""Seed instances 041-055: dates/times, numbers, money & units."""
from ._helpers import B, F, I, R, S, SUBMIT, T, TEST_CMD, V

SPECS = [
    # 41 ---------------------------------------------------------------------------------------
    dict(
        id="seed-041-add-months-end-of-month",
        bug_type="date-arithmetic-overflow", difficulty="medium",
        issue="""\
            Renewal schedule crashes for subscriptions started on the 31st

            A customer who signed up on 2024-01-31 can't open their billing page:

            ```
            Traceback (most recent call last):
              File "<stdin>", line 1, in <module>
              File "billing/subscriptions.py", line 6, in renewal_dates
                return [add_months(start, interval_months * i) for i in range(1, count + 1)]
              File "billing/dates.py", line 9, in add_months
                return d.replace(year=year, month=month)
            ValueError: day is out of range for month
            ```

            Renewals should land on the last day of the month when the month is shorter
            (Jan 31 -> Feb 29 -> Mar 31 -> Apr 30), which is what the docstring promises.
            """,
        files={
            "billing/__init__.py": "",
            "billing/dates.py": '''\
                from datetime import date


                def add_months(d, months):
                    """Return the same day `months` later, clamped to the end of the target month."""
                    total = d.month - 1 + months
                    year = d.year + total // 12
                    month = total % 12 + 1
                    return d.replace(year=year, month=month)


                def month_start(d):
                    return date(d.year, d.month, 1)
                ''',
            "billing/subscriptions.py": '''\
                from .dates import add_months


                def renewal_dates(start, count, interval_months=1):
                    """Next `count` renewal dates, always computed from the anchor to avoid drift."""
                    return [add_months(start, interval_months * i) for i in range(1, count + 1)]
                ''',
            "tests/test_dates.py": '''\
                from datetime import date

                from billing.dates import add_months, month_start
                from billing.subscriptions import renewal_dates


                def test_add_months_simple():
                    assert add_months(date(2024, 1, 15), 1) == date(2024, 2, 15)


                def test_add_months_across_year():
                    assert add_months(date(2023, 11, 10), 3) == date(2024, 2, 10)
                    assert add_months(date(2024, 3, 10), -3) == date(2023, 12, 10)


                def test_month_start():
                    assert month_start(date(2024, 7, 19)) == date(2024, 7, 1)


                def test_renewal_dates_mid_month():
                    assert renewal_dates(date(2024, 1, 10), 3) == [
                        date(2024, 2, 10), date(2024, 3, 10), date(2024, 4, 10)]
                ''',
        },
        hidden_test='''\
            from datetime import date

            from billing.dates import add_months
            from billing.subscriptions import renewal_dates


            def test_clamps_to_end_of_shorter_month():
                assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)
                assert add_months(date(2023, 1, 31), 1) == date(2023, 2, 28)
                assert add_months(date(2024, 5, 31), -1) == date(2024, 4, 30)


            def test_renewals_do_not_drift():
                assert renewal_dates(date(2024, 1, 31), 3) == [
                    date(2024, 2, 29), date(2024, 3, 31), date(2024, 4, 30)]


            def test_quarterly_interval():
                assert renewal_dates(date(2024, 5, 15), 2, interval_months=3) == [
                    date(2024, 8, 15), date(2024, 11, 15)]
            ''',
        p2p=["tests/test_dates.py::test_add_months_simple", "tests/test_dates.py::test_add_months_across_year",
             "tests/test_dates.py::test_month_start", "tests/test_dates.py::test_renewal_dates_mid_month"],
        steps=[
            ("Locate add_months and its callers.", S("add_months", "billing")),
            ("Read the month arithmetic.", V("billing/dates.py")),
            ("Reproduce with the customer's anchor date.",
             B("python -c 'from datetime import date; from billing.subscriptions import renewal_dates; "
               "print(renewal_dates(date(2024, 1, 31), 3))'")),
            ("`d.replace` keeps day=31, which doesn't exist in February. Import calendar so I can "
             "clamp to the target month's length.",
             I("billing/dates.py", 0, "import calendar")),
            ("Clamp the day to the last day of the target month.",
             R("billing/dates.py",
               "    return d.replace(year=year, month=month)\n",
               "    last_day = calendar.monthrange(year, month)[1]\n"
               "    return d.replace(year=year, month=month, day=min(d.day, last_day))\n")),
            ("Re-run the repro plus a non-leap year and a negative offset.",
             B("python -c 'from datetime import date; from billing.subscriptions import renewal_dates; "
               "from billing.dates import add_months; print(renewal_dates(date(2024, 1, 31), 3)); "
               "print(add_months(date(2023, 1, 31), 1), add_months(date(2024, 5, 31), -1))'")),
            ("Run the test suite.", T()),
            ("Renewals now clamp to month end without drifting.", SUBMIT),
        ],
    ),
    # 42 ---------------------------------------------------------------------------------------
    dict(
        id="seed-042-iso-week-year",
        bug_type="iso-week-year-mismatch", difficulty="easy",
        issue="""\
            Weekly report puts Dec 30-31 2024 into "2024-W01"

            In the year-end weekly signup report, the events from Monday 2024-12-30 and
            2024-12-31 are counted under `2024-W01`, i.e. lumped together with the first week
            of January 2024. ISO-wise those days belong to week 1 of **2025**.

                >>> from analytics.report import weekly_counts
                >>> weekly_counts([date(2024, 1, 2), date(2024, 12, 31)])
                {'2024-W01': 2}
            """,
        files={
            "analytics/__init__.py": "",
            "analytics/weeks.py": '''\
                from datetime import timedelta


                def week_key(d):
                    """ISO-8601 week label such as '2024-W07' used to bucket events."""
                    return f"{d.year}-W{d.isocalendar()[1]:02d}"


                def week_start(d):
                    """Monday of the week containing d."""
                    return d - timedelta(days=d.weekday())
                ''',
            "analytics/report.py": '''\
                from collections import Counter

                from .weeks import week_key


                def weekly_counts(dates):
                    return dict(sorted(Counter(week_key(d) for d in dates).items()))
                ''',
            "tests/test_weeks.py": '''\
                from datetime import date

                from analytics.report import weekly_counts
                from analytics.weeks import week_key, week_start


                def test_week_key_mid_year():
                    assert week_key(date(2024, 2, 14)) == "2024-W07"


                def test_week_start_is_monday():
                    assert week_start(date(2024, 2, 15)) == date(2024, 2, 12)


                def test_weekly_counts():
                    assert weekly_counts([date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 11)]) == {
                        "2024-W10": 2, "2024-W11": 1}
                ''',
        },
        hidden_test='''\
            from datetime import date

            from analytics.report import weekly_counts
            from analytics.weeks import week_key


            def test_late_december_belongs_to_next_iso_year():
                assert week_key(date(2024, 12, 30)) == "2025-W01"


            def test_early_january_belongs_to_previous_iso_year():
                assert week_key(date(2021, 1, 3)) == "2020-W53"


            def test_weekly_counts_year_boundary():
                assert weekly_counts([date(2024, 1, 2), date(2024, 12, 31)]) == {
                    "2024-W01": 1, "2025-W01": 1}


            def test_ordinary_week_unchanged():
                assert week_key(date(2024, 6, 5)) == "2024-W23"
            ''',
        p2p=["tests/test_weeks.py::test_week_key_mid_year", "tests/test_weeks.py::test_week_start_is_monday",
             "tests/test_weeks.py::test_weekly_counts"],
        steps=[
            ("Find where week labels are built.", B("grep -rn 'isocalendar\\|def week_key' --include=*.py .")),
            ("Read weeks.py.", V("analytics/weeks.py")),
            ("Reproduce with the dates from the report.",
             B("python -c 'from datetime import date; from analytics.weeks import week_key; "
               "print(week_key(date(2024, 12, 30)), week_key(date(2021, 1, 3)), date(2024, 12, 30).isocalendar())'")),
            ("The label mixes the calendar year with the ISO week number; the ISO year from "
             "isocalendar() must be used together with the week.",
             R("analytics/weeks.py",
               '    return f"{d.year}-W{d.isocalendar()[1]:02d}"\n',
               '    iso_year, iso_week, _ = d.isocalendar()\n    return f"{iso_year}-W{iso_week:02d}"\n')),
            ("Verify the boundary dates and the report.",
             B("python -c 'from datetime import date; from analytics.weeks import week_key; "
               "from analytics.report import weekly_counts; "
               "print(week_key(date(2024, 12, 30)), week_key(date(2021, 1, 3))); "
               "print(weekly_counts([date(2024, 1, 2), date(2024, 12, 31)]))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Fixed.", SUBMIT),
        ],
    ),
    # 43 ---------------------------------------------------------------------------------------
    dict(
        id="seed-043-decimal-half-even-rounding",
        bug_type="rounding-mode", difficulty="easy",
        issue="""\
            Invoice totals are sometimes 1 cent lower than the POS receipt

            Finance flagged a mismatch: a single item at 2.25 with a 10% discount prints
            2.03 on the till receipt, but `invoice_total` gives 2.02. Looks like it happens
            whenever the discounted amount ends in exactly half a cent (e.g. 0.125 -> 0.12).
            Our policy is round-half-up, same as the POS.
            """,
        files={
            "shop/__init__.py": "",
            "shop/money.py": '''\
                from decimal import Decimal

                CENT = Decimal("0.01")


                def to_money(value):
                    """Coerce int/str/Decimal to a Decimal rounded to cents (half-up, like the POS)."""
                    return Decimal(str(value)).quantize(CENT)
                ''',
            "shop/invoice.py": '''\
                from decimal import Decimal

                from .money import to_money


                def line_total(unit_price, qty, discount_pct=0):
                    gross = Decimal(str(unit_price)) * qty
                    discount = gross * Decimal(str(discount_pct)) / 100
                    return to_money(gross - discount)


                def invoice_total(lines):
                    return sum((line_total(*line) for line in lines), Decimal("0.00"))
                ''',
            "tests/test_invoice.py": '''\
                from decimal import Decimal

                from shop.invoice import invoice_total, line_total
                from shop.money import to_money


                def test_to_money_rounds_to_cents():
                    assert to_money("19.999") == Decimal("20.00")
                    assert to_money(3) == Decimal("3.00")


                def test_line_total():
                    assert line_total("10.00", 3) == Decimal("30.00")
                    assert line_total("10.00", 2, 25) == Decimal("15.00")


                def test_invoice_total():
                    assert invoice_total([("1.10", 2), ("0.30", 1)]) == Decimal("2.50")
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from shop.invoice import invoice_total, line_total
            from shop.money import to_money


            def test_half_cent_rounds_up():
                assert to_money("0.125") == Decimal("0.13")
                assert to_money("2.665") == Decimal("2.67")


            def test_negative_half_cent_rounds_away_from_zero():
                assert to_money("-0.125") == Decimal("-0.13")


            def test_discounted_line_matches_pos():
                assert line_total("2.25", 1, 10) == Decimal("2.03")
                assert invoice_total([("2.25", 1, 10), ("1.00", 1)]) == Decimal("3.03")


            def test_odd_half_cent_still_rounds_up():
                assert to_money("0.135") == Decimal("0.14")
            ''',
        p2p=["tests/test_invoice.py::test_to_money_rounds_to_cents", "tests/test_invoice.py::test_line_total",
             "tests/test_invoice.py::test_invoice_total"],
        steps=[
            ("Start from the invoice code that produces the total.", V("shop/invoice.py")),
            ("Rounding happens in to_money; read it.", V("shop/money.py")),
            ("Reproduce: quantize without a rounding argument uses the context default.",
             B("python -c 'from shop.money import to_money; from shop.invoice import line_total; "
               "print(to_money(\"0.125\"), to_money(\"0.135\"), line_total(\"2.25\", 1, 10))'")),
            ("The default context rounding is ROUND_HALF_EVEN (banker's rounding), so 0.125 -> 0.12. "
             "Import ROUND_HALF_UP.",
             R("shop/money.py", "from decimal import Decimal\n", "from decimal import ROUND_HALF_UP, Decimal\n")),
            ("Pass the rounding mode explicitly.",
             R("shop/money.py", "return Decimal(str(value)).quantize(CENT)",
               "return Decimal(str(value)).quantize(CENT, rounding=ROUND_HALF_UP)")),
            ("Re-run the repro.",
             B("python -c 'from shop.money import to_money; from shop.invoice import line_total; "
               "print(to_money(\"0.125\"), to_money(\"0.135\"), to_money(\"-0.125\"), line_total(\"2.25\", 1, 10))'")),
            ("Run the test suite.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 44 ---------------------------------------------------------------------------------------
    dict(
        id="seed-044-parse-drops-utc-offset",
        bug_type="naive-aware-datetime", difficulty="medium",
        issue="""\
            TypeError when checking due reminders with an aware `now`

            The worker passes a timezone-aware UTC `now` and crashes:

            ```
            >>> from datetime import datetime, timezone
            >>> from alerts.scheduler import Reminder, due_reminders
            >>> rems = [Reminder("standup", "2024-03-10T09:00:00+02:00")]
            >>> due_reminders(rems, datetime(2024, 3, 10, 7, 0, tzinfo=timezone.utc))
            Traceback (most recent call last):
              ...
              File "alerts/scheduler.py", line 15, in is_due
                return now >= self.fire_time()
            TypeError: can't compare offset-naive and offset-aware datetimes
            ```

            Reminder times come from the API with explicit offsets. I tried stripping tzinfo
            from `now` as a workaround, but then the standup above fired two hours late, so
            I don't think the offset is being honoured at all.
            """,
        files={
            "alerts/__init__.py": "",
            "alerts/timeparse.py": '''\
                from datetime import datetime, timezone


                def parse_ts(text):
                    """Parse an ISO-8601 timestamp into an aware UTC datetime.

                    Strings without an offset are assumed to already be in UTC.
                    """
                    dt = datetime.fromisoformat(text)
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)
                    return dt
                ''',
            "alerts/scheduler.py": '''\
                from datetime import timedelta

                from .timeparse import parse_ts


                class Reminder:
                    def __init__(self, name, at, lead_minutes=0):
                        self.name = name
                        self.at = parse_ts(at)
                        self.lead = timedelta(minutes=lead_minutes)

                    def fire_time(self):
                        return self.at - self.lead

                    def is_due(self, now):
                        return now >= self.fire_time()


                def due_reminders(reminders, now):
                    return [r.name for r in reminders if r.is_due(now)]
                ''',
            "tests/test_scheduler.py": '''\
                from alerts.scheduler import Reminder
                from alerts.timeparse import parse_ts


                def test_parse_naive_keeps_wall_clock():
                    dt = parse_ts("2024-03-10T09:00:00")
                    assert (dt.hour, dt.minute) == (9, 0)


                def test_lead_time():
                    r = Reminder("x", "2024-03-10T09:00:00", lead_minutes=15)
                    assert r.fire_time().strftime("%H:%M") == "08:45"
                ''',
        },
        hidden_test='''\
            from datetime import datetime, timezone

            from alerts.scheduler import Reminder, due_reminders
            from alerts.timeparse import parse_ts

            UTC = timezone.utc


            def test_offset_is_converted_to_utc():
                assert parse_ts("2024-03-10T09:00:00+02:00") == datetime(2024, 3, 10, 7, 0, tzinfo=UTC)
                assert parse_ts("2024-03-10T09:00:00+02:00").utcoffset().total_seconds() == 0


            def test_naive_input_is_treated_as_utc():
                assert parse_ts("2024-03-10T09:00:00") == datetime(2024, 3, 10, 9, 0, tzinfo=UTC)


            def test_due_with_aware_now():
                rems = [Reminder("standup", "2024-03-10T09:00:00+02:00"),
                        Reminder("lunch", "2024-03-10T12:00:00Z".replace("Z", "+00:00"))]
                assert due_reminders(rems, datetime(2024, 3, 10, 6, 59, tzinfo=UTC)) == []
                assert due_reminders(rems, datetime(2024, 3, 10, 7, 0, tzinfo=UTC)) == ["standup"]


            def test_negative_offset_with_lead():
                r = Reminder("x", "2024-03-10T08:00:00-05:00", lead_minutes=30)
                assert r.fire_time() == datetime(2024, 3, 10, 12, 30, tzinfo=UTC)
            ''',
        p2p=["tests/test_scheduler.py::test_parse_naive_keeps_wall_clock", "tests/test_scheduler.py::test_lead_time"],
        steps=[
            ("The comparison fails in the scheduler, but the datetimes come from parse_ts. Find it.",
             S("parse_ts", "alerts")),
            ("Read the parser.", V("alerts/timeparse.py")),
            ("Reproduce the crash from the report.",
             B("python -c 'from datetime import datetime, timezone; from alerts.scheduler import Reminder, due_reminders; "
               "print(due_reminders([Reminder(\"standup\", \"2024-03-10T09:00:00+02:00\")], "
               "datetime(2024, 3, 10, 7, 0, tzinfo=timezone.utc)))' 2>&1 | tail -1")),
            ("And check what parse_ts returns for an offset timestamp.",
             B("python -c 'from alerts.timeparse import parse_ts; print(repr(parse_ts(\"2024-03-10T09:00:00+02:00\")))'")),
            ("`replace(tzinfo=None)` throws the offset away instead of converting, and returns a naive "
             "value despite the docstring. Convert aware values to UTC and tag naive ones as UTC.",
             R("alerts/timeparse.py",
               "    if dt.tzinfo is not None:\n        dt = dt.replace(tzinfo=None)\n    return dt\n",
               "    if dt.tzinfo is None:\n        return dt.replace(tzinfo=timezone.utc)\n"
               "    return dt.astimezone(timezone.utc)\n")),
            ("Re-run the repro on both sides of the fire time.",
             B("python -c 'from datetime import datetime, timezone; from alerts.scheduler import Reminder, due_reminders; "
               "r = [Reminder(\"standup\", \"2024-03-10T09:00:00+02:00\")]; "
               "print(due_reminders(r, datetime(2024, 3, 10, 6, 59, tzinfo=timezone.utc)), "
               "due_reminders(r, datetime(2024, 3, 10, 7, 0, tzinfo=timezone.utc)))'")),
            ("Run the existing tests.", T()),
            ("Parser now returns aware UTC datetimes.", SUBMIT),
        ],
    ),
    # 45 ---------------------------------------------------------------------------------------
    dict(
        id="seed-045-timedelta-seconds-drops-days",
        bug_type="timedelta-attribute-misuse", difficulty="easy",
        issue="""\
            Job summary under-reports long runs

            The nightly backfill ran from Mon 01:00 to Tue 03:00 (26 hours) but the summary
            line says `backfill took 2h`. Anything over a day seems to lose the days part.
            """,
        files={
            "jobs/__init__.py": "",
            "jobs/durations.py": '''\
                def format_duration(td):
                    """Human-readable duration such as '1d 2h 5m' or '45s'."""
                    secs = td.seconds
                    days, rem = divmod(secs, 86400)
                    hours, rem = divmod(rem, 3600)
                    minutes, seconds = divmod(rem, 60)
                    units = ((days, "d"), (hours, "h"), (minutes, "m"), (seconds, "s"))
                    parts = [f"{value}{unit}" for value, unit in units if value]
                    return " ".join(parts) or "0s"
                ''',
            "jobs/summary.py": '''\
                from .durations import format_duration


                def describe_run(name, started, finished):
                    return f"{name} took {format_duration(finished - started)}"
                ''',
            "tests/test_durations.py": '''\
                from datetime import datetime, timedelta

                from jobs.durations import format_duration
                from jobs.summary import describe_run


                def test_hours_and_minutes():
                    assert format_duration(timedelta(hours=2, minutes=5)) == "2h 5m"


                def test_zero_and_seconds():
                    assert format_duration(timedelta(0)) == "0s"
                    assert format_duration(timedelta(seconds=45)) == "45s"


                def test_describe_run():
                    assert describe_run("etl", datetime(2024, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 30)) == "etl took 30m"
                ''',
        },
        hidden_test='''\
            from datetime import datetime, timedelta

            from jobs.durations import format_duration
            from jobs.summary import describe_run


            def test_days_are_included():
                assert format_duration(timedelta(days=1, hours=2)) == "1d 2h"
                assert format_duration(timedelta(days=3)) == "3d"


            def test_describe_multi_day_run():
                assert describe_run("backfill", datetime(2024, 1, 1, 1, 0), datetime(2024, 1, 2, 3, 0)) == "backfill took 1d 2h"


            def test_sub_day_unchanged():
                assert format_duration(timedelta(minutes=90, seconds=1)) == "1h 30m 1s"
            ''',
        p2p=["tests/test_durations.py::test_hours_and_minutes", "tests/test_durations.py::test_zero_and_seconds",
             "tests/test_durations.py::test_describe_run"],
        steps=[
            ("Find the duration formatter used by the summary.", S("format_duration", ".", "*.py")),
            ("Read it.", V("jobs/durations.py")),
            ("Reproduce with a 26-hour timedelta.",
             B("python -c 'from datetime import timedelta; from jobs.durations import format_duration; "
               "td = timedelta(days=1, hours=2); print(format_duration(td), td.seconds, td.total_seconds())'")),
            ("`timedelta.seconds` is only the seconds component (0..86399), not the total. Use total_seconds().",
             R("jobs/durations.py", "    secs = td.seconds\n", "    secs = int(td.total_seconds())\n")),
            ("Verify.",
             B("python -c 'from datetime import timedelta; from jobs.durations import format_duration; "
               "print(format_duration(timedelta(days=1, hours=2)), format_duration(timedelta(minutes=90)))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 46 ---------------------------------------------------------------------------------------
    dict(
        id="seed-046-fx-rate-direction-inverted",
        bug_type="inverted-conversion-factor", difficulty="medium",
        issue="""\
            EUR -> USD conversion goes the wrong way

            `convert(100, "EUR", "USD")` returns `92.59`. EUR is worth more than USD, so
            100 EUR should come out at about 108 USD with our current table. JPY results
            are wildly off as well (100 USD -> 0.67 JPY). Round-tripping a value comes back
            to the original amount, which is probably why nobody noticed.
            """,
        files={
            "fx/__init__.py": "",
            "fx/rates.py": '''\
                from decimal import Decimal

                # Value of ONE unit of each currency, expressed in USD.
                USD_PER_UNIT = {
                    "USD": Decimal("1"),
                    "EUR": Decimal("1.08"),
                    "GBP": Decimal("1.25"),
                    "JPY": Decimal("0.0067"),
                }


                def rate(code):
                    try:
                        return USD_PER_UNIT[code]
                    except KeyError:
                        raise ValueError(f"unknown currency {code!r}") from None
                ''',
            "fx/convert.py": '''\
                from decimal import ROUND_HALF_UP, Decimal

                from .rates import rate


                def convert(amount, src, dst):
                    amount = Decimal(str(amount))
                    if src == dst:
                        return amount
                    usd = amount / rate(src)
                    return (usd * rate(dst)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ''',
            "tests/test_convert.py": '''\
                from decimal import Decimal

                import pytest

                from fx.convert import convert


                def test_same_currency_is_identity():
                    assert convert("12.34", "EUR", "EUR") == Decimal("12.34")


                def test_unknown_currency():
                    with pytest.raises(ValueError):
                        convert(1, "USD", "XYZ")


                def test_round_trip():
                    there = convert(100, "EUR", "GBP")
                    assert convert(there, "GBP", "EUR") == Decimal("100.00")
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from fx.convert import convert


            def test_eur_to_usd():
                assert convert(100, "EUR", "USD") == Decimal("108.00")


            def test_usd_to_jpy():
                assert convert(100, "USD", "JPY") == Decimal("14925.37")


            def test_cross_rate():
                assert convert("50", "GBP", "EUR") == Decimal("57.87")


            def test_round_trip_still_holds():
                assert convert(convert(250, "USD", "GBP"), "GBP", "USD") == Decimal("250.00")
            ''',
        p2p=["tests/test_convert.py::test_same_currency_is_identity", "tests/test_convert.py::test_unknown_currency",
             "tests/test_convert.py::test_round_trip"],
        steps=[
            ("Get an overview of the package.", F("fx/*.py")),
            ("Check what the rate table means.", V("fx/rates.py")),
            ("Now the conversion itself.", V("fx/convert.py")),
            ("Reproduce.",
             B("python -c 'from fx.convert import convert; print(convert(100, \"EUR\", \"USD\"), convert(100, \"USD\", \"JPY\"))'")),
            ("Rates are USD per unit, so amount_in_usd = amount * rate(src) and the target amount is "
             "usd / rate(dst). The code has both operations inverted.",
             R("fx/convert.py",
               "    usd = amount / rate(src)\n    return (usd * rate(dst)).quantize(",
               "    usd = amount * rate(src)\n    return (usd / rate(dst)).quantize(")),
            ("Verify with known values: 100 EUR = 108 USD, 50 GBP = 62.5/1.08 EUR.",
             B("python -c 'from fx.convert import convert; print(convert(100, \"EUR\", \"USD\"), "
               "convert(100, \"USD\", \"JPY\"), convert(50, \"GBP\", \"EUR\"))'")),
            ("Run the tests (round-trip should still hold).", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 47 ---------------------------------------------------------------------------------------
    dict(
        id="seed-047-percentile-rank-index",
        bug_type="off-by-one", difficulty="medium",
        issue="""\
            percentile(): IndexError at p=100 and p50 doesn't match the median

                >>> from metrics.percentile import percentile
                >>> percentile([1, 2, 3, 4], 50)
                3
                >>> percentile([1, 2, 3, 4], 100)
                IndexError: list index out of range

            The docstring says linear interpolation between closest ranks (numpy's default),
            which would give 2.5 and 4. Our p95 latency numbers are also skewed high, and
            `latency_summary` blows up with the same IndexError for endpoints that only have
            a handful of samples.
            """,
        files={
            "metrics/__init__.py": "",
            "metrics/percentile.py": '''\
                import math


                def percentile(values, p):
                    """p-th percentile (0-100), linear interpolation between closest ranks."""
                    if not values:
                        raise ValueError("percentile of empty data")
                    if not 0 <= p <= 100:
                        raise ValueError("p must be within [0, 100]")
                    s = sorted(values)
                    k = len(s) * p / 100
                    lo, hi = math.floor(k), math.ceil(k)
                    if lo == hi:
                        return s[lo]
                    return s[lo] + (s[hi] - s[lo]) * (k - lo)
                ''',
            "metrics/summary.py": '''\
                from .percentile import percentile


                def latency_summary(samples_ms):
                    return {
                        "count": len(samples_ms),
                        "p50": percentile(samples_ms, 50),
                        "p95": percentile(samples_ms, 95),
                        "max": max(samples_ms),
                    }
                ''',
            "tests/test_percentile.py": '''\
                import pytest

                from metrics.percentile import percentile
                from metrics.summary import latency_summary


                def test_minimum():
                    assert percentile([3, 1, 2], 0) == 1
                    assert percentile([5], 0) == 5


                def test_validates_p():
                    with pytest.raises(ValueError):
                        percentile([1, 2], 101)
                    with pytest.raises(ValueError):
                        percentile([], 50)


                def test_summary_shape():
                    s = latency_summary(list(range(40, 0, -1)))
                    assert s["count"] == 40 and s["max"] == 40
                ''',
        },
        hidden_test='''\
            import pytest

            from metrics.percentile import percentile
            from metrics.summary import latency_summary


            def test_median_interpolates():
                assert percentile([1, 2, 3, 4], 50) == 2.5
                assert percentile([4, 1, 3], 50) == 3


            def test_maximum():
                assert percentile([1, 2, 3, 4], 100) == 4
                assert percentile([7], 100) == 7


            def test_p90_exact_rank():
                assert percentile(list(range(1, 12)), 90) == 10


            def test_summary_p95():
                assert latency_summary(list(range(1, 101)))["p95"] == pytest.approx(95.05)


            def test_summary_few_samples():
                assert latency_summary([5, 1, 9])["p95"] == pytest.approx(8.6)
            ''',
        p2p=["tests/test_percentile.py::test_minimum", "tests/test_percentile.py::test_validates_p",
             "tests/test_percentile.py::test_summary_shape"],
        steps=[
            ("Find the implementation.", S("def percentile", ".", "*.py")),
            ("Read it.", V("metrics/percentile.py")),
            ("Reproduce both symptoms.",
             B("python -c 'from metrics.percentile import percentile; print(percentile([1, 2, 3, 4], 50)); "
               "print(percentile([1, 2, 3, 4], 100))' 2>&1 | tail -2")),
            ("For n sorted values the rank positions run 0..n-1, so the fractional rank is "
             "(n - 1) * p / 100, not n * p / 100.",
             R("metrics/percentile.py", "    k = len(s) * p / 100\n", "    k = (len(s) - 1) * p / 100\n")),
            ("Verify p50, p100 and a p95 over 1..100 (numpy gives 95.05).",
             B("python -c 'from metrics.percentile import percentile; "
               "print(percentile([1, 2, 3, 4], 50), percentile([1, 2, 3, 4], 100), percentile(list(range(1, 101)), 95))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 48 ---------------------------------------------------------------------------------------
    dict(
        id="seed-048-histogram-drops-max",
        bug_type="boundary-condition", difficulty="easy",
        issue="""\
            Histogram drops the largest sample

            When I call `histogram(values, bins)` without explicit bounds, the counts add up
            to `len(values) - 1` (or less, if the max value appears several times). E.g.
            `histogram([1, 2, 3, 4], 3)` gives `[1, 1, 1]`. The docstring says the last bin
            includes `hi`.
            """,
        files={
            "charts/__init__.py": "",
            "charts/histogram.py": '''\
                def bin_edges(lo, hi, bins):
                    width = (hi - lo) / bins
                    return [lo + i * width for i in range(bins + 1)]


                def histogram(values, bins, lo=None, hi=None):
                    """Counts per equal-width bin over [lo, hi]; the last bin includes hi."""
                    if bins < 1:
                        raise ValueError("bins must be >= 1")
                    values = list(values)
                    lo = min(values) if lo is None else lo
                    hi = max(values) if hi is None else hi
                    if hi <= lo:
                        raise ValueError("hi must be greater than lo")
                    width = (hi - lo) / bins
                    counts = [0] * bins
                    for v in values:
                        idx = int((v - lo) / width)
                        if 0 <= idx < bins:
                            counts[idx] += 1
                    return counts
                ''',
            "charts/render.py": '''\
                from .histogram import histogram


                def ascii_hist(values, bins, width=20):
                    counts = histogram(values, bins)
                    peak = max(counts) or 1
                    return "\\n".join("#" * round(c * width / peak) for c in counts)
                ''',
            "tests/test_histogram.py": '''\
                import pytest

                from charts.histogram import bin_edges, histogram


                def test_interior_values():
                    assert histogram([0.5, 1.5, 2.5], 3, 0, 3) == [1, 1, 1]


                def test_out_of_range_ignored():
                    assert histogram([-1, 0.5, 5], 2, 0, 2) == [1, 0]


                def test_bin_edges():
                    assert bin_edges(0, 10, 2) == [0, 5, 10]


                def test_invalid_bounds():
                    with pytest.raises(ValueError):
                        histogram([1, 1], 2)
                ''',
        },
        hidden_test='''\
            from charts.histogram import histogram
            from charts.render import ascii_hist


            def test_max_value_in_last_bin():
                assert histogram([1, 2, 3, 4], 3) == [1, 1, 2]


            def test_explicit_hi_included():
                assert histogram([0, 10, 10], 2, 0, 10) == [1, 2]


            def test_counts_sum_to_sample_size():
                vals = [0.1 * i for i in range(37)]
                assert sum(histogram(vals, 7)) == len(vals)


            def test_values_above_hi_still_ignored():
                assert histogram([0, 11], 2, 0, 10) == [1, 0]


            def test_render_includes_max():
                assert ascii_hist([1, 4, 4], 3, width=4).splitlines()[-1] == "####"
            ''',
        p2p=["tests/test_histogram.py::test_interior_values", "tests/test_histogram.py::test_out_of_range_ignored",
             "tests/test_histogram.py::test_bin_edges", "tests/test_histogram.py::test_invalid_bounds"],
        steps=[
            ("Open the histogram code.", V("charts/histogram.py")),
            ("Reproduce.",
             B("python -c 'from charts.histogram import histogram; print(histogram([1, 2, 3, 4], 3), histogram([0, 10], 2, 0, 10))'")),
            ("A value equal to hi maps to idx == bins and is skipped by the range check. The last bin "
             "is closed on the right, so map v == hi into it (values above hi stay excluded).",
             I("charts/histogram.py", 18, "        if v == hi:\n            idx = bins - 1")),
            ("Check the edited loop.", V("charts/histogram.py", 17, 23)),
            ("Verify, including that out-of-range values are still dropped.",
             B("python -c 'from charts.histogram import histogram; "
               "print(histogram([1, 2, 3, 4], 3), histogram([0, 10], 2, 0, 10), histogram([0, 11], 2, 0, 10))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 49 ---------------------------------------------------------------------------------------
    dict(
        id="seed-049-welford-merge-variance",
        bug_type="incorrect-formula", difficulty="hard",
        issue="""\
            Fleet stddev on the dashboard is far too low

            We compute per-host latency stats and combine them for the fleet view. The
            combined stddev doesn't match what I get from the raw export:

                >>> from telemetry.aggregate import fleet_summary
                >>> fleet_summary({"a": [1, 2, 3], "b": [10, 11, 12]})
                {'n': 6, 'mean': 6.5, 'stddev': 0.894}
                >>> statistics.stdev([1, 2, 3, 10, 11, 12])
                5.009...

            n and mean are right. Per-host numbers look fine too; it only goes wrong once
            hosts are combined, and seems worse the more the host averages differ.
            """,
        files={
            "telemetry/__init__.py": "",
            "telemetry/running.py": '''\
                import math


                class RunningStats:
                    """Streaming mean/variance (Welford). Shards are combined with merge()."""

                    def __init__(self):
                        self.n = 0
                        self.mean = 0.0
                        self.m2 = 0.0

                    def push(self, x):
                        self.n += 1
                        delta = x - self.mean
                        self.mean += delta / self.n
                        self.m2 += delta * (x - self.mean)

                    def merge(self, other):
                        """Return new stats equivalent to having pushed both streams."""
                        out = RunningStats()
                        out.n = self.n + other.n
                        if out.n == 0:
                            return out
                        delta = other.mean - self.mean
                        out.mean = self.mean + delta * other.n / out.n
                        out.m2 = self.m2 + other.m2
                        return out

                    @property
                    def variance(self):
                        """Sample variance (n - 1 denominator)."""
                        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

                    @property
                    def stddev(self):
                        return math.sqrt(self.variance)
                ''',
            "telemetry/aggregate.py": '''\
                from functools import reduce

                from .running import RunningStats


                def shard_stats(samples):
                    rs = RunningStats()
                    for x in samples:
                        rs.push(x)
                    return rs


                def combine(shards):
                    return reduce(lambda a, b: a.merge(b), shards, RunningStats())


                def fleet_summary(samples_by_host):
                    total = combine(shard_stats(s) for s in samples_by_host.values())
                    return {"n": total.n, "mean": round(total.mean, 3), "stddev": round(total.stddev, 3)}
                ''',
            "tests/test_running.py": '''\
                import pytest

                from telemetry.aggregate import combine, fleet_summary, shard_stats
                from telemetry.running import RunningStats


                def test_push_variance():
                    rs = shard_stats([2, 4, 4, 4, 5, 5, 7, 9])
                    assert rs.mean == pytest.approx(5.0)
                    assert rs.variance == pytest.approx(32 / 7)


                def test_merge_with_empty():
                    rs = shard_stats([1.0, 2.0, 4.0])
                    merged = RunningStats().merge(rs)
                    assert (merged.n, merged.mean, merged.m2) == pytest.approx((rs.n, rs.mean, rs.m2))


                def test_combine_same_mean_shards():
                    total = combine([shard_stats([1, 3]), shard_stats([0, 4])])
                    assert total.variance == pytest.approx(shard_stats([1, 3, 0, 4]).variance)


                def test_fleet_counts_and_mean():
                    s = fleet_summary({"a": [1, 2], "b": [3, 4, 5]})
                    assert s["n"] == 5 and s["mean"] == 3.0
                ''',
        },
        hidden_test='''\
            import statistics

            import pytest

            from telemetry.aggregate import combine, fleet_summary, shard_stats


            def test_merge_matches_single_stream():
                a, b = [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]
                merged = shard_stats(a).merge(shard_stats(b))
                assert merged.variance == pytest.approx(statistics.variance(a + b))


            def test_combine_many_uneven_shards():
                shards = [[5.0], [1.0, 9.0, 2.5], [100.0, 101.0], [-3.0, 0.0, 3.0, 6.0]]
                flat = [x for s in shards for x in s]
                total = combine(shard_stats(s) for s in shards)
                assert total.mean == pytest.approx(statistics.mean(flat))
                assert total.stddev == pytest.approx(statistics.stdev(flat))


            def test_fleet_summary_stddev():
                s = fleet_summary({"a": [1, 2, 3], "b": [10, 11, 12]})
                assert s["stddev"] == pytest.approx(round(statistics.stdev([1, 2, 3, 10, 11, 12]), 3))


            def test_merge_is_order_independent():
                a, b = shard_stats([1.0, 2.0]), shard_stats([7.0, 8.0, 9.0])
                assert a.merge(b).m2 == pytest.approx(b.merge(a).m2)
            ''',
        p2p=["tests/test_running.py::test_push_variance", "tests/test_running.py::test_merge_with_empty",
             "tests/test_running.py::test_combine_same_mean_shards", "tests/test_running.py::test_fleet_counts_and_mean"],
        steps=[
            ("Start where the fleet summary is computed.", V("telemetry/aggregate.py")),
            ("Per-host stats are fine per the report, so the suspect is RunningStats.merge.",
             V("telemetry/running.py")),
            ("Confirm push() is correct and merge() is not, against statistics.variance.",
             B("python -c 'import statistics; from telemetry.aggregate import shard_stats; "
               "a, b = [1, 2, 3], [10, 11, 12]; print(shard_stats(a + b).variance, "
               "shard_stats(a).merge(shard_stats(b)).variance, statistics.variance(a + b))'")),
            ("merge() just adds the two M2 sums, which ignores the spread between the shard means. "
             "Chan's parallel formula adds delta^2 * n_a * n_b / n.",
             R("telemetry/running.py",
               "        out.m2 = self.m2 + other.m2\n",
               "        out.m2 = self.m2 + other.m2 + delta * delta * self.n * other.n / out.n\n")),
            ("Re-run the comparison and the dashboard call.",
             B("python -c 'import statistics; from telemetry.aggregate import shard_stats, fleet_summary; "
               "a, b = [1, 2, 3], [10, 11, 12]; print(shard_stats(a).merge(shard_stats(b)).variance, "
               "statistics.variance(a + b)); print(fleet_summary({\"a\": a, \"b\": b}))'")),
            ("Run the suite (merging with an empty shard must still be a no-op).", T()),
            ("Combined variance now matches the single-stream result.", SUBMIT),
        ],
    ),
    # 50 ---------------------------------------------------------------------------------------
    dict(
        id="seed-050-kelvin-to-fahrenheit-sign",
        bug_type="sign-error", difficulty="easy",
        issue="""\
            Weather widget shows -32°F at freezing

            With the unit set to Fahrenheit, a reading of 273.15 K shows as `-32°F` and
            boiling water (373.15 K) as `148°F`. Celsius and Kelvin displays look correct.
            """,
        files={
            "weather/__init__.py": "",
            "weather/units.py": '''\
                def c_to_k(c):
                    return c + 273.15


                def k_to_c(k):
                    return k - 273.15


                def f_to_k(f):
                    return (f - 32) * 5 / 9 + 273.15


                def k_to_f(k):
                    return (k - 273.15) * 9 / 5 - 32


                _TO_K = {"C": c_to_k, "F": f_to_k, "K": lambda k: k}
                _FROM_K = {"C": k_to_c, "F": k_to_f, "K": lambda k: k}


                def convert_temp(value, src, dst):
                    if src not in _TO_K or dst not in _FROM_K:
                        raise ValueError(f"unsupported unit: {src} -> {dst}")
                    return _FROM_K[dst](_TO_K[src](value))
                ''',
            "weather/display.py": '''\
                from .units import convert_temp


                def format_reading(kelvin, unit="C"):
                    return f"{round(convert_temp(kelvin, 'K', unit))}°{unit}"
                ''',
            "tests/test_units.py": '''\
                import pytest

                from weather.display import format_reading
                from weather.units import convert_temp, f_to_k


                def test_f_to_k():
                    assert f_to_k(32) == pytest.approx(273.15)


                def test_celsius_paths():
                    assert convert_temp(100, "C", "K") == pytest.approx(373.15)
                    assert convert_temp(212, "F", "C") == pytest.approx(100)


                def test_display_celsius():
                    assert format_reading(293.15, "C") == "20°C"


                def test_unknown_unit():
                    with pytest.raises(ValueError):
                        convert_temp(1, "C", "R")
                ''',
        },
        hidden_test='''\
            import pytest

            from weather.display import format_reading
            from weather.units import convert_temp


            def test_celsius_to_fahrenheit():
                assert convert_temp(0, "C", "F") == pytest.approx(32)
                assert convert_temp(100, "C", "F") == pytest.approx(212)
                assert convert_temp(-40, "C", "F") == pytest.approx(-40)


            def test_display_fahrenheit():
                assert format_reading(273.15, "F") == "32°F"


            def test_fahrenheit_round_trip():
                assert convert_temp(convert_temp(98.6, "F", "C"), "C", "F") == pytest.approx(98.6)
            ''',
        p2p=["tests/test_units.py::test_f_to_k", "tests/test_units.py::test_celsius_paths",
             "tests/test_units.py::test_display_celsius", "tests/test_units.py::test_unknown_unit"],
        steps=[
            ("Find the Fahrenheit conversions.", S("def (f_to_k|k_to_f)", "weather")),
            ("Reproduce the widget values.",
             B("python -c 'from weather.display import format_reading; print(format_reading(273.15, \"F\"), format_reading(373.15, \"F\"))'")),
            ("Look at k_to_f.", V("weather/units.py", 13, 14)),
            ("F = C * 9/5 + 32; the offset is subtracted instead of added.",
             R("weather/units.py", "return (k - 273.15) * 9 / 5 - 32", "return (k - 273.15) * 9 / 5 + 32")),
            ("Verify freezing, boiling and the -40 crossover.",
             B("python -c 'from weather.units import convert_temp as c; print(c(0, \"C\", \"F\"), c(100, \"C\", \"F\"), c(-40, \"C\", \"F\"))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
    # 51 ---------------------------------------------------------------------------------------
    dict(
        id="seed-051-approx-equal-near-zero",
        bug_type="float-tolerance", difficulty="medium",
        issue="""\
            Ledger reports "unbalanced" for entries that cancel out

                >>> from ledger.book import Ledger
                >>> lg = Ledger()
                >>> lg.post("cash", 0.1); lg.post("cash", 0.2); lg.post("sales", -0.3)
                >>> lg.is_balanced()
                False
                >>> lg.balance()
                5.551115123125783e-17

            I know about float noise, but I thought `is_balanced` already compared with a
            tolerance. Month-end close is blocked by a bunch of these.
            """,
        files={
            "ledger/__init__.py": "",
            "ledger/numutil.py": '''\
                def approx_equal(a, b, rel_tol=1e-9):
                    """True if a and b are equal up to floating-point noise."""
                    return abs(a - b) <= rel_tol * max(abs(a), abs(b))
                ''',
            "ledger/book.py": '''\
                from .numutil import approx_equal


                class Ledger:
                    def __init__(self):
                        self.entries = []

                    def post(self, account, amount):
                        self.entries.append((account, float(amount)))

                    def balance(self, account=None):
                        return sum(amt for acc, amt in self.entries if account is None or acc == account)

                    def is_balanced(self):
                        return approx_equal(self.balance(), 0.0)
                ''',
            "tests/test_ledger.py": '''\
                from ledger.book import Ledger
                from ledger.numutil import approx_equal


                def test_relative_tolerance():
                    assert approx_equal(1e10, 1e10 + 1)
                    assert not approx_equal(1.0, 1.1)


                def test_balance_per_account():
                    lg = Ledger()
                    lg.post("cash", 10)
                    lg.post("sales", -10)
                    lg.post("cash", 5)
                    assert lg.balance("cash") == 15
                    assert lg.balance() == 5


                def test_exact_balance():
                    lg = Ledger()
                    lg.post("cash", 25)
                    lg.post("sales", -25)
                    assert lg.is_balanced()


                def test_real_imbalance_detected():
                    lg = Ledger()
                    lg.post("cash", 10.00)
                    lg.post("sales", -9.99)
                    assert not lg.is_balanced()
                ''',
        },
        hidden_test='''\
            from ledger.book import Ledger
            from ledger.numutil import approx_equal


            def test_float_noise_is_balanced():
                lg = Ledger()
                for acc, amt in [("cash", 0.1), ("cash", 0.2), ("sales", -0.3)]:
                    lg.post(acc, amt)
                assert lg.is_balanced()


            def test_compare_to_zero():
                assert approx_equal(1e-12, 0.0)
                assert approx_equal(0.0, -1e-12)


            def test_cent_difference_not_equal_to_zero():
                assert not approx_equal(0.01, 0.0)


            def test_many_small_entries():
                lg = Ledger()
                for _ in range(10):
                    lg.post("cash", 0.1)
                lg.post("sales", -1.0)
                assert lg.is_balanced()
            ''',
        p2p=["tests/test_ledger.py::test_relative_tolerance", "tests/test_ledger.py::test_balance_per_account",
             "tests/test_ledger.py::test_exact_balance", "tests/test_ledger.py::test_real_imbalance_detected"],
        steps=[
            ("Find is_balanced.", S("is_balanced|approx_equal", "ledger")),
            ("Read approx_equal.", V("ledger/numutil.py")),
            ("Reproduce: compare a tiny residue to zero.",
             B("python -c 'from ledger.numutil import approx_equal; print(approx_equal(0.1 + 0.2 - 0.3, 0.0), approx_equal(1e-12, 0.0))'")),
            ("A purely relative tolerance scales with max(|a|, |b|), which is ~0 when comparing against 0, "
             "so nothing but an exact 0 passes. Add an absolute tolerance floor like math.isclose's abs_tol.",
             R("ledger/numutil.py",
               "def approx_equal(a, b, rel_tol=1e-9):\n"
               '    """True if a and b are equal up to floating-point noise."""\n'
               "    return abs(a - b) <= rel_tol * max(abs(a), abs(b))\n",
               "def approx_equal(a, b, rel_tol=1e-9, abs_tol=1e-9):\n"
               '    """True if a and b are equal up to floating-point noise."""\n'
               "    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)\n")),
            ("Verify the ledger from the report, and that a one-cent difference still counts.",
             B("python -c 'from ledger.book import Ledger; from ledger.numutil import approx_equal; lg = Ledger(); "
               "[lg.post(a, x) for a, x in ((\"cash\", 0.1), (\"cash\", 0.2), (\"sales\", -0.3))]; "
               "print(lg.is_balanced(), approx_equal(0.01, 0.0))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 52 ---------------------------------------------------------------------------------------
    dict(
        id="seed-052-tz-replace-vs-astimezone",
        bug_type="timezone-conversion", difficulty="hard",
        issue="""\
            Daily signup counts for non-UTC customers look like they're bucketed by UTC day

            Customers with a configured UTC offset are seeing late-evening (UTC) signups on
            the wrong local day. Example: a Berlin customer (offset +120 minutes) had a signup
            logged as `2024-05-01T23:30:00Z`, which is 01:30 on May 2 for them, but the daily
            report counts it on 2024-05-01. Same problem the other way for our New York
            customers (-300). Changing the offset setting doesn't move anything at all.
            """,
        files={
            "events/__init__.py": "",
            "events/parse.py": '''\
                from datetime import datetime, timezone


                def parse_event(line):
                    """'2024-05-01T23:30:00Z,signup' -> (aware UTC datetime, kind)."""
                    ts, kind = line.strip().split(",", 1)
                    if ts.endswith("Z"):
                        ts = ts[:-1] + "+00:00"
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc), kind
                ''',
            "events/tz.py": '''\
                from datetime import timedelta, timezone


                def fixed_offset(minutes):
                    """tzinfo for a fixed UTC offset in minutes, e.g. 330 (IST) or -300 (EST)."""
                    return timezone(timedelta(minutes=minutes))


                def to_local(dt, offset_minutes):
                    """Express an aware datetime in the customer's local (fixed-offset) time."""
                    return dt.replace(tzinfo=fixed_offset(offset_minutes))


                def local_date(dt, offset_minutes):
                    return to_local(dt, offset_minutes).date()
                ''',
            "events/daily.py": '''\
                from collections import Counter

                from .parse import parse_event
                from .tz import local_date


                def daily_counts(lines, offset_minutes=0, kind=None):
                    counts = Counter()
                    for line in lines:
                        dt, k = parse_event(line)
                        if kind is None or k == kind:
                            counts[local_date(dt, offset_minutes).isoformat()] += 1
                    return dict(sorted(counts.items()))
                ''',
            "tests/test_daily.py": '''\
                from datetime import datetime, timedelta, timezone

                from events.daily import daily_counts
                from events.parse import parse_event
                from events.tz import fixed_offset


                def test_parse_converts_to_utc():
                    dt, kind = parse_event("2024-05-01T23:30:00+02:00,signup")
                    assert dt == datetime(2024, 5, 1, 21, 30, tzinfo=timezone.utc)
                    assert kind == "signup"


                def test_fixed_offset():
                    assert fixed_offset(330).utcoffset(None) == timedelta(hours=5, minutes=30)


                def test_daily_counts_utc():
                    lines = ["2024-05-01T10:00:00Z,signup", "2024-05-01T23:59:00Z,signup",
                             "2024-05-02T00:01:00Z,login"]
                    assert daily_counts(lines) == {"2024-05-01": 2, "2024-05-02": 1}
                    assert daily_counts(lines, kind="login") == {"2024-05-02": 1}
                ''',
        },
        hidden_test='''\
            from datetime import datetime, timezone

            from events.daily import daily_counts
            from events.tz import to_local


            def test_positive_offset_moves_to_next_day():
                assert daily_counts(["2024-05-01T23:30:00Z,signup"], offset_minutes=120) == {"2024-05-02": 1}


            def test_negative_offset_moves_to_previous_day():
                assert daily_counts(["2024-05-02T03:00:00Z,signup"], offset_minutes=-300) == {"2024-05-01": 1}


            def test_to_local_preserves_instant():
                utc = datetime(2024, 5, 1, 23, 30, tzinfo=timezone.utc)
                local = to_local(utc, 330)
                assert (local.day, local.hour, local.minute) == (2, 5, 0)
                assert local == utc


            def test_zero_offset_unchanged():
                assert daily_counts(["2024-05-01T23:30:00Z,signup"], offset_minutes=0) == {"2024-05-01": 1}
            ''',
        p2p=["tests/test_daily.py::test_parse_converts_to_utc", "tests/test_daily.py::test_fixed_offset",
             "tests/test_daily.py::test_daily_counts_utc"],
        steps=[
            ("Start at the report entry point.", V("events/daily.py")),
            ("Reproduce with the Berlin example.",
             B("python -c 'from events.daily import daily_counts; "
               "print(daily_counts([\"2024-05-01T23:30:00Z,signup\"], offset_minutes=120))'")),
            ("Check the parser output first: is the timestamp correct UTC?",
             B("python -c 'from events.parse import parse_event; print(parse_event(\"2024-05-01T23:30:00Z,signup\"))'")),
            ("Parsing is right, so the local-date conversion is the suspect.", V("events/tz.py")),
            ("Confirm what to_local does to the instant.",
             B("python -c 'from datetime import datetime, timezone; from events.tz import to_local; "
               "u = datetime(2024, 5, 1, 23, 30, tzinfo=timezone.utc); l = to_local(u, 120); print(l, l == u)'")),
            ("`replace(tzinfo=...)` relabels the wall-clock time with a new offset (a different instant) "
             "instead of converting it. astimezone() keeps the instant and shifts the clock.",
             R("events/tz.py",
               "    return dt.replace(tzinfo=fixed_offset(offset_minutes))\n",
               "    return dt.astimezone(fixed_offset(offset_minutes))\n")),
            ("Verify both directions.",
             B("python -c 'from events.daily import daily_counts as dc; "
               "print(dc([\"2024-05-01T23:30:00Z,signup\"], offset_minutes=120), "
               "dc([\"2024-05-02T03:00:00Z,signup\"], offset_minutes=-300))'")),
            ("Run the tests.", T()),
            ("Done.", SUBMIT),
        ],
    ),
    # 53 ---------------------------------------------------------------------------------------
    dict(
        id="seed-053-vat-inclusive-net",
        bug_type="incorrect-formula", difficulty="medium",
        issue="""\
            VAT breakdown on receipts is wrong for tax-inclusive prices

            For a 120.00 gross item at 20% VAT the receipt prints:

                net 96.00   VAT 24.00   total 120.00

            It should be net 100.00 / VAT 20.00. Also, net + VAT on the receipt has to add up
            to the gross amount to the cent; auditors check that line by line.
            """,
        files={
            "pos/__init__.py": "",
            "pos/tax.py": '''\
                from decimal import ROUND_HALF_UP, Decimal

                CENT = Decimal("0.01")


                def q(x):
                    return x.quantize(CENT, rounding=ROUND_HALF_UP)


                def add_tax(net, rate):
                    """Gross price for a net price and a rate like Decimal('0.20')."""
                    return q(Decimal(str(net)) * (1 + Decimal(str(rate))))


                def net_from_gross(gross, rate):
                    """Net price contained in a tax-inclusive gross price."""
                    return q(gross * (1 - rate))
                ''',
            "pos/receipt.py": '''\
                from decimal import Decimal

                from .tax import net_from_gross, q


                def vat_breakdown(gross, rate):
                    gross = Decimal(str(gross))
                    rate = Decimal(str(rate))
                    net = net_from_gross(gross, rate)
                    tax = q(gross * rate)
                    return {"gross": gross, "net": net, "tax": tax}


                def receipt_line(name, gross, rate):
                    b = vat_breakdown(gross, rate)
                    return f"{name}: net {b['net']}  VAT {b['tax']}  total {b['gross']}"
                ''',
            "tests/test_tax.py": '''\
                from decimal import Decimal

                from pos.receipt import receipt_line, vat_breakdown
                from pos.tax import add_tax


                def test_add_tax():
                    assert add_tax("100.00", "0.20") == Decimal("120.00")
                    assert add_tax("9.99", "0.07") == Decimal("10.69")


                def test_zero_rated():
                    assert vat_breakdown("12.50", "0") == {
                        "gross": Decimal("12.50"), "net": Decimal("12.50"), "tax": Decimal("0.00")}


                def test_receipt_line_format():
                    assert receipt_line("Book", "7.00", "0").startswith("Book: net 7.00")
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from pos.receipt import receipt_line, vat_breakdown
            from pos.tax import add_tax, net_from_gross


            def test_breakdown_twenty_percent():
                assert vat_breakdown("120.00", "0.20") == {
                    "gross": Decimal("120.00"), "net": Decimal("100.00"), "tax": Decimal("20.00")}


            def test_net_plus_tax_equals_gross():
                for gross, rate in [("9.99", "0.19"), ("0.99", "0.20"), ("1234.56", "0.075"), ("19.99", "0.21")]:
                    b = vat_breakdown(gross, rate)
                    assert b["net"] + b["tax"] == Decimal(gross)


            def test_net_from_gross_inverts_add_tax():
                rate = Decimal("0.20")
                assert net_from_gross(add_tax("49.99", rate), rate) == Decimal("49.99")


            def test_receipt_line():
                assert receipt_line("Lamp", "120.00", "0.20") == "Lamp: net 100.00  VAT 20.00  total 120.00"
            ''',
        p2p=["tests/test_tax.py::test_add_tax", "tests/test_tax.py::test_zero_rated",
             "tests/test_tax.py::test_receipt_line_format"],
        steps=[
            ("Find where the breakdown is computed.", S("net_from_gross|def vat_breakdown", "pos")),
            ("Read the receipt code.", V("pos/receipt.py")),
            ("And the tax helpers.", V("pos/tax.py")),
            ("Reproduce.", B("python -c 'from pos.receipt import receipt_line; print(receipt_line(\"Lamp\", \"120.00\", \"0.20\"))'")),
            ("gross = net * (1 + rate), so net = gross / (1 + rate); multiplying by (1 - rate) is wrong.",
             R("pos/tax.py", "    return q(gross * (1 - rate))\n", "    return q(gross / (1 + rate))\n")),
            ("Re-check, including a case where rounding matters.",
             B("python -c 'from pos.receipt import vat_breakdown; print(vat_breakdown(\"120.00\", \"0.20\")); "
               "b = vat_breakdown(\"9.99\", \"0.19\"); print(b, b[\"net\"] + b[\"tax\"])'")),
            ("Net is right now but VAT is still gross * rate (24.00), which is the rate applied to the "
             "wrong base. VAT is whatever remains after net, which also guarantees net + VAT == gross.",
             R("pos/receipt.py", "    tax = q(gross * rate)\n", "    tax = gross - net\n")),
            ("Verify the receipt and the sum invariant.",
             B("python -c 'from pos.receipt import receipt_line, vat_breakdown; print(receipt_line(\"Lamp\", \"120.00\", \"0.20\")); "
               "b = vat_breakdown(\"9.99\", \"0.19\"); print(b, b[\"net\"] + b[\"tax\"])'")),
            ("Run the tests.", T()),
            ("Both the net formula and the VAT base are fixed.", SUBMIT),
        ],
    ),
    # 54 ---------------------------------------------------------------------------------------
    dict(
        id="seed-054-allocation-lost-cents",
        bug_type="rounding-remainder", difficulty="hard",
        issue="""\
            Installment plans don't add up to the order total

            A 1000.00 order on a 3-month plan gets three installments of 333.33, so once
            they're all paid the order still shows 0.01 outstanding and never moves to
            "paid". Same with weighted splits (gift cards + card etc). The allocator's
            docstring says the parts always sum exactly to the total.
            """,
        files={
            "payments/__init__.py": "",
            "payments/money.py": '''\
                from decimal import ROUND_HALF_UP, Decimal


                def to_cents(amount):
                    return int((Decimal(str(amount)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


                def from_cents(cents):
                    return (Decimal(cents) / 100).quantize(Decimal("0.01"))
                ''',
            "payments/allocate.py": '''\
                from .money import from_cents, to_cents


                def allocate(total, ratios):
                    """Split `total` across `ratios` proportionally, to the cent.

                    The parts always sum exactly to `total`.
                    """
                    if not ratios or any(r < 0 for r in ratios) or sum(ratios) == 0:
                        raise ValueError("ratios must be non-negative and not all zero")
                    cents = to_cents(total)
                    weight = sum(ratios)
                    parts = [cents * r // weight for r in ratios]
                    return [from_cents(p) for p in parts]


                def split_evenly(total, n):
                    if n < 1:
                        raise ValueError("n must be >= 1")
                    return allocate(total, [1] * n)
                ''',
            "payments/installments.py": '''\
                from .allocate import split_evenly


                def schedule(total, months):
                    """[(month_number, amount), ...] for an interest-free installment plan."""
                    return list(enumerate(split_evenly(total, months), start=1))
                ''',
            "tests/test_allocate.py": '''\
                from decimal import Decimal

                import pytest

                from payments.allocate import allocate, split_evenly
                from payments.installments import schedule
                from payments.money import from_cents, to_cents


                def test_cents_round_trip():
                    assert to_cents("12.345") == 1235
                    assert from_cents(1235) == Decimal("12.35")


                def test_even_split_divisible():
                    assert split_evenly(Decimal("100.00"), 2) == [Decimal("50.00"), Decimal("50.00")]


                def test_weighted_divisible():
                    assert allocate(Decimal("10.00"), [3, 1]) == [Decimal("7.50"), Decimal("2.50")]


                def test_rejects_bad_ratios():
                    with pytest.raises(ValueError):
                        allocate(Decimal("1.00"), [0, 0])


                def test_schedule_numbering():
                    assert [m for m, _ in schedule(Decimal("90.00"), 3)] == [1, 2, 3]
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from payments.allocate import allocate, split_evenly
            from payments.installments import schedule


            def test_split_three_ways_sums_to_total():
                assert split_evenly(Decimal("1000.00"), 3) == [
                    Decimal("333.34"), Decimal("333.33"), Decimal("333.33")]


            def test_small_amount_split():
                assert split_evenly(Decimal("0.05"), 3) == [Decimal("0.02"), Decimal("0.02"), Decimal("0.01")]


            def test_weighted_split_largest_remainder():
                assert allocate(Decimal("0.10"), [1, 1, 1, 7]) == [
                    Decimal("0.01"), Decimal("0.01"), Decimal("0.01"), Decimal("0.07")]
                parts = allocate(Decimal("100.00"), [1, 2, 3])
                assert parts == [Decimal("16.67"), Decimal("33.33"), Decimal("50.00")]


            def test_parts_always_sum_to_total():
                for total, ratios in [("99.99", [1, 1, 1, 1]), ("0.07", [5, 3]), ("250.00", [2, 2, 3])]:
                    assert sum(allocate(Decimal(total), ratios)) == Decimal(total)


            def test_schedule_totals():
                assert sum(amount for _, amount in schedule(Decimal("1000.00"), 3)) == Decimal("1000.00")


            def test_divisible_unchanged():
                assert allocate(Decimal("10.00"), [3, 1]) == [Decimal("7.50"), Decimal("2.50")]
            ''',
        p2p=["tests/test_allocate.py::test_cents_round_trip", "tests/test_allocate.py::test_even_split_divisible",
             "tests/test_allocate.py::test_weighted_divisible", "tests/test_allocate.py::test_rejects_bad_ratios",
             "tests/test_allocate.py::test_schedule_numbering"],
        steps=[
            ("Trace how installment amounts are produced.", S("split_evenly|def allocate", "payments")),
            ("schedule delegates to split_evenly -> allocate. Read the allocator.", V("payments/allocate.py")),
            ("Reproduce the 1000/3 case and a weighted case.",
             B("python -c 'from decimal import Decimal as D; from payments.allocate import allocate, split_evenly; "
               "p = split_evenly(D(\"1000.00\"), 3); print(p, sum(p)); q = allocate(D(\"100.00\"), [1, 2, 3]); print(q, sum(q))'")),
            ("Flooring every share loses up to len(ratios)-1 cents. Use the largest-remainder method: "
             "floor each share, then hand the leftover cents one each to the shares with the biggest "
             "remainders (stable sort, so ties go to the earliest share).",
             R("payments/allocate.py",
               "    parts = [cents * r // weight for r in ratios]\n    return [from_cents(p) for p in parts]\n",
               "    parts = [cents * r // weight for r in ratios]\n"
               "    remainders = [cents * r % weight for r in ratios]\n"
               "    leftover = cents - sum(parts)\n"
               "    for i in sorted(range(len(ratios)), key=lambda i: -remainders[i])[:leftover]:\n"
               "        parts[i] += 1\n"
               "    return [from_cents(p) for p in parts]\n")),
            ("Verify sums for several awkward splits.",
             B("python -c 'from decimal import Decimal as D; from payments.allocate import allocate, split_evenly; "
               "print(split_evenly(D(\"1000.00\"), 3), allocate(D(\"100.00\"), [1, 2, 3])); "
               "print(all(sum(allocate(D(t), r)) == D(t) for t, r in [(\"99.99\", [1, 1, 1, 1]), (\"0.07\", [5, 3]), (\"0.05\", [1, 1, 1])]))'")),
            ("Check the installment schedule end to end.",
             B("python -c 'from decimal import Decimal as D; from payments.installments import schedule; "
               "s = schedule(D(\"1000.00\"), 3); print(s, sum(a for _, a in s))'")),
            ("Run the tests.", T()),
            ("Parts now always sum to the total.", SUBMIT),
        ],
    ),
    # 55 ---------------------------------------------------------------------------------------
    dict(
        id="seed-055-overnight-shift-negative",
        bug_type="time-wraparound", difficulty="medium",
        issue="""\
            Negative pay for night shift

            Payroll preview for one of our night porters shows `-320.00`. Their timesheet
            has a single shift clocked in at 22:00 and out at 06:00, at 20.00/hour, so it
            should be 160.00. Day shifts are fine.
            """,
        files={
            "timesheet/__init__.py": "",
            "timesheet/clock.py": '''\
                def parse_hhmm(text):
                    """'HH:MM' -> minutes since midnight."""
                    h, m = (int(part) for part in text.strip().split(":"))
                    if not (0 <= h < 24 and 0 <= m < 60):
                        raise ValueError(f"invalid time {text!r}")
                    return h * 60 + m


                def shift_minutes(start, end):
                    """Length of a shift from 'HH:MM' clock-in/out times. Shifts may cross midnight."""
                    return parse_hhmm(end) - parse_hhmm(start)
                ''',
            "timesheet/payroll.py": '''\
                from decimal import ROUND_HALF_UP, Decimal

                from .clock import shift_minutes


                def hours_worked(shifts):
                    return Decimal(sum(shift_minutes(start, end) for start, end in shifts)) / 60


                def gross_pay(shifts, hourly_rate):
                    pay = hours_worked(shifts) * Decimal(str(hourly_rate))
                    return pay.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ''',
            "tests/test_payroll.py": '''\
                from decimal import Decimal

                import pytest

                from timesheet.clock import parse_hhmm, shift_minutes
                from timesheet.payroll import gross_pay


                def test_parse():
                    assert parse_hhmm("09:30") == 570
                    with pytest.raises(ValueError):
                        parse_hhmm("24:00")


                def test_day_shift():
                    assert shift_minutes("09:00", "17:30") == 510


                def test_gross_pay_day():
                    assert gross_pay([("09:00", "17:00"), ("12:00", "12:45")], 15) == Decimal("131.25")
                ''',
        },
        hidden_test='''\
            from decimal import Decimal

            from timesheet.clock import shift_minutes
            from timesheet.payroll import gross_pay, hours_worked


            def test_overnight_shift():
                assert shift_minutes("22:00", "06:00") == 480
                assert shift_minutes("23:45", "00:15") == 30


            def test_night_pay():
                assert gross_pay([("22:00", "06:00")], 20) == Decimal("160.00")


            def test_mixed_week():
                shifts = [("09:00", "17:00"), ("20:00", "02:30"), ("08:15", "12:00")]
                assert hours_worked(shifts) == Decimal("18.25")


            def test_day_shift_unchanged():
                assert shift_minutes("06:00", "14:00") == 480
            ''',
        p2p=["tests/test_payroll.py::test_parse", "tests/test_payroll.py::test_day_shift",
             "tests/test_payroll.py::test_gross_pay_day"],
        steps=[
            ("Look at how pay is computed.", V("timesheet/payroll.py")),
            ("Hours come from shift_minutes; find it.", F("timesheet/*.py")),
            ("Read clock.py.", V("timesheet/clock.py")),
            ("Reproduce the night-porter case.",
             B("python -c 'from timesheet.clock import shift_minutes; from timesheet.payroll import gross_pay; "
               "print(shift_minutes(\"22:00\", \"06:00\"), gross_pay([(\"22:00\", \"06:00\")], 20))'")),
            ("When clock-out is earlier than clock-in the shift crossed midnight; wrap the difference "
             "into a 24h day.",
             R("timesheet/clock.py",
               "    return parse_hhmm(end) - parse_hhmm(start)\n",
               "    return (parse_hhmm(end) - parse_hhmm(start)) % (24 * 60)\n")),
            ("Verify overnight and day shifts.",
             B("python -c 'from timesheet.clock import shift_minutes as s; from timesheet.payroll import gross_pay; "
               "print(s(\"22:00\", \"06:00\"), s(\"23:45\", \"00:15\"), s(\"09:00\", \"17:30\"), gross_pay([(\"22:00\", \"06:00\")], 20))'")),
            ("Run the tests.", B(TEST_CMD)),
            ("Done.", SUBMIT),
        ],
    ),
]
