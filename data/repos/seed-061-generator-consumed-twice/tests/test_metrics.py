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
