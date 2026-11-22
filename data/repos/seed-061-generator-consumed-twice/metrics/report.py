from .reader import read_latencies
from .summary import summarize


def report(lines):
    s = summarize(read_latencies(lines))
    if s["count"] == 0:
        return "no samples"
    return f"{s['count']} samples, mean {s['mean']:.1f}ms, max {s['max']:.1f}ms ({s['slowest']})"
