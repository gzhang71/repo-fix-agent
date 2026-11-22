from .percentile import percentile


def latency_summary(samples_ms):
    return {
        "count": len(samples_ms),
        "p50": percentile(samples_ms, 50),
        "p95": percentile(samples_ms, 95),
        "max": max(samples_ms),
    }
