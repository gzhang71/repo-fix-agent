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
