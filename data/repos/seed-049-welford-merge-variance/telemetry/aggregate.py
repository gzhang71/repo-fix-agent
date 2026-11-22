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
