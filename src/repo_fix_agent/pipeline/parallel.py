"""Ordered parallel map for rollouts.

Episodes are I/O-bound (model API calls, sandbox subprocesses / docker exec), so threads are
enough. Results always come back in input order, so outputs are identical to a sequential run
given deterministic policies. Callers that fan out at two levels (instances -> episodes) must use
two pools: a coordinator blocked on episode futures inside the episode pool would deadlock it.
"""
from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Iterable[T], workers: int = 1, pool: Executor | None = None) -> list[R]:
    items = list(items)
    if pool is not None:
        return [f.result() for f in [pool.submit(fn, x) for x in items]]
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(workers, len(items)), thread_name_prefix="rollout") as ex:
        return list(ex.map(fn, items))
