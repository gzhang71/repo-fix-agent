from ratelimit.clock import FakeClock
from ratelimit.limiter import RateLimiter


def test_initial_burst_limited():
    lim = RateLimiter(3, 1.0, FakeClock())
    assert [lim.allow("k") for _ in range(4)] == [True, True, True, False]


def test_refills_over_time():
    clock = FakeClock()
    lim = RateLimiter(2, 1.0, clock)
    lim.allow("k"); lim.allow("k")
    assert not lim.allow("k")
    clock.advance(1.0)
    assert lim.allow("k")


def test_keys_independent():
    lim = RateLimiter(1, 1.0, FakeClock())
    assert lim.allow("a") and lim.allow("b")
    assert not lim.allow("a")
