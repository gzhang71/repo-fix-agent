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
