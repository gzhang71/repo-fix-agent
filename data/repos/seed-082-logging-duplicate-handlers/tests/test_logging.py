import io

from applog.reload import apply_config
from applog.worker import process


def test_worker_logs_info():
    buf = io.StringIO()
    apply_config({"log_level": "info"}, buf)
    process(["a"])
    assert "INFO applog.worker: processing a" in buf.getvalue()


def test_warning_level_hides_info():
    buf = io.StringIO()
    apply_config({"log_level": "warning"}, buf)
    process(["a"])
    assert "processing" not in buf.getvalue()
    assert "WARNING applog.worker: done: 1 jobs" in buf.getvalue()
