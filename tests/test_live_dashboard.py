import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from monitoring import live_dashboard as dash


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "deploy" / "run").mkdir(parents=True)
    monkeypatch.setattr(dash, "ROOT", tmp_path)
    monkeypatch.setattr(dash, "STORE_DIR", tmp_path / "store")
    (tmp_path / "store").mkdir()

    def write(pid=None, heartbeat_age_s=None, started_ago_s=300):
        if pid is not None:
            pidfile = tmp_path / "deploy" / "run" / "consumer.pid"
            pidfile.write_text(str(pid))
            t = time.time() - started_ago_s
            os.utime(pidfile, (t, t))
        if heartbeat_age_s is not None:
            at = datetime.now(timezone.utc) - timedelta(seconds=heartbeat_age_s)
            (tmp_path / "store" / "consumer_metrics.json").write_text(
                json.dumps({"at": at.isoformat(timespec="seconds"), "duplicates": 0, "dead_lettered": 0}))
        return dash.Collector().consumer()
    return write


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_no_pidfile_is_stopped(setup):
    c = setup(heartbeat_age_s=1)
    assert (c["state"], c["alive"]) == ("stopped", False)


def test_dead_pid_is_stopped(setup):
    assert setup(pid=dead_pid(), heartbeat_age_s=1)["state"] == "stopped"


def test_live_pid_fresh_heartbeat_is_running(setup):
    c = setup(pid=os.getpid(), heartbeat_age_s=2)
    assert (c["state"], c["alive"]) == ("running", True)
    assert 0 <= c["heartbeat_age_s"] < 10


def test_live_pid_stale_heartbeat_is_stalled(setup):
    c = setup(pid=os.getpid(), heartbeat_age_s=120)
    assert (c["state"], c["alive"]) == ("stalled", False)


def test_live_pid_without_metrics_is_stalled(setup):
    c = setup(pid=os.getpid())
    assert (c["state"], c["heartbeat_age_s"]) == ("stalled", None)


def test_just_restarted_is_starting_not_running_on_old_heartbeat(setup):
    # previous process wrote a heartbeat 20 s ago; the new one started 5 s ago and has not written yet
    c = setup(pid=os.getpid(), heartbeat_age_s=20, started_ago_s=5)
    assert (c["state"], c["alive"], c["heartbeat_age_s"]) == ("starting", False, None)


def test_restarted_without_heartbeat_after_grace_is_stalled(setup):
    assert setup(pid=os.getpid(), heartbeat_age_s=100, started_ago_s=60)["state"] == "stalled"


def test_probe_never_signals_the_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(5):
            assert dash.pid_alive(child.pid)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()
    assert not dash.pid_alive(dead_pid())
