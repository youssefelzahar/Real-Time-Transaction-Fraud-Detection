"""Graceful stop of the consumer and the API instances (stop file, src/shutdown.py), the
producer ledger, and counter semantics across restarts. No real broker: the API subprocess test
points at 127.0.0.1:1, where nothing listens."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import InMemoryAccountService, app
from monitoring.live_dashboard import ledger_summary
from src.decision_engine import DecisionEngine, DecisionPolicy
from src.event_consumer import ConsumerRunner, EventProcessor
from src.events import KafkaEventPublisher, NullPublisher, ProducerLedger
from src.shutdown import watch_stop_file
from tests.test_event_consumer import FakeDlq, FakeMsg, MemorySink, raw_event
from tests.test_events import FakeProducer

ROOT = Path(__file__).resolve().parent.parent


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


# --- stop file -------------------------------------------------------------------------

def test_stop_file_fires_once_only_after_it_exists(tmp_path):
    calls = []
    stop = tmp_path / "x.stop"
    t = watch_stop_file(stop, lambda: calls.append(1), interval_s=0.01)
    time.sleep(0.1)
    assert calls == []
    stop.write_text("now")
    assert wait_for(lambda: calls == [1])
    t.join(1)
    assert not t.is_alive() and calls == [1]


def test_stop_watching_ends_watcher_without_stopping(tmp_path):
    calls = []
    t = watch_stop_file(tmp_path / "never.stop", lambda: calls.append(1), interval_s=0.01)
    t.stop_watching.set()
    t.join(1)
    assert not t.is_alive() and calls == []


# --- consumer --------------------------------------------------------------------------

class EndlessConsumer:
    """Delivers events forever (offsets 0, 1, 2 ... on partition 0) until the runner stops."""

    def __init__(self):
        self.next_offset = 0
        self.commits = []

    def poll(self, timeout):
        msg = FakeMsg(raw_event(), self.next_offset)
        self.next_offset += 1
        time.sleep(0.001)
        return msg

    def commit(self, offsets, asynchronous):
        self.commits.append({(tp.topic, tp.partition): tp.offset for tp in offsets})

    def assignment(self):
        return []


def test_consumer_stop_file_commits_everything_processed(tmp_path):
    consumer = EndlessConsumer()
    sink = MemorySink()
    runner = ConsumerRunner(consumer, EventProcessor(sink), FakeDlq(), "events.dlq", commit_interval_s=999)
    stop = tmp_path / "consumer.stop"
    watch_stop_file(stop, runner.stop, interval_s=0.01)
    threading.Timer(0.2, lambda: stop.write_text("now")).start()
    assert runner.run(poll_timeout_s=0) == 0
    stored = len(sink.events)
    assert stored > 0
    # Final commit covers exactly the processed events: nothing lost, nothing re-read on restart.
    assert consumer.commits[-1] == {("events", 0): stored}


def test_consumer_finishes_current_message_when_stop_arrives_mid_write():
    """Stop requested while an event is being written: that event is still stored and committed."""
    consumer = EndlessConsumer()

    class StopDuringWrite(MemorySink):
        def write(self, event):
            super().write(event)
            if len(self.events) == 3:
                runner.stop()

    sink = StopDuringWrite()
    runner = ConsumerRunner(consumer, EventProcessor(sink), FakeDlq(), "events.dlq", commit_interval_s=999)
    runner.run(poll_timeout_s=0)
    assert len(sink.events) == 3 and consumer.commits[-1] == {("events", 0): 3}


def test_graceful_stop_does_not_commit_unacknowledged_dead_letters():
    """Offsets are committed only after storage or DLQ acknowledgement, also at shutdown."""
    consumer = EndlessConsumer()
    consumer.poll = lambda timeout: (runner.stop(), FakeMsg(b"{broken", 0))[1]
    runner = ConsumerRunner(consumer, EventProcessor(MemorySink()), FakeDlq(unacked=1), "events.dlq",
                            commit_interval_s=999)
    runner.run(poll_timeout_s=0)
    assert runner.counts["dead_lettered"] == 1 and consumer.commits == []
    assert runner.pending == {("events", 0): 1}        # re-read after restart (at-least-once)


# --- API lifespan + producer ledger ------------------------------------------------------

@pytest.fixture
def api_state(model_dir):
    from src.predict import FraudPredictor
    app.state.predictor = FraudPredictor.load(model_dir)
    app.state.account_service = InMemoryAccountService()
    app.state.decision_engine = DecisionEngine(DecisionPolicy("dp-test", 0.54, 0.999, valid_for_models=("e6_test",)))
    yield app
    for name in ("predictor", "account_service", "decision_engine", "event_publisher"):
        if hasattr(app.state, name):
            delattr(app.state, name)


BODY = {"type": "TRANSFER", "amount": 181.0, "sender_id": "C1", "receiver_id": "C2",
        "sender_balance": 181.0, "receiver_balance": 0.0}


def read_ledger(directory, instance="dev"):
    return [json.loads(line) for line in
            (Path(directory) / f"producer-ledger-instance-{instance}.jsonl").read_text().splitlines()]


def test_stop_file_triggers_graceful_shutdown_request(api_state, tmp_path, monkeypatch):
    requested = []
    monkeypatch.setattr(api_main, "request_shutdown", lambda: requested.append(1))
    monkeypatch.setenv("NILEPAY_STOP_FILE", str(tmp_path / "instance.stop"))
    app.state.event_publisher = NullPublisher()
    with TestClient(app) as c:
        assert c.post("/predict", json=BODY).status_code == 200
        (tmp_path / "instance.stop").write_text("now")
        assert wait_for(lambda: requested == [1])


def test_shutdown_records_pending_events_as_unconfirmed(api_state, tmp_path, monkeypatch):
    """Broker unreachable at shutdown: the bounded flush ends, pending events are recorded as
    unconfirmed (not as delivered, not as proven lost), and shutdown does not hang."""
    monkeypatch.setenv("EVENT_LEDGER_DIR", str(tmp_path))
    pub = KafkaEventPublisher({"queue.buffering.max.messages": 100}, producer=FakeProducer(outcome="stuck"),
                              shutdown_flush_s=0.2, poll_interval_s=0.01)
    app.state.event_publisher = pub
    with TestClient(app) as c:
        for i in range(4):
            assert c.post("/predict", json={**BODY, "transaction_id": f"p-{i}"}).status_code == 200
        assert c.get("/metrics/events").json()["in_flight"] == 4
        started = time.monotonic()
    assert time.monotonic() - started < 3
    start, stop = read_ledger(tmp_path)
    assert (start["record"], stop["record"]) == ("start", "stop") and start["pid"] == stop["pid"] == os.getpid()
    assert (stop["enqueued"], stop["delivered"], stop["unconfirmed_at_shutdown"]) == (4, 0, 4)


def test_shutdown_flush_delivers_and_records_zero_unconfirmed(api_state, tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_LEDGER_DIR", str(tmp_path))
    producer = FakeProducer(outcome="stuck")
    app.state.event_publisher = KafkaEventPublisher({"queue.buffering.max.messages": 100}, producer=producer,
                                                    shutdown_flush_s=1, poll_interval_s=0.01)
    with TestClient(app) as c:
        for i in range(3):
            c.post("/predict", json={**BODY, "transaction_id": f"d-{i}"})
        producer.outcome = None                        # broker back: the shutdown flush delivers them
    stop = read_ledger(tmp_path)[-1]
    assert (stop["delivered"], stop["unconfirmed_at_shutdown"]) == (3, 0)


def test_kafka_disabled_unchanged_with_ledger(api_state, tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_LEDGER_DIR", str(tmp_path))
    app.state.event_publisher = NullPublisher()
    with TestClient(app) as c:
        assert c.get("/metrics/events").json() == {"kafka_enabled": False}
        assert c.post("/predict", json=BODY).status_code == 200
    start, stop = read_ledger(tmp_path)
    assert start["kafka_enabled"] is False and stop["kafka_enabled"] is False and "enqueued" not in stop


def test_ledger_write_failure_never_breaks_shutdown(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    ledger = ProducerLedger(blocker / "sub" / "ledger.jsonl", 1)    # parent is a file: mkdir fails
    ledger.record_start(NullPublisher())
    ledger.record_stop(NullPublisher())


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_uvicorn_instance_stops_gracefully_via_stop_file(model_dir, tmp_path):
    """The deployed mechanism end to end, on this OS: uvicorn subprocess, stop file ->
    in-process SIGINT -> uvicorn graceful shutdown -> lifespan flush -> ledger. Broker
    unreachable (127.0.0.1:1), so the events are still pending at shutdown."""
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"policy_version": "dp-test", "approve_threshold": 0.54, "decline_threshold": 0.999,
                                  "decline_mode": "shadow", "valid_for_models": ["e6_test"]}))
    port, stop = free_port(), tmp_path / "instance.stop"
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAFKA_")}
    env.update(MODEL_DIR=str(model_dir), DECISION_POLICY=str(policy), INSTANCE_ID="9", TIMING_LOG="0",
               KAFKA_BOOTSTRAP_SERVERS="127.0.0.1:1", KAFKA_SHUTDOWN_FLUSH_S="1", KAFKA_TOPIC_PREDICTIONS="t.never",
               NILEPAY_STOP_FILE=str(stop), EVENT_LEDGER_DIR=str(tmp_path))
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port),
                             "--timeout-graceful-shutdown", "2", "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=flags)
    try:
        url = f"http://127.0.0.1:{port}"
        assert wait_for(lambda: _ok(url + "/health"), timeout=30), proc.stdout.read1().decode() if proc.poll() else ""
        for i in range(5):
            assert httpx.post(url + "/predict", json={**BODY, "transaction_id": f"u-{i}"}).status_code == 200
        started = time.monotonic()
        stop.write_text("now")
        proc.wait(timeout=15)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
    out = proc.stdout.read().decode(errors="replace")
    assert "stop requested" in out and "not confirmed" in out
    assert elapsed < 8                                       # graceful (2) + flush (1) + margin, never unbounded
    start, last = read_ledger(tmp_path, instance=9)
    assert (start["record"], last["record"]) == ("start", "stop")
    assert (last["enqueued"], last["delivered"], last["unconfirmed_at_shutdown"]) == (5, 0, 5)


def _ok(url):
    try:
        return httpx.get(url, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False


# --- counter semantics across restarts ----------------------------------------------------

def write_ledger(directory, instance, records):
    with open(Path(directory) / f"producer-ledger-instance-{instance}.jsonl", "a") as f:
        for r in records:
            f.write(json.dumps({"instance_id": instance, "at": "2026-10-29T08:00:00+00:00", **r}) + "\n")


def stop_rec(pid, enqueued, delivered, unconfirmed=0, timed_out=0, dropped=0, started="2026-10-29T07:00:00+00:00"):
    return {"record": "stop", "pid": pid, "enqueued": enqueued, "delivered": delivered,
            "failed": {"timed_out": timed_out, "broker_unavailable": 0, "other": 0}, "buffer_full_drops": dropped,
            "serialization_errors": 0, "produce_errors": 0, "unconfirmed_at_shutdown": unconfirmed,
            "process_started_at": started}


def test_ledger_summary_counts_each_ended_process_once(tmp_path):
    write_ledger(tmp_path, 1, [{"record": "start", "pid": 10}, stop_rec(10, 100, 98, unconfirmed=2),
                               {"record": "start", "pid": 11}, stop_rec(11, 50, 45, timed_out=3, dropped=2),
                               {"record": "start", "pid": 12}])                 # 12 = running now
    write_ledger(tmp_path, 2, [{"record": "start", "pid": 20}])                 # force-killed: no stop line
    s = ledger_summary(tmp_path, current_pids={12})
    assert s["ended_processes"] == 2
    assert (s["enqueued"], s["delivered"], s["unconfirmed_at_shutdown"]) == (150, 143, 2)
    assert (s["delivery_failed"], s["not_sent"]) == (3, 2)
    assert s["ended_without_record"] == 1                                       # unknown, not "lost"
    # Polling again never changes the totals (finals are summed once, not per poll).
    assert ledger_summary(tmp_path, current_pids={12}) == s


def test_ledger_summary_pid_reuse_and_corrupt_lines(tmp_path):
    write_ledger(tmp_path, 3, [{"record": "start", "pid": 7}, {"record": "start", "pid": 7}, stop_rec(7, 5, 5)])
    with open(tmp_path / "producer-ledger-instance-3.jsonl", "a") as f:
        f.write("{not json\n")
    s = ledger_summary(tmp_path, current_pids=set())
    assert (s["ended_processes"], s["delivered"], s["ended_without_record"]) == (1, 5, 1)


def test_restarted_publisher_counters_start_from_zero():
    """Process-local counters reset on restart; the snapshot says so and identifies the process."""
    first = KafkaEventPublisher({}, producer=FakeProducer(), shutdown_flush_s=0.1, poll_interval_s=0.01)
    first.publish({"event_id": "a"})
    assert wait_for(lambda: first.delivered == 1)
    first.close()
    second = KafkaEventPublisher({}, producer=FakeProducer(), shutdown_flush_s=0.1, poll_interval_s=0.01)
    snap = second.snapshot()
    assert (snap["enqueued"], snap["delivered"], snap["counter_scope"]) == (0, 0, "process")
    assert snap["pid"] == os.getpid() and snap["process_started_at"]
    second.close()
    second.close()                                    # idempotent: lifespan + explicit close
