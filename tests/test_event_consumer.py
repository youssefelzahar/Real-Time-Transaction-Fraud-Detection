import json

import pytest
from confluent_kafka import KafkaError

from src.event_consumer import (EXIT_SINK_FAILURE, ConsumerRunner, EventProcessor, InvalidEvent, JsonlSink,
                                SinkFailure)
from src.events import build_prediction_event

RESPONSE = {"transaction_id": "tx-1", "fraud_probability": 0.2, "decision": "approve",
            "reason_code": "LOW_RISK", "threshold": 0.54, "decision_policy_version": "dp-test",
            "model_version": "e7_v2", "missing_features": [], "balance_source": {"sender": "request",
                                                                                  "receiver": "request"}}


def raw_event(**overrides):
    e = build_prediction_event(RESPONSE, transaction_type="PAYMENT", would_decline=False, instance_id=1)
    e.update(overrides)
    return json.dumps(e).encode()


class MemorySink:
    def __init__(self, fail_times=0, always_fail=False):
        self.events, self.fail_times, self.always_fail, self.flushes = [], fail_times, always_fail, 0

    def write(self, event):
        if self.always_fail or self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("disk unavailable")
        self.events.append(event)

    def flush(self):
        self.flushes += 1


def processor(sink=None, **kw):
    return EventProcessor(sink or MemorySink(), sleep=lambda s: None, **kw)


# --- processing ------------------------------------------------------------------------

def test_valid_event_stored_with_aggregates():
    p = processor()
    assert p.handle(raw_event()) == "stored"
    stored = p.sink.events[0]
    assert stored["decision"] == "approve" and stored["model_version"] == "e7_v2"
    assert p.counts["stored"] == 1 and p.aggregates["decision"]["approve"] == 1
    assert len(p.latency_ms) == 1 and p.latency_ms[0] >= 0


@pytest.mark.parametrize("value, reason", [
    (b"{not json", "invalid JSON"),
    (b"\xff\xfe", "invalid JSON"),
    (b"[1, 2]", "not a JSON object"),
    (None, "empty"),
])
def test_invalid_json(value, reason):
    with pytest.raises(InvalidEvent, match=reason):
        processor().handle(value)


@pytest.mark.parametrize("bad", [
    {"event_id": "not-a-uuid"},
    {"fraud_probability": 1.5},
    {"decision": "block"},
    {"event_type": "other"},
])
def test_schema_violations(bad):
    with pytest.raises(InvalidEvent, match="schema violation"):
        processor().handle(raw_event(**bad))


def test_missing_required_fields():
    e = json.loads(raw_event())
    del e["decision"], e["model_version"]
    with pytest.raises(InvalidEvent, match="decision.*model_version|model_version.*decision"):
        processor().handle(json.dumps(e).encode())


def test_duplicate_event_skipped():
    p = processor()
    value = raw_event()
    assert p.handle(value) == "stored"
    assert p.handle(value) == "duplicate"
    assert len(p.sink.events) == 1 and p.counts["duplicates"] == 1


def test_dedup_window_is_bounded():
    p = processor(dedup_size=2)
    first = raw_event()
    p.handle(first)
    p.handle(raw_event())
    p.handle(raw_event())          # pushes `first` out of the window
    assert p.handle(first) == "stored"


def test_transient_failure_retried_then_stored():
    p = processor(MemorySink(fail_times=2), max_retries=3)
    assert p.handle(raw_event()) == "stored"
    assert p.counts["retries"] == 2 and len(p.sink.events) == 1


def test_permanent_failure_raises_after_bounded_retries():
    p = processor(MemorySink(always_fail=True), max_retries=3)
    with pytest.raises(SinkFailure):
        p.handle(raw_event())
    assert p.counts["retries"] == 3 and p.counts["sink_failures"] == 1


def test_jsonl_sink_writes_and_syncs(tmp_path):
    sink = JsonlSink(tmp_path)
    p = EventProcessor(sink)
    p.handle(raw_event())
    sink.flush()
    files = list(tmp_path.glob("prediction-events-*.jsonl"))
    assert len(files) == 1 and json.loads(files[0].read_text().splitlines()[0])["decision"] == "approve"
    sink.close()


# --- runner: offsets, dead letters, shutdown ---------------------------------------------

class FakeMsg:
    def __init__(self, value, offset, partition=0, topic="events", error=None):
        self._v, self._o, self._p, self._t, self._e = value, offset, partition, topic, error

    def value(self): return self._v
    def key(self): return b"k"
    def offset(self): return self._o
    def partition(self): return self._p
    def topic(self): return self._t
    def error(self): return self._e


class FakeConsumer:
    def __init__(self, messages):
        self.messages = list(messages)
        self.commits = []
        self.runner = None

    def poll(self, timeout):
        if self.messages:
            return self.messages.pop(0)
        self.runner.stop()           # end of test input -> graceful shutdown
        return None

    def commit(self, offsets, asynchronous):
        assert asynchronous is False
        self.commits.append({(tp.topic, tp.partition): tp.offset for tp in offsets})

    def assignment(self):
        return []


class FakeDlq:
    def __init__(self, unacked=0):
        self.produced, self.unacked = [], unacked

    def produce(self, topic, value, key, headers):
        self.produced.append((topic, value, dict(headers)))

    def poll(self, timeout): return 0
    def flush(self, timeout): return self.unacked


def run(messages, sink=None, dlq=None, commit_every=500):
    consumer = FakeConsumer(messages)
    runner = ConsumerRunner(consumer, processor(sink), dlq or FakeDlq(), "events.dlq",
                            commit_every=commit_every, commit_interval_s=999)
    consumer.runner = runner
    code = runner.run(poll_timeout_s=0)
    return runner, consumer, code


def test_offsets_committed_after_processing_per_partition():
    msgs = [FakeMsg(raw_event(), 10, partition=0), FakeMsg(raw_event(), 4, partition=1),
            FakeMsg(raw_event(), 11, partition=0)]
    runner, consumer, code = run(msgs)
    assert code == 0
    assert consumer.commits[-1] == {("events", 0): 12, ("events", 1): 5}   # next offset to read
    assert runner.processor.sink.flushes >= 1                               # durable before commit


def test_commit_batches_by_count():
    msgs = [FakeMsg(raw_event(), i) for i in range(5)]
    _, consumer, _ = run(msgs, commit_every=2)
    assert [c[("events", 0)] for c in consumer.commits] == [2, 4, 5]


def test_invalid_message_dead_lettered_and_committed():
    dlq = FakeDlq()
    msgs = [FakeMsg(b"{oops", 7), FakeMsg(raw_event(), 8)]
    runner, consumer, code = run(msgs, dlq=dlq)
    assert code == 0 and runner.counts["dead_lettered"] == 1
    topic, value, headers = dlq.produced[0]
    assert topic == "events.dlq" and value == b"{oops"
    assert headers["error"].startswith(b"invalid JSON") and headers["source_offset"] == b"7"
    assert consumer.commits[-1] == {("events", 0): 9}


def test_no_commit_while_dead_letter_not_acknowledged():
    runner, consumer, code = run([FakeMsg(b"{oops", 7)], dlq=FakeDlq(unacked=1))
    assert code == 0 and consumer.commits == []           # deferred, not crashed
    assert runner.counts["commit_deferred"] >= 1 and runner.pending == {("events", 0): 8}


def test_commit_failure_is_retried_not_fatal():
    """Broker restarting: commit raises NOT_COORDINATOR; the consumer must keep running and
    commit everything (including events processed meanwhile) once Kafka is back."""
    from confluent_kafka import KafkaException

    class FlakyConsumer(FakeConsumer):
        failures = 2

        def commit(self, offsets, asynchronous):
            if self.failures:
                self.failures -= 1
                raise KafkaException(KafkaError(KafkaError.NOT_COORDINATOR))
            super().commit(offsets, asynchronous)

    consumer = FlakyConsumer([FakeMsg(raw_event(), i) for i in range(3)])
    runner = ConsumerRunner(consumer, processor(), FakeDlq(), "events.dlq", commit_every=1, commit_interval_s=0)
    consumer.runner = runner
    assert runner.run(poll_timeout_s=0) == 0
    assert runner.counts["commit_deferred"] == 2
    assert consumer.commits[-1] == {("events", 0): 3}         # nothing skipped or lost
    assert runner.processor.counts["stored"] == 3


def test_permanent_sink_failure_stops_without_committing_failing_event():
    sink = MemorySink()
    msgs = [FakeMsg(raw_event(), 0), FakeMsg(raw_event(), 1), FakeMsg(raw_event(), 2)]

    class FailOnSecond(MemorySink):
        def write(self, event):
            if len(self.events) == 1:
                raise OSError("disk full")
            super().write(event)

    runner, consumer, code = run(msgs, sink=FailOnSecond())
    assert code == EXIT_SINK_FAILURE
    assert consumer.commits == [{("events", 0): 1}]       # first event committed, failing one not
    assert len(consumer.messages) == 1                    # stopped: third message never polled


def test_kafka_errors_counted_partition_eof_ignored():
    msgs = [FakeMsg(None, -1, error=KafkaError(KafkaError._PARTITION_EOF)),
            FakeMsg(None, -1, error=KafkaError(KafkaError._TRANSPORT)),
            FakeMsg(raw_event(), 0)]
    runner, consumer, code = run(msgs)
    assert runner.counts["kafka_errors"] == 1 and runner.processor.counts["stored"] == 1
    assert consumer.commits[-1] == {("events", 0): 1}


def test_duplicate_is_committed_but_not_stored_twice():
    value = raw_event()
    runner, consumer, _ = run([FakeMsg(value, 0), FakeMsg(value, 1)])
    assert len(runner.processor.sink.events) == 1 and runner.processor.counts["duplicates"] == 1
    assert consumer.commits[-1] == {("events", 0): 2}


def test_metrics_snapshot():
    runner, _, _ = run([FakeMsg(raw_event(), 0), FakeMsg(b"x", 1)])
    m = runner.metrics()
    assert m["stored"] == 1 and m["dead_lettered"] == 1 and m["by"]["decision"] == {"approve": 1}
    assert m["event_latency_ms"]["p50"] is not None


def test_lag_after_restart_uses_committed_offset_not_log_start():
    """Right after (re)assignment position() is unset until a partition is fetched; lag must be
    measured from the committed offset (where consumption resumes), not from offset 0."""
    from confluent_kafka import TopicPartition

    class LagConsumer:
        ends = {0: 400, 1: 387}
        positions = {0: -1001, 1: 387}          # 0: not fetched yet since assignment
        commits = {0: 400, 1: 386}

        def assignment(self): return [TopicPartition("events", p) for p in self.ends]
        def get_watermark_offsets(self, tp, timeout, cached): return 0, self.ends[tp.partition]
        def position(self, tps): return [TopicPartition(tp.topic, tp.partition, self.positions[tp.partition]) for tp in tps]
        def committed(self, tps, timeout): return [TopicPartition(tp.topic, tp.partition, self.commits[tp.partition]) for tp in tps]

    runner = ConsumerRunner(LagConsumer(), processor(), FakeDlq(), "events.dlq")
    assert runner.lag() == {"events[0]": 0, "events[1]": 0}


def test_traffic_source_is_validated_and_stored():
    p = processor()
    assert p.handle(raw_event(traffic_source="loadtest")) == "stored"
    assert p.sink.events[-1]["traffic_source"] == "loadtest"
    assert p.handle(raw_event()) == "stored"                     # older events: no tag
    assert p.sink.events[-1]["traffic_source"] is None
    with pytest.raises(InvalidEvent, match="schema violation"):
        p.handle(raw_event(traffic_source="whatever"))
