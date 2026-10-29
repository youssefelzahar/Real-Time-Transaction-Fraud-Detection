import json
import math
import time

import pytest
from confluent_kafka import KafkaError

from src.events import (RESPONSE_FIELDS, KafkaEventPublisher, NullPublisher, build_prediction_event)

RESPONSE = {"transaction_id": "tx-1", "fraud_probability": 0.9995, "decision": "review",
            "reason_code": "HIGH_RISK", "threshold": 0.54, "decision_policy_version": "dp-test",
            "model_version": "e7_v2", "missing_features": [], "balance_source": {"sender": "request",
                                                                                  "receiver": "request"}}


def event(**overrides):
    e = build_prediction_event(RESPONSE, transaction_type="TRANSFER", would_decline=True,
                               instance_id=3, scoring_ms=0.4321)
    e.update(overrides)
    return e


class FakeProducer:
    """Stands in for confluent_kafka.Producer. `outcome`: None = delivered, a KafkaError code =
    failed, "stuck" = never delivered (broker gone)."""

    def __init__(self, outcome=None, capacity=None, produce_error=None):
        self.outcome = outcome
        self.capacity = capacity
        self.produce_error = produce_error
        self.queue = []
        self.sent = []

    def produce(self, topic, value, key, on_delivery):
        if self.produce_error:
            raise self.produce_error
        if self.capacity is not None and len(self.queue) >= self.capacity:
            raise BufferError("Local: Queue full")
        self.queue.append((topic, value, key, on_delivery))

    def poll(self, timeout):
        if self.outcome == "stuck":
            time.sleep(min(timeout, 0.01))
            return 0
        fired, self.queue = self.queue, []
        for topic, value, key, cb in fired:
            err = None if self.outcome is None else KafkaError(self.outcome)
            if err is None:
                self.sent.append((topic, json.loads(value), key))
            cb(err, None)
        if not fired:
            time.sleep(min(timeout, 0.01))
        return len(fired)

    def flush(self, timeout):
        if self.outcome == "stuck":
            return len(self.queue)
        self.poll(0)
        return 0

    def __len__(self):
        return len(self.queue)


def publisher(producer, capacity=100_000):
    return KafkaEventPublisher({"queue.buffering.max.messages": capacity}, topic="t.events",
                               producer=producer, shutdown_flush_s=0.1, poll_interval_s=0.01)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# --- event content ---------------------------------------------------------------------

def test_event_fields_and_serialization():
    e = event()
    assert e["event_type"] == "fraud_prediction" and e["schema_version"] == 1
    assert e["would_decline"] is True and e["instance_id"] == 3 and e["scoring_ms"] == 0.432
    assert e["occurred_at"].endswith("+00:00")
    for k in RESPONSE_FIELDS:
        assert e[k] == RESPONSE[k]
    assert json.loads(json.dumps(e)) == e
    assert event()["event_id"] != event()["event_id"]          # unique per event (dedup key)


def test_event_contains_no_account_or_amount_data():
    sensitive = {"sender_id", "receiver_id", "amount", "sender_balance", "receiver_balance",
                 "oldbalanceOrg", "oldbalanceDest", "nameOrig", "nameDest"}
    e = build_prediction_event({**RESPONSE, "sender_id": "C1", "amount": 10.0},
                               transaction_type="TRANSFER", would_decline=False)
    assert not sensitive & set(e)


# --- producer --------------------------------------------------------------------------

def test_successful_publication():
    prod = FakeProducer()
    pub = publisher(prod)
    assert all(pub.publish(event()) for _ in range(5))
    assert wait_for(lambda: pub.delivered == 5)
    topic, value, key = prod.sent[0]
    assert topic == "t.events" and key == b"tx-1" and value["decision"] == "review"
    s = pub.snapshot()
    assert s["enqueued"] == 5 and s["delivered"] == 5 and s["failed"] == {"timed_out": 0, "broker_unavailable": 0, "other": 0}
    pub.close()


def test_key_falls_back_to_event_id():
    prod = FakeProducer()
    pub = publisher(prod)
    e = event(transaction_id=None)
    pub.publish(e)
    assert wait_for(lambda: pub.delivered == 1)
    assert prod.sent[0][2] == e["event_id"].encode()
    pub.close()


def test_serialization_error_is_counted_not_raised():
    pub = publisher(FakeProducer())
    assert pub.publish(event(fraud_probability=math.nan)) is False   # NaN is not valid JSON
    assert pub.publish(event(extra=object())) is False
    assert pub.snapshot()["serialization_errors"] == 2 and pub.enqueued == 0
    pub.close()


@pytest.mark.parametrize("code", [KafkaError._TRANSPORT, KafkaError._ALL_BROKERS_DOWN])
def test_kafka_unavailable_counted_as_broker_unavailable(code):
    pub = publisher(FakeProducer(outcome=code))
    pub.publish(event())
    assert wait_for(lambda: pub.failed["broker_unavailable"] == 1)
    assert pub.delivered == 0 and pub.snapshot()["last_error"].startswith("broker_unavailable")
    pub.close()


def test_publishing_timeout_counted():
    pub = publisher(FakeProducer(outcome=KafkaError._MSG_TIMED_OUT))
    pub.publish(event())
    assert wait_for(lambda: pub.failed["timed_out"] == 1)
    pub.close()


def test_buffer_full_drops_without_blocking():
    prod = FakeProducer(outcome="stuck", capacity=10)
    pub = publisher(prod, capacity=10)
    results = [pub.publish(event()) for _ in range(10)]
    start = time.perf_counter()
    dropped = [pub.publish(event()) for _ in range(1000)]
    per_call_ms = (time.perf_counter() - start) * 1000 / 1000
    assert all(results) and not any(dropped)
    s = pub.snapshot()
    assert s["buffer_full_drops"] == 1000 and s["in_flight"] == 10 and s["queue_utilization"] == 1.0
    assert per_call_ms < 1.0
    pub.close()


def test_unexpected_produce_error_never_raises():
    pub = publisher(FakeProducer(produce_error=RuntimeError("boom")))
    assert pub.publish(event()) is False and pub.produce_errors == 1
    pub.close()


def test_shutdown_with_pending_events_counts_them_as_unconfirmed():
    pub = publisher(FakeProducer(outcome="stuck"))
    for _ in range(7):
        pub.publish(event())
    start = time.monotonic()
    pub.close()
    assert pub.unconfirmed_at_shutdown == 7
    assert time.monotonic() - start < 3          # bounded: poll thread joins, flush is time-limited


def test_shutdown_flushes_deliverable_events():
    prod = FakeProducer(outcome="stuck")
    pub = publisher(prod)
    for _ in range(4):
        pub.publish(event())
    prod.outcome = None                            # broker back just before shutdown
    pub.close()
    assert pub.unconfirmed_at_shutdown == 0 and pub.delivered == 4


def test_null_publisher_when_kafka_not_configured(monkeypatch):
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    pub = KafkaEventPublisher.from_env("test")
    assert isinstance(pub, NullPublisher) and pub.enabled is False
    assert pub.publish(event()) is False and pub.snapshot() == {"kafka_enabled": False}


def test_real_librdkafka_with_unreachable_broker(monkeypatch):
    """Real producer, nothing listening: publish() must stay fast and events must fail
    (counted) after message.timeout instead of blocking or growing forever."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    monkeypatch.setenv("KAFKA_MESSAGE_TIMEOUT_MS", "1000")
    monkeypatch.setenv("KAFKA_SHUTDOWN_FLUSH_S", "0.2")
    pub = KafkaEventPublisher.from_env("test-unreachable")
    start = time.perf_counter()
    for _ in range(200):
        assert pub.publish(event())
    per_call_ms = (time.perf_counter() - start) * 1000 / 200
    assert per_call_ms < 1.0
    assert wait_for(lambda: sum(pub.failed.values()) == 200, timeout=10)
    assert pub.delivered == 0 and pub.snapshot()["in_flight"] == 0
    pub.close()
