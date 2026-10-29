"""Real-broker integration tests (opt-in):

    deploy/kafka.sh start
    RUN_KAFKA_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 pytest tests/test_kafka_integration.py

Uses temporary topics (created and deleted here), not the live ones.
"""
import json
import os
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_KAFKA_TESTS") != "1", reason="set RUN_KAFKA_TESTS=1 with a running broker")

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")


@pytest.fixture
def topics():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    suffix = uuid.uuid4().hex[:8]
    names = (f"nilepay.test-events-{suffix}", f"nilepay.test-events-{suffix}.dlq")
    for f in admin.create_topics([NewTopic(n, num_partitions=2, replication_factor=1) for n in names]).values():
        f.result(timeout=10)
    yield names
    for f in admin.delete_topics(list(names)).values():
        f.result(timeout=10)


def consume_all(topic, dlq, store_dir, expected, timeout=20):
    from confluent_kafka import Consumer, Producer
    from src.event_consumer import ConsumerRunner, EventProcessor, JsonlSink
    group = f"nilepay-test-{uuid.uuid4().hex[:8]}"
    consumer = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": group,
                         "enable.auto.commit": False, "auto.offset.reset": "earliest"})
    runner = ConsumerRunner(consumer, EventProcessor(JsonlSink(store_dir)),
                            Producer({"bootstrap.servers": BOOTSTRAP}), dlq, commit_interval_s=0.2)
    consumer.subscribe([topic])
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msg = consumer.poll(0.5)
        if msg is not None:
            runner.handle_message(msg)
        if runner.pending:
            runner.commit()
        p = runner.processor.counts
        if p["stored"] + p["duplicates"] + runner.counts["dead_lettered"] >= expected:
            break
    committed = consumer.committed(consumer.assignment(), timeout=5)
    consumer.close()
    runner.processor.sink.close()
    return runner, committed


def test_api_event_reaches_consumer_store(topics, tmp_path, model_dir):
    """POST /predict -> real producer -> broker -> consumer -> JSONL store, same event."""
    from fastapi.testclient import TestClient
    from api.main import InMemoryAccountService, app
    from src.decision_engine import DecisionEngine, DecisionPolicy
    from src.events import KafkaEventPublisher
    from src.predict import FraudPredictor

    topic, dlq = topics
    app.state.predictor = FraudPredictor.load(model_dir)
    app.state.account_service = InMemoryAccountService()
    app.state.decision_engine = DecisionEngine(DecisionPolicy("dp-test", 0.54, 0.999, valid_for_models=("e6_test",)))
    app.state.event_publisher = KafkaEventPublisher(
        {"bootstrap.servers": BOOTSTRAP, "acks": "all", "enable.idempotence": True,
         "message.timeout.ms": 10000}, topic=topic)
    body = {"transaction_id": "it-1", "type": "TRANSFER", "amount": 181.0, "sender_id": "C1", "receiver_id": "C2",
            "sender_balance": 181.0, "receiver_balance": 0.0}
    try:
        with TestClient(app) as c:
            responses = [c.post("/predict", json={**body, "transaction_id": f"it-{i}"}).json() for i in range(5)]
        # leaving the client ran the lifespan shutdown: publisher flushed and closed
        pub = app.state.event_publisher
        assert pub.delivered == 5 and pub.lost_on_shutdown == 0
    finally:
        for name in ("predictor", "account_service", "decision_engine", "event_publisher"):
            delattr(app.state, name)

    runner, committed = consume_all(topic, dlq, tmp_path, expected=5)
    stored = [json.loads(line) for f in tmp_path.glob("prediction-events-*.jsonl") for line in f.read_text().splitlines()]
    assert sorted(e["transaction_id"] for e in stored) == [f"it-{i}" for i in range(5)]
    by_tx = {e["transaction_id"]: e for e in stored}
    for r in responses:
        e = by_tx[r["transaction_id"]]
        assert (e["decision"], e["reason_code"], e["fraud_probability"]) == \
               (r["decision"], r["reason_code"], r["fraud_probability"])
    assert sum(tp.offset for tp in committed if tp.offset > 0) == 5          # all offsets committed


def test_invalid_and_duplicate_messages_on_real_broker(topics, tmp_path):
    from confluent_kafka import Consumer, Producer
    from src.events import build_prediction_event

    topic, dlq = topics
    good = build_prediction_event({"transaction_id": "dup-1", "fraud_probability": 0.1, "decision": "approve",
                                   "reason_code": "LOW_RISK", "threshold": 0.54, "decision_policy_version": "dp",
                                   "model_version": "m", "missing_features": [], "balance_source": {}},
                                  transaction_type="PAYMENT", would_decline=False)
    p = Producer({"bootstrap.servers": BOOTSTRAP})
    for value in (json.dumps(good), json.dumps(good), "{broken", json.dumps({"event_id": "x"})):
        p.produce(topic, value=value.encode(), key=b"same-key")       # same partition, in order
    assert p.flush(10) == 0

    runner, _ = consume_all(topic, dlq, tmp_path, expected=4)
    assert runner.processor.counts["stored"] == 1
    assert runner.processor.counts["duplicates"] == 1
    assert runner.counts["dead_lettered"] == 2

    c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": f"dlq-check-{uuid.uuid4().hex[:6]}",
                  "auto.offset.reset": "earliest"})
    c.subscribe([dlq])
    dead, end = [], time.monotonic() + 15
    while len(dead) < 2 and time.monotonic() < end:
        m = c.poll(0.5)
        if m is not None and not m.error():
            dead.append(dict(m.headers()))
    c.close()
    assert sorted(h["error"].decode().split(":")[0] for h in dead) == ["invalid JSON", "schema violation"]
    assert all(h["source_topic"].decode() == topic for h in dead)
