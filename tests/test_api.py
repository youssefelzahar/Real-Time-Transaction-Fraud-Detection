import pytest
from fastapi.testclient import TestClient

from api.main import InMemoryAccountService, app
from src.decision_engine import DecisionEngine, DecisionPolicy, PolicyError
from src.predict import FraudPredictor

APPROVE_T, DECLINE_T = 0.54, 0.999


def policy(mode="shadow", models=("e6_test",)):
    return DecisionPolicy("dp-test", APPROVE_T, DECLINE_T, decline_mode=mode, valid_for_models=models)


def make_client(model_dir, decision_policy):
    app.state.predictor = FraudPredictor.load(model_dir)
    app.state.account_service = InMemoryAccountService({"C_SENDER": 5000.0, "C_RECEIVER": 0.0})
    app.state.decision_engine = DecisionEngine(decision_policy)
    return TestClient(app, raise_server_exceptions=False)


def reset_state():
    for name in ("predictor", "account_service", "decision_engine", "event_publisher"):
        if hasattr(app.state, name):
            delattr(app.state, name)


@pytest.fixture
def client(model_dir):
    with make_client(model_dir, policy()) as c:
        yield c
    reset_state()


@pytest.fixture
def enforce_client(model_dir):
    with make_client(model_dir, policy("enforce")) as c:
        yield c
    reset_state()


def fixed_score(score):
    """Replace the model output with a fixed score (decision tests only)."""
    def score_fn(record):
        return {"fraud_probability": score, "model_version": "e6_test", "missing_features": []}
    return score_fn


def payload(**kw):
    return {"transaction_id": "tx-1", "type": "TRANSFER", "amount": 5000.0,
            "sender_id": "C_SENDER", "receiver_id": "C_RECEIVER", **kw}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "e6_test"}


RESPONSE_FIELDS = {"transaction_id", "fraud_probability", "decision", "reason_code", "threshold",
                   "decision_policy_version", "model_version", "missing_features", "balance_source"}


def test_response_schema_is_stable(client):
    """Public fields: the original seven plus reason_code and decision_policy_version."""
    body = client.post("/predict", json=payload()).json()
    assert set(body) == RESPONSE_FIELDS
    assert body["threshold"] == APPROVE_T          # same meaning as before: approve threshold
    assert body["decision_policy_version"] == "dp-test"


@pytest.mark.parametrize("score, decision, reason", [
    (0.10, "approve", "LOW_RISK"),
    (APPROVE_T, "review", "REVIEW_THRESHOLD"),
    (0.80, "review", "REVIEW_THRESHOLD"),
    (DECLINE_T, "review", "HIGH_RISK"),         # shadow mode: would decline, reported as review
    (0.9999, "review", "HIGH_RISK"),
])
def test_shadow_policy_through_api(client, monkeypatch, score, decision, reason):
    monkeypatch.setattr(app.state.predictor, "score", fixed_score(score))
    body = client.post("/predict", json=payload()).json()
    assert (body["fraud_probability"], body["decision"], body["reason_code"]) == (score, decision, reason)


@pytest.mark.parametrize("score, decision", [(0.10, "approve"), (0.80, "review"),
                                             (DECLINE_T, "decline"), (1.0, "decline")])
def test_enforce_policy_through_api(enforce_client, monkeypatch, score, decision):
    monkeypatch.setattr(app.state.predictor, "score", fixed_score(score))
    assert enforce_client.post("/predict", json=payload()).json()["decision"] == decision


def test_decision_metrics_endpoint(client, monkeypatch):
    for s in (0.1, 0.6, 0.9995):
        monkeypatch.setattr(app.state.predictor, "score", fixed_score(s))
        client.post("/predict", json=payload())
    m = client.get("/metrics/decisions").json()
    assert m["total"] == 3 and m["would_decline"] == 1
    assert m["by_decision"] == {"approve": 1, "review": 2, "decline": 0}
    assert m["policy_version"] == "dp-test" and m["model_version"] == "e6_test"
    assert "/metrics/decisions" not in client.get("/openapi.json").json()["paths"]


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -0.1, 1.5])
def test_invalid_model_score_is_technical_error(client, monkeypatch, bad_score):
    monkeypatch.setattr(app.state.predictor, "score", fixed_score(bad_score))
    r = client.post("/predict", json=payload())
    assert r.status_code == 500 and r.json() == {"detail": "internal scoring error"}
    assert client.get("/metrics/decisions").json()["invalid_scores"] == 1


def test_model_serving_exception_returns_500_without_details(client, monkeypatch):
    def broken(record):
        raise RuntimeError("booster exploded at /secret/path")
    monkeypatch.setattr(app.state.predictor, "score", broken)
    r = client.post("/predict", json=payload())
    assert r.status_code == 500
    assert r.json() == {"detail": "internal scoring error"}
    assert "secret" not in r.text and "Traceback" not in r.text


def test_startup_refuses_policy_not_validated_for_model(model_dir):
    with pytest.raises(PolicyError, match="not validated"):
        with make_client(model_dir, policy(models=("e7_v2",))):
            pass
    reset_state()


def test_predict_uses_balances_from_account_service(client):
    r = client.post("/predict", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"approve", "review"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["missing_features"] == []
    assert body["model_version"] == "e6_test"


def test_api_matches_direct_prediction(client, model_dir):
    expected = FraudPredictor.load(model_dir).score(
        {"type": "TRANSFER", "amount": 5000.0, "oldbalanceOrg": 5000.0, "oldbalanceDest": 0.0})
    assert client.post("/predict", json=payload()).json()["fraud_probability"] == pytest.approx(
        expected["fraud_probability"])


def test_unknown_accounts_are_scored_with_missing_features(client):
    body = client.post("/predict", json=payload(sender_id="C_UNKNOWN", receiver_id="M_SHOP")).json()
    assert "oldbalanceOrg" in body["missing_features"] and "oldbalanceDest" in body["missing_features"]


@pytest.mark.parametrize("bad", [{"amount": -1}, {"type": "CRYPTO"}, {"sender_id": ""}])
def test_invalid_requests_rejected(client, bad):
    assert client.post("/predict", json=payload(**bad)).status_code == 422


def test_account_service_overrides_request_balances(client):
    r1 = client.post("/predict", json=payload()).json()
    r2 = client.post("/predict", json=payload(sender_balance=1.0, receiver_balance=999.0,
                                              oldbalanceOrg=1.0, newbalanceOrig=0.0)).json()
    assert r1["fraud_probability"] == r2["fraud_probability"]
    assert r2["balance_source"] == {"sender": "account_service", "receiver": "account_service"}


def test_request_balances_used_as_fallback_for_unknown_accounts(client):
    body = client.post("/predict", json=payload(sender_id="C_NEW", receiver_id="C_NEW2",
                                                sender_balance=5000.0, receiver_balance=0.0)).json()
    assert body["missing_features"] == []
    assert body["balance_source"] == {"sender": "request", "receiver": "request"}
    known = client.post("/predict", json=payload()).json()
    assert body["fraud_probability"] == pytest.approx(known["fraud_probability"])


def test_swagger_examples_are_valid_requests(client):
    from api.main import EXAMPLES
    for name, ex in EXAMPLES.items():
        r = client.post("/predict", json=ex["value"])
        assert r.status_code == 200, name
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]
    assert set(body["examples"]) == set(EXAMPLES)


# --- Kafka prediction events (publisher is mocked; see tests/test_events.py) ------------

class RecordingPublisher:
    enabled = True

    def __init__(self, error=None):
        self.events, self.error = [], error

    def publish(self, event):
        if self.error:
            raise self.error
        self.events.append(event)
        return True

    def snapshot(self):
        return {"kafka_enabled": True, "enqueued": len(self.events)}

    def close(self):
        pass


@pytest.fixture
def events_client(model_dir):
    """Client with a recording publisher attached; yields (client, publisher)."""
    publisher = RecordingPublisher()
    app.state.event_publisher = publisher
    with make_client(model_dir, policy()) as c:
        yield c, publisher
    reset_state()


def test_kafka_disabled_by_default(client):
    assert client.get("/metrics/events").json() == {"kafka_enabled": False}
    assert client.post("/predict", json=payload()).status_code == 200


@pytest.mark.parametrize("score, decision, reason, would_decline", [
    (0.10, "approve", "LOW_RISK", False),
    (0.80, "review", "REVIEW_THRESHOLD", False),
    (0.9995, "review", "HIGH_RISK", True),          # shadow mode unchanged by publishing
])
def test_event_published_and_response_unchanged(events_client, monkeypatch, score, decision, reason, would_decline):
    client, publisher = events_client
    monkeypatch.setattr(app.state.predictor, "score", fixed_score(score))
    body = client.post("/predict", json=payload()).json()
    assert set(body) == RESPONSE_FIELDS
    assert (body["decision"], body["reason_code"]) == (decision, reason)
    (event,) = publisher.events
    for k in ("transaction_id", "fraud_probability", "decision", "reason_code", "threshold",
              "decision_policy_version", "model_version", "missing_features", "balance_source"):
        assert event[k] == body[k]
    assert event["would_decline"] is would_decline and event["transaction_type"] == "TRANSFER"
    assert "sender_id" not in event and "amount" not in event


def test_publisher_failure_never_breaks_prediction(model_dir):
    app.state.event_publisher = RecordingPublisher(error=RuntimeError("kafka client exploded"))
    with make_client(model_dir, policy()) as c:
        r = c.post("/predict", json=payload())
    reset_state()
    assert r.status_code == 200 and set(r.json()) == RESPONSE_FIELDS


def test_prediction_does_not_wait_for_delivery(model_dir):
    """Broker never acknowledges: the request still returns at once; the event stays in flight."""
    from tests.test_events import FakeProducer
    from src.events import KafkaEventPublisher
    pub = KafkaEventPublisher({"queue.buffering.max.messages": 10}, producer=FakeProducer(outcome="stuck"),
                              shutdown_flush_s=0.05, poll_interval_s=0.01)
    app.state.event_publisher = pub
    with make_client(model_dir, policy()) as c:
        r = c.post("/predict", json=payload())
        m = c.get("/metrics/events").json()
    reset_state()
    assert r.status_code == 200
    assert m["enqueued"] == 1 and m["delivered"] == 0 and m["in_flight"] == 1


def test_buffer_full_still_returns_predictions(model_dir):
    from tests.test_events import FakeProducer
    from src.events import KafkaEventPublisher
    pub = KafkaEventPublisher({"queue.buffering.max.messages": 2}, producer=FakeProducer(outcome="stuck", capacity=2),
                              shutdown_flush_s=0.05, poll_interval_s=0.01)
    app.state.event_publisher = pub
    with make_client(model_dir, policy()) as c:
        statuses = [c.post("/predict", json=payload()).status_code for _ in range(5)]
        m = c.get("/metrics/events").json()
    reset_state()
    assert statuses == [200] * 5
    assert m["enqueued"] == 2 and m["buffer_full_drops"] == 3 and m["queue_utilization"] == 1.0
    assert pub.lost_on_shutdown == 2                    # counted when the app shut down


def test_event_metrics_endpoint_hidden_from_openapi(events_client):
    client, _ = events_client
    assert "/metrics/events" not in client.get("/openapi.json").json()["paths"]
