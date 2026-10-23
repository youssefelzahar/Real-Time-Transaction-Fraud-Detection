import pytest
from fastapi.testclient import TestClient

from api.main import InMemoryAccountService, app
from src.predict import FraudPredictor


@pytest.fixture
def client(model_dir):
    app.state.predictor = FraudPredictor.load(model_dir)
    app.state.account_service = InMemoryAccountService({"C_SENDER": 5000.0, "C_RECEIVER": 0.0})
    with TestClient(app) as c:
        yield c
    del app.state.predictor, app.state.account_service


def payload(**kw):
    return {"transaction_id": "tx-1", "type": "TRANSFER", "amount": 5000.0,
            "sender_id": "C_SENDER", "receiver_id": "C_RECEIVER", **kw}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "e6_test"}


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
