"""Streamlit UI (streamlit_app/): validation, HTTP error handling, status derivation, page smoke tests.
No network: the HTTP layer is faked. Skipped when the UI dependencies are not installed."""
import math
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
requests = pytest.importorskip("requests")

from streamlit_app import api_client, config, logic  # noqa: E402
from streamlit_app.api_client import Result  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ROOT / "streamlit_app" / "views"

GOOD = {"transaction_id": "tx-1", "fraud_probability": 0.0123, "decision": "approve", "reason_code": "LOW_RISK",
        "threshold": 0.54, "decision_policy_version": "dp-e7-2026-10-26.1", "model_version": "e7_v1",
        "missing_features": [], "balance_source": {"sender": "request", "receiver": "request"}}
FORM = {"transaction_id": " tx-1 ", "type": "TRANSFER", "amount": 181.0, "sender_id": "C1", "receiver_id": "C2",
        "sender_balance": None, "receiver_balance": None, "step": None}


# --- payload ------------------------------------------------------------------------------------------

def test_valid_form_builds_contract_payload_and_omits_empty_balances():
    payload, errors = logic.build_payload(FORM)
    assert errors == []
    assert payload == {"transaction_id": "tx-1", "type": "TRANSFER", "amount": 181.0,
                       "sender_id": "C1", "receiver_id": "C2"}     # no balances, no step, never 0


def test_zero_balance_is_kept_when_entered():
    payload, _ = logic.build_payload({**FORM, "sender_balance": 0.0, "receiver_balance": 5.5, "step": 3})
    assert payload["sender_balance"] == 0.0 and payload["receiver_balance"] == 5.5 and payload["step"] == 3


@pytest.mark.parametrize("change,fragment", [
    ({"amount": None}, "Amount is required"),
    ({"amount": float("inf")}, "finite"),
    ({"amount": float("nan")}, "finite"),
    ({"amount": -1.0}, "negative"),
    ({"amount": "abc"}, "number"),
    ({"amount": True}, "number"),
    ({"sender_id": "  "}, "Sender ID is required"),
    ({"receiver_id": None}, "Receiver ID is required"),
    ({"type": "CRYPTO"}, "Transaction type"),
    ({"type": None}, "Transaction type"),
    ({"sender_balance": -math.inf}, "finite"),
    ({"receiver_balance": -3.0}, "negative"),
    ({"step": 1.5}, "Step"),
    ({"step": -1}, "Step"),
])
def test_invalid_input_is_rejected_before_any_request(change, fragment):
    payload, errors = logic.build_payload({**FORM, **change})
    assert payload is None and any(fragment in e for e in errors)


def test_unknown_form_fields_are_never_sent():
    payload, _ = logic.build_payload({**FORM, "is_fraud": 1, "isFlaggedFraud": 1, "decision": "approve"})
    assert set(payload) <= set(logic.REQUEST_FIELDS)


# --- HTTP client --------------------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status, body=None, text=None):
        self.status_code, self._body, self.reason = status, body, "reason"
        self.text = text if text is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeSession:
    def __init__(self, outcome):
        self.outcome, self.calls = outcome, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    get = post


def submit(outcome):
    s = FakeSession(outcome)
    return api_client.submit_transaction("http://api", {"type": "PAYMENT"}, "demo", 3, session=s), s


def test_success_returns_the_api_response_unchanged_and_posts_once_with_the_tag():
    r, s = submit(FakeResponse(200, GOOD))
    assert r.ok and r.data == GOOD
    assert len(s.calls) == 1
    url, kw = s.calls[0]
    assert url == "http://api/predict" and kw["headers"] == {"X-Traffic-Source": "demo"} and kw["timeout"] == 3


@pytest.mark.parametrize("outcome,kind,fragment", [
    (requests.ConnectionError(), "connection", "Cannot connect"),
    (requests.Timeout(), "timeout", "may have been scored"),
    (FakeResponse(422, {"detail": [{"loc": ["body", "amount"], "msg": "Input should be >= 0"}]}),
     "client_error", "amount: Input should be >= 0"),
    (FakeResponse(422, {"detail": "This model version requires 'step'"}), "client_error", "requires 'step'"),
    (FakeResponse(500, {"detail": "internal scoring error"}), "server_error", "internal scoring error"),
    (FakeResponse(503, {"detail": "fraud scoring temporarily unavailable"}), "server_error", "temporarily unavailable"),
    (FakeResponse(504, {"detail": "fraud scoring timed out"}), "server_error", "do not resubmit"),
    (FakeResponse(502, None, text="<html>bad gateway</html>"), "server_error", "bad gateway"),
    (FakeResponse(200, None, text="<html>"), "invalid_response", "not JSON"),
    (FakeResponse(200, {"decision": "approve"}), "invalid_response", "missing"),
    (FakeResponse(200, {**GOOD, "decision": "block"}), "invalid_response", "unknown decision"),
    (FakeResponse(200, {**GOOD, "fraud_probability": 1.7}), "invalid_response", "fraud_probability"),
    (FakeResponse(200, [GOOD]), "invalid_response", "not a JSON object"),
])
def test_every_failure_is_reported_and_never_retried(outcome, kind, fragment):
    r, s = submit(outcome)
    assert not r.ok and r.data is None and r.error_kind == kind and fragment in r.message
    assert len(s.calls) == 1            # exactly one POST: no automatic resubmission


def test_monitoring_reads_never_raise():
    r = api_client.get_json("http://x/api/snapshot", 1, session=FakeSession(requests.ConnectionError()))
    assert not r.ok and r.error_kind == "connection"
    inst = api_client.instance_status("http://i", 1, session=FakeSession(requests.Timeout()))
    assert inst == {"url": "http://i", "health": None, "decisions": None, "events": None, "up": False}


# --- status and aggregation -----------------------------------------------------------------------------

def inst(up=True, model="e7_v1", total=10, approve=9, review=1, would=0, mode="shadow", kafka=True, failed=0,
         buckets=None):
    if not up:
        return {"url": "u", "up": False, "health": None, "decisions": None, "events": None}
    return {"url": "u", "up": True, "health": {"status": "ok", "model_version": model},
            "decisions": {"policy_version": "dp-1", "decline_mode": mode, "total": total,
                          "by_decision": {"approve": approve, "review": review}, "would_decline": would,
                          "decision_latency_us": {"count": total, "mean": 3.0, "max": 40.0,
                                                  "buckets": buckets or {"<=5us": total, "<=10us": 0, ">1000us": 0}}},
            "events": {"kafka_enabled": kafka, "delivered": total, "failed": {"timed_out": failed},
                       "buffer_full_drops": 0, "produce_errors": 0, "serialization_errors": 0}}


OK_HEALTH = Result(True, {"status": "ok", "model_version": "e7_v1"}, status=200)
DOWN = Result(False, error_kind="connection", message="not reachable")


def snap(kafka_up=True, state="running", lag=0):
    return Result(True, {"kafka": {"up": kafka_up, "lag": lag}, "consumer": {"state": state, "heartbeat_age_s": 3}})


def test_decision_summary_sums_reporting_instances_and_says_when_unavailable():
    s = logic.summarize_decisions([inst(), inst(total=5, approve=3, review=2, would=1), inst(up=False)])
    assert (s["total"], s["approve"], s["review"], s["decline"], s["would_decline"]) == (15, 12, 3, 0, 1)
    assert (s["reporting"], s["instances"], s["decline_modes"]) == (2, 3, ["shadow"])
    none = logic.summarize_decisions([inst(up=False)])
    assert none["total"] is None and logic.fmt_count(none["total"]) == "Not available"


@pytest.mark.parametrize("health,instances,expected", [
    (OK_HEALTH, [inst(), inst()], logic.HEALTHY),
    (OK_HEALTH, [inst(), inst(up=False)], logic.DEGRADED),
    (DOWN, [inst(up=False), inst(up=False)], logic.UNAVAILABLE),
    (DOWN, [inst(), inst()], logic.DEGRADED),          # instances fine, Nginx down
])
def test_api_status(health, instances, expected):
    assert logic.status_api(health, instances)[0] == expected


def test_model_status_detects_mixed_versions():
    assert logic.status_model(OK_HEALTH, [inst(), inst()])[0] == logic.HEALTHY
    assert logic.status_model(OK_HEALTH, [inst(), inst(model="e7_v3")])[0] == logic.DEGRADED
    assert logic.status_model(DOWN, [inst(up=False)])[0] == logic.UNAVAILABLE


@pytest.mark.parametrize("snapshot,instances,expected", [
    (snap(), [inst()], logic.HEALTHY),
    (snap(), [inst(failed=3)], logic.DEGRADED),
    (snap(kafka_up=False), [inst()], logic.UNAVAILABLE),
    (DOWN, [inst()], logic.UNKNOWN),                    # live monitor not running: don't guess
    (snap(), [inst(kafka=False)], logic.UNAVAILABLE),   # publishing disabled
])
def test_kafka_status(snapshot, instances, expected):
    assert logic.status_kafka(snapshot, instances)[0] == expected


@pytest.mark.parametrize("snapshot,expected", [
    (snap(state="running"), logic.HEALTHY), (snap(state="stalled"), logic.DEGRADED),
    (snap(state="starting"), logic.DEGRADED), (snap(state="stopped"), logic.UNAVAILABLE),
    (DOWN, logic.UNKNOWN)])
def test_consumer_status(snapshot, expected):
    assert logic.status_consumer(snapshot)[0] == expected


def test_latency_percentiles_are_bucket_bounds():
    b = {"<=5us": 90, "<=10us": 6, "<=25us": 3, "<=50us": 1, ">1000us": 0}
    lat = logic.latency_from_buckets([inst(total=100, buckets=b)])
    assert (lat["p50"], lat["p95"], lat["p99"]) == ("≤ 5 µs", "≤ 10 µs", "≤ 25 µs")
    assert lat["count"] == 100 and lat["max_us"] == 40.0
    assert logic.latency_from_buckets([inst(up=False)]) is None


# --- shadow mode --------------------------------------------------------------------------------------

def test_shadow_would_decline_is_not_presented_as_blocked():
    notes = logic.shadow_note({**GOOD, "decision": "review", "reason_code": "HIGH_RISK"}, ["shadow"])
    assert any("would-decline" in n and "not blocked" in n for n in notes)
    assert any("does not execute or block payments" in n for n in notes)


def test_decline_mode_unknown_is_said_so():
    notes = logic.shadow_note(GOOD, [])
    assert any("not available" in n for n in notes)


# --- pages (Streamlit AppTest, fake backend) -----------------------------------------------------------

@pytest.fixture
def backend(monkeypatch):
    """Fake every call the pages make; record POSTs."""
    from streamlit.testing.v1 import AppTest
    state = {"post": Result(True, GOOD, status=200), "posts": [], "up": True}

    def fake_submit(base, payload, source, timeout):
        state["posts"].append((payload, source))
        return state["post"]
    monkeypatch.setattr(api_client, "submit_transaction", fake_submit)
    monkeypatch.setattr(api_client, "health", lambda *a, **k: OK_HEALTH if state["up"] else DOWN)
    monkeypatch.setattr(api_client, "instances_status",
                        lambda urls, *a, **k: [inst() if state["up"] else inst(up=False) for _ in range(6)])
    monkeypatch.setattr(api_client, "monitor_snapshot", lambda *a, **k: DOWN)
    state["AppTest"] = AppTest
    return state


def run_page(backend, name):
    at = backend["AppTest"].from_file(str(VIEWS / name), default_timeout=30)
    return at.run()


def texts(at):
    out = []
    for kind in ("markdown", "caption", "info", "error", "warning", "success", "title", "subheader"):
        out += [e.value for e in getattr(at, kind)]
    out += [f"{m.label}={m.value}" for m in at.metric]
    return "\n".join(str(t) for t in out)


def fill(at, **values):
    at.selectbox(key="type").set_value(values.get("type", "TRANSFER"))
    at.number_input(key="amount").set_value(values.get("amount", 181.0))
    at.text_input(key="sender_id").set_value("C1")
    at.text_input(key="receiver_id").set_value("C2")
    return at


def test_analyzer_shows_the_real_response(backend):
    at = fill(run_page(backend, "analyzer.py"))
    at.button[0].click().run()
    assert not at.exception
    assert len(backend["posts"]) == 1 and backend["posts"][0][1] == config.TRAFFIC_SOURCE
    assert "sender_balance" not in backend["posts"][0][0]          # left empty: not sent, not 0
    out = texts(at)
    assert "APPROVE" in out and "e7_v1" in out and "Decline mode: shadow" in out


def test_analyzer_shows_shadow_would_decline(backend):
    backend["post"] = Result(True, {**GOOD, "decision": "review", "reason_code": "HIGH_RISK",
                                    "fraud_probability": 0.9995}, status=200)
    at = fill(run_page(backend, "analyzer.py"))
    at.button[0].click().run()
    out = texts(at)
    assert "REVIEW" in out and "not blocked" in out and "DECLINE" not in out.replace("never returns DECLINE", "")


def test_analyzer_never_shows_a_prediction_on_failure(backend):
    backend["post"] = Result(False, error_kind="connection", message="Cannot connect to the fraud API at http://x.")
    at = fill(run_page(backend, "analyzer.py"))
    at.button[0].click().run()
    assert not at.exception
    out = texts(at)
    assert "Cannot connect" in out and "APPROVE" not in out and "Fraud score" not in out
    assert len(backend["posts"]) == 1


def test_analyzer_validation_blocks_the_request(backend):
    at = run_page(backend, "analyzer.py")
    at.button[0].click().run()                                     # empty form
    assert "Amount is required" in texts(at) and backend["posts"] == []


def test_dashboard_with_backend_up_and_monitor_down(backend):
    out = texts(run_page(backend, "dashboard.py"))
    assert "Transactions processed=60" in out and "Current model version=e7_v1" in out
    assert "Unknown" in out and "not exposed" in out                 # Kafka/consumer unknown; no e2e latency


def test_dashboard_with_everything_down_shows_no_numbers(backend):
    backend["up"] = False
    at = run_page(backend, "dashboard.py")
    assert not at.exception
    out = texts(at)
    assert "Transactions processed=Not available" in out and "API availability=Down" in out
    assert "Unavailable" in out


def test_recent_page_labels_session_data(backend):
    at = run_page(backend, "recent.py")
    assert not at.exception
    out = texts(at)
    assert "Transactions submitted during this session" in out and "not a transaction history" in out
    assert "Not available" in out                                     # monitor feed down


def test_app_entry_point_runs(backend):
    at = backend["AppTest"].from_file(str(ROOT / "streamlit_app" / "app.py"), default_timeout=30).run()
    assert not at.exception
