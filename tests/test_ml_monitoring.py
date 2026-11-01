"""ML monitoring: aggregates, windows, recovery, data quality, baseline, labels, dashboard.
No Kafka broker is used (fakes from test_event_consumer; conftest removes KAFKA_BOOTSTRAP_SERVERS)."""
import json
import math
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import ml_monitoring as mm
from src.data_preprocessing import chronological_split
from src.decision_engine import DecisionPolicy
from src.event_consumer import ConsumerRunner
from src.events import build_prediction_event
from src.ml_performance import add_labels, evaluate_performance, read_labels
from src.monitoring_baseline import build_baseline
from src.monitoring_features import feature_summary, log_bin, log_bins
from src.predict import FraudPredictor
from tests.test_event_consumer import FakeConsumer, FakeDlq, FakeMsg, MemorySink, processor

NOW = datetime(2026, 10, 29, 12, 0, 30, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def event(score=0.01, decision="approve", reason="LOW_RISK", would_decline=False, model="e7_v1",
          policy="dp-1", ttype="PAYMENT", missing=(), ago_s=0, summary=True, tx=None, bins=None,
          non_finite=(), now=NOW):
    e = build_prediction_event(
        {"transaction_id": tx, "fraud_probability": score, "decision": decision, "reason_code": reason,
         "threshold": 0.54, "decision_policy_version": policy, "model_version": model,
         "missing_features": list(missing), "balance_source": {"sender": "request", "receiver": "request"}},
        transaction_type=ttype, would_decline=would_decline, instance_id=1,
        feature_summary={"feature_bins": bins or {"amount": 16, "oldbalanceOrg": 20},
                         "feature_flags": {"orig_bal_zero": 0, "is_max_amount": 0},
                         "non_finite_features": list(non_finite)} if summary else None)
    e["occurred_at"] = datetime.fromtimestamp(now - ago_s, timezone.utc).isoformat(timespec="milliseconds")
    return e


def agg(clock=None, path=None):
    return mm.MonitoringAggregator(path, clock=clock or Clock())


def report(a, minutes=60, baseline=None, key="e7_v1|dp-1", min_events=1):
    merged, _ = a.window(minutes)
    return mm.series_report(merged[key], baseline, min_events=min_events)


# --- prediction behaviour ------------------------------------------------------------------

def test_prediction_counts_and_decision_rates():
    a = agg()
    for _ in range(7):
        a.observe(event())
    for _ in range(2):
        a.observe(event(score=0.6, decision="review", reason="REVIEW_THRESHOLD"))
    a.observe(event(score=0.9995, decision="decline", reason="HIGH_RISK"))
    r = report(a)
    assert r["n"] == 10
    assert {d: v["n"] for d, v in r["decisions"].items()} == {"approve": 7, "review": 2, "decline": 1}
    assert r["decisions"]["approve"]["rate"] == 0.7 and r["decisions"]["review"]["rate"] == 0.2
    assert r["reasons"] == {"LOW_RISK": 7, "REVIEW_THRESHOLD": 2, "HIGH_RISK": 1}


def test_shadow_would_decline_is_not_an_actual_decline():
    a = agg()
    for _ in range(3):   # shadow mode: returned as review / HIGH_RISK, would_decline=True
        a.observe(event(score=0.9995, decision="review", reason="HIGH_RISK", would_decline=True))
    a.observe(event())
    r = report(a)
    assert r["shadow_would_decline"] == {"n": 3, "rate": 0.75}
    assert r["actual_declines"] == {"n": 0, "rate": 0.0}
    assert r["decisions"]["decline"]["n"] == 0 and r["decisions"]["review"]["n"] == 3


def test_score_distribution_mean_exact_percentiles_within_one_bin():
    a = agg()
    scores = list(np.linspace(0.0, 1.0, 201))
    for s in scores:
        a.observe(event(score=float(s)))
    r = report(a)["score"]
    assert r["mean"] == pytest.approx(np.mean(scores), abs=1e-6)
    assert sum(r["hist"]) == 201 and len(r["hist"]) == mm.N_SCORE_BINS
    for q in (50, 90, 95, 99):
        true = float(np.quantile(scores, q / 100))
        i = mm.score_bin(true)
        width = mm.SCORE_EDGES[i + 1] - mm.SCORE_EDGES[i]
        assert abs(r[f"p{q}"] - true) <= width + 1e-9
    assert mm.score_bin(0.54) == mm.SCORE_EDGES.index(0.54)     # threshold is a bin edge
    assert mm.score_bin(1.0) == mm.N_SCORE_BINS - 1


def test_missing_invalid_features_and_unknown_types():
    a = agg()
    a.observe(event(missing=["oldbalanceOrg", "exceeds_orig_bal"]))
    a.observe(event(non_finite=["amount"], bins={"amount": None, "oldbalanceOrg": 3}))
    a.observe(event(ttype="WIRE"))       # not one of the five training types
    a.observe(event(ttype=None))
    a.observe(event(summary=False))      # older event without the feature summary
    r = report(a)
    q = r["quality"]
    assert q["events_with_missing_features"] == {"n": 1, "rate": 0.2}
    assert q["missing_by_feature"]["oldbalanceOrg"]["n"] == 1
    assert q["unknown_type"] == {"n": 2, "rate": 0.4}
    assert q["feature_summary_available"]["n"] == 4
    assert q["events_with_non_finite"] == {"n": 1, "rate": 0.25}      # of events that carry the summary
    assert q["non_finite_by_feature"] == {"amount": 1}
    assert r["types"]["__unknown__"]["n"] == 2


def test_rolling_windows_expire_and_cumulative_does_not():
    clock = Clock()
    a = agg(clock)
    a.observe(event(ago_s=30))              # current minute
    a.observe(event(ago_s=10 * 60))         # 10 min ago
    a.observe(event(ago_s=3 * 3600))        # 3 h ago -> 15-minute slot
    a.observe(event(ago_s=30 * 3600))       # older than 24 h -> cumulative only
    n = lambda minutes: sum(s["n"] for s in a.window(minutes)[0].values())
    assert (n(5), n(60), n(1440)) == (1, 2, 3)
    assert a.counts["older_than_24h"] == 1
    assert a.cumulative["e7_v1|dp-1"]["n"] == 4
    clock.t += 6 * 60                       # 6 minutes later
    assert n(5) == 0 and n(60) == 2
    clock.t += 2 * 3600                     # minutes fold into 15-minute slots
    assert n(60) == 0 and n(1440) == 3
    assert not any(m <= mm._minute(clock.t) - mm.MINUTE_HORIZON for m in a.minutes)
    clock.t += 24 * 3600
    assert n(1440) == 0 and a.coarse == {}
    assert a.cumulative["e7_v1|dp-1"]["n"] == 4


def test_model_versions_are_separate_and_bounded():
    a = agg()
    for _ in range(3):
        a.observe(event(model="e7_v1"))
    a.observe(event(model="e6_test", policy="dp-test", score=0.9))
    merged, _ = a.window(60)
    assert merged["e7_v1|dp-1"]["n"] == 3 and merged["e6_test|dp-test"]["n"] == 1
    for i in range(mm.MAX_SERIES + 5):
        a.observe(event(model=f"m{i}"))
    assert len(a.cumulative) == mm.MAX_SERIES + 1          # + the shared "__other__" series
    assert a.cumulative[f"{mm.OTHER}|{mm.OTHER}"]["n"] == 7


def test_counter_dicts_are_bounded():
    a = agg()
    for i in range(mm.MAX_KEYS + 10):
        a.observe(event(missing=[f"f{i}"]))
    missing = a.cumulative["e7_v1|dp-1"]["missing"]
    assert len(missing) == mm.MAX_KEYS + 1 and missing[mm.OTHER] == 10


def test_malformed_event_fields_are_counted_not_raised():
    a = agg()
    bad = event()
    bad["occurred_at"] = "not a time"
    assert a.observe(bad) is False and a.counts["observe_errors"] == 1
    assert a.observe(event()) is True


# --- recovery ----------------------------------------------------------------------------

def msgs(events, start=0, partition=0):
    return [FakeMsg(json.dumps(e).encode(), start + i, partition=partition) for i, e in enumerate(events)]


def run_consumer(messages, state, clock):
    consumer = FakeConsumer(messages)
    runner = ConsumerRunner(consumer, processor(MemorySink()), FakeDlq(), "dlq", commit_every=3,
                            commit_interval_s=999, aggregator=mm.MonitoringAggregator(state, clock=clock))
    consumer.runner = runner
    runner.run(poll_timeout_s=0)
    return runner, consumer


def test_state_saved_at_commit_and_restored_after_restart(tmp_path):
    state, clock = tmp_path / "state.json", Clock()
    batch = [event(tx=f"t{i}") for i in range(5)]
    runner, consumer = run_consumer(msgs(batch), state, clock)
    assert consumer.commits[-1] == {("events", 0): 5}
    saved = json.loads(state.read_text())
    assert saved["offsets"] == {"events:0": 5} and saved["cumulative"]["e7_v1|dp-1"]["n"] == 5

    # Crash before the commit of the last messages: Kafka re-delivers offsets 3..4 (new process,
    # empty dedup cache) followed by new ones. Monitoring counts each event exactly once.
    redelivered = msgs(batch[3:], start=3) + msgs([event(), event()], start=5)
    runner2, _ = run_consumer(redelivered, state, clock)
    a = runner2.aggregator
    assert a.counts["replay_skipped"] == 2
    assert a.cumulative["e7_v1|dp-1"]["n"] == 7
    assert sum(s["n"] for s in a.window(60)[0].values()) == 7


def test_dead_letters_and_duplicates_advance_the_mark_without_gaps(tmp_path):
    good = event()
    messages = [FakeMsg(json.dumps(good).encode(), 0), FakeMsg(b"{broken", 1),
                FakeMsg(json.dumps(good).encode(), 2), FakeMsg(json.dumps(event()).encode(), 3)]
    runner, _ = run_consumer(messages, tmp_path / "s.json", Clock())
    a = runner.aggregator
    assert a.cumulative["e7_v1|dp-1"]["n"] == 2          # duplicate and dead letter not counted
    assert a.offsets == {"events:0": 4} and a.counts["offset_gap_not_aggregated"] == 0


def test_lost_state_reports_the_unaggregated_range(tmp_path):
    state = tmp_path / "s.json"
    run_consumer(msgs([event()] * 1, start=0), state, Clock())
    # Consumer resumes at offset 40 (e.g. state from an older run restored): 39 events were
    # consumed without being aggregated. That is reported, not hidden.
    runner, _ = run_consumer(msgs([event()], start=40), state, Clock())
    assert runner.aggregator.counts["offset_gap_not_aggregated"] == 39


def test_corrupt_state_starts_empty_and_is_kept_aside(tmp_path):
    state = tmp_path / "s.json"
    state.write_text("{not json")
    a = mm.MonitoringAggregator(state, clock=Clock())
    assert a.counts["state_load_failed"] == 1 and a.cumulative == {}
    assert list(tmp_path.glob("s.json.corrupt-*"))
    a.observe(event(), "events", 0, 0)
    assert a.save() and json.loads(state.read_text())["cumulative"]["e7_v1|dp-1"]["n"] == 1


def test_save_failure_never_raises(tmp_path):
    a = mm.MonitoringAggregator(tmp_path / "missing-dir" / "s.json", clock=Clock())
    a.observe(event(), "events", 0, 0)
    assert a.save() is False and a.counts["save_failures"] == 1


def test_state_size_is_bounded(tmp_path):
    clock = Clock()
    a = mm.MonitoringAggregator(tmp_path / "s.json", clock=clock)
    for minute in range(26 * 60):            # one event per minute for 26 hours
        clock.t = NOW + minute * 60
        a.observe(event(now=clock.t), "events", 0, minute)
    a.save()
    assert len(a.minutes) <= mm.MINUTE_HORIZON
    assert len(a.coarse) <= mm.HORIZON_MINUTES // mm.COARSE_MINUTES + 2
    assert (tmp_path / "s.json").stat().st_size < 600_000


# --- feature summary, API event, input rejections ---------------------------------------------

def test_log_bins_scalar_and_vector_agree():
    values = [0.0, 0.5, 1.0, 9.0, 999.0, 1e6, 1e7, 1e30, -5.0, math.nan, math.inf]
    vec = log_bins(values)
    for v, b in zip(values, vec):
        s = log_bin(v)
        assert (s is None and np.isnan(b)) or s == b
    assert (log_bin(0.0), log_bin(9.0), log_bin(1e6), log_bin(1e30)) == (0, 4, 24, 63)


def test_feature_summary_matches_the_pipeline(model_dir):
    predictor = FraudPredictor.load(model_dir)
    record = {"type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0, "oldbalanceDest": None}
    x = predictor.pipeline.transform_one(record)
    s = feature_summary(x, predictor.feature_names)
    assert s["feature_bins"]["amount"] == log_bin(181.0)
    assert s["feature_bins"]["oldbalanceDest"] is None              # missing
    assert s["feature_flags"]["orig_bal_zero"] == 0 and s["feature_flags"]["dest_bal_zero"] is None
    assert s["non_finite_features"] == []


def test_api_event_has_bins_but_no_exact_amounts(model_dir):
    from api.main import app
    from tests.test_api import RecordingPublisher, make_client, policy, reset_state
    publisher = RecordingPublisher()
    app.state.event_publisher = publisher
    try:
        with make_client(model_dir, policy()) as c:
            body = {"transaction_id": "m-1", "type": "TRANSFER", "amount": 12345.67, "sender_id": "C9",
                    "receiver_id": "C8", "sender_balance": 98765.43, "receiver_balance": 0.0}
            assert c.post("/predict", json=body).status_code == 200
    finally:
        reset_state()
    e = publisher.events[0]
    assert e["feature_bins"]["amount"] == log_bin(12345.67)
    assert e["feature_flags"]["dest_bal_zero"] == 1
    text = json.dumps(e)
    assert "12345.67" not in text and "98765.43" not in text and "C9" not in text


def test_input_rejections_are_counted(client):
    from api import main
    before = client.get("/metrics/decisions").json()["input_rejections"]["total"]
    r = client.post("/predict", json={"type": "WIRE", "amount": 1.0, "sender_id": "a", "receiver_id": "b"})
    assert r.status_code == 422 and r.json()["detail"]          # default FastAPI response unchanged
    client.post("/predict", json={"type": "PAYMENT", "amount": -1.0, "sender_id": "a", "receiver_id": "b"})
    m = client.get("/metrics/decisions").json()["input_rejections"]
    assert m["total"] - before == 2
    assert m["by_field"]["type"] >= 1 and m["by_field"]["amount"] >= 1
    assert set(m["by_field"]) <= main.INPUT_REJECTION_FIELDS | {"other"}


@pytest.fixture
def client(model_dir):
    from tests.test_api import make_client, policy, reset_state
    with make_client(model_dir, policy()) as c:
        yield c
    reset_state()


# --- baseline ----------------------------------------------------------------------------

@pytest.fixture
def baseline_model(model_dir, transactions, tmp_path):
    """The test model with the split it would have recorded if trained by src.train on this data."""
    out = tmp_path / "model"
    shutil.copytree(model_dir, out)
    meta = json.loads((out / "metadata.json").read_text())
    meta["split_boundaries"] = chronological_split(transactions)[3]
    meta["n_rows"] = len(transactions)
    (out / "metadata.json").write_text(json.dumps(meta))
    return out


def test_baseline_is_reproducible_and_linked_to_the_model(baseline_model, transactions):
    pol = DecisionPolicy("dp-test", 0.54, 0.999, valid_for_models=("e6_test",))
    kw = dict(baseline_version="bl-e6_test-1", policy=pol, dataset={"sha256": "x", "n_rows": len(transactions)})
    b1 = build_baseline(transactions, baseline_model, **kw)
    b2 = build_baseline(transactions.copy(), baseline_model, **kw)
    assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)
    assert b1["model_version"] == "e6_test" and set(b1["artifacts"]) == {"model.json", "pipeline.json",
                                                                         "metadata.json"}
    train = chronological_split(transactions)[0]
    assert b1["split"]["train_rows"] == len(train) and b1["split"]["statistics_from"] == "train"
    assert b1["feature_names"] == FraudPredictor.load(baseline_model).feature_names
    assert not {"isFraud", "newbalanceOrig", "newbalanceDest", "isFlaggedFraud"} & set(b1["features"])
    assert sum(b1["transaction_types"].values()) == pytest.approx(1.0)
    assert sum(b1["reference_scores"]["score_hist"]) == b1["split"]["val_rows"]
    assert b1["features"]["amount"]["count"] == len(train)


def test_baseline_refuses_data_that_does_not_reproduce_the_split(baseline_model, transactions):
    pol = DecisionPolicy("dp-test", 0.54, 0.999, valid_for_models=("e6_test",))
    with pytest.raises(SystemExit):
        build_baseline(transactions.iloc[:2000], baseline_model, baseline_version="bl", policy=pol, dataset={})


def test_report_against_baseline_flags_ranges_and_type_mix():
    baseline = {"transaction_types": {"PAYMENT": 0.5, "TRANSFER": 0.5, "CASH_IN": 0, "CASH_OUT": 0, "DEBIT": 0},
                "bin_range": {"amount": [0, 20]}, "bin_p999": {"amount": 18}, "flags": {"is_max_amount": 0.0},
                "features": {"amount": {"missing_rate": 0.0}}}
    a = agg()
    for _ in range(3):
        a.observe(event(ttype="PAYMENT", bins={"amount": 10}))
    a.observe(event(ttype="PAYMENT", bins={"amount": 19}))     # > p99.9, inside range
    a.observe(event(ttype="PAYMENT", bins={"amount": 25}))     # outside the training range
    r = report(a, baseline=baseline)
    assert r["types_vs_baseline"]["tvd"] == pytest.approx(0.5)
    assert r["feature_ranges"]["amount"]["outside_training_range"]["n"] == 1
    assert r["quality"]["unexpected_amount"] == {"n": 2, "rate": 0.4}


# --- labels and performance ------------------------------------------------------------------

def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def stored(n_fraud_caught=8, n_fraud_missed=2, n_fp=10, n_tn=90, ago_days=40):
    out, i = [], 0
    for decision, score, count in (("review", 0.9, n_fraud_caught), ("approve", 0.1, n_fraud_missed),
                                   ("review", 0.7, n_fp), ("approve", 0.01, n_tn)):
        for _ in range(count):
            e = event(score=score, decision=decision, tx=f"tx-{i}", ago_s=ago_days * 86400)
            out.append(e)
            i += 1
    labels = {}
    for j, e in enumerate(out):
        fraud = j < n_fraud_caught + n_fraud_missed
        labels[e["transaction_id"]] = {"transaction_id": e["transaction_id"], "is_fraud": int(fraud),
                                       "label_at": NOW - 86400, "source": "test", "dataset_step": None}
    return out, labels


def test_no_labels_means_unavailable_not_zero():
    events, _ = stored()
    r = evaluate_performance(events, {}, {}, now=NOW)
    assert r["status"] == "unavailable" and "no actual labels" in r["reason"]
    assert "precision" not in json.dumps(r) and r["counts"]["predictions"] == len(events)


def test_performance_with_valid_labels():
    events, labels = stored()
    r = evaluate_performance(events, labels, {}, now=NOW, min_labels=50)
    m = r["at_approve_threshold"]
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (8, 2, 10, 90)
    assert m["precision"] == pytest.approx(8 / 18) and m["recall"] == pytest.approx(0.8)
    assert m["f1"] == pytest.approx(2 * (8 / 18) * 0.8 / ((8 / 18) + 0.8))
    assert m["fpr"] == pytest.approx(10 / 100) and m["cost"] == 100 * 2 + 10
    assert 0 < r["pr_auc"] <= 1 and r["counts"]["label_coverage"] == 1.0


def test_unlabeled_predictions_are_excluded_not_legitimate():
    events, labels = stored()
    extra = [event(score=0.9, decision="review", tx=f"u-{i}") for i in range(500)]
    r = evaluate_performance(events + extra, labels, {}, now=NOW, min_labels=50)
    assert r["at_approve_threshold"]["FP"] == 10                     # the 500 unlabeled are not FP
    assert r["counts"]["label_coverage"] == pytest.approx(110 / 610)


def test_insufficient_labels():
    events, labels = stored()
    few = dict(list(labels.items())[:20])
    r = evaluate_performance(events, few, {}, now=NOW, min_labels=100)
    assert r["status"] == "insufficient_labels" and "at_approve_threshold" not in r
    only_legit = {k: v for k, v in labels.items() if not v["is_fraud"]}
    assert evaluate_performance(events, only_legit, {}, now=NOW, min_labels=10)["status"] == "insufficient_labels"


def test_incorrect_and_delayed_labels(tmp_path):
    e_early = event(tx="early", ago_s=3600)
    e_conf, e_tune, e_young = event(tx="conflict"), event(tx="tune"), event(tx="young", ago_s=60)
    lines = [
        {"transaction_id": "early", "is_fraud": 1, "label_at": iso(NOW - 7200), "source": "x"},   # before prediction
        {"transaction_id": "conflict", "is_fraud": 1, "label_at": iso(NOW + 10), "source": "chargeback"},
        {"transaction_id": "conflict", "is_fraud": 0, "label_at": iso(NOW + 20), "source": "reversal"},  # latest wins
        {"transaction_id": "tune", "is_fraud": 1, "label_at": iso(NOW + 10), "source": "replay", "dataset_step": 300},
        {"transaction_id": "nobody", "is_fraud": 0, "label_at": iso(NOW + 10), "source": "x"},
        {"transaction_id": "young", "is_fraud": 0, "label_at": iso(NOW + 30), "source": "x"},
        {"transaction_id": "bad", "is_fraud": "maybe", "label_at": iso(NOW), "source": "x"},
    ]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(x) for x in lines) + "\n{broken\n")
    res = add_labels(src, tmp_path / "labels")
    assert (res["appended"], res["rejected"]) == (6, 2)
    labels, counts = read_labels(tmp_path / "labels")
    assert labels["conflict"]["is_fraud"] == 0 and counts["label_conflicts"] == 1
    r = evaluate_performance([e_early, e_conf, e_tune, e_young], labels, counts, now=NOW + 100,
                             tuning_steps=[(1, 280), (281, 354)], min_labels=1)
    c = r["counts"]
    assert c["label_before_prediction"] == 1 and c["excluded_tuning_data"] == 1
    assert c["labels_without_prediction"] == 1 and c["labeled_predictions"] == 2
    assert c["mature_predictions"] == 0                      # all younger than the 30-day maturity
    assert r["status"] == "insufficient_labels"              # only legitimate labels remain


# --- dashboard -------------------------------------------------------------------------------

@pytest.fixture
def dash(tmp_path, monkeypatch):
    from monitoring import live_dashboard as d
    store, baselines = tmp_path / "store", tmp_path / "baselines"
    store.mkdir()
    (baselines / "e7_v1").mkdir(parents=True)
    from tests.test_drift_detection import baseline as valid_baseline
    b = valid_baseline("e7_v1")       # the dashboard only shows baselines that pass validation
    b["baseline_version"] = "bl-e7_v1-1"
    b["feature_names"] = json.loads((d.ROOT / "models/e7_v1/metadata.json").read_text())["feature_names"] \
        if (d.ROOT / "models/e7_v1/metadata.json").exists() else b["feature_names"]
    (baselines / "e7_v1" / "bl-e7_v1-1.json").write_text(json.dumps(b))
    monkeypatch.setattr(d, "STORE_DIR", store)
    monkeypatch.setattr(d, "BASELINE_DIR", baselines)
    monkeypatch.setattr(d.collector, "performance", d.PerformanceCache(store))
    monkeypatch.setattr(d.collector, "last_kafka", None)
    return d, store, TestClient(d.app)


def test_dashboard_reports_missing_state(dash):
    _, _, c = dash
    r = c.get("/api/ml").json()
    assert r["available"] is False and "no monitoring state" in r["reason"]


def test_dashboard_ml_freshness_and_insufficient_data(dash):
    import time as _time
    d, store, c = dash
    a = mm.MonitoringAggregator(store / "ml_monitoring_state.json")
    for i in range(3):
        a.observe(event(now=_time.time()), "events", 0, i)
    a.save()
    d.collector.last_kafka = (_time.time(), {"up": True, "lag": 0})
    r = c.get("/api/ml?window=5m").json()
    s = r["series"]["e7_v1|dp-1"]
    assert s["n"] == 3 and s["insufficient_data"] is True
    assert s["baseline_version"] == "bl-e7_v1-1"
    assert s["performance"]["status"] == "unavailable"
    f = r["freshness"]
    assert f["stale"] is False and f["state_age_s"] < 5 and f["newest_prediction_age_s"] < 5

    # Old state while events wait in Kafka -> stale. Old state with no lag (idle) -> not stale.
    old = json.loads((store / "ml_monitoring_state.json").read_text())
    old["saved_at"] = datetime.fromtimestamp(_time.time() - 120, timezone.utc).isoformat(timespec="seconds")
    (store / "ml_monitoring_state.json").write_text(json.dumps(old))
    d.collector.last_kafka = (_time.time(), {"up": True, "lag": 25})
    assert c.get("/api/ml").json()["freshness"]["stale"] is True
    d.collector.last_kafka = (_time.time(), {"up": True, "lag": 0})
    assert c.get("/api/ml").json()["freshness"]["stale"] is False
    d.collector.last_kafka = None
    assert c.get("/api/ml").json()["freshness"]["stale"] is None       # lag unknown
    assert c.get("/api/ml?window=7d").status_code == 422


def test_traffic_sources_are_separate_series_and_old_keys_are_production():
    a = agg()
    tagged = event()
    tagged["traffic_source"] = "loadtest"
    a.observe(event())
    a.observe(tagged)
    untagged_old = event()                   # event written before the tag existed
    a.observe(untagged_old)
    merged, _ = a.window(60)
    assert merged["e7_v1|dp-1"]["n"] == 2 and merged["e7_v1|dp-1|loadtest"]["n"] == 1
    assert mm.series_parts("e7_v1|dp-1") == ("e7_v1", "dp-1", "production")
    assert mm.series_parts("e7_v1|dp-1|demo") == ("e7_v1", "dp-1", "demo")
    rep = mm.monitoring_report(a, 60, "/nonexistent", now=NOW, min_events=1)
    assert rep["series"]["e7_v1|dp-1|loadtest"]["traffic_source"] == "loadtest"
    assert rep["series"]["e7_v1|dp-1"]["traffic_source"] == "production"
