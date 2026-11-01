"""Drift monitor (monitoring/drift_monitor.py): windows over the consumer's real aggregation state,
model/baseline versions, restart recovery, staleness, alerts and their deduplication, dashboard API.
No Kafka broker is used (conftest removes KAFKA_BOOTSTRAP_SERVERS, so the lag is 'unknown')."""
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from monitoring import drift_monitor as dm
from src import drift_detection as dd
from src.ml_monitoring import MonitoringAggregator, load_state
from tests.test_drift_detection import AMOUNT, FLAG, TYPES, baseline, scaled

NOW = datetime(2026, 10, 30, 12, 0, 30, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def expand(counts):
    for k, c in counts.items():
        for _ in range(c):
            yield k


def events(n=1000, model="e7_v1", ago_s=0, shift=0, now=NOW, source=None):
    """n events whose amount/flag/type mix matches the reference (amount shifted by `shift` bins)."""
    amounts = list(expand({k + shift: v for k, v in scaled(AMOUNT, n).items()}))
    flags = list(expand(scaled(FLAG, n)))
    types = list(expand(scaled(TYPES, n)))
    t = datetime.fromtimestamp(now - ago_s, timezone.utc).isoformat(timespec="milliseconds")
    for i in range(n):
        yield {"occurred_at": t, "model_version": model, "decision_policy_version": "dp-1", "decision": "approve",
               "reason_code": "LOW_RISK", "fraud_probability": 1e-6, "would_decline": False,
               "transaction_type": types[i], "missing_features": [],
               "feature_bins": {"amount": amounts[i]}, "feature_flags": {"orig_bal_zero": flags[i]},
               "non_finite_features": [], **({"traffic_source": source} if source else {})}


@pytest.fixture
def env(tmp_path):
    store, bdir, models = tmp_path / "store", tmp_path / "baselines", tmp_path / "models"
    store.mkdir()
    (bdir / "e7_v1").mkdir(parents=True)
    (bdir / "e7_v1" / "bl-e7_v1-1.json").write_text(json.dumps(baseline("e7_v1")))
    cfg = tmp_path / "drift.json"
    cfg.write_text(json.dumps({"min_samples": 200, "alerts": {"low_volume_min_per_hour": None}}))
    return {"store": store, "baselines": bdir, "models": models, "config": cfg}


def fill(env, evs, clock=None, offset0=0):
    a = MonitoringAggregator(env["store"] / "ml_monitoring_state.json", clock=clock or Clock())
    for i, e in enumerate(evs):
        a.observe(e, "events", 0, offset0 + i)
    a.save()
    return a


def run(env, now=NOW, **kw):
    return dm.run_once(env["store"], baseline_dir=env["baselines"], models_dir=env["models"],
                       config_path=env["config"], now=now, **kw)


def feat(report, window, feature, model="e7_v1"):
    return next(r for r in report["windows"][window]["models"][model]["features"] if r["feature"] == feature)


def alerts(env):
    p = env["store"] / dm.ALERTS_FILE
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# --- windows, versions, recovery ------------------------------------------------------------------

def test_windows_use_prediction_time(env):
    fill(env, events(600, ago_s=2 * 3600))                 # two hours ago: in 24 h, not in 1 h
    r = run(env)
    assert r["status"] == "ok"
    assert r["windows"]["1h"]["models"] == {}
    s = r["windows"]["24h"]["models"]["e7_v1"]["summary"]
    assert s["status"] == dd.NO_DRIFT and s["production_n"] == 600
    a = feat(r, "24h", "amount")
    assert a["window"] == "24h" and a["baseline_version"] == "bl-e7_v1-1" and a["model_version"] == "e7_v1"
    assert a["reference_n"] == sum(AMOUNT.values()) and a["production_n"] == 600 and a["detected_at"]


def test_models_are_separate_and_a_model_without_baseline_is_insufficient(env):
    fill(env, [*events(400, model="e7_v1"), *events(400, model="e9_v1", shift=6)])
    r = run(env)
    w = r["windows"]["1h"]["models"]
    assert w["e7_v1"]["summary"]["status"] == dd.NO_DRIFT and w["e7_v1"]["summary"]["production_n"] == 400
    assert w["e9_v1"]["summary"]["status"] == dd.INSUFFICIENT
    assert "no baseline" in w["e9_v1"]["summary"]["reason"] and w["e9_v1"]["baseline_version"] is None


def test_invalid_newer_baseline_is_skipped_for_the_valid_one(env):
    bad = baseline("e7_v1")
    bad["baseline_version"] = "bl-e7_v1-2"
    bad["model_version"] = "e7_v2"
    (env["baselines"] / "e7_v1" / "bl-e7_v1-2.json").write_text(json.dumps(bad))
    fill(env, events(400))
    m = run(env)["windows"]["1h"]["models"]["e7_v1"]
    assert m["baseline_version"] == "bl-e7_v1-1" and m["baselines_skipped"][0]["file"] == "bl-e7_v1-2.json"


def test_report_is_identical_after_consumer_restart(env):
    a = fill(env, [*events(500, shift=4), *events(300, ago_s=3 * 3600)])
    cfg = dd.load_config(env["config"])
    kw = dict(baseline_dir=env["baselines"], models_dir=env["models"], now=NOW)
    before = dm.compute(a, cfg, **kw)["windows"]
    after = dm.compute(load_state(env["store"] / "ml_monitoring_state.json"), cfg, **kw)["windows"]
    assert before == after
    restarted = MonitoringAggregator(env["store"] / "ml_monitoring_state.json", clock=Clock())
    for i, e in enumerate(events(500, shift=4)):          # re-delivered after a crash: offsets below the mark
        assert restarted.observe(e, "events", 0, i) is False
    assert dm.compute(restarted, cfg, **kw)["windows"] == before


# --- alerts -------------------------------------------------------------------------------------

def test_feature_drift_alert_fires_once_and_resolves_only_with_enough_data(env):
    fill(env, events(500, shift=4))
    r = run(env)
    fired = [a for a in alerts(env) if a["event"] == "fired" and a["rule"] == "feature_drift"]
    assert {(a["feature"], a["window"]) for a in fired} == {("amount", "1h"), ("amount", "24h")}
    a = fired[0]
    for k in ("drift_score", "threshold", "window", "window_start", "sample_count", "baseline_version",
              "model_version", "at"):
        assert a[k] is not None
    assert len(r["active_alerts"]) == 2

    run(env, now=NOW + 60)                                # unchanged condition: no new records
    assert len(alerts(env)) == len(fired)

    (env["store"] / "ml_monitoring_state.json").unlink()
    fill(env, events(50, shift=4))                        # too few events: insufficient, alert stays
    run(env, now=NOW + 120)
    assert len(alerts(env)) == len(fired) and len(dm.AlertManager(env["store"]).active) == 2

    (env["store"] / "ml_monitoring_state.json").unlink()
    fill(env, events(500))                                # back to the reference: resolved
    run(env, now=NOW + 180)
    resolved = [a for a in alerts(env) if a["event"] == "resolved"]
    assert len(resolved) == 2 and dm.AlertManager(env["store"]).active == {}


def test_alert_state_survives_restart_and_escalation_refires(env):
    m = dm.AlertManager(env["store"])
    cond = {"key": "feature_drift|m|1h|amount", "rule": "feature_drift", "state": "active", "severity": "moderate",
            "details": {"feature": "amount", "drift_score": 0.12}}
    assert [r["event"] for r in m.update([cond], NOW)] == ["fired"]
    m2 = dm.AlertManager(env["store"])                    # process restart
    assert m2.update([cond], NOW + 60) == []
    assert [r["event"] for r in m2.update([{**cond, "severity": "substantial"}], NOW + 120)] == ["escalated"]
    assert m2.update([{**cond, "severity": "substantial"}], NOW + 180) == []
    assert m2.update([{**cond, "state": "unknown"}], NOW + 240) == []        # unknown never resolves
    assert [r["event"] for r in m2.update([{**cond, "state": "clear"}], NOW + 300)] == ["resolved"]


def test_corrupt_alert_state_starts_clean(env):
    (env["store"] / dm.ALERT_STATE_FILE).write_text("{not json")
    assert dm.AlertManager(env["store"]).active == {}


def test_sustained_drift_needs_continuity_and_the_long_window(env):
    a = fill(env, events(500, shift=4))
    cfg = dd.load_config(env["config"])
    report = dm.compute(a, cfg, baseline_dir=env["baselines"], models_dir=env["models"], now=NOW)
    since = {}
    first = {c["key"]: c for c in dm.conditions(report, cfg, since, NOW)}
    assert first["sustained_drift|e7_v1|amount"]["state"] == "unknown"      # just started
    later = {c["key"]: c for c in dm.conditions(report, cfg, since, NOW + 61 * 60)}
    assert later["sustained_drift|e7_v1|amount"]["state"] == "active"
    assert later["sustained_drift|e7_v1|amount"]["details"]["drifting_since"].startswith("2026-10-30T12:00")
    # the long window back to normal: not sustained
    report["windows"]["24h"]["models"]["e7_v1"]["features"] = [
        {**r, "status": dd.NO_DRIFT} for r in report["windows"]["24h"]["models"]["e7_v1"]["features"]]
    again = {c["key"]: c for c in dm.conditions(report, cfg, since, NOW + 62 * 60)}
    assert again["sustained_drift|e7_v1|amount"]["state"] == "clear"


def test_stale_monitoring_and_dead_consumer(env):
    fill(env, events(300))
    cfg = dd.load_config(env["config"])
    kw = dict(baseline_dir=env["baselines"], models_dir=env["models"])
    agg = load_state(env["store"] / "ml_monitoring_state.json")
    assert dm.compute(agg, cfg, now=NOW + 600, lag=0, **kw)["freshness"]["stale"] is False    # idle, nothing waiting
    assert dm.compute(agg, cfg, now=NOW + 600, lag=40, **kw)["freshness"]["stale"] is True    # events wait, state old
    assert dm.compute(agg, cfg, now=NOW + 60, lag=40, **kw)["freshness"]["stale"] is False    # recent enough
    assert dm.compute(agg, cfg, now=NOW + 10, lag=None, **kw)["freshness"]["stale"] is None   # unknown
    run(env, now=NOW + 600, alive=False)
    assert any(a["rule"] == "monitoring_stale" and a["event"] == "fired" for a in alerts(env))


def test_low_volume_alert(env):
    env["config"].write_text(json.dumps({"min_samples": 200, "alerts": {"low_volume_min_per_hour": 1000}}))
    fill(env, events(300))
    run(env)
    low = [a for a in alerts(env) if a["rule"] == "low_volume"]
    assert low and low[0]["sample_count"] == 300 and low[0]["threshold"] == 1000


def test_no_state_and_failed_calculation(env):
    r = run(env)
    assert r["status"] == "no_state" and r["last_successful_at"] == r["computed_at"]
    fill(env, events(300))
    ok = run(env, now=NOW + 60)
    env["config"].write_text(json.dumps({"psi": {"moderate": 0.5, "substantial": 0.1}}))     # invalid config
    bad = run(env, now=NOW + 120)
    assert bad["status"] == dd.ERROR and "ConfigError" in bad["reason"]
    assert bad["last_successful_at"] == ok["computed_at"]
    saved = json.loads((env["store"] / dm.REPORT_FILE).read_text())
    assert saved["status"] == dd.ERROR


# --- dashboard ------------------------------------------------------------------------------------

@pytest.fixture
def dash(tmp_path, monkeypatch):
    from monitoring import live_dashboard as d
    monkeypatch.setattr(d, "STORE_DIR", tmp_path)
    return d, tmp_path, TestClient(d.app)


def test_dashboard_drift_endpoint(dash, env):
    d, store, c = dash
    r = c.get("/api/drift").json()
    assert r["available"] is False and "start the drift monitor" in r["reason"]
    fill(env, events(500, shift=4))
    report = run(env)
    report["computed_at"] = "2026-01-01T00:00:00+00:00"   # written long ago: the monitor stopped
    (store / dm.REPORT_FILE).write_text(json.dumps(report))
    (store / dm.ALERTS_FILE).write_text((env["store"] / dm.ALERTS_FILE).read_text() + "garbage\n")
    r = c.get("/api/drift").json()
    assert r["available"] is True and r["report_stale"] is True
    assert r["windows"]["1h"]["models"]["e7_v1"]["summary"]["features_drifting"] >= 1
    assert len(r["recent_alerts"]) >= 2 and all("event" in a for a in r["recent_alerts"])


# --- traffic sources and acknowledgements -----------------------------------------------------------

def test_load_test_traffic_is_evaluated_but_never_alerts(env):
    fill(env, [*events(500), *events(500, shift=4, source="loadtest")])
    r = run(env)
    w = r["windows"]["1h"]
    assert w["traffic_by_source"] == {"production": 500, "loadtest": 500}
    assert w["models"]["e7_v1"]["summary"]["status"] == dd.NO_DRIFT
    lt = w["models"]["e7_v1|loadtest"]
    assert lt["traffic_source"] == "loadtest" and lt["model_version"] == "e7_v1"
    assert lt["summary"]["status"] == dd.DRIFT                     # shown...
    assert alerts(env) == [] and r["active_alerts"] == []           # ...never alerted
    assert feat(r, "1h", "amount", model="e7_v1|loadtest")["traffic_source"] == "loadtest"


def test_low_volume_counts_monitored_sources_only(env):
    env["config"].write_text(json.dumps({"min_samples": 200, "alerts": {"low_volume_min_per_hour": 400}}))
    fill(env, [*events(300), *events(600, source="loadtest")])
    run(env)
    low = [a for a in alerts(env) if a["rule"] == "low_volume"]
    assert low and low[0]["sample_count"] == 300


def test_acknowledgement_silences_records_reason_and_expires(env):
    fill(env, events(500, shift=4))
    run(env)
    fired = len(alerts(env))
    dm.request_ack(env["store"], "feature_drift|e7_v1|1h|amount", "synthetic wrk traffic, see investigation",
                   hours=0.25, by="tester", now=NOW + 30)
    run(env, now=NOW + 60)
    acks = [a for a in alerts(env) if a["event"] == "acknowledged"]
    assert len(acks) == 1 and acks[0]["reason"].startswith("synthetic") and acks[0]["by"] == "tester"
    st = dm.AlertManager(env["store"])
    assert st.active["feature_drift|e7_v1|1h|amount"]["ack"]["by"] == "tester"
    assert "ack" not in st.active["feature_drift|e7_v1|24h|amount"]
    run(env, now=NOW + 120)
    assert len(alerts(env)) == fired + 1                         # silent while acknowledged
    run(env, now=NOW + 30 + 900 + 60)                            # expired, condition still there: fires again
    refired = [a for a in alerts(env) if a["event"] == "fired" and "note" in a]
    assert len(refired) == 1 and "expired" in refired[0]["note"]
    dm.request_ack(env["store"], "no-such-alert", "typo", now=NOW)
    run(env, now=NOW + 3800)                                     # unmatched request: ignored, no crash
