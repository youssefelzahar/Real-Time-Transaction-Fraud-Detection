"""Phase 1 drift audit: independent checks of the drift statistics, their behaviour under no drift
and known drift, window boundaries, and regression tests for defects D1-D7
(investigations/2026-10-31-phase1-drift-audit.md). Synthetic data only; no broker."""
import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest
from scipy import stats as sps

from monitoring import drift_monitor as dm
from src import drift_detection as dd
from src import ml_monitoring as mm
from tests.test_drift_detection import AMOUNT, FLAG, SCORES, TYPES, baseline
from tests.test_drift_detection import stats as make_stats
from tests.test_drift_monitor import NOW, Clock, alerts, env, events, feat, fill, run  # noqa: F401 (fixture)

CFG = dd.load_config(None)


# --- independent cross-checks ---------------------------------------------------------------------

def numpy_psi(ref, prod, alpha=0.5):
    ref, prod = np.asarray(ref, float), np.asarray(prod, float)
    k = len(ref)
    pr = (ref + alpha) / (ref.sum() + alpha * k)
    pp = (prod + alpha) / (prod.sum() + alpha * k)
    return float(np.sum((pp - pr) * np.log(pp / pr)))


def test_psi_matches_an_independent_implementation_and_is_symmetric():
    rng = np.random.default_rng(1)
    for _ in range(200):
        k = int(rng.integers(2, 12))
        ref = rng.integers(0, 1000, k).tolist()
        prod = rng.integers(0, 1000, k).tolist()
        if sum(ref) == 0 or sum(prod) == 0:
            continue
        assert dd.psi(ref, prod) == pytest.approx(numpy_psi(ref, prod), rel=1e-12, abs=1e-15)
        # symmetric when both sides have the same total (same smoothing denominators)
        assert numpy_psi(ref, ref) == pytest.approx(0.0, abs=1e-15)
    a, b = [100, 300, 600], [600, 300, 100]
    assert dd.psi(a, b) == pytest.approx(dd.psi(b, a))


def test_chi_square_matches_scipy_when_no_merging_is_needed():
    rng = np.random.default_rng(2)
    for _ in range(100):
        ref = rng.integers(500, 5000, 6)
        prod = rng.integers(200, 2000, 6)
        t = dd.chi_square_gof(prod.tolist(), ref.tolist(), min_expected=5)
        expected = prod.sum() * ref / ref.sum()
        s = sps.chisquare(prod, expected)
        assert t["applicable"] and t["df"] == 5
        assert t["statistic"] == pytest.approx(s.statistic, abs=1e-3)
        assert t["p_value"] == pytest.approx(s.pvalue, rel=1e-9, abs=1e-300)


def test_holm_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(100):
        p = rng.random(int(rng.integers(1, 15))) ** 3
        adj = dd.holm({i: float(v) for i, v in enumerate(p)})
        order = np.argsort(p)
        m = len(p)
        brute = {}
        for rank, i in enumerate(order):
            brute[i] = min(1.0, max((m - j) * p[order[j]] for j in range(rank + 1)))
        for i in range(m):
            assert adj[i] == pytest.approx(brute[i])


def test_ks_distance_matches_numpy_cdf():
    rng = np.random.default_rng(4)
    a = dict(zip(range(20), rng.integers(0, 100, 20).tolist()))
    b = dict(zip(range(5, 25), rng.integers(0, 100, 20).tolist()))
    keys = sorted(set(a) | set(b))
    ca = np.cumsum([a.get(k, 0) for k in keys]) / sum(a.values())
    cb = np.cumsum([b.get(k, 0) for k in keys]) / sum(b.values())
    assert dd.ks_distance(a, b) == pytest.approx(float(np.max(np.abs(ca - cb))))


# --- behaviour under no drift and known drift (Monte Carlo, fixed seeds) ---------------------------

def sample_stats(rng, n, amount_shift=0, missing=0, types=None):
    amount = dict(zip(AMOUNT, rng.multinomial(n - missing, np.array(list(AMOUNT.values())) / sum(AMOUNT.values()))))
    amount = {k + amount_shift: int(v) for k, v in amount.items() if v}
    flag = dict(zip(FLAG, rng.multinomial(n, np.array(list(FLAG.values())) / sum(FLAG.values())).tolist()))
    tp = types or TYPES
    t = dict(zip(tp, rng.multinomial(n, np.array(list(tp.values())) / sum(tp.values())).tolist()))
    scores = dict(enumerate(rng.multinomial(n, np.array(SCORES) / sum(SCORES)).tolist()))
    return make_stats(n=n, amount=amount, flag=flag, types=t, scores={k: v for k, v in scores.items() if v},
                      missing=missing)


def test_no_drift_false_alarm_rate_is_low_at_min_samples():
    rng = np.random.default_rng(10)
    n, reps = CFG["min_samples"], 300
    flagged, psis = {}, []
    for _ in range(reps):
        out = dd.evaluate_window(sample_stats(rng, n), baseline(), CFG)
        for r in out["features"]:
            flagged[r["feature"]] = flagged.get(r["feature"], 0) + (r["status"] == dd.DRIFT)
            if r["feature"] == "amount":
                psis.append(r["psi"])
        flagged["prediction"] = flagged.get("prediction", 0) + (out["prediction"]["status"] == dd.DRIFT)
    for feature, count in flagged.items():
        assert count / reps <= 0.02, (feature, count)
    # Small-sample bias: about (K-1)/N = 11/500 = 0.022 for amount's 12 groups, plus smoothing.
    # Measured mean 0.027: well below the 0.10 "moderate" band, which is what keeps false alarms rare.
    assert np.mean(psis) < 0.035


@pytest.mark.parametrize("kind", ["amount_x10", "missing_20pct", "type_mix"])
def test_known_drift_is_detected(kind):
    rng = np.random.default_rng(11)
    n, reps, hits = CFG["min_samples"], 100, 0
    shifted_types = {"CASH_IN": 1, "CASH_OUT": 1, "DEBIT": 1, "PAYMENT": 1, "TRANSFER": 6}
    feature = {"amount_x10": "amount", "missing_20pct": "amount", "type_mix": "transaction_type"}[kind]
    for _ in range(reps):
        s = sample_stats(rng, n, amount_shift=4 if kind == "amount_x10" else 0,
                         missing=n // 5 if kind == "missing_20pct" else 0,
                         types=shifted_types if kind == "type_mix" else None)
        r = next(f for f in dd.evaluate_window(s, baseline(), CFG)["features"] if f["feature"] == feature)
        hits += r["status"] == dd.DRIFT
    assert hits / reps >= 0.99


# --- window boundaries ------------------------------------------------------------------------------

def test_one_hour_window_edges():
    clock = Clock()
    a = mm.MonitoringAggregator(None, clock=clock)
    now_m = int(NOW // 60)
    first = (now_m - 60 + 1) * 60          # first minute inside the 60-minute window
    for i, t in enumerate((first, first - 60)):
        e = next(events(1, now=t + 1))     # +1 s: inside that minute
        a.observe(e, "events", 0, i)
    merged, _ = a.window(60)
    assert merged["e7_v1|dp-1"]["n"] == 1   # the minute before the window is excluded


def test_long_window_reports_its_effective_start(env):  # D7
    fill(env, events(300, ago_s=24 * 3600 - 60))   # 23 h 59 min ago: inside the 24 h window
    r = run(env)
    w = r["windows"]["24h"]
    assert w["effective_start"] <= w["start"]
    start = datetime.fromisoformat(w["start"]).timestamp()
    eff = datetime.fromisoformat(w["effective_start"]).timestamp()
    assert 0 <= start - eff < 15 * 60


# --- regression tests for the confirmed defects ----------------------------------------------------

def test_d1_production_series_listed_first(env):
    fill(env, [*events(200, source="loadtest"), *events(300, source="loadtest"), *events(250)])
    r = run(env)
    assert list(r["windows"]["1h"]["models"])[0] == "e7_v1"
    a = mm.load_state(env["store"] / "ml_monitoring_state.json")
    a.clock = Clock()
    rep = mm.monitoring_report(a, 60, env["baselines"], now=NOW, min_events=1)
    first = next(iter(rep["series"].values()))
    assert first["traffic_source"] == "production"


def test_d2_non_finite_inputs_get_a_clean_422(client):
    from tests.test_api import payload
    def body(field, value):
        p = payload()
        p[field] = "__VALUE__"
        return json.dumps(p).replace('"__VALUE__"', value)    # raw JSON tokens json.dumps would refuse
    for value in ("Infinity", "-Infinity", "NaN", "1e400"):
        r = client.post("/predict", content=body("amount", value), headers={"Content-Type": "application/json"})
        assert r.status_code == 422, (value, r.text)
        assert r.json()["detail"][0]["loc"] == ["body", "amount"]
    for field in ("sender_balance", "receiver_balance"):
        r = client.post("/predict", content=body(field, "Infinity"), headers={"Content-Type": "application/json"})
        assert r.status_code == 422, (field, r.text)
    assert client.post("/predict", json=payload()).status_code == 200


def test_d3_non_finite_values_are_not_reported_as_missing():
    s = make_stats(n=1000, missing=30)
    s["non_finite"] = {"amount": 30}            # the 30 null bins were inf, not missing
    r = next(f for f in dd.evaluate_window(s, baseline(), CFG)["features"] if f["feature"] == "amount")
    assert r["missing_rate"]["production"] == 0.0
    assert r["non_finite_rate"]["n"] == 30 and r["non_finite_rate"]["production"] == 0.03
    assert r["status"] == dd.DRIFT and any("non-finite" in x for x in r["reasons"])
    s["non_finite"] = {"amount": 10}            # 10 inf + 20 truly missing
    r = next(f for f in dd.evaluate_window(s, baseline(), CFG)["features"] if f["feature"] == "amount")
    assert r["missing_rate"]["production"] == 0.02 and r["non_finite_rate"]["n"] == 10


def test_d4_dashboard_and_drift_monitor_choose_the_same_baseline(env):
    bad = baseline("e7_v1")
    bad["baseline_version"] = "bl-e7_v1-2"
    bad["bins"]["amount"]["13"] = 5.0            # shares no longer sum to 1: invalid
    (env["baselines"] / "e7_v1" / "bl-e7_v1-2.json").write_text(json.dumps(bad))
    fill(env, events(300))
    drift_choice = run(env)["windows"]["1h"]["models"]["e7_v1"]["baseline_version"]
    a = mm.load_state(env["store"] / "ml_monitoring_state.json")
    a.clock = Clock()
    ml = mm.monitoring_report(a, 60, env["baselines"], now=NOW, min_events=1)["series"]["e7_v1|dp-1"]
    assert drift_choice == ml["baseline_version"] == "bl-e7_v1-1"
    assert ml["baselines_skipped"][0]["file"] == "bl-e7_v1-2.json"


def test_d5_a_gap_without_data_breaks_sustained_drift(env):
    a = fill(env, events(500, shift=4))
    cfg = dd.load_config(env["config"])
    kw = dict(baseline_dir=env["baselines"], models_dir=env["models"])
    report = dm.compute(a, cfg, now=NOW, **kw)
    since = {}
    dm.conditions(report, cfg, since, NOW)
    assert "e7_v1|amount" in since
    empty = {**report, "windows": {k: {**w, "models": {}} for k, w in report["windows"].items()}}
    dm.conditions(empty, cfg, since, NOW + 3 * 3600)          # hours with no data
    assert since == {}
    later = {c["key"]: c for c in dm.conditions(report, cfg, since, NOW + 4 * 3600)}
    assert later["sustained_drift|e7_v1|amount"]["state"] == "unknown"     # continuity starts again


def test_d6_alerts_expire_after_a_window_without_data(env):
    fill(env, events(500, shift=4))
    run(env)
    keys = {a["key"] for a in alerts(env) if a["event"] == "fired"}
    assert {"feature_drift|e7_v1|1h|amount", "feature_drift|e7_v1|24h|amount"} <= keys
    (env["store"] / "ml_monitoring_state.json").unlink()
    fill(env, [], clock=Clock(NOW + 30 * 60))
    run(env, now=NOW + 30 * 60)                       # 30 min without data: nothing expires yet
    assert not [a for a in alerts(env) if a["event"] == "expired_no_data"]
    run(env, now=NOW + 61 * 60 + 30)                  # > 1 h without evidence: the 1 h alert expires
    exp = [a for a in alerts(env) if a["event"] == "expired_no_data"]
    assert {a["key"] for a in exp} == {"feature_drift|e7_v1|1h|amount"}
    assert exp[0]["last_evidence"] and "closed without a verdict" in exp[0]["message"]
    assert "feature_drift|e7_v1|24h|amount" in dm.AlertManager(env["store"]).active   # 24 h still waiting
    run(env, now=NOW + 24 * 3600 + 120)
    assert "feature_drift|e7_v1|24h|amount" not in dm.AlertManager(env["store"]).active


def test_d6_expiry_can_be_disabled(env):
    env["config"].write_text(json.dumps({"min_samples": 200, "alerts": {"low_volume_min_per_hour": None,
                                                                        "expire_without_data": False}}))
    fill(env, events(500, shift=4))
    run(env)
    (env["store"] / "ml_monitoring_state.json").unlink()
    fill(env, [])
    run(env, now=NOW + 3 * 3600)
    assert not [a for a in alerts(env) if a["event"] == "expired_no_data"]


@pytest.fixture
def client(model_dir):
    from tests.test_api import make_client, policy, reset_state
    with make_client(model_dir, policy()) as c:
        yield c
    reset_state()
