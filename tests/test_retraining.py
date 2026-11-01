"""Retraining policy (RETRAINING_STRATEGY.md): report-only triggers, data eligibility, artifact safety.

Deterministic, no broker, nothing written outside tmp_path. Numbers in the test names refer to the
Phase 3 brief's test list.
"""
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import retraining as rt
from src.drift_detection import DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_800_000_000.0          # fixed clock (2027-01-15)
DAY = 86400.0
DRIFT_CFG = copy.deepcopy(DEFAULT_CONFIG)


@pytest.fixture
def policy():
    return rt.load_policy()


def candidates(n=6, **overrides):
    """Mature, production-tagged, confirmed-label rows; override columns per test."""
    base = {
        "event_id": [f"e{i}" for i in range(n)],
        "transaction_id": [f"t{i}" for i in range(n)],
        "occurred_at": [NOW - 60 * DAY + i * 3600 for i in range(n)],
        "traffic_source": ["production"] * n,
        "type": ["TRANSFER"] * n,
        "amount": [100.0 + i for i in range(n)],
        "oldbalanceOrg": [1000.0] * n,
        "oldbalanceDest": [0.0] * n,
        "label": [i % 2 for i in range(n)],
        "label_at": [NOW - 20 * DAY] * n,
        "label_source": ["chargeback"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def categories(df, policy, **kw):
    out, counts = rt.select_training_records(df, policy, now=NOW, **kw)
    return list(out["eligibility"]), counts


# --- policy -----------------------------------------------------------------------------------------

def test_repository_policy_is_report_only_and_protects_the_champion(policy):
    assert policy["mode"] == "report_only"
    assert policy["auto_train"] is False and policy["auto_promote"] is False
    assert "models/e7_v1" in policy["artifacts"]["protected_model_dirs"]
    gates = policy["performance_gates"]
    assert all(gates[k] is None for k in ("recall_floor", "fpr_ceiling", "max_cost_per_1k_transactions"))


@pytest.mark.parametrize("change", [{"auto_train": True}, {"auto_promote": True}, {"mode": "auto"},
                                    {"auto_train": "false"}])
def test_11_policy_cannot_enable_automatic_training_or_promotion(tmp_path, change):
    p = json.loads((ROOT / "config" / "retraining_policy.json").read_text())
    p.update(change)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(p))
    with pytest.raises(rt.PolicyViolation):
        rt.load_policy(path)


# --- eligibility --------------------------------------------------------------------------------------

def test_1_synthetic_and_untagged_traffic_are_never_eligible(policy):
    df = candidates(5, traffic_source=["production", "loadtest", "test", "demo", None])
    cats, counts = categories(df, policy)
    assert cats == ["eligible", "synthetic_traffic", "synthetic_traffic", "synthetic_traffic", "untagged_traffic"]
    assert counts["eligible"] == 1


def test_2_unlabeled_rows_are_not_treated_as_legitimate(policy):
    df = candidates(4, label=[1, None, np.nan, 0])
    cats, counts = categories(df, policy)
    assert cats == ["eligible", "unlabeled", "unlabeled", "eligible"]
    out, _ = rt.select_training_records(df, policy, now=NOW)
    assert out.loc[out["eligibility"] == "unlabeled", "label"].isna().all()   # not filled with 0


@pytest.mark.parametrize("column", ["fraud_probability", "decision", "would_decline", "reason_code"])
def test_2_model_outputs_cannot_become_labels(policy, column):
    df = candidates(2)
    df[column] = [0.9, 0.1]
    with pytest.raises(rt.PolicyViolation):
        rt.select_training_records(df, policy, now=NOW, label_column=column)


def test_3_isflaggedfraud_cannot_be_the_target(policy, tmp_path):
    df = candidates(2)
    df["isFlaggedFraud"] = [0, 1]
    with pytest.raises(rt.PolicyViolation):
        rt.select_training_records(df, policy, now=NOW, label_column="isFlaggedFraud")
    with pytest.raises(rt.PolicyViolation):
        rt.check_target("isFlaggedFraud", policy)
    p = json.loads((ROOT / "config" / "retraining_policy.json").read_text())
    p["labels"]["target_column"] = "isFlaggedFraud"
    (tmp_path / "p.json").write_text(json.dumps(p))
    with pytest.raises(rt.PolicyViolation):
        rt.load_policy(tmp_path / "p.json")


def test_4_invalid_early_and_pending_labels(policy):
    df = candidates(7, label=[1, 2, "1", True, 0, 1, 0],
                    label_source=["chargeback", "chargeback", "chargeback", "chargeback", "rumour",
                                  "chargeback", "manual_review"])
    df.loc[5, "label_at"] = df.loc[5, "occurred_at"] - 10        # label before the prediction
    df.loc[6, "occurred_at"] = NOW - 5 * DAY                     # younger than the 30-day maturity
    df.loc[6, "label_at"] = NOW - 1 * DAY
    cats, _ = categories(df, policy)
    assert cats == ["eligible", "invalid_label", "invalid_label", "invalid_label", "invalid_label",
                    "invalid_label", "label_pending"]


def test_invalid_features_are_excluded(policy):
    df = candidates(5, type=["TRANSFER", "CRYPTO", "TRANSFER", "TRANSFER", "TRANSFER"],
                    amount=[1.0, 1.0, np.inf, -5.0, None])
    cats, _ = categories(df, policy)
    assert cats == ["eligible"] + ["invalid_features"] * 4


def test_7_duplicates_are_resolved_deterministically(policy):
    # e0 re-delivered; t1 predicted twice (the earlier prediction is kept, as in src.ml_performance).
    df = candidates(4, event_id=["e0", "e0", "e2", "e3"], transaction_id=["t0", "t0", "t1", "t1"])
    df.loc[2, "occurred_at"], df.loc[3, "occurred_at"] = NOW - 40 * DAY, NOW - 50 * DAY
    cats, _ = categories(df, policy)
    assert cats == ["eligible", "duplicate", "duplicate", "eligible"]
    shuffled = df.iloc[[3, 1, 0, 2]]
    out, _ = rt.select_training_records(shuffled, policy, now=NOW)
    assert dict(zip(out.index, out["eligibility"])) == {0: "eligible", 1: "duplicate", 2: "duplicate", 3: "eligible"}


def test_every_row_gets_exactly_one_category(policy):
    rng = np.random.default_rng(0)
    n = 300
    df = candidates(n, traffic_source=rng.choice(["production", "loadtest", None], n),
                    label=rng.choice([0, 1, None], n),
                    event_id=[f"e{i}" for i in rng.integers(0, 250, n)])
    out, counts = rt.select_training_records(df, policy, now=NOW)
    assert sum(counts.values()) == n and set(out["eligibility"]) <= set(rt.CATEGORIES)


# --- splits -------------------------------------------------------------------------------------------

def test_6_chronological_split_has_no_future_rows_and_no_late_labels():
    t = np.arange(10) * DAY
    df = pd.DataFrame({"occurred_at": t, "label_at": t + 2 * DAY, "label": [0, 1] * 5})
    df.loc[2, "label_at"] = 9 * DAY            # train row whose label arrived after training
    train, val, test, info = rt.chronological_split_with_label_cutoff(df, train_end=5 * DAY, val_end=8 * DAY)
    assert train["occurred_at"].max() < val["occurred_at"].min() <= val["occurred_at"].max() < test["occurred_at"].min()
    assert (train["label_at"] <= 5 * DAY).all() and 2 not in train.index
    assert info["dropped_label_after_train_end"] >= 1
    assert set(train.index).isdisjoint(val.index) and set(val.index).isdisjoint(test.index)
    with pytest.raises(ValueError):
        rt.chronological_split_with_label_cutoff(df, train_end=8 * DAY, val_end=5 * DAY)


# --- features and schema ----------------------------------------------------------------------------

def test_leakage_and_shortcut_gate(policy):
    assert rt.check_feature_set(["amount", "newbalanceOrig"], policy)["status"] == "rejected"
    assert rt.check_feature_set(["amount", "drains_orig"], policy)["status"] == "rejected"
    assert rt.check_feature_set(["amount", "hour"], policy)["status"] == "rejected"
    r = rt.check_feature_set(["amount", "exceeds_orig_bal"], policy)
    assert r["status"] == "needs_justification" and "exceeds_orig_bal" in r["needs_justification"]
    ok = rt.check_feature_set(["amount", "exceeds_orig_bal"], policy, {"exceeds_orig_bal": "validated on replay X"})
    assert ok["status"] == "ok"


def test_8_schema_mismatch_is_reported():
    exp = ["amount", "oldbalanceOrg", "type_code"]
    assert rt.check_schema(exp, exp)["compatible"]
    r = rt.check_schema(["amount", "type_code", "hour"], exp)
    assert not r["compatible"] and r["missing"] == ["oldbalanceOrg"] and r["unexpected"] == ["hour"]
    assert rt.check_schema(["type_code", "amount", "oldbalanceOrg"], exp)["order_differs"]


@pytest.mark.skipif(not (ROOT / "models" / "e7_v1" / "metadata.json").exists(), reason="artifact missing")
def test_check_model_on_the_champion_flags_exceeds_orig_bal(policy):
    r = rt.check_model(ROOT / "models" / "e7_v1", policy)
    assert r["feature_policy"]["status"] == "needs_justification"
    assert r["schema"]["compatible"]


# --- artifact immutability ----------------------------------------------------------------------------

def test_10_the_champion_directory_is_protected(policy):
    with pytest.raises(rt.PolicyViolation):
        rt.guard_new_artifact_dir(ROOT / "models" / "e7_v1", policy)
    with pytest.raises(rt.PolicyViolation):
        rt.guard_new_artifact_dir("models/e7_v1/../e7_v1", policy, root=ROOT)


def test_10_the_champion_path_stays_protected_even_if_missing(policy, tmp_path):
    """Protection is by name, not only by content: a deleted champion cannot be silently recreated."""
    with pytest.raises(rt.PolicyViolation, match="protected"):
        rt.guard_new_artifact_dir(tmp_path / "models" / "e7_v1", policy, root=tmp_path)


def test_10_existing_artifacts_are_never_overwritten(policy, tmp_path):
    from src.train import save_artifacts
    full = tmp_path / "e7_v9"
    full.mkdir()
    (full / "model.json").write_text("{}")
    with pytest.raises(rt.PolicyViolation):
        save_artifacts({"model": None, "pipeline": None}, full, {}, policy)
    assert (full / "model.json").read_text() == "{}"
    assert rt.guard_new_artifact_dir(tmp_path / "new", policy) == (tmp_path / "new").resolve()
    empty = tmp_path / "empty"
    empty.mkdir()
    rt.guard_new_artifact_dir(empty, policy)       # an empty directory has nothing to overwrite


def test_10_train_cli_refuses_an_existing_version_before_loading_data(tmp_path, monkeypatch):
    from src import train
    (tmp_path / "e7_v1").mkdir()
    (tmp_path / "e7_v1" / "model.json").write_text("{}")

    def no_data(*a, **k):
        raise AssertionError("training started")
    monkeypatch.setattr(train, "load_data", no_data)
    monkeypatch.setattr(sys, "argv", ["train", "--models-dir", str(tmp_path), "--version", "v1",
                                      "--feature-sets", "E7"])
    with pytest.raises(SystemExit) as e:
        train.main()
    assert e.value.code == 2
    (tmp_path / "comparison_v2.md").write_text("old")
    monkeypatch.setattr(sys, "argv", ["train", "--models-dir", str(tmp_path), "--version", "v2",
                                      "--feature-sets", "E7"])
    with pytest.raises(SystemExit):
        train.main()
    assert (tmp_path / "comparison_v2.md").read_text() == "old"


def test_11_retraining_module_never_trains_or_promotes():
    tree = ast.parse((ROOT / "src" / "retraining.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
               {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {"src.train", "src.predict", "api.main", "src.decision_engine"} & imported
    called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {"fit", "save_model", "run_experiment", "save_artifacts"} & called


# --- triggers ---------------------------------------------------------------------------------------

def iso(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def drift_report(*, n=5000, dist=(), quality=(), pred="no_significant_drift", status="ok", age=60,
                 source="production", alerts=(), summary_status=None, model="e7_v1"):
    feats = [{"feature": f, "drift_kinds": ["distribution"], "status": "drift_detected"} for f in dist]
    feats += [{"feature": f, "drift_kinds": [k], "status": "drift_detected"} for f, k in quality]
    st = summary_status or ("drift_detected" if feats else ("insufficient_data" if n < 500 else "no_significant_drift"))
    series = {"model_version": model, "traffic_source": source, "baseline_problems": [],
              "features": feats, "prediction": {"status": pred}, "summary": {"status": st, "production_n": n}}
    return {"status": status, "computed_at": iso(NOW - age), "freshness": {"stale": False},
            "windows": {"1h": {"models": {model: series}}, "24h": {"models": {model: series}}},
            "active_alerts": list(alerts)}


def perf(tp=0, fn=0, fp=0, tn=0, status="ok"):
    if status != "ok":
        return {"status": status, "reason": "no actual labels ingested"}
    n = tp + fn + fp + tn
    return {"status": "ok", "n": n, "positives": tp + fn, "negatives": fp + tn, "pr_auc": 0.5,
            "at_approve_threshold": {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "recall": tp / (tp + fn),
                                     "fpr": fp / (fp + tn), "cost": 100 * fn + fp}}


def rec(report, performance, policy):
    return rt.evaluate_triggers(report, performance, policy, DRIFT_CFG, "e7_v1", now=NOW)


def gated(policy, **gates):
    p = copy.deepcopy(policy)
    p["performance_gates"].update(gates)
    return p


def test_9_drift_without_labels_only_asks_for_investigation(policy):
    r = rec(drift_report(dist=["oldbalanceDest", "amount"], pred="drift_detected",
                         alerts=[{"key": "sustained_drift|e7_v1|amount", "rule": "sustained_drift"}]),
            perf(status="unavailable"), policy)
    assert r["recommendation"] == "investigate_drift"
    assert r["action_taken"] == "none" and any("labels" in b for b in r["blockers"])


def test_drift_on_loadtest_traffic_is_ignored(policy):
    r = rec(drift_report(dist=["amount"], source="loadtest"), perf(status="unavailable"), policy)
    assert r["recommendation"] == "insufficient_data"       # no production series at all


def test_insufficient_production_volume(policy):
    r = rec(drift_report(n=20), perf(status="unavailable"), policy)
    assert r["recommendation"] == "insufficient_data"


def test_no_drift_no_labels_means_labels_required(policy):
    r = rec(drift_report(), perf(status="insufficient_labels"), policy)
    assert r["recommendation"] == "labels_required"


@pytest.mark.parametrize("report", [
    None,
    drift_report(age=10_000),
    drift_report(status="monitoring_error"),
    drift_report(summary_status="monitoring_error"),
    drift_report(quality=[("amount", "non_finite")]),
    drift_report(quality=[("transaction_type", "unknown_category")]),
    drift_report(alerts=[{"key": "monitoring_stale", "rule": "monitoring_stale"}]),
])
def test_data_quality_problems_are_not_retraining_signals(policy, report):
    good = perf(tp=95, fn=5, fp=50, tn=9850)
    r = rec(report, good, gated(policy, recall_floor=0.99))
    assert r["recommendation"] == "investigate_data_quality"


def test_5_too_few_fraud_labels_give_no_performance_claim(policy):
    r = rec(drift_report(), perf(tp=10, fn=5, fp=10, tn=500), gated(policy, recall_floor=0.9))
    assert r["recommendation"] == "labels_required"
    assert "recall" not in r["evidence"]["labels"]          # no metric claimed on 15 frauds


def test_unset_business_gates_block_any_conclusion(policy):
    r = rec(drift_report(), perf(tp=90, fn=10, fp=100, tn=9800), policy)
    assert r["recommendation"] == "retraining_blocked"
    assert any("business approval" in b for b in r["blockers"])


def test_degraded_performance_is_blocked_without_a_feature_source(policy):
    p = gated(policy, recall_floor=0.95, fpr_ceiling=0.05, max_cost_per_1k_transactions=1e9)
    r = rec(drift_report(), perf(tp=60, fn=40, fp=100, tn=9800), p)    # recall 0.60, CI far below 0.95
    assert r["recommendation"] == "retraining_blocked"
    assert any("point-in-time feature source" in b for b in r["blockers"])
    p["training_data"]["point_in_time_feature_source"] = "feature-log-v1"
    r = rec(drift_report(), perf(tp=60, fn=40, fp=100, tn=9800), p)
    assert r["recommendation"] == "eligible_for_candidate_training" and r["action_taken"] == "none"


def test_uncertain_degradation_is_not_degradation(policy):
    p = gated(policy, recall_floor=0.95, fpr_ceiling=0.05, max_cost_per_1k_transactions=1e9)
    r = rec(drift_report(), perf(tp=27, fn=3, fp=10, tn=1000), p)   # recall 0.90, CI reaches above 0.95
    assert r["evidence"]["labels"]["recall_ci"][1] > 0.95
    assert r["recommendation"] == "retraining_not_recommended"


def test_cli_is_report_only_on_an_empty_store(tmp_path, capsys):
    assert rt.main(["triggers", "--model", "e7_v1", "--store", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["recommendation"] == "investigate_data_quality" and out["action_taken"] == "none"
    assert list(tmp_path.iterdir()) == []                   # nothing written


def test_drift_results_expose_drift_kinds():
    """The evaluator reads drift_kinds; make sure drift_detection produces it."""
    from src.drift_detection import _finalize
    res = {"feature": "amount", "threshold": {"moderate": 0.1, "substantial": 0.2}, "psi": 0.5,
           "test": {"applicable": False}, "missing_test": None, "reasons": [], "production_total": 1000,
           "non_finite_rate": {"production": 0.01}, "unknown_rate": None}
    out = _finalize(res, DRIFT_CFG, {})
    assert out["status"] == "drift_detected" and out["drift_kinds"] == ["distribution", "non_finite"]
