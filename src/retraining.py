"""Retraining policy (RETRAINING_STRATEGY.md): report-only trigger evaluation, training-data
eligibility, leakage/shortcut checks and artifact immutability.

Nothing in this module trains, promotes, writes model artifacts or changes serving. A
recommendation is input for a human decision.

    python -m src.retraining triggers --model e7_v1     # what the evidence supports now
    python -m src.retraining check-model models/e7_v1   # feature policy + schema of an artifact

Recommendations, in order of precedence:
  investigate_data_quality         monitoring is broken, stale, or inputs are invalid (not a model problem)
  eligible_for_candidate_training  labelled performance is below an approved gate and data exists to train
  retraining_blocked               labelled evidence exists, but a gate or the training data is missing
  retraining_not_recommended       labelled performance is within every approved gate
  investigate_drift                production inputs drift, but there are no labels to judge accuracy
  insufficient_data                too few production predictions to say anything
  labels_required                  no drift signal and no labels: performance is unknown
Drift alone never makes a model eligible for retraining.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.data_preprocessing import LEAKY_COLUMNS, TYPE_CODES
from src.drift_detection import DRIFT, ERROR, load_config as load_drift_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = ROOT / "config" / "retraining_policy.json"
ARTIFACT_FILES = ("model.json", "pipeline.json", "metadata.json")
RAW_FEATURES = ("type", "amount", "oldbalanceOrg", "oldbalanceDest")
DATA_QUALITY_KINDS = ("missing", "unknown_category", "non_finite")

INVESTIGATE_DATA_QUALITY = "investigate_data_quality"
ELIGIBLE = "eligible_for_candidate_training"
RETRAINING_BLOCKED = "retraining_blocked"
NOT_RECOMMENDED = "retraining_not_recommended"
INVESTIGATE_DRIFT = "investigate_drift"
INSUFFICIENT_DATA = "insufficient_data"
LABELS_REQUIRED = "labels_required"

# Training-data categories (select_training_records); only ELIGIBLE_ROW may be trained on.
ELIGIBLE_ROW = "eligible"
CATEGORIES = ("duplicate", "synthetic_traffic", "untagged_traffic", "unlabeled", "invalid_label",
              "label_pending", "invalid_features", ELIGIBLE_ROW)


class PolicyViolation(Exception):
    """An action the retraining policy forbids (e.g. overwriting an artifact, a forbidden target)."""


# --- policy -----------------------------------------------------------------------------------------

def load_policy(path=DEFAULT_POLICY_PATH):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("mode") != "report_only":
        raise PolicyViolation(f"mode must be 'report_only', got {policy.get('mode')!r}")
    for flag in ("auto_train", "auto_promote"):
        if policy.get(flag) is not False:
            raise PolicyViolation(f"{flag} must be false: training and promotion need human approval")
    if not policy["artifacts"]["protected_model_dirs"]:
        raise PolicyViolation("protected_model_dirs must name the champion")
    check_target(policy["labels"]["target_column"], policy)
    lab = policy["labels"]
    if int(lab["min_labels"]) < 1 or int(lab["min_fraud_labels"]) < 1 or float(lab["maturity_days"]) < 0:
        raise PolicyViolation("min_labels, min_fraud_labels >= 1 and maturity_days >= 0")
    return policy


def check_target(target, policy):
    """The label must be a confirmed outcome, never the legacy rule flag or a model output."""
    if target in policy["labels"]["forbidden_targets"]:
        raise PolicyViolation(f"{target!r} cannot be the training target (forbidden_targets)")
    return target


# --- triggers (report-only) -------------------------------------------------------------------------

def wilson(k, n, confidence=0.95):
    """Wilson score interval for k successes in n trials; None when n == 0."""
    if n <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _ts(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _drift_evidence(report, policy, drift_cfg, model_version, now):
    """Production series of this model in the drift report -> (evidence, data-quality reasons, drift reasons)."""
    trig = policy["triggers"]
    sources = set(trig["monitored_traffic_sources"])
    quality, drift, windows = [], [], {}
    if report is None:
        return {"available": False}, ["no drift report (is the drift monitor running?)"], []
    if report.get("status") not in ("ok",):
        quality.append(f"drift report status {report.get('status')!r}")
    try:
        age = now - _ts(report["computed_at"])
    except (KeyError, ValueError, TypeError):
        age = None
    if age is None or age > trig["max_report_age_s"]:
        quality.append(f"drift report is stale (age {None if age is None else round(age)} s > "
                       f"{trig['max_report_age_s']} s)")
    if (report.get("freshness") or {}).get("stale"):
        quality.append("monitoring state is stale (consumer behind or not saving)")
    for label, win in (report.get("windows") or {}).items():
        for sid, m in (win.get("models") or {}).items():
            if m.get("model_version") != model_version or m.get("traffic_source") not in sources:
                continue
            s = m.get("summary") or {}
            dist = [r["feature"] for r in m.get("features", []) if "distribution" in r.get("drift_kinds", [])]
            dq = sorted({f"{r['feature']}: {k}" for r in m.get("features", [])
                         for k in r.get("drift_kinds", []) if k in DATA_QUALITY_KINDS})
            pred = (m.get("prediction") or {}).get("status")
            windows[label] = {"status": s.get("status"), "production_n": s.get("production_n", 0),
                              "distribution_drift": dist, "data_quality_drift": dq, "prediction_drift": pred}
            if s.get("status") == ERROR:
                quality.append(f"{label}: monitoring_error ({s.get('reason')})")
            if m.get("baseline_problems"):
                quality.append(f"{label}: baseline problems {m['baseline_problems']}")
            quality += [f"{label}: invalid or missing inputs ({x})" for x in dq]
            if label == trig["long_window"] and (dist or pred == DRIFT):
                drift.append(f"{label}: distribution drift in {dist or []}"
                             + ("; prediction drift" if pred == DRIFT else ""))
    alerts = report.get("active_alerts") or []
    sustained = [a["key"] for a in alerts if a.get("rule") == "sustained_drift"
                 and a["key"].split("|")[1:2] == [model_version]]
    quality += [f"active alert {a['key']}" for a in alerts if a.get("rule") in ("monitoring_error", "monitoring_stale")]
    drift += [f"active alert {k}" for k in sustained]
    max_n = max((w["production_n"] for w in windows.values()), default=0)
    evidence = {"available": True, "computed_at": report.get("computed_at"), "age_s": age, "windows": windows,
                "max_production_n": max_n, "min_samples": drift_cfg["min_samples"],
                "active_sustained_drift": sustained}
    return evidence, quality, drift


def _performance_evidence(perf, policy):
    """(evidence, labels_ok, reason) from src.ml_performance.evaluate_performance output."""
    lab = policy["labels"]
    perf = perf or {"status": "unavailable", "reason": "no performance report"}
    ev = {"status": perf.get("status"), "reason": perf.get("reason"),
          "positives": perf.get("positives"), "negatives": perf.get("negatives"), "n": perf.get("n")}
    if perf.get("status") != "ok":
        return ev, False, f"labels: {perf.get('status')} ({perf.get('reason')})"
    if perf["n"] < lab["min_labels"] or perf["positives"] < lab["min_fraud_labels"]:
        return ev, False, (f"labels: {perf['n']} usable ({perf['positives']} fraud); need >= {lab['min_labels']} "
                           f"with >= {lab['min_fraud_labels']} fraud")
    m = perf["at_approve_threshold"]
    conf = policy["performance_gates"]["confidence"]
    ev.update({"recall": m["recall"], "recall_ci": wilson(m["TP"], m["TP"] + m["FN"], conf),
               "fpr": m["fpr"], "fpr_ci": wilson(m["FP"], m["FP"] + m["TN"], conf),
               "cost": m["cost"], "cost_per_1k_transactions": 1000 * m["cost"] / perf["n"],
               "pr_auc": perf.get("pr_auc"), "confusion": {k: m[k] for k in ("TP", "FP", "FN", "TN")},
               "fraud_prevalence": perf["positives"] / perf["n"], "confidence": conf})
    return ev, True, None


def evaluate_triggers(drift_report, performance, policy, drift_cfg, model_version, now=None):
    """Report-only recommendation for one model version. Never starts training."""
    now = time.time() if now is None else now
    drift_ev, quality, drift = _drift_evidence(drift_report, policy, drift_cfg, model_version, now)
    perf_ev, labels_ok, label_reason = _performance_evidence(performance, policy)
    reasons, blockers = [], []
    if quality:
        rec, reasons = INVESTIGATE_DATA_QUALITY, quality + ["fix the data or monitoring path first; "
                                                            "retraining on broken inputs would learn the defect"]
    elif labels_ok:
        gates = policy["performance_gates"]
        degraded = []
        if gates["recall_floor"] is not None and perf_ev["recall_ci"][1] < gates["recall_floor"]:
            degraded.append(f"recall CI {perf_ev['recall_ci']} entirely below floor {gates['recall_floor']}")
        if gates["fpr_ceiling"] is not None and perf_ev["fpr_ci"][0] > gates["fpr_ceiling"]:
            degraded.append(f"FPR CI {perf_ev['fpr_ci']} entirely above ceiling {gates['fpr_ceiling']}")
        if gates["max_cost_per_1k_transactions"] is not None and \
                perf_ev["cost_per_1k_transactions"] > gates["max_cost_per_1k_transactions"]:
            degraded.append(f"cost per 1k {perf_ev['cost_per_1k_transactions']:.1f} > "
                            f"{gates['max_cost_per_1k_transactions']} (point estimate)")
        unset = [k for k in ("recall_floor", "fpr_ceiling", "max_cost_per_1k_transactions") if gates[k] is None]
        if policy["training_data"]["point_in_time_feature_source"] is None:
            source_blocker = ("no point-in-time feature source: prediction events carry no amounts or "
                              "balances, so production outcomes can be evaluated but not trained on")
        else:
            source_blocker = None
        if degraded:
            reasons = degraded + drift
            blockers = [source_blocker] if source_blocker else []
            rec = RETRAINING_BLOCKED if blockers else ELIGIBLE
        elif unset:
            rec = RETRAINING_BLOCKED
            reasons = ["no approved gate is violated"] + drift
            blockers = [f"performance gates need business approval: {', '.join(unset)} are null"]
        else:
            rec, reasons = NOT_RECOMMENDED, ["labelled performance is within every approved gate"] + drift
    else:
        blockers = [label_reason]
        if drift:
            rec, reasons = INVESTIGATE_DRIFT, drift + ["drift does not show that accuracy dropped; "
                                                       "investigate the cause and obtain labels"]
        elif drift_ev.get("max_production_n", 0) < drift_cfg["min_samples"]:
            rec = INSUFFICIENT_DATA
            reasons = [f"at most {drift_ev.get('max_production_n', 0)} production predictions per window "
                       f"(< min_samples {drift_cfg['min_samples']})"]
        else:
            rec, reasons = LABELS_REQUIRED, ["no drift signal; model performance is unknown without labels"]
    return {"policy_version": policy["policy_version"], "model_version": model_version,
            "evaluated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "mode": "report_only", "recommendation": rec, "reasons": reasons, "blockers": blockers,
            "evidence": {"drift": drift_ev, "labels": perf_ev}, "action_taken": "none",
            "note": "A recommendation for a human decision; nothing was trained, promoted or changed."}


# --- training-data eligibility ----------------------------------------------------------------------

def _epoch(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    return pd.to_datetime(series, utc=True, errors="coerce").map(lambda t: t.timestamp() if pd.notna(t) else np.nan)


def select_training_records(df, policy, now=None, label_column="label"):
    """Assign every candidate row exactly one category (CATEGORIES). Expected columns:
    event_id, transaction_id, occurred_at, traffic_source, type, amount, oldbalanceOrg, oldbalanceDest,
    <label_column>, label_at, label_source. One label per row: join the latest label per transaction
    first (src.ml_performance.read_labels). Returns (df with an `eligibility` column, counts)."""
    check_target(label_column, policy)
    now = time.time() if now is None else now
    lab = policy["labels"]
    d = df.copy()
    d["_t"] = _epoch(d["occurred_at"])
    d["_lt"] = _epoch(d["label_at"]) if "label_at" in d else np.nan
    d["_order"] = np.arange(len(d))
    cat = pd.Series(ELIGIBLE_ROW, index=d.index, dtype=object)

    def mark(mask, name):
        free = cat.eq(ELIGIBLE_ROW) & mask.fillna(False).astype(bool)
        cat[free] = name

    # Duplicates: a re-delivered event_id, or a second prediction for the same transaction
    # (the first prediction is the one the outcome belongs to, as in src.ml_performance).
    ordered = d.sort_values(["_t", "_order"], kind="mergesort")
    dup = ordered["event_id"].duplicated(keep="first") | (
        ordered["transaction_id"].notna() & ordered["transaction_id"].duplicated(keep="first"))
    mark(dup.reindex(d.index), "duplicate")
    src = d["traffic_source"]
    allowed = set(policy["triggers"]["monitored_traffic_sources"])
    mark(src.notna() & ~src.isin(allowed), "synthetic_traffic")
    if policy["training_data"].get("require_explicit_production_tag", True):
        mark(src.isna(), "untagged_traffic")
    y = d[label_column]
    mark(y.isna(), "unlabeled")
    valid_y = y.isin([0, 1]) & ~y.map(lambda v: isinstance(v, bool))
    bad_label = ~valid_y | ~d["label_source"].isin(lab["allowed_label_sources"]) | d["_lt"].isna() | (d["_lt"] < d["_t"])
    mark(bad_label, "invalid_label")
    mark(d["_t"].isna() | (now - d["_t"] < lab["maturity_days"] * 86400), "label_pending")
    bad_feat = ~d["type"].isin(list(TYPE_CODES))
    for f in RAW_FEATURES[1:]:
        v = pd.to_numeric(d[f], errors="coerce")
        bad_feat |= v.isna() | ~np.isfinite(v.fillna(0)) | (v < 0)
    mark(bad_feat, "invalid_features")

    out = df.copy()
    out["eligibility"] = cat.values
    counts = {c: int((cat == c).sum()) for c in CATEGORIES}
    return out, counts


def chronological_split_with_label_cutoff(df, train_end, val_end, time_col="occurred_at", label_time_col="label_at"):
    """train: t < train_end; validation: train_end <= t < val_end; test: t >= val_end (epoch seconds or
    timestamps). A training or validation row whose label arrived after its period ended is dropped:
    that label did not exist when the model was trained or its threshold chosen."""
    t, lt = _epoch(df[time_col]), _epoch(df[label_time_col])
    train_end, val_end = float(_epoch(pd.Series([train_end])).iloc[0]), float(_epoch(pd.Series([val_end])).iloc[0])
    if not train_end < val_end:
        raise ValueError("train_end must precede val_end")
    tr, va, te = t < train_end, (t >= train_end) & (t < val_end), t >= val_end
    late_tr, late_va = tr & (lt > train_end), va & (lt > val_end)
    train, val, test = df[tr & ~late_tr], df[va & ~late_va], df[te]
    assert _epoch(train[time_col]).max(skipna=True) < train_end if len(train) else True
    assert len(set(train.index) & set(val.index)) == len(set(val.index) & set(test.index)) == 0
    info = {"train": len(train), "validation": len(val), "test": len(test),
            "dropped_label_after_train_end": int(late_tr.sum()), "dropped_label_after_val_end": int(late_va.sum())}
    return train, val, test, info


# --- feature, schema and artifact checks ------------------------------------------------------------

def check_feature_set(feature_names, policy, justifications=None):
    """Leakage and known-shortcut gate: rejected > needs_justification > ok."""
    feats = policy["features"]
    forbidden = set(feats["forbidden"]) | set(LEAKY_COLUMNS)
    rejected = [f for f in feature_names if f in forbidden]
    pending = {f: note for f, note in feats["justification_required"].items()
               if f in feature_names and f not in (justifications or {})}
    status = "rejected" if rejected else ("needs_justification" if pending else "ok")
    return {"status": status, "rejected": rejected, "needs_justification": pending,
            "justified": {f: j for f, j in (justifications or {}).items() if f in feature_names}}


def check_schema(actual, expected):
    actual, expected = list(actual), list(expected)
    return {"compatible": actual == expected,
            "missing": [f for f in expected if f not in actual],
            "unexpected": [f for f in actual if f not in expected],
            "order_differs": sorted(actual) == sorted(expected) and actual != expected}


def guard_new_artifact_dir(path, policy=None, root=ROOT):
    """Refuse to write a model into a protected or non-empty directory: artifacts are immutable."""
    policy = policy or load_policy()
    p = Path(path).resolve()
    protected = {(Path(root) / d).resolve() for d in policy["artifacts"]["protected_model_dirs"]}
    if p in protected:
        raise PolicyViolation(f"{path} is a protected model directory (the champion); use a new version")
    if p.exists() and (not p.is_dir() or any(p.iterdir())):
        raise PolicyViolation(f"{path} already exists and is not empty; artifact directories are immutable, "
                              "use a new version")
    return p


# --- reproducibility --------------------------------------------------------------------------------

def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def records_fingerprint(df, columns):
    """Order-independent fingerprint of selected columns (for datasets that are not a single file)."""
    canon = df[list(columns)].astype(str).apply("\x1f".join, axis=1).sort_values()
    return hashlib.sha256("\n".join(canon).encode()).hexdigest()


def code_revision(root=ROOT):
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, timeout=10, check=True).stdout.strip())
        return {"commit": rev, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return None


def dataset_manifest(splits, *, label_column, time_column, fingerprint, label_definition, feature_names,
                     seed=None, pipeline_id=None, policy=None):
    """What a candidate was trained and evaluated on, for metadata.json and the evaluation report."""
    def part(d):
        y = d[label_column]
        return {"rows": int(len(d)), "fraud": int(y.sum()), "prevalence": float(y.mean()) if len(d) else None,
                "time_range": [d[time_column].min(), d[time_column].max()] if len(d) else None,
                "label_sources": d["label_source"].value_counts().to_dict() if "label_source" in d else None}
    import sklearn
    import xgboost
    return {"fingerprint": fingerprint, "label_column": label_column, "label_definition": label_definition,
            "splits": {k: part(v) for k, v in splits.items()}, "feature_names": list(feature_names),
            "pipeline_id": pipeline_id, "random_seed": seed, "code_revision": code_revision(),
            "library_versions": {"python": platform.python_version(), "xgboost": xgboost.__version__,
                                 "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
            "retraining_policy_version": (policy or {}).get("policy_version"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


# --- CLI --------------------------------------------------------------------------------------------

def _store_dir():
    return Path(os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events")))


def triggers_report(model_version, store_dir=None, policy=None, now=None):
    from src import ml_performance
    from monitoring.drift_monitor import REPORT_FILE

    policy = policy or load_policy()
    store_dir = Path(store_dir) if store_dir else _store_dir()
    drift_cfg = load_drift_config(ROOT / policy["triggers"]["drift_config"])
    path = store_dir / REPORT_FILE
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    labels_dir = store_dir / "labels"
    labels, counts = ml_performance.read_labels(labels_dir) if labels_dir.exists() else ({}, {})
    steps = ml_performance.tuning_steps_for(ROOT / policy["artifacts"]["models_dir"] / model_version)
    perf = ml_performance.evaluate_performance(ml_performance.iter_events(store_dir) if labels else (), labels,
                                               counts, model_version=model_version, tuning_steps=steps,
                                               maturity_days=policy["labels"]["maturity_days"],
                                               min_labels=policy["labels"]["min_labels"])
    return evaluate_triggers(report, perf, policy, drift_cfg, model_version, now=now)


def check_model(model_dir, policy=None, justifications=None):
    policy = policy or load_policy()
    d = Path(model_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    pipe = json.loads((d / "pipeline.json").read_text(encoding="utf-8"))
    return {"model_dir": str(model_dir), "feature_set": meta.get("feature_set"),
            "sha256": {f: file_sha256(d / f) for f in ARTIFACT_FILES if (d / f).exists()},
            "feature_policy": check_feature_set(meta["feature_names"], policy, justifications),
            "schema": check_schema(pipe["feature_names"], meta["feature_names"]),
            "threshold": meta.get("threshold"), "split_boundaries": meta.get("split_boundaries")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triggers", help="report-only retraining recommendation")
    t.add_argument("--model", default="e7_v1")
    t.add_argument("--store", default=None, help="event store directory (default $EVENT_STORE_DIR)")
    t.add_argument("--out", default=None, help="also write the JSON report to this file")
    c = sub.add_parser("check-model", help="feature policy and schema check of a model artifact")
    c.add_argument("model_dir")
    args = ap.parse_args(argv)
    policy = load_policy(args.policy)
    if args.cmd == "triggers":
        result = triggers_report(args.model, args.store, policy)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    else:
        result = check_model(args.model_dir, policy)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
