"""Model performance from actual fraud labels, when they exist (see ML_MONITORING.md).

No label source exists yet. This module defines the contract and the evaluation, and reports
"unavailable" until labels are ingested. It never infers labels from predictions and never
treats an unlabeled prediction as legitimate.

Label contract: append-only JSONL files in <EVENT_STORE_DIR>/labels/, one record per line:
    {"transaction_id": "tx-1", "is_fraud": 1, "label_at": "2026-11-20T10:00:00+00:00",
     "source": "chargeback", "dataset_step": 400}        # dataset_step: optional, offline replays only
    python -m src.ml_performance add-labels <file.jsonl>  # validates, then appends
    python -m src.ml_performance report [--model e7_v1]

Join: label.transaction_id == prediction event.transaction_id (stored events). Prediction time
(occurred_at) and label arrival time (label_at) are kept separate: a label that arrives before
its prediction is invalid; label delay is reported; coverage is reported against all predictions
and against "mature" predictions (older than the label maturity window, default 30 days).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.evaluate import COST_FN, COST_FP, business_cost

MIN_LABELS = 100
LABEL_MATURITY_DAYS = 30.0
SOURCES_MAX_LEN = 40


def _ts(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp without timezone")
    return dt.timestamp()


def validate_label(rec):
    """Returns a normalised label dict or raises ValueError."""
    if not isinstance(rec, dict):
        raise ValueError("not an object")
    tx = rec.get("transaction_id")
    if not isinstance(tx, str) or not tx:
        raise ValueError("transaction_id missing")
    y = rec.get("is_fraud")
    if isinstance(y, bool):
        y = int(y)
    if y not in (0, 1):
        raise ValueError("is_fraud must be 0 or 1")
    label_at = _ts(rec.get("label_at"))
    source = str(rec.get("source") or "unknown")[:SOURCES_MAX_LEN]
    step = rec.get("dataset_step")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
        raise ValueError("dataset_step must be an integer")
    return {"transaction_id": tx, "is_fraud": y, "label_at": label_at, "source": source, "dataset_step": step}


def add_labels(src, labels_dir):
    """Validate a JSONL file and append the valid records to labels/labels-<date>.jsonl."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    good, bad = [], 0
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            validate_label(rec)
            good.append(rec)
        except (ValueError, TypeError):
            bad += 1
    out = labels_dir / f"labels-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        for rec in good:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return {"appended": len(good), "rejected": bad, "file": str(out)}


def read_labels(labels_dir):
    """Latest label per transaction_id (by label_at); counts conflicts and invalid lines."""
    labels, counts = {}, {"label_records": 0, "invalid_label_records": 0, "label_conflicts": 0,
                          "label_updates": 0}
    for path in sorted(Path(labels_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts["label_records"] += 1
            try:
                lab = validate_label(json.loads(line))
            except (ValueError, TypeError):
                counts["invalid_label_records"] += 1
                continue
            old = labels.get(lab["transaction_id"])
            if old is not None:
                counts["label_updates"] += 1
                if old["is_fraud"] != lab["is_fraud"]:
                    counts["label_conflicts"] += 1      # e.g. a chargeback later reversed
                if lab["label_at"] < old["label_at"]:
                    continue
            labels[lab["transaction_id"]] = lab
    return labels, counts


def iter_events(store_dir):
    for path in sorted(Path(store_dir).glob("prediction-events-*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def _metrics(y, flagged, scores):
    y, flagged = np.asarray(y, bool), np.asarray(flagged, bool)
    tp, fp = int(np.sum(y & flagged)), int(np.sum(~y & flagged))
    fn, tn = int(np.sum(y & ~flagged)), int(np.sum(~y & ~flagged))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else None, "false_negatives": fn,
            "cost": business_cost(fn, fp), "cost_formula": f"{COST_FN} * FN + {COST_FP} * FP"}


def evaluate_performance(events, labels, label_counts, *, model_version=None, now=None,
                         maturity_days=LABEL_MATURITY_DAYS, min_labels=MIN_LABELS, tuning_steps=()):
    """events: iterable of stored prediction events; labels: {transaction_id: label}.
    tuning_steps: [(lo, hi), ...] dataset steps used for training/threshold tuning (excluded)."""
    now = now if now is not None else time.time()
    maturity_s = maturity_days * 86400
    c = {"predictions": 0, "predictions_without_transaction_id": 0, "mature_predictions": 0,
         "labeled_predictions": 0, "mature_labeled_predictions": 0, "label_before_prediction": 0,
         "excluded_tuning_data": 0, "repeat_predictions_for_transaction": 0, **label_counts}
    pairs, matched, seen_events = {}, set(), set()
    for e in events:
        if model_version and e.get("model_version") != model_version:
            continue
        if e.get("event_id") in seen_events:
            continue
        seen_events.add(e.get("event_id"))
        c["predictions"] += 1
        try:
            occurred = _ts(e["occurred_at"])
        except (KeyError, ValueError, TypeError):
            continue
        mature = now - occurred >= maturity_s
        c["mature_predictions"] += mature
        tx = e.get("transaction_id")
        if not tx:
            c["predictions_without_transaction_id"] += 1
            continue
        lab = labels.get(tx)
        if lab is None:
            continue                                   # unlabeled: excluded, never "legitimate"
        if tx in pairs:
            c["repeat_predictions_for_transaction"] += 1   # the first prediction is the one evaluated
            continue
        matched.add(tx)
        if lab["label_at"] < occurred:
            c["label_before_prediction"] += 1
            continue
        if lab["dataset_step"] is not None and any(lo <= lab["dataset_step"] <= hi for lo, hi in tuning_steps):
            c["excluded_tuning_data"] += 1
            continue
        c["labeled_predictions"] += 1
        c["mature_labeled_predictions"] += mature
        pairs[tx] = (lab["is_fraud"], float(e["fraud_probability"]), e.get("decision"),
                     bool(e.get("would_decline")), lab["label_at"] - occurred)
    c["labels_without_prediction"] = len(set(labels) - matched)
    c["label_coverage"] = c["labeled_predictions"] / c["predictions"] if c["predictions"] else None
    c["mature_label_coverage"] = (c["mature_labeled_predictions"] / c["mature_predictions"]
                                  if c["mature_predictions"] else None)
    out = {"model_version": model_version, "counts": c, "label_maturity_days": maturity_days,
           "min_labels": min_labels}
    if not labels:
        return {**out, "status": "unavailable", "reason": "no actual labels ingested (no label source exists yet)"}
    if not pairs:
        return {**out, "status": "unavailable", "reason": "no label matches a stored prediction"}
    y = np.array([p[0] for p in pairs.values()])
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    out["positives"], out["negatives"] = positives, negatives
    delays = np.array([p[4] for p in pairs.values()]) / 86400
    out["label_delay_days"] = {"p50": float(np.quantile(delays, 0.5)), "p95": float(np.quantile(delays, 0.95))}
    if len(y) < min_labels or positives == 0 or negatives == 0:
        return {**out, "status": "insufficient_labels",
                "reason": f"{len(y)} usable labels ({positives} fraud, {negatives} legitimate); "
                          f"need >= {min_labels} with both classes"}
    scores = np.array([p[1] for p in pairs.values()])
    # Decision actually returned: anything not approved went to review (or decline).
    not_approved = np.array([p[2] in ("review", "decline") for p in pairs.values()])
    # Decline threshold, including shadow would-declines (hypothetical, not executed).
    decline_or_would = np.array([p[2] == "decline" or p[3] for p in pairs.values()])
    return {**out, "status": "ok", "n": int(len(y)),
            "at_approve_threshold": _metrics(y, not_approved, scores),
            "at_decline_threshold_incl_shadow": _metrics(y, decline_or_would, scores),
            "pr_auc": float(average_precision_score(y, scores))}


def tuning_steps_for(model_dir):
    """Train and validation step ranges from the model's metadata (used to fit / tune thresholds)."""
    try:
        meta = json.loads((Path(model_dir) / "metadata.json").read_text())
        b = meta["split_boundaries"]
        return [tuple(b["train"]), tuple(b["val"])]
    except (OSError, ValueError, KeyError):
        return []


class PerformanceCache:
    """Recompute at most every `ttl_s` (the dashboard polls; the evaluation scans the store)."""

    def __init__(self, store_dir, model_dirs=(), ttl_s=60):
        self.store_dir, self.ttl_s = Path(store_dir), ttl_s
        self.model_dirs = {Path(d).name: d for d in model_dirs}
        self._cache = {}

    def get(self, model_version):
        hit = self._cache.get(model_version)
        if hit and time.monotonic() - hit[0] < self.ttl_s:
            return hit[1]
        labels_dir = self.store_dir / "labels"
        labels, counts = read_labels(labels_dir) if labels_dir.exists() else ({}, {})
        events = iter_events(self.store_dir) if labels else ()
        steps = tuning_steps_for(self.model_dirs[model_version]) if model_version in self.model_dirs else []
        result = evaluate_performance(events, labels, counts, model_version=model_version, tuning_steps=steps)
        self._cache[model_version] = (time.monotonic(), result)
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-labels")
    a.add_argument("file")
    r = sub.add_parser("report")
    r.add_argument("--model", default=None)
    r.add_argument("--model-dir", default=None, help="excludes labels from its train/val steps")
    args = ap.parse_args(argv)
    store = Path(os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events")))
    if args.cmd == "add-labels":
        print(json.dumps(add_labels(args.file, store / "labels"), indent=2))
        return 0
    labels_dir = store / "labels"
    labels, counts = read_labels(labels_dir) if labels_dir.exists() else ({}, {})
    steps = tuning_steps_for(args.model_dir) if args.model_dir else []
    print(json.dumps(evaluate_performance(iter_events(store) if labels else (), labels, counts,
                                          model_version=args.model, tuning_steps=steps), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
