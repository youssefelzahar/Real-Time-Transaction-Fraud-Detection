"""Versioned monitoring baseline from the model's own training data (see ML_MONITORING.md).

    python -m src.monitoring_baseline --model-dir models/e7_v1 --data data/fraud_data.csv

Reproducible and linked to the model: it re-creates the exact chronological split recorded in
the model's metadata.json (and refuses to run if the data does not reproduce it), computes the
features with the model's saved FeaturePipeline (inference-time inputs only, no leaky columns)
and records hashes of the dataset and the model artifacts. Feature and type statistics come
from the TRAIN split only. The validation split is scored once to record a reference score and
decision distribution under the current policy; it is never used to evaluate the model here.
Never fitted on production data. Writes monitoring/baselines/<model_version>/bl-<model_version>-<n>.json
and never overwrites an existing version.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data_preprocessing import LEAKY_COLUMNS, TIME_COL, chronological_split, load_data
from src.decision_engine import DEFAULT_POLICY_PATH, DecisionEngine, DecisionPolicy
from src.ml_monitoring import KNOWN_TYPES, N_SCORE_BINS, SCORE_EDGES, score_bin
from src.monitoring_features import BINNED_FEATURES, FLAG_FEATURES, log_bins
from src.predict import FraudPredictor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "monitoring" / "baselines"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def feature_stats(X):
    out, bins, ranges, p999 = {}, {}, {}, {}
    for name in X.columns:
        v = X[name].to_numpy(dtype="float64")
        finite = v[np.isfinite(v)]
        out[name] = {
            "count": int(len(v)), "missing_rate": float(np.isnan(v).mean()) if len(v) else None,
            "non_finite_rate": float(np.isinf(v).mean()) if len(v) else None,
            "mean": _f(finite.mean()) if len(finite) else None, "std": _f(finite.std()) if len(finite) else None,
            "min": _f(finite.min()) if len(finite) else None, "max": _f(finite.max()) if len(finite) else None,
            "quantiles": {f"p{q * 100:g}": _f(np.quantile(finite, q)) for q in QUANTILES} if len(finite) else {},
        }
        if name in BINNED_FEATURES:
            b = log_bins(v)
            b = b[~np.isnan(b)].astype(int)
            values, counts = np.unique(b, return_counts=True)
            bins[name] = {str(k): float(c / len(b)) for k, c in zip(values, counts)}
            if len(b):
                ranges[name] = [int(b.min()), int(b.max())]
                p999[name] = int(np.quantile(b, 0.999, method="higher"))
    return out, bins, ranges, p999


def build_baseline(df, model_dir, *, baseline_version, policy, dataset):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    predictor = FraudPredictor.load(model_dir)
    train, val, _, boundaries = chronological_split(df)
    if boundaries["train"] != metadata["split_boundaries"]["train"] or \
            boundaries["val"] != metadata["split_boundaries"]["val"]:
        raise SystemExit(f"data does not reproduce the model's split: {boundaries} != {metadata['split_boundaries']}")
    if metadata.get("n_rows") is not None and len(df) != metadata["n_rows"]:
        raise SystemExit(f"dataset has {len(df)} rows, model metadata says {metadata['n_rows']}")
    names = predictor.feature_names
    assert not set(names) & set(LEAKY_COLUMNS)

    X_train = predictor.pipeline.transform(train)
    stats, bins, ranges, p999 = feature_stats(X_train)
    flags = {n: float(np.nanmean(X_train[n].to_numpy())) for n in names if n in FLAG_FEATURES}
    types = train["type"].astype(str).value_counts(normalize=True)

    # Reference only (not an evaluation): validation scores and the current policy's decisions.
    scores = predictor.model.predict_proba(predictor.pipeline.transform(val))[:, 1]
    hist = [0] * N_SCORE_BINS
    for i, c in zip(*np.unique([score_bin(float(p)) for p in scores], return_counts=True)):
        hist[int(i)] = int(c)
    engine = DecisionEngine(policy)
    decisions, would_decline = {"approve": 0, "review": 0, "decline": 0}, 0
    for p in scores:
        r = engine.decide(float(p))
        decisions[r.decision] += 1
        would_decline += r.would_decline

    n_val = len(scores)
    return {
        "baseline_version": baseline_version,
        "model_version": predictor.model_version,
        "dataset": dataset,
        "artifacts": {f: sha256(model_dir / f) for f in ("model.json", "pipeline.json", "metadata.json")},
        "model_dir": model_dir.as_posix(),
        "split": {"source": "chronological_split, as recorded in metadata.json",
                  "statistics_from": "train", "train_steps": boundaries["train"], "val_steps": boundaries["val"],
                  "train_rows": int(len(train)), "val_rows": int(n_val), "time_col": TIME_COL},
        "feature_names": names,
        "features": stats,
        "bins": bins, "bin_range": ranges, "bin_p999": p999,
        "binning": "floor(4*log10(1+x)), src/monitoring_features.py",
        "flags": flags,
        "transaction_types": {t: float(types.get(t, 0.0)) for t in KNOWN_TYPES},
        "reference_scores": {
            "split": "val", "purpose": "reference distribution only; thresholds were tuned on this split",
            "n": n_val, "score_edges": list(SCORE_EDGES), "score_hist": hist, "mean": float(np.mean(scores)),
            "policy_version": policy.policy_version,
            "decision_rates": {d: c / n_val for d, c in decisions.items()},
            "would_decline_rate": would_decline / n_val},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", default="data/fraud_data.csv")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--version", type=int, default=1, help="baseline number; an existing one is never overwritten")
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    model_version = f"{metadata['feature_set'].lower()}_{metadata['version']}"
    baseline_version = f"bl-{model_version}-{args.version}"
    out = Path(args.out_dir) / model_version / f"{baseline_version}.json"
    if out.exists():
        sys.exit(f"{out} exists; baselines are immutable, use --version {args.version + 1}")
    data = Path(args.data)
    print(f"hashing {data} ...", flush=True)
    dataset = {"path": data.as_posix(), "sha256": sha256(data), "bytes": data.stat().st_size}
    print("loading ...", flush=True)
    df = load_data(data)
    dataset["n_rows"] = int(len(df))
    policy = DecisionPolicy.from_file(args.policy)
    policy.check_model(model_version)
    baseline = build_baseline(df, model_dir, baseline_version=baseline_version, policy=policy, dataset=dataset)
    baseline["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    baseline["generator"] = ("python -m src.monitoring_baseline --model-dir " + model_dir.as_posix() +
                             f" --data {data.as_posix()} --version {args.version}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
