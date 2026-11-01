"""Train E5, E6 and E7 on the same chronological split, save versioned artifacts and a comparison report.

Usage:
    python -m src.train --data data/fraud_data.csv --version v3

Artifact directories are immutable: an existing, non-empty `models/<set>_<version>` (and the
champion listed in config/retraining_policy.json) is never overwritten. Pick a new version.
Training is manual; nothing here promotes or serves the result (RETRAINING_STRATEGY.md).
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from xgboost import XGBClassifier

from src.data_preprocessing import (DEFAULT_MAX_AMOUNT, FEATURE_SETS, LEAKY_COLUMNS, TARGET,
                                    FeaturePipeline, chronological_split, load_data)
from src.evaluate import (COST_FN, COST_FP, choose_threshold, evaluate_at_threshold,
                          shortcut_checks)
from src.retraining import (PolicyViolation, check_feature_set, check_target, code_revision, file_sha256,
                            guard_new_artifact_dir, load_policy)

LABEL_DEFINITION = ("isFraud: transactions made by PaySim's fraud agents (simulator ground truth). "
                    "isFlaggedFraud (the legacy >200k rule flag) is excluded as a target and as a feature.")

XGB_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                  tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42)


def train_xgb(X_train, y_train, **overrides):
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": neg / pos, **overrides})
    model.fit(X_train, y_train)
    return model


def run_experiment(feature_set, train, val, test, max_amount=DEFAULT_MAX_AMOUNT,
                   cost_fn=COST_FN, cost_fp=COST_FP, xgb_overrides=None):
    """Fit preprocessing + model on train, threshold on val, evaluate once on test."""
    pipeline = FeaturePipeline(feature_set, max_amount).fit(train)
    X_train, X_val, X_test = (pipeline.transform(d) for d in (train, val, test))
    assert not set(LEAKY_COLUMNS) & set(X_train.columns)

    model = train_xgb(X_train, train[TARGET].to_numpy(), **(xgb_overrides or {}))
    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, val_cost = choose_threshold(val[TARGET], proba_val, cost_fn, cost_fp)
    val_metrics = evaluate_at_threshold(val[TARGET], proba_val, threshold, cost_fn, cost_fp)
    test_metrics = evaluate_at_threshold(test[TARGET], model.predict_proba(X_test)[:, 1],
                                         threshold, cost_fn, cost_fp)
    checks = shortcut_checks(X_val, val[TARGET].to_numpy(), val_metrics, val_cost)
    importance = model.get_booster().get_score(importance_type="gain")
    return {"feature_set": feature_set, "pipeline": pipeline, "model": model,
            "threshold": threshold, "val": val_metrics, "test": test_metrics, "checks": checks,
            "importance": dict(sorted(importance.items(), key=lambda kv: -kv[1]))}


def save_artifacts(result, out_dir, metadata, policy=None):
    """Write model.json, pipeline.json (medians + feature schema) and metadata.json into a new or
    empty directory. Raises PolicyViolation rather than overwrite an existing artifact."""
    out_dir = guard_new_artifact_dir(out_dir, policy)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["model"].save_model(out_dir / "model.json")
    result["pipeline"].save(out_dir / "pipeline.json")
    meta = {**metadata,
            "feature_set": result["feature_set"],
            "threshold": result["threshold"],
            "feature_names": result["pipeline"].feature_names,
            "feature_dtype": "float64",
            "required_inputs": result["pipeline"].required_inputs,
            "validation_metrics": result["val"],
            "test_metrics": result["test"],
            "shortcut_checks": result["checks"],
            "feature_policy_check": check_feature_set(result["pipeline"].feature_names, policy or load_policy()),
            "feature_importance_gain": result["importance"]}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    return out_dir


def comparison_report(results, boundaries, fraud_rates, cost_fn, cost_fp):
    rows = ["| Exp | Threshold | Precision | Recall | PR-AUC | FPR | TP | FP | FN | Cost |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["test"]
        rows.append(f"| {r['feature_set']} | {m['threshold']:.2f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                    f"{m['pr_auc']:.4f} | {m['fpr']:.5f} | {m['TP']:,} | {m['FP']:,} | {m['FN']:,} | {m['cost']:,} |")
    lines = [
        "# " + " vs ".join(r["feature_set"] for r in results) + " comparison", "",
        f"Business cost = {cost_fn} x FN + {cost_fp} x FP. Threshold picked on validation only; test evaluated once.", "",
        "| Period | Steps | Fraud rate |", "|---|---|---|",
        *[f"| {k} | {v[0]}-{v[1]} | {fraud_rates[k]:.4%} |" for k, v in boundaries.items()], "",
        "## Test results", "", *rows, "",
    ]
    for r in results:
        v = r["val"]
        lines += [f"## {r['feature_set']}", "",
                  f"- Features: {', '.join(r['pipeline'].feature_names)}",
                  f"- Validation: cost {v['cost']:,}, PR-AUC {v['pr_auc']:.4f}, FP {v['FP']:,}, FN {v['FN']:,}",
                  "- Top features by gain: " + ", ".join(f"{k} ({g:,.0f})" for k, g in list(r["importance"].items())[:5]),
                  "- Strongest single features (validation PR-AUC alone): " + ", ".join(
                      f"{k} {s:.3f}" for k, s in list(r["checks"]["single_feature_pr_auc"].items())[:3]),
                  "- Shortcut warnings: " + ("; ".join(r["checks"]["warnings"]) or "none"), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fraud_data.csv")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--max-amount", type=float, default=DEFAULT_MAX_AMOUNT)
    ap.add_argument("--cost-fn", type=float, default=COST_FN)
    ap.add_argument("--cost-fp", type=float, default=COST_FP)
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=FEATURE_SETS)
    args = ap.parse_args()

    policy = load_policy()
    check_target(TARGET, policy)
    out_dirs = {fs: Path(args.models_dir) / f"{fs.lower()}_{args.version}" for fs in args.feature_sets}
    report_path = Path(args.models_dir) / f"comparison_{args.version}.md"
    try:  # check every output before spending time on training
        for d in out_dirs.values():
            guard_new_artifact_dir(d, policy)
        if report_path.exists():
            raise PolicyViolation(f"{report_path} already exists; use a new version")
    except PolicyViolation as e:
        ap.error(str(e))

    df = load_data(args.data)
    train, val, test, boundaries = chronological_split(df)
    fraud_rates = {k: float(d[TARGET].mean()) for k, d in zip(("train", "val", "test"), (train, val, test))}
    for k, (lo, hi) in boundaries.items():
        print(f"{k:5s} steps {lo}-{hi} | fraud rate {fraud_rates[k]:.4%}")

    metadata = {"version": args.version,
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "data_path": str(args.data), "n_rows": int(len(df)),
                "dataset_sha256": file_sha256(args.data),
                "target": TARGET, "label_definition": LABEL_DEFINITION,
                "random_seed": XGB_PARAMS["random_state"],
                "code_revision": code_revision(),
                "retraining_policy_version": policy["policy_version"],
                "split_boundaries": boundaries, "fraud_rates": fraud_rates,
                "cost_function": {"cost_fn": args.cost_fn, "cost_fp": args.cost_fp},
                "xgb_params": XGB_PARAMS,
                "library_versions": {"python": platform.python_version(), "xgboost": xgboost.__version__,
                                     "scikit-learn": sklearn.__version__}}

    results = []
    for fs in args.feature_sets:
        print(f"\n===== {fs} =====")
        r = run_experiment(fs, train, val, test, args.max_amount, args.cost_fn, args.cost_fp)
        out = save_artifacts(r, out_dirs[fs], metadata, policy)
        t = r["test"]
        print(f"threshold {r['threshold']:.2f} | test PR-AUC {t['pr_auc']:.4f} | FP {t['FP']} | FN {t['FN']} "
              f"| cost {t['cost']:,} -> {out}")
        for w in r["checks"]["warnings"]:
            print("  WARNING:", w)
        results.append(r)

    report = comparison_report(results, boundaries, fraud_rates, args.cost_fn, args.cost_fp)
    with open(report_path, "x") as f:  # "x": never overwrite
        f.write(report)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
