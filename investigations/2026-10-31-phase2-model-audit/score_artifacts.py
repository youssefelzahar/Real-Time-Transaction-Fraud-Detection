"""Re-score every saved model artifact and compare with its recorded metrics.

For each models/<name>: load model + pipeline exactly as serving does, score the
validation and test splits, re-select the threshold on VALIDATION only (same grid
and tie rule as src.evaluate.choose_threshold), and recompute every metric from
the raw scores. Writes artifact_results.json; caches scores for later scripts.

    python investigations/2026-10-31-phase2-model-audit/score_artifacts.py
"""
import json

import numpy as np

from common import (DATA, ROOT, TARGET, compare, load_splits, metrics, save_scores, sha256,
                    write_json)
from src.evaluate import business_cost, choose_threshold, confusion_counts
from src.predict import FraudPredictor

ARTIFACTS = ("e5_v1", "e6_v1", "e7_v1", "e7_v2")
METRIC_KEYS = ("threshold", "precision", "recall", "pr_auc", "fpr", "TP", "FP", "FN", "TN", "cost",
               "cost_per_fraud")


def cost_at(y, p, t):
    _, fp, fn, _ = confusion_counts(y, p, t)
    return business_cost(fn, fp)


def grid_costs(y, p):
    ts = np.round(np.linspace(0.01, 0.99, 99), 2)
    return ts, np.array([cost_at(y, p, t) for t in ts])


def main():
    _, _, val, test, bounds = load_splits()
    y_val, y_test = val[TARGET].to_numpy(), test[TARGET].to_numpy()
    save_scores("labels", y_val=y_val, y_test=y_test, step_val=val["step"].to_numpy(),
                step_test=test["step"].to_numpy())
    out = {"data": {"path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA)}, "splits": {}}
    for name, d in (("val", val), ("test", test)):
        out["splits"][name] = {"steps": bounds[name], "rows": len(d), "fraud": int(d[TARGET].sum())}

    for name in ARTIFACTS:
        d = ROOT / "models" / name
        pred = FraudPredictor.load(d)
        pred.model.set_params(n_jobs=8)  # batch scoring; serving keeps 1 thread
        meta = pred.metadata
        p_val = pred.model.predict_proba(pred.pipeline.transform(val))[:, 1]
        p_test = pred.model.predict_proba(pred.pipeline.transform(test))[:, 1]
        save_scores(name, val=p_val, test=p_test)

        t, val_cost = choose_threshold(y_val, p_val)
        ts, costs = grid_costs(y_val, p_val)
        tied = ts[costs == costs.min()]
        fine = np.round(np.arange(0.990, 1.0, 0.001), 3)
        fine_costs = [cost_at(y_val, p_val, f) for f in fine]
        val_m, test_m = metrics(y_val, p_val, t), metrics(y_test, p_test, meta["threshold"])
        out[name] = {
            "sha256": {f: sha256(d / f) for f in ("model.json", "pipeline.json", "metadata.json")},
            "feature_set": meta["feature_set"], "trained_at": meta["trained_at"],
            "feature_names": meta["feature_names"], "required_inputs": meta["required_inputs"],
            "threshold": {"recorded": meta["threshold"], "reselected_on_val": t, "val_cost": val_cost,
                          "grid": [0.01, 0.99], "tied_minimum": [float(tied.min()), float(tied.max())],
                          "at_grid_edge": t in (0.01, 0.99),
                          "val_cost_above_grid": dict(zip(map(str, fine), map(int, fine_costs)))},
            "validation": val_m, "test": test_m,
            "validation_vs_recorded": compare(meta["validation_metrics"], val_m, METRIC_KEYS),
            "test_vs_recorded": compare(meta["test_metrics"], test_m, METRIC_KEYS),
        }
        mism = [k for part in ("validation_vs_recorded", "test_vs_recorded")
                for k, v in out[name][part].items() if not v["match"]]
        out[name]["all_recorded_metrics_reproduced"] = not mism and t == meta["threshold"]
        print(f"{name}: threshold {meta['threshold']} (re-selected {t}, ties {tied.min()}-{tied.max()}) | "
              f"test cost {test_m['cost']:,} FN {test_m['FN']} FP {test_m['FP']} | "
              f"{'all recorded metrics reproduced' if not mism else 'MISMATCH ' + str(mism)}")

    same = out["e7_v1"]["sha256"]["model.json"] == out["e7_v2"]["sha256"]["model.json"]
    out["e7_v1_equals_e7_v2"] = {"model.json": same,
                                 "pipeline.json": out["e7_v1"]["sha256"]["pipeline.json"]
                                 == out["e7_v2"]["sha256"]["pipeline.json"]}
    write_json("artifact_results.json", out)
    print(json.dumps(out["e7_v1_equals_e7_v2"]))


if __name__ == "__main__":
    main()
