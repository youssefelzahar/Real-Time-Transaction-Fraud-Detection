"""Leakage and shortcut analysis (why E4 is so strong, what E5/E7 rely on), plus
uncertainty of the cost comparison. Uses cached scores; no training.

    python investigations/2026-10-31-phase2-model-audit/shortcut_analysis.py
"""
import numpy as np
import pandas as pd

from common import TARGET, load_scores, load_splits, metrics, write_json
from src.evaluate import business_cost, confusion_counts

MAX_AMOUNT = 10_000_000
FRAUD_TYPES = ("TRANSFER", "CASH_OUT")
# name -> (score file, threshold selected on validation)
CANDIDATES = {"E1": ("nb_E1", 0.36), "E2": ("nb_E2", 0.67), "E3": ("nb_E3", 0.79), "E4": ("nb_E4", 0.97),
              "E5": ("e5_v1", 0.53), "E6": ("e6_v1", 0.99), "E7": ("e7_v1", 0.54)}


def cost_at(y, p, t):
    _, fp, fn, _ = confusion_counts(y, p, t)
    return business_cost(fn, fp)


def flags(d):
    drains = ((d["amount"] - d["oldbalanceOrg"]).abs() < 0.01) & (d["oldbalanceOrg"] > 0)
    return pd.DataFrame({"fraud": d[TARGET].astype(bool), "type": d["type"],
                         "drains": drains, "is_max": d["amount"] >= MAX_AMOUNT,
                         "exceeds": d["amount"] > d["oldbalanceOrg"],
                         "hour": d["step"] % 24, "step": d["step"]})


def share(mask, within):
    return float(mask[within].mean()) if within.any() else None


def mechanism(f):
    fr, lg = f["fraud"], ~f["fraud"]
    tx = f["type"].isin(FRAUD_TYPES)
    rule = tx & (f["drains"] | f["is_max"])
    return {
        "rows": len(f), "fraud": int(fr.sum()),
        "fraud_by_type": f.loc[fr, "type"].value_counts().to_dict(),
        "drains_share": {"fraud": share(f["drains"], fr), "legit": share(f["drains"], lg),
                         "legit_TRANSFER_CASH_OUT": share(f["drains"], lg & tx)},
        "drains_or_max_share_of_fraud": share(f["drains"] | f["is_max"], fr),
        "rule_flags": {"fraud": int((rule & fr).sum()), "legit": int((rule & lg).sum())},
        "exceeds_orig_bal_share": {"fraud": share(f["exceeds"], fr), "legit": share(f["exceeds"], lg),
                                   "legit_TRANSFER_CASH_OUT": share(f["exceeds"], lg & tx),
                                   "legit_by_type": {t: share(f["exceeds"], lg & (f["type"] == t))
                                                     for t in sorted(f["type"].unique())}},
    }


def hour_pattern(f):
    night = f["hour"].between(0, 6)
    return {"legit_share_hours_0_6": share(night, ~f["fraud"]), "fraud_share_hours_0_6": share(night, f["fraud"]),
            "fraud_rate_hours_0_6": float(f.loc[night, "fraud"].mean()),
            "fraud_rate_hours_7_23": float(f.loc[~night, "fraud"].mean())}


def rule_metrics(f):
    """A one-line rule, no model: TRANSFER/CASH_OUT that empties the sender or hits the 10M cap."""
    score = (f["type"].isin(FRAUD_TYPES) & (f["drains"] | f["is_max"])).astype(float).to_numpy()
    return metrics(f["fraud"].to_numpy(), score, 0.5)


def sliced(f, y, p, t):
    """Model errors on TRANSFER/CASH_OUT split by whether the sender can cover the amount."""
    out = {}
    tx = f["type"].isin(FRAUD_TYPES).to_numpy()
    for name, m in (("sufficient_funds", ~f["exceeds"].to_numpy()), ("exceeds_balance", f["exceeds"].to_numpy())):
        s = tx & m
        tp, fp, fn, tn = confusion_counts(y[s], p[s], t)
        out[name] = {"rows": int(s.sum()), "fraud": int(y[s].sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "fpr": fp / (fp + tn) if fp + tn else None, "recall": tp / (tp + fn) if tp + fn else None,
                     "precision": tp / (tp + fp) if tp + fp else None}
    return out


def per_step_errors(y, p, t, steps, uniq):
    pred = p >= t
    idx = np.searchsorted(uniq, steps)
    fn = np.bincount(idx, weights=(y & ~pred), minlength=len(uniq))
    fp = np.bincount(idx, weights=(~y & pred), minlength=len(uniq))
    return fn, fp


def bootstrap_cost(scores, y, steps, pairs, n_boot=2000, seed=0):
    """Paired block bootstrap over time steps (fraud arrives in per-step bursts, so rows are not iid)."""
    uniq = np.unique(steps)
    per = {k: per_step_errors(y, s, t, steps, uniq) for k, (s, t) in scores.items()}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    out = {}
    for a, b in pairs:
        ca = 100 * per[a][0] + per[a][1]
        cb = 100 * per[b][0] + per[b][1]
        diff = (ca - cb)[draws].sum(axis=1)
        out[f"{a}-{b}"] = {"observed": float((ca - cb).sum()), "ci95": [float(np.percentile(diff, 2.5)),
                                                                         float(np.percentile(diff, 97.5))],
                           "share_of_resamples_a_cheaper": float((diff < 0).mean())}
    return out


def by_period(f, y, p, t, edges):
    rows = []
    s = f["step"].to_numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        tp, fp, fn, tn = confusion_counts(y[m], p[m], t)
        rows.append({"steps": [int(lo), int(hi - 1)], "rows": int(m.sum()), "fraud": int(y[m].sum()),
                     "fraud_rate": float(y[m].mean()), "FN": fn, "FP": fp, "cost": business_cost(fn, fp)})
    return rows


def main():
    _, train, val, test, _ = load_splits()
    fs = {"train": flags(train), "val": flags(val), "test": flags(test)}
    lab = load_scores("labels")
    y = {"val": lab["y_val"].astype(bool), "test": lab["y_test"].astype(bool)}
    sc = {k: load_scores(f) for k, (f, _) in CANDIDATES.items()}

    out = {"mechanism": {k: mechanism(f) for k, f in fs.items()},
           "hour_pattern_train": hour_pattern(fs["train"]),
           "rule_drains_or_max": {k: rule_metrics(fs[k]) for k in ("val", "test")}}

    # E4 threshold: how many grid thresholds tie at the minimum validation cost, and what each gives on test.
    ts = np.linspace(0.01, 0.99, 99)
    vc = np.array([cost_at(y["val"], sc["E4"]["val"], t) for t in ts])
    tied = ts[vc == vc.min()]
    out["e4_threshold_ties"] = {"val_cost_min": int(vc.min()), "tied_range": [float(tied.min()), float(tied.max())],
                                "n_tied": int(len(tied)),
                                "test_cost_over_tied": sorted({cost_at(y["test"], sc["E4"]["test"], t) for t in tied})}

    out["sliced_test_TRANSFER_CASH_OUT"] = {k: sliced(fs["test"], y["test"], sc[k]["test"], t)
                                            for k, (_, t) in CANDIDATES.items()}
    test_scores = {k: (sc[k]["test"], t) for k, (_, t) in CANDIDATES.items()}
    val_scores = {k: (sc[k]["val"], t) for k, (_, t) in CANDIDATES.items()}
    pairs = [("E7", "E5"), ("E7", "E2"), ("E7", "E1"), ("E5", "E2")]
    out["bootstrap_cost_diff"] = {
        "test": bootstrap_cost(test_scores, y["test"], lab["step_test"], pairs),
        "validation": bootstrap_cost(val_scores, y["val"], lab["step_val"], pairs)}
    edges = np.array([355, 452, 549, 646, 744])
    out["e7_test_by_period"] = by_period(fs["test"], y["test"], sc["E7"]["test"], 0.54, edges)
    write_json("shortcut_results.json", out)

    m = out["mechanism"]
    for k in ("train", "val", "test"):
        print(f"{k}: fraud by type {m[k]['fraud_by_type']} | drains: fraud {m[k]['drains_share']['fraud']:.4f} "
              f"legit {m[k]['drains_share']['legit']:.6f} | drains-or-max of fraud {m[k]['drains_or_max_share_of_fraud']:.4f}"
              f" | exceeds: fraud {m[k]['exceeds_orig_bal_share']['fraud']:.4f} legit TX/CO "
              f"{m[k]['exceeds_orig_bal_share']['legit_TRANSFER_CASH_OUT']:.4f}")
    print("hour:", out["hour_pattern_train"])
    for k, r in out["rule_drains_or_max"].items():
        print(f"rule on {k}: TP {r['TP']} FP {r['FP']} FN {r['FN']} cost {r['cost']:,} precision {r['precision']:.4f} recall {r['recall']:.4f}")
    print("E4 ties:", out["e4_threshold_ties"])
    for k, s in out["sliced_test_TRANSFER_CASH_OUT"].items():
        a, b = s["sufficient_funds"], s["exceeds_balance"]
        print(f"{k}: sufficient funds rows {a['rows']:,} fraud {a['fraud']} FP {a['FP']:,} FPR {a['fpr']:.4f} recall {a['recall']:.4f}"
              f" | exceeds rows {b['rows']:,} fraud {b['fraud']} FP {b['FP']:,} FPR {b['fpr']:.5f}")
    for part, d in out["bootstrap_cost_diff"].items():
        for k, v in d.items():
            print(f"bootstrap {part} {k}: observed {v['observed']:,.0f} CI95 {v['ci95'][0]:,.0f}..{v['ci95'][1]:,.0f} "
                  f"P(a cheaper) {v['share_of_resamples_a_cheaper']:.3f}")
    for r in out["e7_test_by_period"]:
        print("E7 period", r)


if __name__ == "__main__":
    main()
