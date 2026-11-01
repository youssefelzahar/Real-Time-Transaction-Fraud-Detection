"""Re-execute the notebook experiments E1-E5 and compare with the notebook's recorded results.

E1-E4 have no saved model, pipeline or predictions; the only way to verify their
numbers is to re-run the code. The functions below are copied from
notebooks/01_eda_and_modeling.ipynb (cells 19, 26, 30-34) with plotting removed
and nothing else changed (unrounded threshold grid, int-cast flags, one-hot types,
`hour`). Models stay in memory: nothing is written to models/ or served.

Controls that the re-run is faithful and deterministic:
  - the notebook's E5 code must give scores identical to the saved models/e5_v1;
  - src.train.run_experiment("E7") must give scores identical to models/e7_v1.

    python investigations/2026-10-31-phase2-model-audit/rerun_notebook.py
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from common import ROOT, TARGET, load_scores, load_splits, metrics, save_scores, write_json
from src import train as src_train
from src.evaluate import single_feature_pr_auc

NOTEBOOK = ROOT / "notebooks" / "01_eda_and_modeling.ipynb"
COST_FN, COST_FP = 100, 1
MAX_AMOUNT = 10_000_000


# --- notebook code (plots removed) ------------------------------------------------
def prepare_features(df, num_cols, target="isFraud"):
    X = df[num_cols].copy()
    X["hour"] = df["step"] % 24
    X = pd.concat([X, pd.get_dummies(df["type"], prefix="type", dtype=int)], axis=1)
    return X, df[target]


def train_xgb(X_train, y_train):
    neg, pos = np.bincount(y_train)
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=neg / pos,
                          tree_method="hist", eval_metric="aucpr", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


def choose_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP):
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_true, (y_proba >= t).astype(int), labels=[0, 1]).ravel()
        costs.append(fn * cost_fn + fp * cost_fp)
    best = int(np.argmin(costs))
    return thresholds[best], int(costs[best])


def safe_ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def fit_feature_stats(train_df):
    return {"type_median_amount": train_df.groupby("type")["amount"].median().to_dict()}


def build_engineered_features(df, stats):
    X = df[["amount", "oldbalanceOrg", "oldbalanceDest"]].copy()
    X["amt_to_orig_bal"] = safe_ratio(df["amount"], df["oldbalanceOrg"])
    X["drains_orig"] = ((df["amount"] - df["oldbalanceOrg"]).abs() < 0.01) & (df["oldbalanceOrg"] > 0)
    X["exceeds_orig_bal"] = df["amount"] > df["oldbalanceOrg"]
    X["orig_bal_zero"] = df["oldbalanceOrg"] == 0
    X["dest_bal_zero"] = df["oldbalanceDest"] == 0
    X["amt_to_dest_bal"] = safe_ratio(df["amount"], df["oldbalanceDest"])
    X["is_max_amount"] = df["amount"] >= MAX_AMOUNT
    type_median = df["type"].map(stats["type_median_amount"])
    X["amt_vs_type_median"] = safe_ratio(df["amount"], type_median)
    X["hour"] = df["step"] % 24
    X = pd.concat([X, pd.get_dummies(df["type"], prefix="type", dtype=int)], axis=1)
    bool_cols = X.select_dtypes(bool).columns
    X[bool_cols] = X[bool_cols].astype(int)
    return X


def run_on_splits(make_X, splits):
    train_df, val_df, test_df = splits
    X_train, y_train = make_X(train_df)
    X_val, y_val = make_X(val_df)
    X_test, y_test = make_X(test_df)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    model = train_xgb(X_train, y_train)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    threshold, val_cost = choose_threshold(y_val, p_val)
    return model, list(X_train.columns), X_val, p_val, p_test, threshold, val_cost
# ----------------------------------------------------------------------------------


def recorded_from_notebook():
    """Parse the notebook's own printed results: the cell 34 table and the 'Best threshold' lines."""
    nb = json.loads(NOTEBOOK.read_text())
    outputs = ["\n\n".join("".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", []))
                           for o in c.get("outputs", [])) for c in nb["cells"]]
    rec = {}
    # pandas wraps wide frames into blocks: a header line, a 'model' line, one line per experiment.
    for block in outputs[34].split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3 or lines[1].strip() != "model":
            continue
        header = lines[0].replace("\\", "").split()
        for ln in lines[2:]:
            exp = ln.split()[0]
            values = re.findall(r"\d+\.\d+|\d+", ln)[-len(header):]
            rec.setdefault(exp, {}).update(zip(header, map(float, values)))
    for exp, cost in re.findall(r"===== (E\d) [^=]*=====\s*Best threshold on validation: [\d.]+ \(cost=([\d,]+)\)",
                                outputs[33] + outputs[34]):
        rec[exp]["val_cost"] = float(cost.replace(",", ""))
    for r in rec.values():
        r.pop("cost_per_fraud", None)
    return rec


def compare_rounded(rec, m, val_cost):
    """The notebook printed 4 decimals; compare at that precision, counts exactly."""
    out = {}
    for k, r in rec.items():
        if r is None:
            continue
        q = val_cost if k == "val_cost" else m[k]
        ok = abs(round(float(q), 4) - r) < 5e-5 if k in ("threshold", "precision", "recall", "pr_auc", "fpr") \
            else int(q) == int(r)
        out[k] = {"recorded": r, "reproduced": q, "match": bool(ok)}
    return out


def main():
    _, train, val, test, _ = load_splits()
    splits = (train, val, test)
    y_val, y_test = val[TARGET].to_numpy(), test[TARGET].to_numpy()
    log = {c: c + "_log" for c in ("amount", "oldbalanceOrg", "oldbalanceDest")}
    splits_log = tuple(d.assign(**{v: np.log1p(d[k].clip(lower=0)) for k, v in log.items()}) for d in splits)
    stats = fit_feature_stats(train)
    pre, post = list(log), list(log) + ["newbalanceOrig", "newbalanceDest"]
    ablate = ["drains_orig", "amt_to_orig_bal"]

    experiments = {
        "E1": ("post-tx balances (leaky reference)", lambda d: prepare_features(d, post), splits),
        "E2": ("pre-tx raw", lambda d: prepare_features(d, pre), splits),
        "E3": ("pre-tx log1p", lambda d: prepare_features(d, list(log.values())), splits_log),
        "E4": ("pre-tx engineered", lambda d: (build_engineered_features(d, stats), d[TARGET]), splits),
        "E5": ("engineered without drain signal",
               lambda d: (build_engineered_features(d, stats).drop(columns=ablate), d[TARGET]), splits),
    }
    recorded = recorded_from_notebook()
    out = {"source": "notebooks/01_eda_and_modeling.ipynb cells 19, 26, 30-34", "experiments": {}}
    for exp, (desc, make_X, sp) in experiments.items():
        model, cols, X_val, p_val, p_test, t, val_cost = run_on_splits(make_X, sp)
        save_scores(f"nb_{exp}", val=p_val, test=p_test)
        v, m = metrics(y_val, p_val, t), metrics(y_test, p_test, t)
        imp = model.get_booster().get_score(importance_type="gain")
        entry = {"description": desc, "features": cols, "threshold": float(t), "val_cost": val_cost,
                 "validation": v, "test": m,
                 "gain_top5": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:5]),
                 "vs_notebook": compare_rounded(recorded.get(exp, {}), m, val_cost)}
        if exp == "E4":
            entry["single_feature_pr_auc_val_top5"] = dict(list(single_feature_pr_auc(X_val, y_val).items())[:5])
        out["experiments"][exp] = entry
        bad = [k for k, c in entry["vs_notebook"].items() if not c["match"]]
        print(f"{exp} {desc}: t={t:.2f} val cost {val_cost:,} | test cost {m['cost']:,} FN {m['FN']} FP {m['FP']} "
              f"| {'matches notebook' if not bad else 'DIFFERS: ' + str(bad)}")

    # Controls
    e5 = load_scores("e5_v1")
    nb_e5 = np.load(__import__("common").SCORES_DIR / "nb_E5.npz")
    r7 = src_train.run_experiment("E7", train, val, test)
    p7_val = r7["model"].predict_proba(r7["pipeline"].transform(val))[:, 1]
    p7_test = r7["model"].predict_proba(r7["pipeline"].transform(test))[:, 1]
    e7 = load_scores("e7_v1")
    out["controls"] = {
        "notebook_E5_code_equals_models_e5_v1": {
            "val_identical": bool(np.array_equal(nb_e5["val"], e5["val"])),
            "test_identical": bool(np.array_equal(nb_e5["test"], e5["test"])),
            "max_abs_diff": float(max(np.abs(nb_e5["val"] - e5["val"]).max(),
                                      np.abs(nb_e5["test"] - e5["test"]).max()))},
        "src_train_E7_refit_equals_models_e7_v1": {
            "val_identical": bool(np.array_equal(p7_val, e7["val"])),
            "test_identical": bool(np.array_equal(p7_test, e7["test"])),
            "max_abs_diff": float(max(np.abs(p7_val - e7["val"]).max(), np.abs(p7_test - e7["test"]).max())),
            "threshold": r7["threshold"]},
    }
    print(json.dumps(out["controls"], indent=1))
    write_json("notebook_results.json", out)


if __name__ == "__main__":
    main()
