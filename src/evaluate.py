"""Threshold selection, test evaluation and shortcut/leakage checks."""
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

# Business cost: one missed fraud costs as much as reviewing 100 false alerts.
COST_FN = 100
COST_FP = 1


def confusion_counts(y_true, y_proba, threshold):
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_proba) >= threshold
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    return tp, fp, fn, tn


def business_cost(fn, fp, cost_fn=COST_FN, cost_fp=COST_FP):
    return fn * cost_fn + fp * cost_fp


def choose_threshold(y_val, proba_val, cost_fn=COST_FN, cost_fp=COST_FP):
    """Threshold in 0.01..0.99 minimizing validation cost (ties -> lowest threshold)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        _, fp, fn, _ = confusion_counts(y_val, proba_val, t)
        costs.append(business_cost(fn, fp, cost_fn, cost_fp))
    best = int(np.argmin(costs))
    return float(round(thresholds[best], 2)), int(costs[best])


def evaluate_at_threshold(y_true, y_proba, threshold, cost_fn=COST_FN, cost_fp=COST_FP):
    tp, fp, fn, tn = confusion_counts(y_true, y_proba, threshold)
    cost = business_cost(fn, fp, cost_fn, cost_fp)
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "cost": cost,
        "cost_per_fraud": cost / (tp + fn) if tp + fn else 0.0,
    }


def single_feature_pr_auc(X, y, max_depth=3):
    """PR-AUC of a depth-3 tree using one feature alone (NaN kept as its own value).
    A tree catches non-monotonic shortcuts such as a spike at ratio == 1.
    A single feature near 1.0 points to leakage or a simulator shortcut."""
    out = {}
    for col in X.columns:
        v = X[[col]].fillna(-1e18).to_numpy()
        tree = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=0)
        out[col] = float(average_precision_score(y, tree.fit(v, y).predict_proba(v)[:, 1]))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def shortcut_checks(X_val, y_val, val_metrics, val_cost):
    """Flags for suspiciously high performance on the validation period."""
    y = np.asarray(y_val).astype(bool)
    single = single_feature_pr_auc(X_val, y)
    warnings = []
    if val_cost == 0:
        warnings.append("Validation cost is 0 (perfect separation) - likely a shortcut feature.")
    if val_metrics["pr_auc"] > 0.99:
        warnings.append(f"Validation PR-AUC {val_metrics['pr_auc']:.4f} > 0.99.")
    for col, score in single.items():
        if score > 0.8:
            warnings.append(f"Feature '{col}' alone reaches PR-AUC {score:.3f} on validation.")
    drain_share = None
    if "amt_to_orig_bal" in X_val:
        ratio = X_val["amt_to_orig_bal"].to_numpy()
        drained = np.isclose(ratio, 1.0, atol=1e-6)
        drain_share = {"fraud": float(drained[y].mean()), "non_fraud": float(drained[~y].mean())}
        if drain_share["fraud"] > 0.5 and drain_share["non_fraud"] < 0.01:
            warnings.append(
                f"amt_to_orig_bal == 1 (account fully drained) for {drain_share['fraud']:.1%} of fraud vs "
                f"{drain_share['non_fraud']:.2%} of non-fraud: PaySim fraud-agent rule, not a general pattern.")
    return {"single_feature_pr_auc": single, "drain_share": drain_share, "warnings": warnings}
