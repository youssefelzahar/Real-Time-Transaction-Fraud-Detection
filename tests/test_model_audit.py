"""Phase 2 model audit: metric correctness, threshold protocol, and artifact reproducibility.

The full-data test needs data/fraud_data.csv and the artifacts in models/; it is opt-in:
    RUN_DATA_TESTS=1 pytest tests/test_model_audit.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.evaluate import choose_threshold, confusion_counts, evaluate_at_threshold

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _have(*names):
    return all((MODELS / n / "metadata.json").exists() for n in names)


@pytest.mark.parametrize("seed", range(20))
def test_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 3000))
    y = rng.random(n) < rng.uniform(0.001, 0.3)
    # Rounded scores create exact ties at the threshold (score == t counts as flagged).
    p = np.round(np.clip(rng.normal(0.3 + 0.4 * y, 0.25), 0, 1), 2)
    t = float(rng.choice([0.0, 0.3, 0.5, 0.54, 1.0]))
    m = evaluate_at_threshold(y, p, t)
    pred = p >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (tp, fp, fn, tn)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y, pred, zero_division=0))
    assert m["f1"] == pytest.approx(f1_score(y, pred, zero_division=0))
    assert m["fpr"] == pytest.approx(fp / (fp + tn) if fp + tn else 0.0)
    assert m["cost"] == 100 * fn + fp
    if y.any():
        assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))


def test_accuracy_is_high_for_a_useless_model():
    """Why accuracy is never the selection metric here: approving everything scores 99.9%."""
    y = np.zeros(10_000, dtype=bool)
    y[:10] = True
    m = evaluate_at_threshold(y, np.zeros(len(y)), 0.5)
    assert m["accuracy"] == pytest.approx(0.999)
    assert m["recall"] == 0 and m["f1"] == 0 and m["cost"] == 1000


def test_threshold_ties_pick_the_lowest_and_cost_is_consistent():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])  # every t in (0.2, 0.8] has cost 0
    t, cost = choose_threshold(y, p)
    assert t == 0.21 and cost == 0
    assert evaluate_at_threshold(y, p, t)["cost"] == cost


def test_threshold_grid_stops_at_099():
    """The grid's upper edge: a model whose optimum lies above 0.99 is clipped (E6 case)."""
    y = np.array([0] * 5 + [1])
    p = np.array([0.5, 0.98, 0.992, 0.993, 0.994, 0.999])  # cost falls up to 0.99 and beyond
    t, cost = choose_threshold(y, p)
    assert t == 0.99
    _, fp, fn, _ = confusion_counts(y, p, 0.995)
    assert cost > 100 * fn + fp  # a better threshold exists outside the grid


@pytest.mark.skipif(not _have("e7_v1", "e7_v2"), reason="model artifacts not present")
def test_served_policy_threshold_is_the_artifact_threshold():
    policy = json.loads((ROOT / "config" / "decision_policy.json").read_text())
    for name in policy["valid_for_models"]:
        meta = json.loads((MODELS / name / "metadata.json").read_text())
        assert meta["threshold"] == policy["approve_threshold"]
        assert meta["threshold"] == meta["validation_metrics"]["threshold"]


@pytest.mark.skipif(not _have("e7_v1", "e7_v2"), reason="model artifacts not present")
def test_e7_v1_and_e7_v2_are_the_same_model():
    for f in ("model.json", "pipeline.json"):
        assert (MODELS / "e7_v1" / f).read_bytes() == (MODELS / "e7_v2" / f).read_bytes()


@pytest.mark.skipif(os.getenv("RUN_DATA_TESTS") != "1", reason="set RUN_DATA_TESTS=1 (needs the 6.4M-row CSV)")
@pytest.mark.parametrize("name", ["e5_v1", "e6_v1", "e7_v1"])
def test_artifact_metrics_reproduce_from_raw_predictions(name):
    from src.data_preprocessing import TARGET, chronological_split, load_data
    from src.predict import FraudPredictor

    if not _have(name):
        pytest.skip("artifact missing")
    _, val, test, bounds = chronological_split(load_data(ROOT / "data" / "fraud_data.csv"))
    pred = FraudPredictor.load(MODELS / name)
    meta = pred.metadata
    assert bounds == meta["split_boundaries"]
    p_val = pred.model.predict_proba(pred.pipeline.transform(val))[:, 1]
    t, _ = choose_threshold(val[TARGET], p_val)
    assert t == meta["threshold"]  # selected on validation only
    p_test = pred.model.predict_proba(pred.pipeline.transform(test))[:, 1]
    for recorded, y, p in ((meta["validation_metrics"], val[TARGET], p_val), (meta["test_metrics"], test[TARGET], p_test)):
        m = evaluate_at_threshold(y, p, t)
        for k, v in recorded.items():
            assert m[k] == pytest.approx(v, rel=1e-12), k
