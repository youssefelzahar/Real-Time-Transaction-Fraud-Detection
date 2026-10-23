import json

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (E6_FEATURES, E7_FEATURES, LEAKY_COLUMNS, TYPE_CODES, FeaturePipeline,
                                    chronological_split, safe_ratio)
from src.evaluate import choose_threshold, evaluate_at_threshold
from src.predict import FraudPredictor

TRAIN = pd.DataFrame({
    "type": ["TRANSFER", "TRANSFER", "PAYMENT", "PAYMENT", "CASH_OUT"],
    "amount": [100.0, 300.0, 10.0, 30.0, 50.0],
})


def pipeline(feature_set="E6", max_amount=1000.0):
    return FeaturePipeline(feature_set, max_amount=max_amount).fit(TRAIN)


def row(**kw):
    base = {"type": "TRANSFER", "amount": 200.0, "oldbalanceOrg": 400.0, "oldbalanceDest": 50.0, "step": 25}
    return pd.DataFrame([{**base, **kw}])


def test_type_medians_fitted_on_training_data_only():
    p = pipeline()
    assert p.type_medians == {"CASH_OUT": 50.0, "PAYMENT": 20.0, "TRANSFER": 200.0}


def test_feature_values():
    X = pipeline().transform(row())
    r = X.iloc[0]
    assert r["amt_to_orig_bal"] == 0.5
    assert r["exceeds_orig_bal"] == 0.0
    assert r["orig_bal_zero"] == 0.0
    assert r["dest_bal_zero"] == 0.0
    assert r["is_max_amount"] == 0.0
    assert r["amt_vs_type_median"] == 1.0


def test_exceeds_and_max_amount_threshold_is_configurable():
    r = pipeline(max_amount=150.0).transform(row(amount=500.0)).iloc[0]
    assert r["exceeds_orig_bal"] == 1.0
    assert r["is_max_amount"] == 1.0
    assert pipeline(max_amount=1000.0).transform(row(amount=500.0)).iloc[0]["is_max_amount"] == 0.0
    assert pipeline(max_amount=500.0).transform(row(amount=500.0)).iloc[0]["is_max_amount"] == 1.0  # ">=" cap


def test_zero_balances_do_not_divide_by_zero():
    r = pipeline().transform(row(oldbalanceOrg=0.0, oldbalanceDest=0.0)).iloc[0]
    assert np.isnan(r["amt_to_orig_bal"])
    assert r["orig_bal_zero"] == 1.0
    assert r["dest_bal_zero"] == 1.0
    assert r["exceeds_orig_bal"] == 1.0


def test_safe_ratio():
    out = safe_ratio([1.0, 1.0, np.nan, 4.0], [0.0, np.nan, 2.0, 2.0])
    assert np.isnan(out[:3]).all() and out[3] == 2.0


def test_missing_values_become_nan_not_false():
    r = pipeline().transform(row(oldbalanceOrg=None, oldbalanceDest=np.nan, amount=None)).iloc[0]
    for col in ["amount", "oldbalanceOrg", "oldbalanceDest", "amt_to_orig_bal", "exceeds_orig_bal",
                "orig_bal_zero", "dest_bal_zero", "is_max_amount", "amt_vs_type_median"]:
        assert np.isnan(r[col]), col


def test_missing_balance_columns_entirely():
    X = pipeline().transform(pd.DataFrame([{"type": "TRANSFER", "amount": 10.0}]))
    assert np.isnan(X.iloc[0]["oldbalanceOrg"]) and np.isnan(X.iloc[0]["dest_bal_zero"])


def test_unseen_transaction_type():
    X = pipeline().transform(row(type="CRYPTO"))
    assert np.isnan(X.iloc[0]["amt_vs_type_median"])
    X5 = pipeline("E5").transform(row(type="CRYPTO"))
    assert X5.filter(like="type_").sum(axis=1).iloc[0] == 0.0


def test_schema_order_dtype_and_no_leakage():
    X = pipeline().transform(row(newbalanceOrig=0.0, newbalanceDest=1.0, isFraud=1, isFlaggedFraud=1))
    assert list(X.columns) == E6_FEATURES
    assert (X.dtypes == "float64").all()
    assert not set(LEAKY_COLUMNS) & set(X.columns)


def test_e5_schema_includes_hour_and_type_dummies():
    p = pipeline("E5")
    assert p.feature_names[-4:] == ["hour", "type_CASH_OUT", "type_PAYMENT", "type_TRANSFER"]
    assert p.transform(row(step=25)).iloc[0]["hour"] == 1.0


def test_e7_type_code_mapping_and_no_shortcut_ratio():
    p = pipeline("E7")
    assert p.feature_names == E7_FEATURES and "amt_to_orig_bal" not in p.feature_names
    types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER", "CRYPTO", None]
    X = p.transform(pd.concat([row(type=t) for t in types], ignore_index=True))
    assert X["type_code"].iloc[:5].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert X["type_code"].iloc[5:].isna().all()  # unseen / missing type
    assert (X.dtypes == "float64").all()


def test_e7_mapping_saved_and_restored(tmp_path):
    p = pipeline("E7")
    p.save(tmp_path / "pipeline.json")
    assert json.loads((tmp_path / "pipeline.json").read_text())["type_codes"] == TYPE_CODES
    restored = FeaturePipeline.load(tmp_path / "pipeline.json")
    pd.testing.assert_frame_equal(p.transform(row()), restored.transform(row()))


def test_pipeline_roundtrip_is_identical(tmp_path):
    p = pipeline()
    p.save(tmp_path / "pipeline.json")
    restored = FeaturePipeline.load(tmp_path / "pipeline.json")
    df = pd.concat([row(), row(oldbalanceOrg=0.0), row(type="CRYPTO"), row(amount=None)], ignore_index=True)
    pd.testing.assert_frame_equal(p.transform(df), restored.transform(df))


def test_tampered_schema_is_rejected(tmp_path):
    d = pipeline().to_dict()
    d["feature_names"] = list(reversed(d["feature_names"]))
    with pytest.raises(ValueError):
        FeaturePipeline.from_dict(d)


def test_chronological_split_has_no_overlap(transactions):
    train, val, test, b = chronological_split(transactions)
    assert train["step"].max() < val["step"].min() <= val["step"].max() < test["step"].min()
    assert len(train) + len(val) + len(test) == len(transactions)


def test_threshold_selection_and_cost():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    t, cost = choose_threshold(y, p)
    assert 0.6 < t <= 0.7 and cost == 0
    m = evaluate_at_threshold(y, p, 0.5)
    assert (m["FP"], m["FN"], m["cost"]) == (1, 0, 1)


def test_training_serving_consistency(model_dir, transactions):
    """Batch features/probabilities from training code == record-by-record serving path."""
    predictor = FraudPredictor.load(model_dir)
    sample = transactions.sample(50, random_state=1)
    train_pipeline = FeaturePipeline.load(model_dir / "pipeline.json")
    X_batch = train_pipeline.transform(sample)
    proba_batch = predictor.model.predict_proba(X_batch)[:, 1]

    records = sample[predictor.pipeline.required_inputs].to_dict("records")
    serving = np.array([predictor.score(r)["fraud_probability"] for r in records])
    pd.testing.assert_frame_equal(predictor.features(records), X_batch.reset_index(drop=True))
    np.testing.assert_allclose(serving, proba_batch, rtol=1e-6)


def test_artifacts_contain_schema_threshold_and_medians(model_dir):
    meta = json.loads((model_dir / "metadata.json").read_text())
    pipe = json.loads((model_dir / "pipeline.json").read_text())
    assert meta["feature_names"] == E6_FEATURES == pipe["feature_names"]
    assert 0 < meta["threshold"] < 1
    assert set(pipe["type_medians"]) >= {"TRANSFER", "CASH_OUT"}
    assert meta["required_inputs"] == ["type", "amount", "oldbalanceOrg", "oldbalanceDest"]
