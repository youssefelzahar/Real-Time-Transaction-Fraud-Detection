import numpy as np
import pandas as pd
import pytest

from src.train import run_experiment, save_artifacts

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def make_transactions(n=3000, seed=0):
    """Small PaySim-shaped dataset: fraud is TRANSFER/CASH_OUT that drains the sender."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "step": np.sort(rng.integers(1, 100, n)),
        "type": rng.choice(TYPES, n),
        "amount": rng.lognormal(10, 1.5, n).round(2),
        "oldbalanceOrg": rng.choice([0.0, 1.0], n, p=[0.4, 0.6]) * rng.lognormal(11, 1.5, n).round(2),
        "oldbalanceDest": rng.choice([0.0, 1.0], n, p=[0.3, 0.7]) * rng.lognormal(12, 1.5, n).round(2),
        "isFraud": 0,
    })
    fraud = rng.random(n) < 0.05
    df.loc[fraud, "type"] = rng.choice(["TRANSFER", "CASH_OUT"], fraud.sum())
    df.loc[fraud, "oldbalanceOrg"] = df.loc[fraud, "amount"]
    df.loc[fraud, "isFraud"] = 1
    df["newbalanceOrig"] = (df["oldbalanceOrg"] - df["amount"]).clip(lower=0)
    df["newbalanceDest"] = df["oldbalanceDest"] + df["amount"]
    df["isFlaggedFraud"] = 0
    return df


@pytest.fixture(scope="session")
def transactions():
    return make_transactions()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, transactions):
    """A tiny trained E6 model saved with the real artifact writer."""
    df = transactions
    train, val, test = df.iloc[:1800], df.iloc[1800:2400], df.iloc[2400:]
    result = run_experiment("E6", train, val, test, xgb_overrides={"n_estimators": 20, "n_jobs": 1})
    out = tmp_path_factory.mktemp("e6_test")
    save_artifacts(result, out, {"version": "test"})
    return out
