"""Data loading, chronological split and the feature pipeline shared by training and serving."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "isFraud"
TIME_COL = "step"

# Never allowed as model inputs: the label, the legacy rule flag, and balances
# that only exist after the transaction has executed.
LEAKY_COLUMNS = ("isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest")

# PaySim caps a single transaction at 10M; fraudsters split larger thefts into
# chunks of exactly this size. Override per deployment via FeaturePipeline(max_amount=...).
DEFAULT_MAX_AMOUNT = 10_000_000.0

E5_BASE_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "exceeds_orig_bal", "orig_bal_zero", "dest_bal_zero", "amt_to_dest_bal",
    "is_max_amount", "amt_vs_type_median", "hour",
]  # + one-hot type_* columns for the types seen in training

E6_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "amt_to_orig_bal", "exceeds_orig_bal", "orig_bal_zero", "dest_bal_zero",
    "is_max_amount", "amt_vs_type_median",
]

# E6 without the drained-account ratio (a PaySim shortcut), plus the type as a code.
E7_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "exceeds_orig_bal", "orig_bal_zero", "dest_bal_zero",
    "is_max_amount", "amt_vs_type_median", "type_code",
]

# Fixed string -> number mapping for `type`; unseen types map to NaN.
TYPE_CODES = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

FEATURE_SETS = ("E5", "E6", "E7")


def load_data(path):
    return pd.read_csv(path)


def chronological_split(df, train_frac=0.6, val_frac=0.2, time_col=TIME_COL):
    """Earliest steps -> train, next -> val, latest -> test. No step spans two periods."""
    step_counts = df[time_col].value_counts().sort_index()
    cum = step_counts.cumsum() / len(df)
    train_end = int(cum[cum <= train_frac].index.max())
    val_end = int(cum[cum <= train_frac + val_frac].index.max())

    train = df[df[time_col] <= train_end]
    val = df[(df[time_col] > train_end) & (df[time_col] <= val_end)]
    test = df[df[time_col] > val_end]
    boundaries = {"train": [int(train[time_col].min()), train_end],
                  "val": [train_end + 1, val_end],
                  "test": [val_end + 1, int(test[time_col].max())]}
    return train, val, test, boundaries


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce").astype("float64") if col in df else \
        pd.Series(np.nan, index=df.index, dtype="float64")


def safe_ratio(num, den):
    """num / den; NaN when den is missing or <= 0, or num is missing."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    ok = (den > 0) & ~np.isnan(num)
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def _flag(condition, *inputs):
    """1.0/0.0 flag, NaN when any input is missing (so 'unknown' != 'false')."""
    missing = np.zeros(len(condition), dtype=bool)
    for s in inputs:
        missing |= np.isnan(np.asarray(s, dtype="float64"))
    return np.where(missing, np.nan, np.asarray(condition, dtype="float64"))


class FeaturePipeline:
    """Builds model features from raw transaction fields.

    The only learned state is the median amount per transaction type, fitted on the
    training split. The same instance (or one restored with from_dict) is used for
    validation, test and API inference, so features are computed identically.
    """

    def __init__(self, feature_set="E6", max_amount=DEFAULT_MAX_AMOUNT, type_codes=None):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature_set {feature_set!r}; expected one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.max_amount = float(max_amount)
        self.type_codes = dict(type_codes or TYPE_CODES)
        self.type_medians = None
        self.known_types = None

    @property
    def feature_names(self):
        if self.feature_set == "E6":
            return list(E6_FEATURES)
        if self.feature_set == "E7":
            return list(E7_FEATURES)
        return E5_BASE_FEATURES + [f"type_{t}" for t in self.known_types]

    @property
    def required_inputs(self):
        cols = ["type", "amount", "oldbalanceOrg", "oldbalanceDest"]
        return cols + ["step"] if self.feature_set == "E5" else cols

    def fit(self, train_df):
        medians = train_df.groupby("type")["amount"].median()
        self.type_medians = {str(k): float(v) for k, v in medians.items()}
        self.known_types = sorted(self.type_medians)
        return self

    def transform(self, df):
        if self.type_medians is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform")
        leaky = [c for c in LEAKY_COLUMNS if c in self.feature_names]
        assert not leaky, f"Leaky columns in feature set: {leaky}"

        amount = _num(df, "amount")
        orig = _num(df, "oldbalanceOrg")
        dest = _num(df, "oldbalanceDest")
        types = df["type"].astype("string") if "type" in df else pd.Series(pd.NA, index=df.index, dtype="string")
        type_median = types.map(self.type_medians).astype("float64")  # unseen/missing type -> NaN

        X = pd.DataFrame(index=df.index)
        X["amount"] = amount
        X["oldbalanceOrg"] = orig
        X["oldbalanceDest"] = dest
        X["amt_to_orig_bal"] = safe_ratio(amount, orig)
        X["exceeds_orig_bal"] = _flag(amount > orig, amount, orig)
        X["orig_bal_zero"] = _flag(orig == 0, orig)
        X["dest_bal_zero"] = _flag(dest == 0, dest)
        X["amt_to_dest_bal"] = safe_ratio(amount, dest)
        X["is_max_amount"] = _flag(amount >= self.max_amount, amount)
        X["amt_vs_type_median"] = safe_ratio(amount, type_median)
        X["type_code"] = types.map(self.type_codes).astype("float64")
        if self.feature_set == "E5":
            X["hour"] = _num(df, "step") % 24
            for t in self.known_types:
                X[f"type_{t}"] = (types == t).fillna(False).astype("float64")  # unseen type -> all zeros

        return X[self.feature_names].astype("float64")

    def to_dict(self):
        return {"feature_set": self.feature_set, "max_amount": self.max_amount,
                "type_codes": self.type_codes,
                "type_medians": self.type_medians, "known_types": self.known_types,
                "feature_names": self.feature_names, "required_inputs": self.required_inputs}

    @classmethod
    def from_dict(cls, d):
        p = cls(d["feature_set"], d["max_amount"], d.get("type_codes"))
        p.type_medians = {str(k): float(v) for k, v in d["type_medians"].items()}
        p.known_types = list(d["known_types"])
        if p.feature_names != d["feature_names"]:
            raise ValueError("Saved feature schema does not match the pipeline definition")
        return p

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
