"""Coarse, non-identifying feature summary for monitoring (shared by the API event, the training
baseline and the consumer-side aggregator, so all three bin identically).

Exact amounts and balances never leave the API. Each continuous feature is reduced to a bin of
a quarter decade, floor(4 * log10(1 + x)):  0 -> 0,  ~1.4 -> 1, 9 -> 4, 1k -> 12, 1M -> 24, 10M -> 28.
Model flags (0/1) are kept as they are. NaN (missing) -> None; +-inf -> listed in non_finite.
"""
import math

import numpy as np

BINS_PER_DECADE = 4
MAX_BIN = 63               # values above ~10^15.75 share the last bin (bounded cardinality)
NEGATIVE_BIN = -1          # cannot happen through the API (ge=0), kept so it is counted, not dropped

# Continuous features that get a log bin and model flags kept as 0/1, when the model has them.
BINNED_FEATURES = ("amount", "oldbalanceOrg", "oldbalanceDest", "amt_vs_type_median",
                   "amt_to_orig_bal", "amt_to_dest_bal")
FLAG_FEATURES = ("exceeds_orig_bal", "orig_bal_zero", "dest_bal_zero", "is_max_amount")


def log_bin(x):
    """Scalar bin; None for NaN/None, None for +-inf (reported separately as non-finite)."""
    if x is None or x != x or math.isinf(x):
        return None
    if x < 0:
        return NEGATIVE_BIN
    return min(int(BINS_PER_DECADE * math.log10(1.0 + x)), MAX_BIN)


def log_bins(values):
    """Vectorised log_bin for the baseline: float array -> float array of bins (NaN where missing/inf)."""
    v = np.asarray(values, dtype="float64")
    out = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    pos = finite & (v >= 0)
    out[pos] = np.minimum(np.floor(BINS_PER_DECADE * np.log10(1.0 + v[pos])), MAX_BIN)
    out[finite & (v < 0)] = NEGATIVE_BIN
    return out


def bin_lower_edge(b):
    """Smallest raw value that falls into bin b (for display)."""
    return 10 ** (b / BINS_PER_DECADE) - 1.0


def feature_summary(features, feature_names):
    """features: the model's feature row (same order as feature_names)."""
    bins, flags, non_finite = {}, {}, []
    for name, value in zip(feature_names, features):
        value = float(value)
        if math.isinf(value):
            non_finite.append(name)
        if name in BINNED_FEATURES:
            bins[name] = log_bin(value)
        elif name in FLAG_FEATURES:
            flags[name] = None if value != value or math.isinf(value) else int(value)
    return {"feature_bins": bins, "feature_flags": flags, "non_finite_features": non_finite}
