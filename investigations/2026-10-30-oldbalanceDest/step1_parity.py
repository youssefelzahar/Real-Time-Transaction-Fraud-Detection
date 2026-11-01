"""Step 1: online (API path) vs offline (baseline path) features and monitoring bins, on train rows."""
import math, sys, numpy as np
sys.path.insert(0, ".")
from src.data_preprocessing import chronological_split, load_data
from src.monitoring_features import BINNED_FEATURES, FLAG_FEATURES, feature_summary, log_bins
from src.predict import FraudPredictor
p = FraudPredictor.load("models/e7_v1")
train, _, _, _ = chronological_split(load_data("data/fraud_data.csv"))
s = train.sample(20000, random_state=7)
X = p.pipeline.transform(s)
print("offline dtypes:", sorted({str(t) for t in X.dtypes}))
feat_mismatch = bin_mismatch = flag_mismatch = 0
for i, (_, row) in enumerate(s.iterrows()):
    rec = {"type": row["type"], "amount": float(row["amount"]), "oldbalanceOrg": float(row["oldbalanceOrg"]),
           "oldbalanceDest": float(row["oldbalanceDest"]), "step": int(row["step"])}
    online = p.pipeline.transform_one(rec)
    off = X.iloc[i].to_numpy(dtype="float64")
    if not all((a == b) or (math.isnan(a) and math.isnan(b)) for a, b in zip(online, off)):
        feat_mismatch += 1
    summ = feature_summary(online, p.feature_names)
    for f in BINNED_FEATURES:
        if f in p.feature_names:
            ob = log_bins(np.array([X.iloc[i][f]]))[0]
            ob = None if np.isnan(ob) else int(ob)
            bin_mismatch += summ["feature_bins"][f] != ob
    for f in FLAG_FEATURES:
        if f in p.feature_names:
            flag_mismatch += summ["feature_flags"][f] != int(X.iloc[i][f])
print(f"rows={len(s)} feature_row_mismatches={feat_mismatch} bin_mismatches={bin_mismatch} flag_mismatches={flag_mismatch}")
print("online value type:", type(online), getattr(online, "dtype", None))
