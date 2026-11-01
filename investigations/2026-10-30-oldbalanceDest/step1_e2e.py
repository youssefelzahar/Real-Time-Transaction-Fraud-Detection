"""Step 1 end to end: known train rows -> Nginx -> API -> Kafka -> consumer -> store; bins must equal offline."""
import json, sys, time, glob, os
import httpx, numpy as np
sys.path.insert(0, ".")
from src.data_preprocessing import chronological_split, load_data
from src.monitoring_features import BINNED_FEATURES, FLAG_FEATURES, log_bins
from src.predict import FraudPredictor
p = FraudPredictor.load("models/e7_v1")
train, _, _, _ = chronological_split(load_data("data/fraud_data.csv"))
s = train.sample(200, random_state=11).reset_index(drop=True)
X = p.pipeline.transform(s)
run = f"inv-e2e-{int(time.time())}"
with httpx.Client(timeout=5) as c:
    for i, row in s.iterrows():
        body = {"transaction_id": f"{run}-{i}", "type": row["type"], "amount": float(row["amount"]),
                "sender_id": "C-inv", "receiver_id": "C-inv", "sender_balance": float(row["oldbalanceOrg"]),
                "receiver_balance": float(row["oldbalanceDest"])}
        assert c.post("http://127.0.0.1:8080/predict", json=body, headers={"X-Traffic-Source": "test"}).status_code == 200
time.sleep(8)                                      # consumer flushes and commits every 5 s
store = sorted(glob.glob(os.path.expanduser("~/.local/share/nilepay-events/prediction-events-*.jsonl")))[-1]
found = {}
with open(store, "rb") as f:
    f.seek(max(os.path.getsize(store) - 20_000_000, 0))
    for line in f:
        if run.encode() in line:
            e = json.loads(line); found[e["transaction_id"]] = e
bad = 0
for i in range(len(s)):
    e = found.get(f"{run}-{i}")
    if e is None:
        bad += 1; continue
    for fname in BINNED_FEATURES:
        if fname in p.feature_names:
            ob = log_bins(np.array([X.iloc[i][fname]]))[0]
            bad += e["feature_bins"][fname] != (None if np.isnan(ob) else int(ob))
    for fname in FLAG_FEATURES:
        if fname in p.feature_names:
            bad += e["feature_flags"][fname] != int(X.iloc[i][fname])
srcs = {e.get("traffic_source") for e in found.values()}
print(f"sent 200, stored {len(found)}, mismatches {bad}, traffic_source {srcs}, receiver source {set(e['balance_source']['receiver'] for e in found.values())}")
print("zero oldbalanceDest rows in sample:", int((s['oldbalanceDest'] == 0).sum()), "| stored with bin 0:",
      sum(e['feature_bins']['oldbalanceDest'] == 0 for e in found.values()))
