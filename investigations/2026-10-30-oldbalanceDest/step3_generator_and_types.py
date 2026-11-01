"""Step 3: does wrk's receiver_balance = lognormal(11, 2) reproduce the production distribution?
And what does PaySim training data look like per type (zero destination balance, merchant receivers)?"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from src.data_preprocessing import chronological_split, load_data
from src.drift_detection import make_groups, grouped, psi, ks_distance
from src.monitoring_features import log_bins
from pathlib import Path
HERE = Path(__file__).parent
w = json.load(open(HERE / "scan_result.json"))["classes"]["loadtest(wrk-)"]
prod = {int(k): v for k, v in w["dest_bin"].items() if k not in ("None", "null")}
rng = np.random.default_rng(0)
sim = log_bins(np.round(rng.lognormal(11, 2, 2_000_000), 2)).astype(int)
v, c = np.unique(sim, return_counts=True)
gen = dict(zip(v.tolist(), c.tolist()))
g = make_groups(gen, 10)
print(f"generator lognormal(11,2) vs production: PSI {psi(grouped(gen, g), grouped(prod, g)):.5f}  KS {ks_distance(gen, prod):.5f}")
train, _, _, _ = chronological_split(load_data("data/fraud_data.csv"))
t = train.assign(zero=train["oldbalanceDest"] == 0, merchant=train["nameDest"].str.startswith("M"))
print(t.groupby("type").agg(rows=("zero", "size"), zero_dest_share=("zero", "mean"), merchant_share=("merchant", "mean")).round(4).to_string())
print("zero dest share overall", round(t["zero"].mean(), 4), "| zero among merchant receivers", round(t.loc[t["merchant"], "zero"].mean(), 4),
      "| zero among customer receivers", round(t.loc[~t["merchant"], "zero"].mean(), 4))
