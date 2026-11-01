"""Phase 1 live check: replay real PaySim test-period rows (tagged `test`), a few untagged requests
(production) and one non-finite request, through Nginx. Requires the stack (see RUN.md).

    python investigations/2026-10-31-phase1-drift-audit/live_traffic.py
"""
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:8080/predict"
df = pd.read_csv(ROOT / "data/fraud_data.csv", usecols=["step", "type", "amount", "nameOrig", "oldbalanceOrg",
                                                        "nameDest", "oldbalanceDest"])
test = df[(df.step >= 355) & (df.step <= 743)].sample(1000, random_state=31)


def body(row, i, prefix):
    return {"transaction_id": f"{prefix}-{i}", "type": row.type, "amount": float(row.amount),
            "sender_id": "C-replay", "receiver_id": "M-replay" if row.nameDest.startswith("M") else "C-replay",
            "sender_balance": float(row.oldbalanceOrg), "receiver_balance": float(row.oldbalanceDest)}


codes = {}
with httpx.Client(timeout=5) as c:
    for i, row in enumerate(test.itertuples()):
        r = c.post(URL, json=body(row, i, "audit-replay"), headers={"X-Traffic-Source": "test"})
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
    for i, row in enumerate(test.head(5).itertuples()):
        r = c.post(URL, json=body(row, i, "audit-untagged"))
        codes[r.status_code] = codes.get(r.status_code, 0) + 1
    raw = json.dumps(body(next(test.itertuples()), 0, "audit-inf")).replace(
        f'"amount": {float(test.iloc[0].amount)}', '"amount": Infinity')
    r = c.post(URL, content=raw, headers={"Content-Type": "application/json", "X-Traffic-Source": "test"})
    print("Infinity amount ->", r.status_code, r.text[:110])
print("replay (tagged test) + untagged status codes:", codes)
sys.exit(0 if codes == {200: 1005} and r.status_code == 422 else 1)
