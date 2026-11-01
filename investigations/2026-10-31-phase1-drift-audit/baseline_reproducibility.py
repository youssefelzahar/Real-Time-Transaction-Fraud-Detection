"""Phase 1 evidence: the stored training baseline can be regenerated bit-for-bit (except its
creation time), it comes from the train split only, and it belongs to the served model.

    python investigations/2026-10-31-phase1-drift-audit/baseline_reproducibility.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
stored_path = ROOT / "monitoring/baselines/e7_v1/bl-e7_v1-2.json"
stored = json.loads(stored_path.read_text())

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, "-m", "src.monitoring_baseline", "--model-dir", "models/e7_v1",
                    "--data", "data/fraud_data.csv", "--version", "2", "--out-dir", tmp],
                   cwd=ROOT, check=True, capture_output=True)
    fresh = json.loads((Path(tmp) / "e7_v1" / "bl-e7_v1-2.json").read_text())
diff = sorted(k for k in set(stored) | set(fresh) if k != "created_at" and stored.get(k) != fresh.get(k))
print("regenerated vs stored, keys that differ (created_at ignored):", diff or "none")

sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
for f in ("model.json", "pipeline.json", "metadata.json"):
    print(f"artifact {f}: {'match' if stored['artifacts'][f] == sha('models/e7_v1/' + f) else 'MISMATCH'}")
print("feature_pipeline:", stored["feature_pipeline"]["id"],
      "match" if stored["feature_pipeline"]["sha256"] == sha("models/e7_v1/pipeline.json") else "MISMATCH")
print("split:", stored["split"])

# Split facts, recomputed from the CSV (no reuse of recorded numbers).
df = pd.read_csv(ROOT / "data/fraud_data.csv", usecols=["step", "isFraud"])
meta = json.loads((ROOT / "models/e7_v1/metadata.json").read_text())
for name, (lo, hi) in meta["split_boundaries"].items():
    part = df[(df.step >= lo) & (df.step <= hi)]
    print(f"{name:<5} steps {lo}-{hi}: rows {len(part):,}  fraud {int(part.isFraud.sum()):,}  "
          f"rate {part.isFraud.mean():.6f}  (metadata: {meta['fraud_rates'][name]:.6f})")
