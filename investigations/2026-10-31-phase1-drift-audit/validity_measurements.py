"""Phase 1 evidence: false-alarm rate under no drift and detection rate under known drift, at the
configured min_samples (Monte Carlo, fixed seeds; same generators as tests/test_drift_audit.py).

    python investigations/2026-10-31-phase1-drift-audit/validity_measurements.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import drift_detection as dd  # noqa: E402
from tests.test_drift_audit import CFG, sample_stats  # noqa: E402
from tests.test_drift_detection import baseline  # noqa: E402

n, reps = CFG["min_samples"], 1000
rng = np.random.default_rng(10)
flagged, psi = {}, {}
for _ in range(reps):
    out = dd.evaluate_window(sample_stats(rng, n), baseline(), CFG)
    for r in out["features"]:
        flagged[r["feature"]] = flagged.get(r["feature"], 0) + (r["status"] == dd.DRIFT)
        psi.setdefault(r["feature"], []).append(r["psi"])
    flagged["prediction"] = flagged.get("prediction", 0) + (out["prediction"]["status"] == dd.DRIFT)
    psi.setdefault("prediction", []).append(out["prediction"]["psi"])
print(f"NO DRIFT (n={n} per window, {reps} windows): false-alarm rate and PSI mean / p99")
for f in flagged:
    print(f"  {f:<18} false alarms {flagged[f] / reps:.3%}   PSI mean {np.mean(psi[f]):.4f}  p99 {np.quantile(psi[f], .99):.4f}")

shifted_types = {"CASH_IN": 1, "CASH_OUT": 1, "DEBIT": 1, "PAYMENT": 1, "TRANSFER": 6}
print(f"KNOWN DRIFT (n={n}, 300 windows each): detection rate")
for name, kw, feature in (("amount x10 (+4 bins)", {"amount_shift": 4}, "amount"),
                          ("amount x1.8 (+1 bin)", {"amount_shift": 1}, "amount"),
                          ("20% missing amount", {"missing": n // 5}, "amount"),
                          ("type mix, TRANSFER 60%", {"types": shifted_types}, "transaction_type")):
    rng = np.random.default_rng(11)
    hits = sum(next(r for r in dd.evaluate_window(sample_stats(rng, n, **kw), baseline(), CFG)["features"]
                    if r["feature"] == feature)["status"] == dd.DRIFT for _ in range(300))
    print(f"  {name:<24} detected {hits / 300:.1%}")
