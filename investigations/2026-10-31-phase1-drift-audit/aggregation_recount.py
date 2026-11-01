"""Phase 1 evidence: recount the event store and compare with the consumer's monitoring state.

The consumer aggregates every stored event into ml_monitoring_state.json (cumulative per series
since `created_at`). If aggregation is correct, a recount of the store over the same period gives
the same counts per series and the same oldbalanceDest bin counts. Read-only; bounded memory.

    python investigations/2026-10-31-phase1-drift-audit/aggregation_recount.py
"""
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

STORE = os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events"))
state = json.load(open(os.path.join(STORE, "ml_monitoring_state.json")))
created = datetime.fromisoformat(state["created_at"]).timestamp()
saved = datetime.fromisoformat(state["saved_at"]).timestamp()

n, bins, before, after = Counter(), defaultdict(Counter), 0, 0
for path in sorted(glob.glob(os.path.join(STORE, "prediction-events-*.jsonl"))):
    with open(path, "rb") as f:
        for line in f:
            e = json.loads(line)
            t = datetime.fromisoformat(e["occurred_at"].replace("Z", "+00:00")).timestamp()
            if t < created:
                before += 1
                continue
            if t > saved:
                after += 1
                continue
            src = e.get("traffic_source") or "production"
            key = f"{e['model_version']}|{e['decision_policy_version']}" + ("" if src == "production" else f"|{src}")
            n[key] += 1
            b = (e.get("feature_bins") or {}).get("oldbalanceDest", "absent")
            bins[key]["null" if b is None else str(b)] += 1

print(f"state created {state['created_at']} saved {state['saved_at']}; store events before {before:,}, after {after:,}")
ok = True
for key in sorted(set(n) | set(state["cumulative"])):
    s = state["cumulative"].get(key, {})
    sb = s.get("bins", {}).get("oldbalanceDest", {})
    match_n = s.get("n", 0) == n[key]
    match_b = {k: v for k, v in sb.items()} == {k: v for k, v in bins[key].items() if k != "absent"}
    ok &= match_n and match_b
    print(f"{key:<45} state n={s.get('n', 0):>10,}  store n={n[key]:>10,}  n {'OK' if match_n else 'MISMATCH'}"
          f"  oldbalanceDest bins {'OK' if match_b else 'MISMATCH'}")
print("RESULT:", "aggregation matches the store exactly" if ok else "MISMATCH")
