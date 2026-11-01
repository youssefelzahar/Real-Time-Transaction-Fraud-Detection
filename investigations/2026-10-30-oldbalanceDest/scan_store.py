"""Steps 1-4: one bounded-memory pass over the event store. Counters only; no raw events kept."""
import json, sys, time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, ".")
from src.ml_monitoring import score_bin
paths = sys.argv[1:]
def cls(tx):
    if tx is None: return "no_id"
    for p, c in (("wrk-", "loadtest(wrk-)"), ("demo-", "dashboard demo(demo-)"), ("it-", "integration test(it-)"), ("dup-", "integration test(dup-)")):
        if tx.startswith(p): return c
    return "other"
agg = defaultdict(lambda: {"n": 0, "type": Counter(), "dest_bin": Counter(), "dest_zero": Counter(), "recv_src": Counter(),
                            "send_src": Counter(), "decision": Counter(), "score_bin": Counter(), "missing": 0,
                            "non_finite": 0, "no_summary": 0, "model": Counter(), "dest_bin_by_type": defaultdict(Counter)})
hours = Counter(); others = Counter(); bad = 0; t0 = time.time()
for path in paths:
    with open(path, "rb") as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                bad += 1; continue
            c = cls(e.get("transaction_id"))
            a = agg[c]; a["n"] += 1
            t = e.get("transaction_type"); a["type"][t] += 1
            a["model"][e.get("model_version")] += 1
            bins = e.get("feature_bins"); flags = e.get("feature_flags")
            if bins is None: a["no_summary"] += 1
            else:
                b = bins.get("oldbalanceDest"); a["dest_bin"][b] += 1; a["dest_bin_by_type"][t][b] += 1
            if flags is not None: a["dest_zero"][flags.get("dest_bal_zero")] += 1
            bs = e.get("balance_source") or {}
            a["recv_src"][bs.get("receiver")] += 1; a["send_src"][bs.get("sender")] += 1
            a["decision"][e.get("decision")] += 1
            a["score_bin"][score_bin(float(e["fraud_probability"]))] += 1
            a["missing"] += bool(e.get("missing_features")); a["non_finite"] += bool(e.get("non_finite_features"))
            hours[(e["occurred_at"][:13], c)] += 1
            if c == "other" and len(others) < 50: others[e.get("transaction_id")] += 1
out = {"files": paths, "bad_lines": bad, "seconds": round(time.time() - t0, 1),
       "classes": {c: {k: (dict(v) if isinstance(v, Counter) else ({kk: dict(vv) for kk, vv in v.items()} if isinstance(v, defaultdict) else v))
                       for k, v in a.items()} for c, a in agg.items()},
       "hours": [[h, c, n] for (h, c), n in sorted(hours.items())], "other_ids_sample": dict(others)}
json.dump(out, open(Path(__file__).with_name("scan_result.json"), "w"), default=str)
print("done", out["seconds"], "s", {c: a["n"] for c, a in agg.items()})
