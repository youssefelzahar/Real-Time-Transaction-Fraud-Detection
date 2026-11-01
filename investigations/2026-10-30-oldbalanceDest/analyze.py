"""Steps 1-4 from the scan counters + baseline bl-e7_v1-2 (reference = train split)."""
import json, sys
sys.path.insert(0, ".")
from src.drift_detection import psi, chi_square_gof, make_groups, grouped, ks_distance
from src.monitoring_features import bin_lower_edge
from pathlib import Path
HERE = Path(__file__).parent
scan = json.load(open(HERE / "scan_result.json"))
b = json.load(open("monitoring/baselines/e7_v1/bl-e7_v1-2.json"))
C = scan["classes"]
print("== classes:", {k: v["n"] for k, v in C.items()}, "| other ids:", scan["other_ids_sample"], "| bad lines:", scan["bad_lines"])
hrs = {}
for h, c, n in scan["hours"]:
    hrs.setdefault(c, []).append(h)
for c, hs in hrs.items():
    print(f"   {c}: first hour {hs[0]}, last hour {hs[-1]}, active hours {len(hs)}")
w = C["loadtest(wrk-)"]
print("\n== step 1 quality (wrk):", "no_summary", w["no_summary"], "missing", w["missing"], "non_finite", w["non_finite"],
      "| models", w["model"], "| receiver src", w["recv_src"], "| sender src", w["send_src"])
for c in C:
    if c != "loadtest(wrk-)":
        print("   ", c, "receiver src", C[c]["recv_src"], "models", C[c]["model"])
ref = {int(k): v for k, v in b["bin_counts"]["oldbalanceDest"].items()}
prod = {int(k): v for k, v in w["dest_bin"].items() if k not in ("None", "null")}
nr, npd = sum(ref.values()), sum(prod.values())
def q(d, p):
    n, cum = sum(d.values()), 0
    for k in sorted(d):
        cum += d[k]
        if cum >= p * n: return k
print("\n== step 3 oldbalanceDest (fine log bins, value >= 10^(bin/4)-1)")
print(f"   zero share  ref {ref.get(0,0)/nr:.4f}  prod {prod.get(0,0)/npd:.6f}")
for p in (0.25, 0.5, 0.75, 0.95):
    rb, pb = q(ref, p), q(prod, p)
    print(f"   p{int(p*100):<3} bin ref {rb} (>= {bin_lower_edge(rb):,.0f})  prod {pb} (>= {bin_lower_edge(pb):,.0f})")
print(f"   range ref {min(ref)}..{max(ref)}  prod {min(prod)}..{max(prod)}")
print("   ref non-zero dist p50 bin:", q({k: v for k, v in ref.items() if k > 0}, 0.5))
g = make_groups(ref, 10)
print(f"   PSI all wrk events: {psi(grouped(ref, g), grouped(prod, g)):.4f}  KS distance {ks_distance(ref, prod):.4f}")
refz = {k: v for k, v in ref.items() if k > 0}; prodz = {k: v for k, v in prod.items() if k > 0}
g2 = make_groups(refz, 10)
print(f"   PSI on non-zero balances only: {psi(grouped(refz, g2), grouped(prodz, g2)):.4f}  KS {ks_distance(refz, prodz):.4f}")
print("   dest_bal_zero flag ref", round(b["flag_counts"]["dest_bal_zero"]["1"] / sum(b["flag_counts"]["dest_bal_zero"].values()), 4),
      "prod", w["dest_zero"])
print("   zero share by type (prod wrk):", {t: round(c.get("0", 0) / sum(c.values()), 4) for t, c in w["dest_bin_by_type"].items()})
print("   type mix ref", {t: round(v, 4) for t, v in b["categorical"]["transaction_type"]["proportions"].items()})
print("   type mix prod", {t: round(c / w["n"], 4) for t, c in w["type"].items()})
# reference zero share per type requires the training data: done in step3_by_type.py
print("\n== step 4 prediction drift (wrk vs validation reference)")
rh = {i: c for i, c in enumerate(b["reference_scores"]["score_hist"])}
ph = {int(k): v for k, v in w["score_bin"].items()}
g3 = make_groups(rh, 10)
print(f"   score PSI {psi(grouped(rh, g3), grouped(ph, g3)):.4f}  chi2 {chi_square_gof(grouped(ph, g3), grouped(rh, g3))}")
dec = w["decision"]; n = w["n"]
print("   decisions prod", {k: round(v / n, 5) for k, v in dec.items()}, " ref", {k: round(v, 5) for k, v in b["reference_scores"]["decision_rates"].items()})
print("   ref would_decline", round(b["reference_scores"]["would_decline_rate"], 6))
