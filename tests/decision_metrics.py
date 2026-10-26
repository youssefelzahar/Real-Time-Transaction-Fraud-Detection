"""Sum the decision metrics of every running instance (read-only).

    python tests/decision_metrics.py

Reads the instance ports from deploy/nginx/nginx.conf and calls GET /metrics/decisions
on each instance directly (127.0.0.1:800N); the endpoint is blocked on the public :8080.
Counters are per process and reset when an instance restarts.
"""
import re
from pathlib import Path

import httpx

CONF = Path(__file__).resolve().parents[1] / "deploy" / "nginx" / "nginx.conf"


def instance_ports():
    return [int(p) for p in re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", CONF.read_text(), flags=re.M)]


def main():
    totals = {"total": 0, "would_decline": 0, "invalid_scores": 0}
    by_decision, by_reason, policies = {}, {}, set()
    lat_count, lat_sum, lat_max = 0, 0.0, 0.0
    print(f"{'instance':<10} {'total':>8} {'approve':>8} {'review':>8} {'decline':>8} {'would_decl':>11} "
          f"{'mean_us':>8} {'policy'}")
    for n, port in enumerate(instance_ports(), 1):
        try:
            m = httpx.get(f"http://127.0.0.1:{port}/metrics/decisions", timeout=2).json()
        except httpx.HTTPError as e:
            print(f"{'i' + str(n):<10} unreachable ({type(e).__name__})")
            continue
        d, lat = m["by_decision"], m["decision_latency_us"]
        print(f"{'i' + str(n):<10} {m['total']:>8,} {d['approve']:>8,} {d['review']:>8,} {d['decline']:>8,} "
              f"{m['would_decline']:>11,} {lat['mean']:>8} {m['policy_version']} ({m['decline_mode']})")
        for k in totals:
            totals[k] += m[k]
        for src, dst in ((d, by_decision), (m["by_reason"], by_reason)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0) + v
        lat_count += lat["count"]
        lat_sum += lat["mean"] * lat["count"]
        lat_max = max(lat_max, lat["max"])
        policies.add(m["policy_version"])

    print(f"\nall instances: {totals['total']:,} decisions | {by_decision} | reasons {by_reason}")
    print(f"would_decline (shadow) {totals['would_decline']:,} | invalid scores {totals['invalid_scores']:,} | "
          f"decision latency mean {lat_sum / max(lat_count, 1):.1f} us, max {lat_max:.1f} us")
    if len(policies) > 1:
        print(f"WARNING: instances run different policies: {sorted(policies)}")


if __name__ == "__main__":
    main()
