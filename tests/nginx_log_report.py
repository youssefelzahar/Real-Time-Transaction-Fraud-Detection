"""Summarize the Nginx access log per instance (read-only diagnostics).

    python tests/nginx_log_report.py --last 30            # requests from the last 30 s
    python tests/nginx_log_report.py --last 60 --timeline  # plus requests/s per instance
    python tests/nginx_log_report.py --since-offset 12345  # from a byte offset (scripts)

Columns:
  share          fraction of /predict requests served by the instance
  nginx p95      total time inside Nginx for the request (rt)
  inst p50/p95   time the instance took (upstream_rt)
  wait p95       time spent in Nginx outside the instance (rt - upstream_rt): queueing,
                 retries, client-side delays. Near 0 = Nginx is not the bottleneck.
  new conns      requests that needed a new TCP connection (upstream_ct > 0); high
                 values mean the keepalive pool is not being reused.
  retried        requests delivered after a connection error on another instance.
"""
import argparse
import collections
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

LINE = re.compile(
    r"\[(?P<ts>[^\]]+)\] \"(?P<method>\S+) (?P<path>\S+)[^\"]*\" (?P<status>\d{3}) rt=(?P<rt>[\d.]+) "
    r"upstream=(?P<up>.*?) upstream_status=(?P<ust>.*?) upstream_rt=(?P<urt>.*?)(?: upstream_ct=(?P<uct>.*))?$")


def instance_name(addr):
    port = addr.rsplit(":", 1)[-1]
    return f"instance {int(port) - 8000}" if port.isdigit() and 8001 <= int(port) <= 8099 else addr


def last_value(field):
    """Nginx logs one value per attempt ('a, b'); the last is the one that answered."""
    v = field.split(",")[-1].strip() if field else "-"
    return None if v in ("-", "") else float(v)


def parse(path, since_ts=None, offset=0):
    with open(path) as f:
        f.seek(offset)
        for line in f:
            m = LINE.search(line)
            if not m or m["path"] != "/predict":
                continue
            ts = datetime.strptime(m["ts"], "%d/%b/%Y:%H:%M:%S %z").timestamp()
            if since_ts and ts < since_ts:
                continue
            ups = [u.strip() for u in m["up"].split(",")]
            yield {"ts": int(ts), "status": m["status"], "rt": float(m["rt"]) * 1000,
                   "up": ups[-1], "retried": len(ups) > 1,
                   "urt": (last_value(m["urt"]) or 0) * 1000, "uct": last_value(m["uct"] or "-")}


def pct(values, q):
    return f"{np.percentile(values, q):.1f}" if values else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="deploy/run/access.log")
    ap.add_argument("--last", type=float, help="only requests from the last N seconds")
    ap.add_argument("--since-offset", type=int, default=0, help="start reading at this byte offset")
    ap.add_argument("--timeline", action="store_true", help="requests per second per instance")
    args = ap.parse_args()
    if not Path(args.log).exists():
        raise SystemExit(f"{args.log} not found: is Nginx running (deploy/start.sh)?")

    since = time.time() - args.last if args.last else None
    rows = list(parse(args.log, since, args.since_offset))
    if not rows:
        raise SystemExit("no POST /predict requests in the selected window")

    span = max(r["ts"] for r in rows) - min(r["ts"] for r in rows) + 1
    statuses = collections.Counter(r["status"] for r in rows)
    print(f"{len(rows):,} requests over ~{span}s ({len(rows) / span:,.0f}/s) | statuses: "
          + ", ".join(f"{k}={v:,}" for k, v in sorted(statuses.items())))

    by_up = collections.defaultdict(list)
    for r in rows:
        by_up[r["up"]].append(r)
    print(f"\n{'upstream':<14} {'requests':>9} {'share':>6} {'non-200':>8} {'nginx p95':>10} {'inst p50':>9} "
          f"{'inst p95':>9} {'inst p99':>9} {'wait p95':>9} {'new conns':>10} {'retried':>8}")
    for up in sorted(by_up, key=lambda u: (not u.startswith("127."), u)):
        rs = by_up[up]
        ok = [r for r in rs if r["status"] == "200"]
        total = [r["rt"] for r in ok]
        inst = [r["urt"] for r in ok]
        wait = [max(r["rt"] - r["urt"], 0) for r in ok]
        new = sum(1 for r in rs if r["uct"] and r["uct"] > 0)
        print(f"{instance_name(up):<14} {len(rs):>9,} {len(rs) / len(rows):>6.1%} "
              f"{sum(r['status'] != '200' for r in rs):>8,} {pct(total, 95):>10} {pct(inst, 50):>9} {pct(inst, 95):>9} "
              f"{pct(inst, 99):>9} {pct(wait, 95):>9} {new:>10,} {sum(r['retried'] for r in rs):>8,}")
    print("(times in ms; 'fraud_api' = answered by Nginx itself, e.g. 503 when no instance was available)")

    if args.timeline:
        names = sorted({r["up"] for r in rows if r["up"].startswith("127.")})
        t0 = min(r["ts"] for r in rows)
        per_sec = collections.defaultdict(collections.Counter)
        for r in rows:
            per_sec[r["ts"] - t0][r["up"] if r["status"] == "200" else f"{r['status']}"] += 1
        print("\n  sec " + "".join(f"{instance_name(n).replace('instance ', 'i'):>7}" for n in names) + "   other")
        for sec in sorted(per_sec):
            c = per_sec[sec]
            other = " ".join(f"{k}={v}" for k, v in c.items() if k not in names)
            print(f"  {sec:>3} " + "".join(f"{c.get(n, 0):>7}" for n in names) + f"   {other}")


if __name__ == "__main__":
    main()
