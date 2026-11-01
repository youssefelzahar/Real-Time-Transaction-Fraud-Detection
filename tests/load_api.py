"""Load test for POST /predict (not collected by pytest).

Start the API first, e.g.:
    MODEL_DIR=models/e7_v1 uvicorn api.main:app --workers 4
Then:
    python tests/load_api.py --url http://127.0.0.1:8000 --requests 2000 --concurrency 1 10 50 100 --processes 8

A single Python client process tops out around 170 req/s, so the load is spread over
--processes client processes; otherwise the client, not the API, is what gets measured.
"""
import argparse
import asyncio
import random
import time
from concurrent.futures import ProcessPoolExecutor

import httpx
import numpy as np

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def random_payload(i, rng):
    balance = round(rng.lognormvariate(10, 2), 2)
    return {"transaction_id": f"load-{i}", "type": rng.choice(TYPES),
            "amount": round(rng.lognormvariate(10, 1.5), 2),
            "sender_id": f"C{rng.randrange(10**9)}", "receiver_id": f"C{rng.randrange(10**9)}",
            "sender_balance": rng.choice([0.0, balance]), "receiver_balance": round(rng.lognormvariate(11, 2), 2)}


async def send_batch(url, n_requests, concurrency, seed):
    """One client process: n_requests with `concurrency` in flight. Returns latencies (ms), errors."""
    rng = random.Random(seed)
    payloads = [random_payload(i, rng) for i in range(n_requests)]
    latencies, errors = [], 0
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        nonlocal errors
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"{url}/predict", json=p, headers={"X-Traffic-Source": "loadtest"})
                if r.status_code != 200:
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            latencies.append((time.perf_counter() - t0) * 1000)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        await asyncio.gather(*(one(p) for p in payloads[:min(20, n_requests)]))  # warm-up
        latencies, errors = [], 0
        await asyncio.gather(*(one(p) for p in payloads))
    return latencies, errors


def _worker(args):
    return asyncio.run(send_batch(*args))


def run_level(url, n_requests, concurrency, processes, seed=42):
    processes = max(1, min(processes, concurrency))
    jobs = [(url, n_requests // processes, max(1, concurrency // processes), seed + i) for i in range(processes)]
    start = time.perf_counter()
    with ProcessPoolExecutor(processes) as pool:
        results = list(pool.map(_worker, jobs))
    elapsed = time.perf_counter() - start
    lat = np.concatenate([np.array(r[0]) for r in results])
    errors = sum(r[1] for r in results)
    n_requests = len(lat)
    return {"concurrency": concurrency, "requests": n_requests, "errors": errors,
            "rps": n_requests / elapsed, "p50": np.percentile(lat, 50),
            "p95": np.percentile(lat, 95), "p99": np.percentile(lat, 99), "max": lat.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 50, 100])
    ap.add_argument("--processes", type=int, default=8, help="client processes generating load")
    ap.add_argument("--p95-target-ms", type=float, default=100.0)
    args = ap.parse_args()

    print(f"{'conc':>5} {'reqs':>6} {'errors':>6} {'req/s':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8} {'max ms':>8}  p95<target")
    for c in args.concurrency:
        r = run_level(args.url, args.requests, c, args.processes)
        ok = "yes" if r["p95"] <= args.p95_target_ms and r["errors"] == 0 else "NO"
        print(f"{r['concurrency']:>5} {r['requests']:>6} {r['errors']:>6} {r['rps']:>8.0f} {r['p50']:>8.1f} "
              f"{r['p95']:>8.1f} {r['p99']:>8.1f} {r['max']:>8.1f}  {ok}")


if __name__ == "__main__":
    main()
