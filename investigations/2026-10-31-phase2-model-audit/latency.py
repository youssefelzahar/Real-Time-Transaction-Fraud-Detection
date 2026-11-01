"""Offline single-request scoring latency per saved artifact (FraudPredictor.score, the API hot path:
transform_one + booster.inplace_predict, 1 thread). Not end-to-end: no HTTP, Nginx, decision engine or Kafka.

    TIMING_LOG=0 taskset -c 12 python investigations/2026-10-31-phase2-model-audit/latency.py
"""
import os
import platform
from time import perf_counter

os.environ.setdefault("TIMING_LOG", "0")

import numpy as np  # noqa: E402

from common import ROOT, load_splits, write_json  # noqa: E402
from src.predict import FraudPredictor  # noqa: E402

ARTIFACTS = ("e5_v1", "e6_v1", "e7_v1")
N, WARMUP = 5000, 500


def main():
    _, _, _, test, _ = load_splits()
    rows = test.sample(N, random_state=0)
    out = {"what": "FraudPredictor.score() per request, 1 thread, real test-period rows",
           "cpu_affinity": sorted(os.sched_getaffinity(0)), "machine": platform.processor() or platform.machine(),
           "n": N}
    for name in ARTIFACTS:
        pred = FraudPredictor.load(ROOT / "models" / name)
        records = rows[pred.pipeline.required_inputs].to_dict("records")
        for r in records[:WARMUP]:
            pred.score(r)
        times = np.empty(N)
        for i, r in enumerate(records):
            t0 = perf_counter()
            pred.score(r)
            times[i] = (perf_counter() - t0) * 1e6
        out[name] = {f"p{q}_us": float(np.percentile(times, q)) for q in (50, 95, 99)}
        print(name, {k: round(v, 1) for k, v in out[name].items()})
    write_json("latency_results.json", out)


if __name__ == "__main__":
    main()
