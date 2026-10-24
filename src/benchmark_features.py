
from time import perf_counter
import statistics

import pandas as pd

from src.predict import FraudPredictor

predictor = FraudPredictor.load("models/e6_v1")

record = {
    "type": "TRANSFER",
    "amount": 25000.0,
    "oldbalanceOrg": 300000.0,
    "oldbalanceDest": 150000.0,
}

# Warm-up
for _ in range(100):
    predictor.score(record)

# Measure repeated single-request scoring
times_ms = []

for _ in range(2000):
    start = perf_counter()
    predictor.score(record)
    times_ms.append((perf_counter() - start) * 1000)

print("Current score() benchmark")
print(f"Median: {statistics.median(times_ms):.3f} ms")
print(
    "p95: "
    f"{sorted(times_ms)[int(0.95 * len(times_ms))]:.3f} ms"
)
