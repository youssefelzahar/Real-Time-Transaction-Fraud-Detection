# Real-Time-Transaction-Fraud-Detection
The system receives a transaction, estimates its fraud risk, and helps the business decide whether to approve it, reject it, or send it for manual review.  The important business trade-off is that missing fraudulent transactions costs money, but incorrectly blocking legitimate customers also costs money.  Your system should balance both.


## How to run

Step-by-step commands for everything (setup, training, tests, API, load balancer, load and failure tests): **[RUN.md](RUN.md)**.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train E5 (baseline), E6 and E7 on the same chronological split; writes
# models/e5_v1/, models/e6_v1/, models/e7_v1/ and models/comparison_v1.md
python -m src.train --data data/fraud_data.csv --version v1
#   --max-amount 10000000   transaction cap used by is_max_amount
#   --cost-fn 100 --cost-fp 1   business cost used to pick the threshold

pytest                                        # unit + API tests (no dataset needed)

MODEL_DIR=models/e6_v1 ACCOUNT_SNAPSHOT=balances.json uvicorn api.main:app --reload
```

`ACCOUNT_SNAPSHOT` is optional: a JSON file `{"account_id": balance}` used by the local `InMemoryAccountService` stand-in.

## Pipeline

| File | Role |
|---|---|
| `src/data_preprocessing.py` | `chronological_split` and `FeaturePipeline`: the single feature function used by training **and** the API |
| `src/train.py` | Trains E5/E6/E7, picks the threshold on validation, evaluates once on test, runs shortcut checks, saves artifacts + report |
| `src/evaluate.py` | Cost function, threshold selection, metrics, leakage/shortcut checks |
| `src/predict.py` | `FraudPredictor`: loads artifacts, checks the model and pipeline schemas match, scores records |
| `src/decision_engine.py` | Probability to `approve` / `review` |
| `api/main.py` | FastAPI `/predict` and `/health` |

Chronological split: train steps 1-280, validation 281-354, test 355-743. Business cost = `100 x FN + 1 x FP`.

### Features (no `isFraud`, `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`)

| Feature | Definition | E5 | E6 | E7 |
|---|---|---|---|---|
| `amount`, `oldbalanceOrg`, `oldbalanceDest` | raw pre-transaction values | ✓ | ✓ | ✓ |
| `amt_to_orig_bal` | amount / oldbalanceOrg, NaN if balance is 0 or missing | | ✓ | |
| `exceeds_orig_bal` | amount > oldbalanceOrg | ✓ | ✓ | ✓ |
| `orig_bal_zero` / `dest_bal_zero` | balance == 0 | ✓ | ✓ | ✓ |
| `amt_to_dest_bal` | amount / oldbalanceDest | ✓ | | |
| `is_max_amount` | amount >= `max_amount` (configurable, default 10,000,000) | ✓ | ✓ | ✓ |
| `amt_vs_type_median` | amount / train-period median amount of its type | ✓ | ✓ | ✓ |
| `hour`, `type_*` | step % 24, one-hot type | ✓ | | |
| `type_code` | fixed mapping CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4 (unseen -> NaN), saved in `pipeline.json` | | | ✓ |

Missing inputs stay NaN (a flag is NaN when its input is unknown, not 0), and XGBoost routes NaN natively. An unseen type gives a NaN median ratio and all-zero `type_*` columns. All features are `float64` in a fixed order, which is saved in the artifacts.

### Artifacts (`models/<feature_set>_<version>/`)

- `model.json`: the XGBoost model
- `pipeline.json`: type medians, `max_amount`, feature names/order, required inputs
- `metadata.json`: threshold, cost function, split boundaries, validation/test metrics, shortcut checks, feature importance, library versions

## E5 vs E6 vs E7 results (test period, threshold chosen on validation)

| Exp | Threshold | Precision | Recall | PR-AUC | FPR | FP | FN | Cost |
|---|---|---|---|---|---|---|---|---|
| E5 | 0.53 | 0.7013 | 0.9814 | 0.9711 | 0.00138 | 1,780 | 79 | 9,680 |
| E6 | 0.99 | 0.9988 | 0.9993 | 0.9998 | 0.00000 | 5 | 3 | 305 |
| **E7** | 0.54 | 0.7163 | 0.9880 | 0.9682 | 0.00129 | 1,666 | 51 | **6,766** |

**E6's score is a simulator shortcut, not a real-world estimate.** `amt_to_orig_bal == 1` (sender's balance fully drained) holds for 99.1% of validation fraud and 0.00% of non-fraud. That ratio alone reaches PR-AUC 0.992, and validation PR-AUC is 1.0000. This is how PaySim's fraud agent behaves, and real fraud will not follow it this cleanly. E5 deliberately excludes this signal. Use E6 for PaySim scoring only, and don't quote its numbers as expected production performance.

**E7** = E6 without `amt_to_orig_bal`, plus `type_code`. It has no shortcut warnings and is the best realistic model: test cost is 30% lower than E5 (6,766 vs 9,680), with fewer missed frauds (51 vs 79) and fewer false alerts (1,666 vs 1,780). Its PR-AUC is slightly lower (0.968 vs 0.971 on test, 0.892 vs 0.920 on validation), so the gain comes mainly at the chosen operating threshold rather than in ranking overall.

## API contract

The client sends only transaction fields. Pre-transaction balances are **not** accepted from the client: the server fetches them through an `AccountService` (`get_balance(account_id)`). In production that is the core-banking/account service; locally it is `InMemoryAccountService`.

| Field | Source |
|---|---|
| `transaction_id` (optional), `type`, `amount`, `sender_id`, `receiver_id` | request |
| `step` | request, only for E5 models (`hour`) |
| `oldbalanceOrg`, `oldbalanceDest` | account service, looked up by sender/receiver ID (source of truth) |
| `sender_balance`, `receiver_balance` (optional) | request: fallback only when the account service has no balance for that account |
| type medians, `max_amount`, threshold | model artifacts |

Unknown accounts are still scored, and the affected features are listed in `missing_features`. `balance_source` in the response shows where each balance came from (`account_service`, `request` or `missing`). Swagger (`/docs`) has ready-made request examples in the **Examples** dropdown of `POST /predict`.

```http
POST /predict
{"transaction_id": "tx-001", "type": "TRANSFER", "amount": 181.0,
 "sender_id": "C1305486145", "receiver_id": "C553264065"}
```

```json
{"transaction_id": "tx-001", "fraud_probability": 0.99999988, "decision": "review",
 "threshold": 0.99, "model_version": "e6_v1", "missing_features": []}
```

A merchant receiver with no balance on record:

```json
{"transaction_id": "tx-002", "fraud_probability": 1.45e-08, "decision": "approve",
 "threshold": 0.99, "model_version": "e6_v1", "missing_features": ["oldbalanceDest", "dest_bal_zero"]}
```

## Deployment: Nginx load balancer + two API instances

```
Client / Payment Service
          |
          v
   Nginx :8080  (round-robin, max_conns=64 per instance, 503 when saturated)
       |                    |
       v                    v
 Instance A :8001     Instance B :8002      (FastAPI + uvicorn, 1 worker each)
       |                    |
       +--------+-----------+
                v
   FraudPredictor (same models/e7_v1 artifacts) -> decision response
```

### Workers vs instances vs load balancer

| | What it is | What it gives you |
|---|---|---|
| **uvicorn `--workers N`** | N processes behind one uvicorn supervisor, sharing one port | Uses N CPU cores. Still one service: one config, one start/stop, one machine |
| **Multiple instances** | Independent uvicorn processes on separate ports (A, B) | Each can be stopped, restarted or upgraded on its own |
| **Reverse proxy / load balancer (Nginx)** | One public port in front of the instances | Spreads requests, skips failed instances, sheds load when saturated, and is the place to add more instances or machines later |

The instances share no mutable state: each loads the same read-only artifacts (`MODEL_DIR`) and its own copy of the account snapshot (`ACCOUNT_SNAPSHOT`). Any instance can serve any request.

### Run

```bash
sudo apt install nginx      # once; the system nginx is not used (ours runs as a normal user)
deploy/start.sh             # instance A, instance B, then `nginx -t` and Nginx on :8080
deploy/stop.sh              # everything
deploy/stop.sh a            # one part only: a | b | nginx  (same for start.sh)

curl localhost:8080/health
# Options (env): MODEL_DIR=models/e7_v1  WORKERS_PER_INSTANCE=1  ACCOUNT_SNAPSHOT=...
#                CPUS_A=0-1 CPUS_B=2-3 CPUS_NGINX=4-5   (taskset pinning, used for benchmarks)
```

PID files, logs and Nginx temp files are written to `deploy/run/` (gitignored). The Nginx access log records which instance served each request (`upstream=...`), so routing can be checked without adding fields to the API response.

### Nginx behavior (`deploy/nginx/nginx.conf`)

- **Balancing:** round-robin, with keepalive connections to the instances.
- **Headers:** `Host`, `X-Real-IP`, `X-Forwarded-For` and `X-Forwarded-Proto` are forwarded.
- **Timeouts:** connect 1 s, send/read 2 s.
- **Retries:** only on connection errors (`proxy_next_upstream error`), so only when the request never reached an instance. `non_idempotent` is not set and timeouts are not retried, so a delivered transaction is never scored twice.
- **Health checks are passive only.** Open-source Nginx never calls `/health` itself. After 3 failures (errors or timeouts) within 5 s, an instance is skipped for 5 s, then tried again.
- **Load shedding:** at most 64 in-flight requests per instance (`max_conns`). Above that, Nginx answers `503 {"detail":"fraud scoring temporarily unavailable"}` with `Retry-After: 1`, so the payment service can apply its fallback.

Why load shedding: without it, a benchmark at 200 concurrent requests made the single-worker instances queue unfairly (p99 4–6 s). Requests hit the 2 s timeout, the timeouts disabled both instances, and **~92% of requests failed** with `no live upstreams`, while the instances sat mostly idle.

### Benchmark (same machine; `wrk -t4 -d30s`, `tests/wrk_predict.lua`)

A on CPUs 0–1, B on 2–3, Nginx on 4–5, `wrk` on 8–15 (separate physical cores). One worker per instance.

| Target | Concurrency | req/s | p50 | p95 | p99 | max | Errors | CPU |
|---|---|---|---|---|---|---|---|---|
| Instance A direct | 100 | 2,868 | 34.0 ms | 36.9 ms | 323.8 ms | 1,638 ms | 0 | A 100% |
| Instance B direct | 100 | 2,881 | 33.9 ms | 35.7 ms | 302.2 ms | 1,645 ms | 0 | B 100% |
| **Nginx → A+B** | 100 | **5,311** | 18.4 ms | 38.1 ms | 70.0 ms | 413 ms | 0 | A 100%, B 100%, Nginx 31% |

Under overload (accepted requests only, from the Nginx log, 20 s runs):

| Concurrency | Accepted req/s | Accepted p50 / p95 / p99 | Shed (503) | Timeouts (504) |
|---|---|---|---|---|
| 100 | 5,072 | 19 / 36 / 55 ms | 1 | 0 |
| 200 | 4,504 | 24 / 51 / 97 ms | 814,382 | 0 |
| 400 | 4,159 | 24 / 67 / 133 ms | 696,430 | 0 |

`wrk` resends immediately after every 503, so the shed count shows how fast rejections are, not real traffic. The meaningful numbers are the accepted throughput and latency.

Memory: ~196 MB per instance, ~16 MB for Nginx.

Two instances nearly double throughput (2,870 to 5,310 req/s), and through Nginx the p99 is lower than hitting one instance directly. The proxy itself used about a third of a core. This setup has less capacity than the 8-worker single service measured earlier (~13,000 req/s), because it uses 2 cores instead of 8. Capacity grows by adding instances, not by the proxy.

### Failure tests (`wrk -c20`, 20 s, Nginx log per second)

| Scenario | Observed |
|---|---|
| Both healthy | Even split between A and B |
| A stopped gracefully (SIGTERM) under load | **18 failed requests (0.02%)**, in flight on A's connections when it shut down; not retried because they had been delivered. All traffic to B within the same second |
| B crashed (SIGKILL) under load | **20 failed requests (0.03%)**, same cause; all traffic to A immediately |
| Instance restarted | Healthy ~1.3 s after start. Traffic back **~2.5 s later**, when the 5 s `fail_timeout` expired (one probe request was refused and retried on the other instance) |
| Both instances down | Immediate `503` JSON (~1 ms after the first attempt); `/health` through Nginx returns 503 |
| Both restarted | Healthy in 2.6 s; requests succeed right away, round-robin resumes |

**Not zero-downtime:** stopping or losing an instance drops the requests in flight on it. A rolling restart without errors would need draining first (mark the server `down`, `nginx -s reload`, wait, then stop).

### Limitations

- **One machine.** Nginx and both instances share one host, so this is a single point of failure: no real high availability, and no basis for a 99.9% availability claim.
- **Nginx itself is not redundant.** If it stops, the service is down.
- **Passive health checks only.** A hung instance that accepts connections but never answers is only detected through timeouts.
- **Client and server ran on the same machine**, pinned to separate cores. Numbers need confirming with the load generator on separate hardware.
- `docker-compose.yml` is still empty; Docker was not usable in this environment, so a compose setup could not be verified.

### Tests

`tests/test_deploy_config.py` checks the config statically: both upstreams with `max_conns`, no `non_idempotent`, retry only on `error`, explicit timeouts, forwarded headers and the 503 page. It runs `nginx -t` when Nginx is installed and checks the scripts parse. An opt-in live test starts the stack and checks that both instances receive traffic and give identical predictions:

```bash
RUN_DEPLOY_TESTS=1 pytest tests/test_deploy_config.py
```
