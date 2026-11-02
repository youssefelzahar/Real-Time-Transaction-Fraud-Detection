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
| `src/decision_engine.py` | Score to `approve` / `review` / `decline` with a configurable policy (`config/decision_policy.json`, see [DECISION_POLICY.md](DECISION_POLICY.md)) |
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
| type medians, `max_amount` | model artifacts |
| `approve_threshold`, `decline_threshold`, `decline_mode` | decision policy (`config/decision_policy.json`) |

Unknown accounts are still scored, and the affected features are listed in `missing_features`. `balance_source` in the response shows where each balance came from (`account_service`, `request` or `missing`). Swagger (`/docs`) has ready-made request examples in the **Examples** dropdown of `POST /predict`.

```http
POST /predict
{"transaction_id": "tx-001", "type": "TRANSFER", "amount": 181.0,
 "sender_id": "C1305486145", "receiver_id": "C553264065"}
```

```json
{"transaction_id": "tx-001", "fraud_probability": 0.9999, "decision": "review",
 "reason_code": "HIGH_RISK", "threshold": 0.54, "decision_policy_version": "dp-e7-2026-10-26.1",
 "model_version": "e7_v2", "missing_features": [],
 "balance_source": {"sender": "request", "receiver": "request"}}
```

| Field | Meaning |
|---|---|
| `fraud_probability` | Model risk score in [0, 1]. **Not calibrated**: read it as a score (see DECISION_POLICY.md) |
| `decision` | `approve`, `review` or `decline`. `decline` only appears when the policy's `decline_mode` is `enforce`; the current policy runs decline in **shadow** mode |
| `reason_code` | `LOW_RISK` (approve), `REVIEW_THRESHOLD` (review band), `HIGH_RISK` (decline, or review in shadow mode) |
| `threshold` | Unchanged meaning: scores at or above it are not auto-approved (the policy's `approve_threshold`) |
| `decision_policy_version` | Policy that made the decision |
| `model_version` | Model that produced the score |

A scoring failure returns `500 {"detail": "internal scoring error"}`. That's a technical error, not a decline, and should not be resent automatically.

## Deployment: Nginx load balancer + six API instances

```
                    Client / Payment Service
                              |
                              v
        Nginx :8080  (round-robin, max_conns=64 per instance, 503 when saturated)
   |         |         |         |         |         |
   v         v         v         v         v         v
 :8001     :8002     :8003     :8004     :8005     :8006     6 FastAPI instances,
 inst 1    inst 2    inst 3    inst 4    inst 5    inst 6    1 uvicorn worker each
   |         |         |         |         |         |
   +---------+---------+----+----+---------+---------+
                            v
        FraudPredictor (same models/e7_v1 artifacts) -> decision response
```

The upstream list in `deploy/nginx/nginx.conf` is the single source of truth: `deploy/start.sh` starts one instance per `server` line, so Nginx and the running instances always match.

### Workers vs instances vs load balancer

| | What it is | What it gives you |
|---|---|---|
| **uvicorn `--workers N`** | N processes behind one uvicorn supervisor, sharing one port | Uses N CPU cores. Still one service: one config, one start/stop, one machine |
| **Multiple instances** | Independent uvicorn processes on separate ports (here 6 × 1 worker) | Each can be stopped, restarted or upgraded on its own |
| **Reverse proxy / load balancer (Nginx)** | One public port in front of the instances | Spreads requests, skips failed instances, sheds load when saturated, and is the place to add instances on other machines later |

The instances share no mutable state: each loads the same read-only artifacts (`MODEL_DIR`) and its own copy of the account snapshot (`ACCOUNT_SNAPSHOT`). Any instance can serve any request.

### Run

```bash
sudo apt install nginx      # once; the system nginx is not used (ours runs as a normal user)
deploy/start.sh             # instances 1-6, then `nginx -t` and Nginx on :8080
deploy/stop.sh              # everything
deploy/stop.sh 3            # one part only: 1..6 | nginx  (same for start.sh)

curl localhost:8080/health
# Options (env): MODEL_DIR=models/e7_v1  WORKERS_PER_INSTANCE=1  ACCOUNT_SNAPSHOT=...  LOG_LEVEL=info
#                CPUS_INSTANCES=0-7 CPUS_NGINX=0-7   (taskset pinning, used for benchmarks)
```

PID files, logs (`instance_<n>.log`) and Nginx temp files are written to `deploy/run/` (gitignored). The Nginx access log records which instance served each request (`upstream=...`), so routing can be checked without adding fields to the API response.

### Resources

Hardware: Ryzen 7 5800H, 8 cores / 16 threads, 13.8 GB RAM. Starting the six instances: ~195 MB RSS each, and available memory dropped by **817 MB** (6.5 GB to 5.7 GB; pages shared between processes). Nginx: ~14 MB. Memory is not a constraint for six model copies on this machine.

### Nginx behavior (`deploy/nginx/nginx.conf`)

- **Balancing:** round-robin across the six instances. Verified: 60 requests gave exactly 10 per instance.
- **Upstream keepalive:** pool of 256 idle connections per Nginx worker, closed after 4 s idle (shorter than uvicorn's 5 s keep-alive, see below).
- **Headers:** `Host`, `X-Real-IP`, `X-Forwarded-For` and `X-Forwarded-Proto` are forwarded.
- **Timeouts:** connect 1 s, send/read 2 s.
- **Retries:** only on connection errors (`proxy_next_upstream error`), so only when the request never reached an instance. `non_idempotent` is not set and timeouts are not retried, so a delivered transaction is never scored twice.
- **Health checks are passive only.** Open-source Nginx never calls `/health` itself. After 3 failures (errors or timeouts) within 5 s, an instance is skipped for 5 s, then tried again.
- **Load shedding:** at most 64 in-flight requests per instance (`max_conns`), 384 in total. Above that, Nginx answers `503 {"detail":"fraud scoring temporarily unavailable"}` with `Retry-After: 1`, so the payment service can apply its fallback.

Three settings came from problems found while testing:
1. **`max_conns` (load shedding).** Without it, at 200 concurrent requests on two instances, overloaded single-worker instances queued requests for 4–6 s. The 2 s timeouts disabled every instance, and **~92% of requests failed** with `no live upstreams` while the instances sat mostly idle.
2. **`keepalive 256`.** With 64, the pool was too small for six instances. Connections kept being opened and closed, and p95 at 100 concurrent was 82–126 ms. With 256 it is 44 ms.
3. **`keepalive_timeout 4s`.** uvicorn closes idle connections after 5 s and Nginx's default is 60 s. Nginx sometimes reused a connection the instance was closing, and that POST failed with `connection reset by peer`.

### Benchmark: 6 instances vs the 8-worker service

Same session, same machine, same settings for every row: server side on **CPUs 0–7** (4 physical cores), `wrk -t8 -d30s` with `tests/wrk_predict.lua` on CPUs 8–15.

| Setup | Concurrency | req/s | p50 | p95 | p99 | max | Errors | CPU | Memory |
|---|---|---|---|---|---|---|---|---|---|
| 1 uvicorn × 8 workers, direct | 100 | **10,630** | 8.7 ms | **12.5 ms** | **18.6 ms** | 160 ms | 0 | server 799% | 1,589 MB |
| 1 uvicorn × 8 workers, direct | 200 | 11,469 | 16.7 ms | 22.6 ms | 31.3 ms | 186 ms | 0 | server 800% | 1,590 MB |
| 1 uvicorn × 8 workers, direct | 400 | 12,287 | 31.9 ms | 35.5 ms | 89.8 ms | 759 ms | 0 | server 800% | 1,592 MB |
| 1 instance (1 worker), direct | 100 | 2,892 | 32.5 ms | 34.0 ms | 270 ms | 1,528 ms | 0 | instance 100% | 195 MB |
| **6 instances via Nginx** | 100 | **10,082** | 8.7 ms | **44.3 ms** | **186.1 ms** | 758 ms | 0 | instances 584%, Nginx 84% | 1,185 MB |
| **6 instances via Nginx** | 200 | 10,122 | 20.1 ms | 61.2 ms | 170.2 ms | 833 ms | 0 | instances 589%, Nginx 87% | 1,185 MB |
| **6 instances via Nginx** | 400 | see below | | | | | 0 timeouts | instances 595%, Nginx 166% | 1,185 MB |

At 400 concurrent the six instances are past the 384 in-flight cap, so Nginx sheds the excess. Accepted requests (from the Nginx log): **9,117 req/s**, p50 39 ms, p95 72 ms, p99 117 ms, 0 timeouts. `wrk` resends immediately after each 503, so the 695,443 rejections show how fast shedding is, not real traffic.

Accepted-request latency measured inside Nginx at 100 concurrent: p50 5 ms, p95 32 ms, p99 55 ms. `wrk`'s numbers above also include the client side.

**Result:** six instances behind Nginx deliver **~5% less throughput than eight workers** and a **clearly worse tail** (p95 44 vs 12.5 ms, p99 186 vs 19 ms at 100 concurrent). The two setups differ in topology and resources:

| | 8 workers | 6 instances + Nginx |
|---|---|---|
| Python processes serving requests | 8 | 6 |
| Proxy hop | none | Nginx, ~0.85 core at 100 concurrent, taken from the same CPUs 0–7 |
| Who spreads connections | kernel (shared listening socket) | Nginx round-robin over pooled connections |
| Restart one part without stopping the service | no | yes |
| Path to more machines | no | yes (add `server` lines) |

On one machine, the six-instance layout buys **independent restarts, per-instance failure isolation and a path to scale out**, not speed. If raw single-machine throughput were the only goal, one uvicorn with 8 workers is faster.

Earlier two-instance run (A/B, each on its own physical core, Nginx on a third): 5,311 req/s through Nginx vs ~2,870 for one instance directly.

### Failure tests (`wrk -t4 -c60 -d20s` through Nginx, per-second counts from the Nginx log)

| Scenario | Observed |
|---|---|
| All six healthy | Even split, ~1,800–2,200 req/s per instance |
| Instance 3 stopped gracefully (SIGTERM) under load | **9–45 failed requests** per run (≤0.02% of ~215k), all in the second it stopped: in flight on its connections, not retried because they had been delivered. The other five took all traffic within the same second |
| Instance 5 crashed (SIGKILL) under load | **32–53 failed requests** per run (≤0.03%), same pattern |
| Stopped/crashed instance restarted | Healthy ~1.3 s after start. Traffic returns **~2.5 s later**, when the 5 s `fail_timeout` expires (one probe request was refused and retried on another instance) |
| All six down | Immediate `503` JSON with `Retry-After: 1`; `/health` through Nginx returns 503. Under 3 s of load: 199,964 requests, all 503, p99 18 ms, no hangs |
| All six restarted after being down under load | Healthy after 1.9 s, but Nginx kept answering **503 for up to 5 s** more, because every instance had been marked failed (`fail_timeout`). Then round-robin resumed (2 requests to each of the six) |

**Not zero-downtime:** stopping or losing an instance drops the requests in flight on it. A rolling restart without errors would need draining first (mark the server `down`, `nginx -s reload`, wait, then stop).

### Prediction events (Kafka)

Optional: with `KAFKA_BOOTSTRAP_SERVERS` set, each instance also publishes every prediction (no account IDs, amounts or balances) to the Kafka topic `nilepay.prediction-events`, without blocking the request. A separate consumer (`python -m src.event_consumer`) validates, deduplicates and stores the events, and dead-letters invalid ones. Delivery is at-least-once while the broker is reachable within 30 s; otherwise best-effort, with every lost event counted. The broker runs natively (`deploy/kafka.sh`). Architecture, failure behavior, measurements and limits: **[KAFKA_EVENTS.md](KAFKA_EVENTS.md)**.

### ML monitoring

The consumer also aggregates prediction behaviour and input data quality (rolling 5 min / 1 h / 24 h windows, per model and policy version) and compares them with a versioned baseline built from the model's training data; the live dashboard shows it. Model performance (precision, recall, PR-AUC, cost) is computed only from actual labels, and none exist yet. **[ML_MONITORING.md](ML_MONITORING.md)**.

A separate drift monitor compares production inputs with the training baseline per feature. It uses PSI, a chi-square test with Holm adjustment, and missing-rate and unknown-category checks, and raises deduplicated alerts. Drift is a distribution change, not a measured loss of accuracy. **[DATA_DRIFT.md](DATA_DRIFT.md)**.

When retraining may be considered, which data may be used, and how a candidate would be judged against the champion is set by a report-only policy (`config/retraining_policy.json`, `python -m src.retraining triggers`). Nothing retrains or promotes automatically, and existing model directories can't be overwritten. **[RETRAINING_STRATEGY.md](RETRAINING_STRATEGY.md)**.

### Streamlit UI

A small UI (`streamlit run streamlit_app/app.py`) submits transactions through Nginx and shows the decision, the system's health and recent transactions. It holds no model and reads only existing endpoints. **[streamlit_app/README.md](streamlit_app/README.md)**.

### Limitations

- **Six instances on one machine do not protect against machine-level failure.** Power, kernel, disk, network or memory failure on this host takes down all six instances and Nginx together. There's no real high availability, and nothing here supports a 99.9% availability claim.
- **Nginx itself is not redundant.** If it stops, the service is down even with six healthy instances.
- **Passive health checks only.** A hung instance that accepts connections but never answers is only detected through timeouts.
- **Shared CPUs.** Instances, Nginx and the benchmark client share one machine (pinned to separate cores). Numbers need confirming with the load generator on separate hardware.
- `docker-compose.yml` is still empty; Docker was not usable in this environment, so a compose setup could not be verified.

### Tests

`tests/test_deploy_config.py` checks the config statically: the six upstreams with `max_conns`, no `non_idempotent`, retry only on `error`, explicit timeouts, forwarded headers and the 503 page. It runs `nginx -t` when Nginx is installed and checks the scripts parse. An opt-in live test starts the stack and checks that all six instances receive traffic and give identical predictions:

```bash
RUN_DEPLOY_TESTS=1 pytest tests/test_deploy_config.py
```
