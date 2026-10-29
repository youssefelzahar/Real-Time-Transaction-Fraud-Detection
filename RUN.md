# How to run NilePay fraud detection

All commands run from the project root. On Windows, see [RUN_WINDOWS.md](RUN_WINDOWS.md).

## 1. One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

sudo apt install nginx wrk               # load balancer + load generator
sudo systemctl disable --now nginx       # optional: we run our own Nginx on :8080, not the system one on :80
```

Put the PaySim CSV at `data/fraud_data.csv`.

> Every later command assumes the venv is active (`source venv/bin/activate`).

## 2. Train the models

```bash
python -m src.train --data data/fraud_data.csv --version v1
```

Writes `models/e5_v1/`, `models/e6_v1/`, `models/e7_v1/` and the report `models/comparison_v1.md`.
**E7 is the model to serve.** Takes a few minutes on the full dataset.

## 3. Run the tests

```bash
pytest                                                    # all unit + API + config tests (no dataset needed)
RUN_DEPLOY_TESTS=1 pytest tests/test_deploy_config.py     # also start Nginx + 6 instances and check routing
```

The second command needs ports 8001–8006 and 8080 free, and stops everything when it finishes.

## 4. Run the API: single instance (development)

```bash
MODEL_DIR=models/e7_v1 uvicorn api.main:app --reload --reload-dir src --reload-dir api
```

- Swagger: <http://127.0.0.1:8000/docs> → `POST /predict` → **Try it out** → pick a request from **Examples**.
- `--reload-dir` keeps the file watcher off `venv/`; plain `--reload` watches everything and burns CPU.
- Each request prints its timing; add `TIMING_LOG=0` in front to turn that off.

## 5. Run the API: load balanced (Nginx + 6 instances)

```bash
deploy/start.sh        # instances 1-6 on :8001-:8006, then nginx -t and Nginx :8080
deploy/stop.sh         # stop everything
```

Start or stop one part only (e.g. to test failures):

```bash
deploy/stop.sh 3       # 1..6 | nginx
deploy/start.sh 3
```

Check it:

```bash
curl localhost:8080/health

curl -X POST localhost:8080/predict -H 'content-type: application/json' -d '{
  "transaction_id": "tx-001", "type": "TRANSFER", "amount": 181.0,
  "sender_id": "C1305486145", "receiver_id": "C553264065",
  "sender_balance": 181.0, "receiver_balance": 0.0}'
```

See which instance served each request:

```bash
tail -f deploy/run/access.log          # upstream=127.0.0.1:800N = instance N
```

Options (environment variables for `deploy/start.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models/e7_v1` | Model artifacts both instances serve |
| `WORKERS_PER_INSTANCE` | `1` | uvicorn workers per instance |
| `ACCOUNT_SNAPSHOT` | none | JSON file `{"account_id": balance}` for the in-memory account service |
| `CPUS_INSTANCES` | none | CPU pinning with `taskset`: one list for all instances (`0-7`), or one per instance (`"0 1 2 3 4 5"`) |
| `CPUS_NGINX` | none | CPU pinning for Nginx, e.g. `0-7` |
| `LOG_LEVEL` | `warning` | `info` logs one line per request in each instance log |
| `TIMING_LOG` | `0` | `1` prints per-request timings (feature / model / total ms) |
| `DECISION_POLICY` | `config/decision_policy.json` | Decision policy file (thresholds, decline mode). Instances refuse to start if it isn't validated for `MODEL_DIR`'s model |

### Logs

| File | Contents |
|---|---|
| `deploy/run/instance_<n>.log` | Instance n (:800n), n = 1..6 |
| `deploy/run/access.log` | Nginx: every request, its status, time, and which instance served it (`upstream=`) |
| `deploy/run/error.log` | Nginx: instance down, refused connections, timeouts |

By default the instance logs only show startup, warnings and errors. To see every request per instance:

```bash
deploy/stop.sh
LOG_LEVEL=info deploy/start.sh                  # add TIMING_LOG=1 for timings too

tail -f deploy/run/instance_1.log               # one instance
tail -f deploy/run/instance_*.log               # all six together, labelled by file
```

Don't benchmark with `LOG_LEVEL=info` or `TIMING_LOG=1`: writing a line per request slows the instances down.

The log files are appended to across restarts. To start clean: `deploy/stop.sh && rm deploy/run/*.log`.

## 6. Load test

Start the stack pinned to separate cores, then run `wrk` on the remaining cores:

```bash
CPUS_INSTANCES=0-7 CPUS_NGINX=0-7 deploy/start.sh

taskset -c 8-15 wrk -t8 -c100 -d30s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080   # through Nginx
taskset -c 8-15 wrk -t8 -c100 -d30s --latency -s tests/wrk_predict.lua http://127.0.0.1:8001   # instance 1 only

deploy/stop.sh
```

The last line of output (`RESULT ...`) has req/s, p50/p95/p99, non-200 responses and socket errors. Watch CPU in a second terminal with `top`.

Benchmark rules: stop any other API server first (`ss -ltnp | grep -E ':800[0-6]|:8080'` should show only yours), and keep duration, concurrency and CPU pinning the same between runs you compare.

Python load tester (no `wrk` needed, but tops out around 2,000 req/s by itself):

```bash
python tests/load_api.py --url http://127.0.0.1:8080 --requests 4000 --concurrency 100 --processes 8
```

## 7. Failure test (manual)

Full manual guide (baseline, distribution, failover, slow instances, 80% capacity budget, saturation): **[RELIABILITY_TESTING.md](RELIABILITY_TESTING.md)**. Quick version:

```bash
deploy/start.sh
taskset -c 8-15 wrk -t4 -c60 -d20s -s tests/wrk_predict.lua http://127.0.0.1:8080 &
sleep 5;  deploy/stop.sh 3        # instance 3 goes down: the other five take its traffic
sleep 7;  deploy/start.sh 3       # it comes back: traffic returns after ~2.5 s
wait
grep -o 'upstream=[^ ]*' deploy/run/access.log | sort | uniq -c
deploy/stop.sh
```

Expected: a few dozen failed requests at most (in flight on instance 3 when it stopped), then the other five serve everything, then all six again.

### Automated high-traffic test (capacity, throughput, availability)

```bash
python -m monitoring.live_dashboard &      # optional: watch it at http://127.0.0.1:8090
tests/stress_suite.sh                      # ~5 min; report in deploy/run/stress-<time>.txt
PHASES="ramp" tests/stress_suite.sh        # only the capacity ramp (also: sustain, avail)
```

1. **Ramp:** 25 to 800 connections, 15 s each. It reports throughput, successful req/s, p50/p95/p99 and errors, plus the highest load that stays within p95 < 100 ms with 0 errors. 503 above ~384 connections is Nginx load shedding (6 × `max_conns=64`).
2. **Sustain:** 60 s at that load, with per-second min/median/max.
3. **Avail:** the same load for 75 s. Instance 3 is stopped at +10 s and restarted at +25 s; the Kafka broker is stopped at +40 s and restarted at +50 s. It reports availability, the seconds that had errors, and Kafka events delivered or lost.

## 8. Decision policy and metrics

```bash
python -m src.decision_policy_analysis --model-dir models/e7_v2   # validation-only threshold evidence
curl -s localhost:8001/metrics/decisions | python3 -m json.tool    # one instance's decision counters
python tests/decision_metrics.py                                  # all six instances summed
```

`/metrics/decisions` is internal: it answers on each instance (`:8001`–`:8006`) and returns 404 through Nginx (`:8080`). See [DECISION_POLICY.md](DECISION_POLICY.md) for the policy, its evidence, and how to switch decline from shadow to enforce.

## 8b. Prediction events on Kafka (optional)

```bash
sudo apt install openjdk-21-jre-headless       # once
deploy/kafka.sh install                         # once: Kafka 4.0.0 into ~/.local/nilepay-kafka
deploy/kafka.sh start                           # broker on 127.0.0.1:9092 + topics
export KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
deploy/start.sh consumer                        # event consumer (log: deploy/run/consumer.log)
MODEL_DIR=models/e7_v2 deploy/start.sh          # instances publish events while the variable is set

deploy/kafka.sh status                          # topics + consumer lag
curl -s localhost:8001/metrics/events | python3 -m json.tool
RUN_KAFKA_TESTS=1 pytest tests/test_kafka_integration.py

deploy/stop.sh; deploy/stop.sh consumer; deploy/kafka.sh stop
```

Without `KAFKA_BOOTSTRAP_SERVERS` the API runs exactly as before. Details: [KAFKA_EVENTS.md](KAFKA_EVENTS.md).

### Live view (see requests and events moving)

```bash
python -m monitoring.live_dashboard            # open http://127.0.0.1:8090
# in another terminal, from the project root: a slow, steady load you can watch
wrk -t2 -c8 -d120s -s tests/wrk_paced.lua http://127.0.0.1:8080 -- 20
```

It shows client → Nginx → 6 instances → Kafka → consumer → event store with live rates, instances or broker going down (red dashed), Kafka lag, decision totals, the latest stored events, and buttons that send one sample transaction through Nginx. It only reads existing metrics and listens on 127.0.0.1.

## 9. Benchmark only the scoring code

```bash
TIMING_LOG=0 python -m src.benchmark_features
```

Run it as a module (`-m`). `python src/benchmark_features.py` fails to import `src`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `address already in use` | Something is on 8000–8006/8080: `ss -ltnp \| grep -E ':800[0-6]\|:8080'`, then stop it or run `deploy/stop.sh` |
| `deploy/start.sh` says an instance didn't become healthy | Read `deploy/run/instance_<n>.log`; usually a wrong `MODEL_DIR` or models not trained yet |
| Nginx won't start | `nginx -t -p "$PWD/deploy/" -c nginx/nginx.conf -e run/error.log`, then read `deploy/run/error.log` |
| `503 fraud scoring temporarily unavailable` | Both instances are down, or saturated (over 64 in-flight requests each) |
| `ModuleNotFoundError: src` | Run from the project root, with `python -m ...` |
| ~40 ms extra latency with `--workers > 1` | `uvloop`/`httptools` missing: `pip install "uvicorn[standard]"` |
