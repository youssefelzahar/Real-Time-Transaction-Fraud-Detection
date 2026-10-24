# How to run NilePay fraud detection

All commands run from the project root.

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
RUN_DEPLOY_TESTS=1 pytest tests/test_deploy_config.py     # also start Nginx + A + B and check routing
```

The second command needs ports 8001, 8002 and 8080 free, and stops everything when it finishes.

## 4. Run the API: single instance (development)

```bash
MODEL_DIR=models/e7_v1 uvicorn api.main:app --reload --reload-dir src --reload-dir api
```

- Swagger: <http://127.0.0.1:8000/docs> → `POST /predict` → **Try it out** → pick a request from **Examples**.
- `--reload-dir` keeps the file watcher off `venv/`; plain `--reload` watches everything and burns CPU.
- Each request prints its timing; add `TIMING_LOG=0` in front to turn that off.

## 5. Run the API: load balanced (Nginx + 2 instances)

```bash
deploy/start.sh        # instance A :8001, instance B :8002, then nginx -t and Nginx :8080
deploy/stop.sh         # stop everything
```

Start or stop one part only (e.g. to test failures):

```bash
deploy/stop.sh a       # a | b | nginx
deploy/start.sh a
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
tail -f deploy/run/access.log          # upstream=127.0.0.1:8001 (A) / 127.0.0.1:8002 (B)
```

Options (environment variables for `deploy/start.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models/e7_v1` | Model artifacts both instances serve |
| `WORKERS_PER_INSTANCE` | `1` | uvicorn workers per instance |
| `ACCOUNT_SNAPSHOT` | none | JSON file `{"account_id": balance}` for the in-memory account service |
| `CPUS_A`, `CPUS_B`, `CPUS_NGINX` | none | CPU pinning with `taskset`, e.g. `0-1` |
| `LOG_LEVEL` | `warning` | `info` logs one line per request in each instance log |
| `TIMING_LOG` | `0` | `1` prints per-request timings (feature / model / total ms) |

### Logs

| File | Contents |
|---|---|
| `deploy/run/instance_a.log` | Instance A (:8001) |
| `deploy/run/instance_b.log` | Instance B (:8002) |
| `deploy/run/access.log` | Nginx: every request, its status, time, and which instance served it (`upstream=`) |
| `deploy/run/error.log` | Nginx: instance down, refused connections, timeouts |

By default the instance logs only show startup, warnings and errors. To see every request per instance:

```bash
deploy/stop.sh
LOG_LEVEL=info deploy/start.sh                  # add TIMING_LOG=1 for timings too

tail -f deploy/run/instance_a.log               # one terminal per instance
tail -f deploy/run/instance_b.log
tail -f deploy/run/instance_*.log               # or both together, labelled by file
```

Don't benchmark with `LOG_LEVEL=info` or `TIMING_LOG=1`: writing a line per request slows the instances down.

The log files are appended to across restarts. To start clean: `deploy/stop.sh && rm deploy/run/*.log`.

## 6. Load test

Start the stack pinned to separate cores, then run `wrk` on the remaining cores:

```bash
CPUS_A=0-1 CPUS_B=2-3 CPUS_NGINX=4-5 deploy/start.sh

taskset -c 8-15 wrk -t4 -c100 -d30s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080   # through Nginx
taskset -c 8-15 wrk -t4 -c100 -d30s --latency -s tests/wrk_predict.lua http://127.0.0.1:8001   # instance A only

deploy/stop.sh
```

The last line of output (`RESULT ...`) has req/s, p50/p95/p99, non-200 responses and socket errors. Watch CPU in a second terminal with `top`.

Benchmark rules: stop any other API server first (`ss -ltnp | grep -E ':800[0-2]|:8080'` should show only yours), and keep duration, concurrency and CPU pinning the same between runs you compare.

Python load tester (no `wrk` needed, but tops out around 2,000 req/s by itself):

```bash
python tests/load_api.py --url http://127.0.0.1:8080 --requests 4000 --concurrency 100 --processes 8
```

## 7. Failure test (manual)

```bash
deploy/start.sh
taskset -c 8-15 wrk -t2 -c20 -d20s -s tests/wrk_predict.lua http://127.0.0.1:8080 &
sleep 5;  deploy/stop.sh a        # A goes down: traffic should move to B
sleep 7;  deploy/start.sh a       # A comes back: traffic returns after ~2.5 s
wait
grep -o 'upstream=[^ ]*' deploy/run/access.log | sort | uniq -c
deploy/stop.sh
```

Expected: a few failed requests (in flight on A when it stopped), then everything served by B, then both again.

## 8. Benchmark only the scoring code

```bash
TIMING_LOG=0 python -m src.benchmark_features
```

Run it as a module (`-m`). `python src/benchmark_features.py` fails to import `src`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `address already in use` | Something is on 8000/8001/8002/8080: `ss -ltnp \| grep -E ':800[0-2]\|:8080'`, then stop it or run `deploy/stop.sh` |
| `deploy/start.sh` says an instance didn't become healthy | Read `deploy/run/instance_a.log` (or `_b`); usually a wrong `MODEL_DIR` or models not trained yet |
| Nginx won't start | `nginx -t -p "$PWD/deploy/" -c nginx/nginx.conf -e run/error.log`, then read `deploy/run/error.log` |
| `503 fraud scoring temporarily unavailable` | Both instances are down, or saturated (over 64 in-flight requests each) |
| `ModuleNotFoundError: src` | Run from the project root, with `python -m ...` |
| ~40 ms extra latency with `--workers > 1` | `uvloop`/`httptools` missing: `pip install "uvicorn[standard]"` |
