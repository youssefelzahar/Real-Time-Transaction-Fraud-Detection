# How to run NilePay fraud detection on Windows

Native Windows, no WSL, Docker or admin rights. Commands are for PowerShell, run from the project root.
`deploy\windows.py` replaces `deploy/start.sh`, `deploy/stop.sh` and `deploy/kafka.sh`; it reads the
same `deploy/nginx/nginx.conf` and `deploy/kafka/server.properties`. Linux instructions: [RUN.md](RUN.md).

## 1. One-time setup

```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

python deploy\windows.py install     # nginx, a portable Java 21 JRE and Kafka into %LOCALAPPDATA%\nilepay
```

Put the PaySim CSV at `data\fraud_data.csv`. Every later command assumes the venv is active.

`install` takes Kafka 4.0.0 (the Linux version) from `archive.apache.org`. That server is often
rate-limited; if the download fails, pick a version from <https://dlcdn.apache.org/kafka/>:
`$env:KAFKA_VERSION = "4.3.2"; python deploy\windows.py install`. Later commands use whichever version is installed.

## 2. Train the models

```powershell
python -m src.train --data data/fraud_data.csv --version v1
```

Writes `models\e5_v1\`, `models\e6_v1\`, `models\e7_v1\` and `models\comparison_v1.md`. **E7 is the model to serve.**

## 3. Run the tests

```powershell
pytest
```

`RUN_DEPLOY_TESTS=1` and `tests/stress_suite.sh` drive the Linux scripts and `wrk`, so they don't run on Windows.

## 4. Single instance (development)

```powershell
$env:MODEL_DIR = "models/e7_v1"
uvicorn api.main:app --reload --reload-dir src --reload-dir api
```

Swagger: <http://127.0.0.1:8000/docs>.

## 5. Load balanced: Nginx + 6 instances

```powershell
python deploy\windows.py start       # instances 1-6 on :8001-:8006, then Nginx on :8080
python deploy\windows.py status
python deploy\windows.py stop        # Nginx + every instance, gracefully

python deploy\windows.py stop 3      # one part only: 1..6 | nginx | consumer | kafka
python deploy\windows.py start 3
python deploy\windows.py stop 3 --force   # skip the graceful stop (taskkill /T /F at once)
```

The processes run in the background and keep running after you close the terminal; stop them with `stop`.

**How `stop` works.** Windows has no SIGTERM, and Ctrl+Break only reaches processes on the sender's
console, so `stop` creates `deploy\run\<name>.stop`, and the process shuts itself down the way SIGTERM
does on Linux ([src/shutdown.py](src/shutdown.py)):

| | What happens | Bound |
|---|---|---|
| Instance | uvicorn stops accepting, finishes in-flight requests, then the Kafka producer flushes pending events and writes its final counters to `deploy\run\producer-ledger-instance-<n>.jsonl` | `GRACEFUL_SHUTDOWN_S` (5) + `KAFKA_SHUTDOWN_FLUSH_S` (5) |
| Consumer | finishes the current message, commits what is stored (fsynced) or dead-lettered (acknowledged), closes the Kafka consumer so it **leaves the group** | one poll (1 s) + DLQ flush (5) |
| Fallback | anything still running after `STOP_TIMEOUT_S` (20 s) is force-killed, and `stop` says so | |

Graceful instance stop needs `WORKERS_PER_INSTANCE=1` (the default). With more workers, `stop` falls back to the force-kill after 20 s.
Settings are environment variables, the same as in RUN.md (`MODEL_DIR`, `WORKERS_PER_INSTANCE`, `LOG_LEVEL`,
`TIMING_LOG`, `DECISION_POLICY`, `ACCOUNT_SNAPSHOT`), for example `$env:LOG_LEVEL = "info"`. CPU pinning (`CPUS_*`) is not supported.

Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health

Invoke-RestMethod -Method Post http://127.0.0.1:8080/predict -ContentType application/json -Body (@{
  transaction_id = "tx-001"; type = "TRANSFER"; amount = 181.0
  sender_id = "C1305486145"; receiver_id = "C553264065"
  sender_balance = 181.0; receiver_balance = 0.0 } | ConvertTo-Json)

Get-Content deploy\run\access.log -Wait -Tail 20     # upstream=127.0.0.1:800N = instance N
```

Use `127.0.0.1`, not `localhost`: Windows tries IPv6 `::1` first, which adds ~200 ms per request.
Logs are the same files as on Linux (`deploy\run\instance_<n>.log`, `access.log`, `error.log`).

Load test (no `wrk` on Windows):

```powershell
python tests\load_api.py --url http://127.0.0.1:8080 --requests 4000 --concurrency 100 --processes 8
```

## 6. Kafka prediction events (optional)

```powershell
python deploy\windows.py start kafka              # broker on 127.0.0.1:9092 + topics
$env:KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
python deploy\windows.py start consumer           # log: deploy\run\consumer.log
python deploy\windows.py stop; python deploy\windows.py start   # instances publish while the variable is set

python deploy\windows.py status                   # includes topics + consumer lag
Invoke-RestMethod http://127.0.0.1:8001/metrics/events
python -m monitoring.live_dashboard               # http://127.0.0.1:8090

python deploy\windows.py stop; python deploy\windows.py stop consumer; python deploy\windows.py stop kafka
Remove-Item Env:KAFKA_BOOTSTRAP_SERVERS           # back to running without events
```

Broker data and logs: `%LOCALAPPDATA%\nilepay\kafka-data` (`server.log` there if the broker won't start).

### What the counters mean

Each dashboard number has a scope; numbers with different scopes don't have to match.

| Scope | Numbers | Resets when |
|---|---|---|
| **Current processes** (each running instance since it started) | enqueued, delivered (acked by the broker), in flight, dropped before Kafka, delivery not confirmed | that instance restarts |
| **Ended processes** (producer ledger, one final line per gracefully stopped instance process, each counted once) | delivered, unconfirmed at shutdown; *instance exits without accounting* = processes that were force-killed or crashed | never (append-only file) |
| **Dashboard session** | stored since dashboard start | the dashboard restarts |
| **Persistent** | events in Kafka topic (end offsets), Kafka lag (committed vs end) | never |

What each outcome proves:
- **Delivered:** the broker acknowledged the event (`acks=all`).
- **Dropped before Kafka:** the buffer was full or the event couldn't be serialized, so it was never sent. This is the only proven loss.
- **Delivery not confirmed:** the client reported a failure, usually a timeout after `KAFKA_MESSAGE_TIMEOUT_MS` (30 s). The event may still have been written.
- **Unconfirmed at shutdown:** the event was still pending when the bounded shutdown flush ended. Its outcome is unknown.
- **Exits without accounting:** a process that was force-killed or crashed. Its pending events are unknown, not counted as lost.
- **Enqueued:** handed to the client queue, which is not delivery.

The consumer status is `running` only if its process is alive **and** it wrote a heartbeat in the last 30 s. It shows `starting` for the first heartbeat after a restart and `stalled` if the process is alive but silent.

### Delivery guarantees

- **API → Kafka: at most once, and every outcome is counted.**
  - The producer is idempotent, so its retries can't create duplicates.
  - Events can be lost when the broker is down for longer than `KAFKA_MESSAGE_TIMEOUT_MS`. They're counted as *not confirmed*.
  - Events can also be lost when a graceful stop's 5 s flush can't reach the broker. They're recorded in the ledger as *unconfirmed at shutdown*.
  - A force-killed or crashed instance loses its pending events **without a count**. That's the remaining unaccounted loss window: normally the 5 ms linger, but everything queued while Kafka is down.
- **Kafka → store: at least once.**
  - Offsets are committed only after the events are fsynced to the store or dead-lettered.
  - A graceful consumer stop commits everything processed, so a restart re-reads nothing.
  - A force-killed consumer re-reads events since its last commit (≤ 5 s). The duplicate filter is in memory and starts empty, so these can be stored twice.
- **Neither path is exactly-once.** The measurements below found 0 lost and 0 duplicates in graceful runs, but that's a measured result, not a guarantee.

### Reliability tests (repeatable)

```powershell
$env:RUN_KAFKA_TESTS = "1"; pytest tests\test_kafka_integration.py   # temp topics: consumer stop/restart/reassignment
python tests\windows_reliability.py consumer-restart [--force]         # stop + start the consumer under 40 req/s
python tests\windows_reliability.py instance-stop [--force]            # stop instance 3 under 60 req/s
python tests\windows_reliability.py instance-stop-kafka-down [--force] # ... with events pending (broker down)
python tests\windows_reliability.py latency [--kafka-down]             # p95, client side and at Nginx
```

Results are printed and appended to `deploy\run\reliability-results.jsonl`. `--force` reproduces the old force-kill for comparison.
The client-side p95 is only meaningful at moderate concurrency: the Python client saturates near 300 req/s on Windows (about 50 concurrent), while the Nginx-side p95 stays at 3–8 ms.

## Differences from Linux

| | Linux | Windows |
|---|---|---|
| Nginx | 2 workers, 4096 connections each | 1 worker, 1024 connections: nginx for Windows can't do more. Generated as `deploy\run\nginx.windows.conf` |
| Instance down | Connection refused at once, retried on the next instance | Windows reports a closed loopback port only after ~2 s, so nginx retries on connect **timeout** too. The first 3 requests after an instance dies take about 1 s longer, then it is skipped for 5 s. A POST that reached an instance is still never resent |
| `stop` | SIGTERM: graceful shutdown | Stop file, then the same graceful path; force-kill only after 20 s or with `--force` (see section 5) |
| Kafka | `bin/*.sh` | `java -cp libs\*` directly: Kafka's `.bat` scripts overflow cmd's 8191-character line limit |
| Kafka broker stop | Graceful | Force-kill (`taskkill /F`): the next start runs log recovery, about 15 s here |
| **Deleting Kafka data** | Works | **Breaks the broker.** Windows can't rename or delete segment files the broker has memory-mapped (KAFKA-1194), so it marks the log dir failed and **shuts down**. That covers deleting a topic and also **retention deleting old segments**: a closed segment whose data is older than `retention.ms`, 7 days for the predictions topic. Expect the first failure after about a week of continuous running. Never delete topics on this broker (the integration tests skip it on Windows). For anything long-running, run Kafka on Linux (WSL or Docker) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `address already in use` / instance not healthy | `Get-NetTCPConnection -State Listen -LocalPort 8001,8002,8003,8004,8005,8006,8080`, then `python deploy\windows.py stop`; read `deploy\run\instance_<n>.log` |
| Nginx won't start | `deploy\run\error.log` |
| `This program is blocked by group policy` | Your organisation's AppLocker/SRP blocks the venv's `python.exe`; run from another shell (e.g. Git Bash: `venv/Scripts/python.exe deploy/windows.py start`) or ask IT |
| Windows Firewall prompt | Everything listens on 127.0.0.1 only; you can cancel the prompt |
