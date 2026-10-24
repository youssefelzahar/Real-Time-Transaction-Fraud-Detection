# Manual reliability, failover and capacity testing

A step-by-step guide to test the NilePay deployment by hand: **Nginx on `:8080` in front of six FastAPI instances on `:8001`–`:8006`**, one uvicorn worker each. Every test explains what it checks, the exact commands, what we saw when we ran it on this machine, and how to read the result.

All commands run from the project root with the venv active (`source venv/bin/activate`). Sample outputs come from a Ryzen 7 5800H laptop (8 cores / 16 threads), **on battery, `balanced` power profile**. Your numbers will differ; the patterns should not.

---

## 0. Before you start

### What the deployment does (inspected, not assumed)

| Item | Actual setup |
|---|---|
| Start / stop | `deploy/start.sh [all\|1..6\|nginx]`, `deploy/stop.sh [all\|1..6\|nginx]`. One uvicorn process per `server` line in `deploy/nginx/nginx.conf` |
| Ports | Instances `127.0.0.1:8001`–`8006`, Nginx `127.0.0.1:8080`, Nginx status page `127.0.0.1:8081/nginx_status` (internal only) |
| `/health` | Liveness, plus "model loaded at startup" (the model loads before the port opens). It does **not** run a prediction or check the account service |
| Which instance served a request | Nginx access log `deploy/run/access.log`, field `upstream=127.0.0.1:800N`. The API response does not say, by design |
| Balancing | Round-robin |
| Health checks | **Passive only.** An instance is skipped for 5 s after 3 failures (connection errors **or timeouts**) within 5 s. Nginx never calls `/health` itself |
| Timeouts | Connect 1 s, read/send 2 s. A timeout gives `504 {"detail":"fraud scoring timed out"}` |
| Retries | Only on connection errors (the request never reached an instance), at most 2 tries. A delivered POST is **never** resent |
| Load shedding | Max 64 in-flight requests per instance. Above that: `503 {"detail":"fraud scoring temporarily unavailable"}` + `Retry-After: 1` |
| Keep-alive | Nginx→instance pool of 256 per Nginx worker, idle 4 s (uvicorn closes at 5 s) |
| Memory | ~195 MB per instance; six instances took ~820 MB of available memory |

### Tools

```bash
sudo apt install wrk sysstat     # wrk = load generator; sysstat = pidstat + mpstat
```

Helpers in this repo:

| Tool | Use |
|---|---|
| `tests/wrk_predict.lua` | `wrk` script: varied `/predict` payloads, prints one `RESULT` line |
| `tests/wrk_paced.lua` | Same payloads, rate-limited: `-- <pause_ms>` after each response, per connection |
| `tests/nginx_log_report.py` | Per-instance summary of the Nginx log: share, status codes, times, new connections, retries |
| `curl localhost:8081/nginx_status` | Nginx connection counters (see A6) |

### Record the power state (capacity numbers depend on it)

```bash
cat /sys/class/power_supply/*/online /sys/class/power_supply/*/status 2>/dev/null
powerprofilesctl get
```

On this laptop, the same six-instance setup did **~10,000 req/s on AC power** and **~6,700 req/s on battery** (`balanced`). Run all capacity tests (A5, A6) in the same power state, and write it down next to the results. For stable numbers: plug in AC and use `powerprofilesctl set performance`.

### Terminal layout

| Terminal | Use |
|---|---|
| **T1, control** | Start/stop instances and Nginx |
| **T2, traffic** | Send requests |
| **T3, observe** | Logs, CPU, Nginx status |

### Start clean

```bash
# T1
deploy/stop.sh
rm -f deploy/run/*.log
deploy/start.sh
```

Expected:

```
  instance 1 (:8001) ready
  ...
  instance 6 (:8006) ready
  nginx (:8080) ready
```

For the capacity tests (A5, A6), pin the server side to CPUs 0–7 and run the load generator on 8–15 so they don't compete:

```bash
deploy/stop.sh && CPUS_INSTANCES=0-7 CPUS_NGINX=0-7 deploy/start.sh
```

Handy variables for the other terminals:

```bash
BODY='{"transaction_id":"tx-001","type":"TRANSFER","amount":181.0,"sender_id":"C1305486145","receiver_id":"C553264065","sender_balance":181.0,"receiver_balance":0.0}'
PIDS=$(cat deploy/run/instance_*.pid | paste -sd,)      # the six instance PIDs, comma separated
```

---

## A1. Baseline health

**Purpose:** prove every instance works on its own and through Nginx before testing failures. Without a clean baseline, you can't tell a failure you caused from one that was already there.

```bash
# T2: each instance directly, then through Nginx
for p in 8001 8002 8003 8004 8005 8006 8080; do
  curl -s -w "  :$p  HTTP %{http_code}  %{time_total}s\n" localhost:$p/health
done

# Response schema through Nginx
curl -s -X POST localhost:8080/predict -H 'content-type: application/json' -d "$BODY" | python3 -m json.tool

# Baseline latency: median of 20 requests, direct vs through Nginx
for p in 8001 8080; do
  for i in $(seq 20); do curl -s -o /dev/null -w "%{time_total}\n" -X POST localhost:$p/predict -H 'content-type: application/json' -d "$BODY"; done \
    | sort -n | awk -v p=$p '{a[NR]=$1} END{printf "  :%s  median %.1f ms  max %.1f ms\n", p, a[int(NR/2)]*1000, a[NR]*1000}'
done
```

Observed:

```
{"status":"ok","model_version":"e7_v1"}  :8001  HTTP 200  0.000882s
...                                      (all seven: HTTP 200, ~1 ms)
{
    "transaction_id": "tx-001",
    "fraud_probability": 0.999896764755249,
    "decision": "review",
    "threshold": 0.54,
    "model_version": "e7_v1",
    "missing_features": [],
    "balance_source": {"sender": "request", "receiver": "request"}
}
  :8001  median 1.2 ms  max 2.1 ms
  :8080  median 1.4 ms  max 2.6 ms
```

**Pass if:** all seven return 200 with the same `model_version`, the response has exactly those seven fields, and Nginx adds well under a millisecond. **Record** the medians; they're your "healthy" reference for later tests.

---

## A2. Load distribution

**Purpose:** confirm all six instances get traffic and the split matches round-robin. Also rule out a common trap: with keep-alive, a client reuses one connection, and some balancers (including uvicorn's own `--workers`) pin a whole connection to one process.

```bash
# T2: (a) 60 requests, a new connection each
sleep 1.2; OFF=$(stat -c %s deploy/run/access.log)          # remember where the log ends
for i in $(seq 60); do curl -s -o /dev/null -X POST localhost:8080/predict -H 'content-type: application/json' -d "$BODY"; done
sleep 1.2; tail -c +$((OFF+1)) deploy/run/access.log | grep -o 'upstream=[^ ]*' | sort | uniq -c

# (b) 60 requests over ONE keep-alive connection (curl reuses it for repeated URLs)
sleep 1.2; OFF=$(stat -c %s deploy/run/access.log)
A0=$(curl -s localhost:8081/nginx_status | awk 'NR==3{print $1}')
curl -s -X POST -H 'content-type: application/json' -d "$BODY" $(for i in $(seq 60); do printf 'http://127.0.0.1:8080/predict '; done) > /dev/null
A1=$(curl -s localhost:8081/nginx_status | awk 'NR==3{print $1}')
echo "client connections used: $((A1 - A0 - 1))"
sleep 1.2; tail -c +$((OFF+1)) deploy/run/access.log | grep -o 'upstream=[^ ]*' | sort | uniq -c

# (c) wrk with 6 persistent connections for 5 s, then the per-instance report
sleep 1.2; OFF=$(stat -c %s deploy/run/access.log)
wrk -t1 -c6 -d5s -s tests/wrk_predict.lua http://127.0.0.1:8080 | grep RESULT
sleep 1.3; python tests/nginx_log_report.py --since-offset $OFF
```

The access log is flushed every second, hence the `sleep 1.2` before reading it.

Observed:

```
(b) client connections used: 1
     10 upstream=127.0.0.1:8001
     10 upstream=127.0.0.1:8002
     ...                          (10 for each of the six)

(c) upstream    requests  share  non-200  nginx p95  inst p50 ...
    instance 1     6,760  16.7%        0 ...
    ...                          (16.7% each)
```

**Pass if:** every instance gets about 1/6. **How to read it:** (b) shows Nginx balances **per request, not per connection**: one client connection still spread perfectly over all six. So keep-alive between client and Nginx doesn't skew the split. A count of 0 for an instance means it's down or marked failed; check `deploy/run/error.log`.

---

## A3. Stop one instance under traffic

**Purpose:** show that when instance 3 disappears, Nginx keeps serving from the other five, and that it rejoins after restart.

Two kinds of requests behave differently:
- **New requests** that Nginx tries to send to a stopped instance get "connection refused". The request never arrived, so Nginx **retries it on another instance**. The client sees a normal 200.
- **In-flight requests**, already sent to the instance when it stops, are lost. They return an error, because a delivered POST is never resent: the instance may already have scored it, and sending it again could score the transaction twice.

```bash
# T2: visible traffic, one request every 20 ms, prints status + seconds
while true; do curl -s -o /dev/null -w "%{http_code} %{time_total}\n" -X POST localhost:8080/predict -H 'content-type: application/json' -d "$BODY"; sleep 0.02; done

# T3: live view of where requests go
tail -f deploy/run/access.log | grep --line-buffered -o 'upstream=[^ ]*'

# T1: stop instance 3, wait, start it again
deploy/stop.sh 3
sleep 10
deploy/start.sh 3
```

After stopping T2 with Ctrl+C:

```bash
python tests/nginx_log_report.py --last 30 --timeline
tail -5 deploy/run/error.log
```

Observed at ~31 requests/s (light traffic):

```
489 requests over ~16s (31/s) | statuses: 200=489
upstream     requests  share  non-200 ...  retried
instance 3         22   4.5%        0 ...        0
instance 4         94  19.2%        0 ...        3
  sec     i1  i2  i3  i4  i5  i6
    4      6   6   5   5   5   6
    5      6   6   0   7   7   6      <- instance 3 stopped: the other five take its share
   ...
error.log: connect() failed (111: Connection refused) ... upstream server temporarily disabled
```

Observed at heavy traffic instead (T2 replaced by `wrk -t4 -c60 -d20s -s tests/wrk_predict.lua http://127.0.0.1:8080`): **9–45 failed requests** on a normal stop (`deploy/stop.sh 3`) and **32–53** on a crash (`kill -9 $(cat deploy/run/instance_3.pid)`), out of ~210,000. All of them happened in the second the instance went away.

**How to read it:**
- `retried` > 0 with all statuses 200 means Nginx moved undelivered requests to another instance. That's the safe failover.
- A few non-200s in the exact second of the stop are in-flight requests. That's expected and is why this setup is **not zero-downtime**.
- **After restart, traffic comes back up to ~5 s later.** If Nginx tried instance 3 while it was still starting, that failure restarts the 5 s penalty (`fail_timeout`). In our light-traffic run, instance 3 was healthy but got nothing for the rest of the run for this reason.

---

## A4. Slow or overloaded instance

**Purpose:** find out whether Nginx avoids a slow instance. Nginx's passive checks only notice **errors and timeouts**, not slowness, so the answer depends on how slow it is.

Both simulations are local and reversible. Don't run them against shared or production systems.

### A4a. Hung instance (accepts connections, never answers)

`SIGSTOP` freezes the process. The kernel still accepts TCP connections, so from Nginx's side it looks alive but silent.

```bash
# T2: steady load for 25 s
wrk -t2 -c24 -d25s -s tests/wrk_predict.lua http://127.0.0.1:8080

# T1: while T2 runs
kill -STOP $(cat deploy/run/instance_3.pid)     # freeze
sleep 15
kill -CONT $(cat deploy/run/instance_3.pid)     # resume (always do this!)

# afterwards
python tests/nginx_log_report.py --last 30 --timeline
grep -c "timed out" deploy/run/error.log
```

Observed:

```
RESULT req/s=9532 p50=1.9ms p95=12.9ms p99=1395.1ms max=2000.0ms ... non_200=25 socket_errors=14
statuses: 200=210,367, 504=26
  sec     i1    i2    i3    i4    i5    i6   other
    5   1782  1769     0  1783  1784  1783   504=25     <- requests sent to frozen instance 3
   13   2139  2134     0  2136  2137  2133   504=1      <- Nginx tries it again after 5 s, fails again
   19   1754  1749  1751  1754  1762  1756              <- resumed: back in rotation
```

What a client gets: `HTTP 504 {"detail":"fraud scoring timed out"}` after **2 s**.

**How to read it:**
- Nginx **does not** detect a hung instance right away. Each request sent to it waits the full 2 s `proxy_read_timeout`, then returns 504.
- After 3 timeouts it skips the instance for 5 s, then sends one test request (another 2 s wait and 504 if it's still stuck), and so on.
- **504s are never retried:** the instance may have scored the transaction.
- `socket_errors` in `wrk` are its own 2 s client timeout (`wrk` default) hitting the same requests.

### A4b. Overloaded instance (slow but answering)

Pin instance 3 to its own CPU, then take most of that CPU away with busy loops:

```bash
# T1: restart the stack with instance 3 alone on CPU 15 (load generator then uses 8-14)
deploy/stop.sh
CPUS_INSTANCES="0-7 0-7 15 0-7 0-7 0-7" CPUS_NGINX=0-7 deploy/start.sh

# T2: reference run, instance 3 healthy
taskset -c 8-14 wrk -t4 -c60 -d20s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080 | grep RESULT
sleep 1.3; python tests/nginx_log_report.py --last 21

# T1: starve instance 3 (two busy loops on its CPU)
taskset -c 15 python3 -c 'while True: pass' &
taskset -c 15 python3 -c 'while True: pass' &

# T2: same run again
taskset -c 8-14 wrk -t4 -c60 -d20s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080 | grep RESULT
sleep 1.3; python tests/nginx_log_report.py --last 21

# T1: clean up
kill %1 %2
deploy/stop.sh && deploy/start.sh
```

Observed:

| Run | req/s | p95 | p99 | 504 | Instance 3 share | Instance 3 p95 |
|---|---|---|---|---|---|---|
| Instance 3 healthy | 7,208 | 38 ms | 80 ms | 0 | 16.7% | 1 ms |
| **Instance 3 starved** | 5,933 | **254 ms** | **1,173 ms** | 15 | 4.3% | 149 ms (p99 969 ms) |
| Starved, with `least_conn` (experiment) | 8,408 | 20 ms | 36 ms | 0 | 5.2% | 37 ms |

**How to read it:**
- With round-robin, Nginx keeps giving the slow instance its full share until requests start hitting the 2 s timeout. One slow instance made **the whole service's** p99 go from 80 ms to 1.2 s.
- In the report, a slow instance stands out by its `inst p95` / `inst p99` compared with the others.
- `least_conn` sends each request to the instance with the fewest active requests, so a slow instance automatically gets less. In this test it removed the problem. It isn't enabled; it was a measured experiment and is a recommended change.

---

## A5. The 80% utilization target

**Purpose:** decide how much traffic the deployment can take while keeping 20% headroom, including after losing one instance.

### Define "80%" first

Two different things get called "utilization". Don't mix them up:

| Kind | Definition | How to measure |
|---|---|---|
| **CPU, per instance** | % of one CPU used by an instance's process. One uvicorn worker is single-threaded, so its maximum is ~100% | `pidstat -u -p ${PIDS:?set PIDS first, see section 0} 5 1` |
| **CPU, host** | % busy of the CPUs given to the server (here 0–7) | `mpstat -P 0-7 5 1` |
| **Request capacity** | Offered req/s ÷ the highest req/s that still meets the latency/error SLO | Load test (below) |

**They aren't interchangeable.** On this machine an instance reached 100% CPU at very different request rates depending on conditions (next section). Also, CPUs 0–7 are 4 physical cores with 2 threads each: "50% host busy" doesn't mean half the compute is left.

**SLO used in this guide (example; set your own):** p99 ≤ 100 ms through Nginx, and zero 5xx errors.

### Step 1: measure C, the per-instance capacity, with all six loaded

Measure capacity **through Nginx with all six instances busy**, then divide by 6. Measuring one instance alone gives a very different number:

| Measured (battery) | Throughput | Instance CPU | Clock |
|---|---|---|---|
| One instance alone, others idle | ~2,150 req/s | 100% | up to 4.46 GHz |
| All six busy through Nginx | ~6,650 req/s total = **~1,100 per instance** | ~97% each | 3.47 GHz (all-core) |

Alone, an instance gets a full physical core and a higher clock. When all six are busy, they share 4 physical cores and run at all-core speed. Using the "alone" number in `6 × 0.8 × C` would **overestimate capacity by about 2×**.

Find the largest load that meets the SLO:

```bash
# T2: increase concurrency until p99 breaks the SLO
for c in 12 24 48 96; do
  echo "c=$c $(taskset -c 8-15 wrk -t8 -c$c -d20s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080 | grep RESULT)"
done
```

Observed (battery):

```
c=12  req/s=6281  p99=2.6ms     ok
c=24  req/s=6676  p99=24.5ms    ok   <- highest level within the SLO
c=48  req/s=6705  p99=262.1ms   breaks p99 <= 100 ms
```

So the deployment capacity at the SLO is ~6,680 req/s, and **C ≈ 6,680 / 6 ≈ 1,110 req/s per instance**.

### Step 2: compute the budgets

| Situation | Formula | With C = 1,110 |
|---|---|---|
| Six healthy, 80% target | 6 × 0.8 × C | **5,330 req/s** |
| One instance down, 80% target | 5 × 0.8 × C | **4,440 req/s** |

To survive losing an instance without going above 80%, plan for the **lower** number, 4,440 req/s here. With all six healthy that's 4,440 / (6 × 1,110) ≈ 67% utilization.

These formulas are only valid if C was measured in the **same** setup you're budgeting for: same CPUs, same power state, all instances busy. With five instances on the same 4 cores, each gets a bit more CPU than with six, so the "5 healthy" budget is slightly pessimistic. Step 3 measures it.

### Step 3: test the budget

Use the paced script to send a steady rate near the budget. The rate is about `connections × 1000 / (pause_ms + latency_ms)`. Start with 24 connections and a 4 ms pause, check the achieved `req/s` in the RESULT line, and adjust the pause until you're close to 4,440:

```bash
# T2: steady load at the budget, all six healthy
taskset -c 8-15 wrk -t4 -c24 -d30s --latency -s tests/wrk_paced.lua http://127.0.0.1:8080 -- 4 | grep RESULT

# T3: during the run
pidstat -u -p ${PIDS:?set PIDS first, see section 0} 5 1 | grep Average     # %CPU per instance: target <= 80
mpstat -P 0-7 5 1 | awk '/^Average/ && $2 ~ /^[0-9]+$/ {b+=100-$NF; n++} END{printf "host busy (cpu 0-7): %.0f%%\n", b/n}'
```

Then repeat with one instance down:

```bash
# T1
deploy/stop.sh 3
PIDS=$(cat deploy/run/instance_*.pid | paste -sd,)    # now five PIDs

# T2: same command, same pause
taskset -c 8-15 wrk -t4 -c24 -d30s --latency -s tests/wrk_paced.lua http://127.0.0.1:8080 -- 4 | grep RESULT
# T3: same pidstat / mpstat

# T1
deploy/start.sh 3
```

**Pass if**, for both runs at (about) the budget rate: per-instance CPU ≤ ~80%, p99 within the SLO and no 5xx. Record the achieved rate, the per-instance CPU and p99 for six and for five instances.

Observed, six healthy instances (battery), 24 connections, 4 ms pause:

```
RESULT req/s=3812 p50=1.8ms p95=5.1ms p99=16.0ms ... non_200=0
pidstat: each instance 78.2-80.0 %CPU
```

At 3,812 req/s, only **~57% of the request capacity** (6,680), every instance is already at **~79% CPU**. Request-based and CPU-based "80%" give **different budgets** on this machine: by CPU, the 80% line is reached around 3,800 req/s, not 5,330. Decide which one your target refers to and budget by the one that's hit first.

**If it fails:** per-instance CPU near 100% at the budget rate means C was overestimated (re-measure in the same power state). CPU under 80% but p99 breaking the SLO means the bottleneck isn't CPU; use the diagnosis table in A6.

Pacing is still a closed loop: if the server slows down, the rate drops instead of requests piling up. So near saturation it **underestimates** how bad latency would get with real traffic, which keeps arriving regardless.

---

## A6. Saturation test

**Purpose:** find where the deployment stops scaling and see *why*, by raising the load step by step and recording everything at each step.

```bash
# T3, run during each level: per-instance CPU, host CPU, Nginx counters
pidstat -u -p ${PIDS:?set PIDS first, see section 0} 5 1 | grep Average
mpstat -P 0-7 5 1 | awk '/^Average/ && $2 ~ /^[0-9]+$/ {b+=100-$NF; n++} END{printf "host busy (cpu 0-7): %.0f%%\n", b/n}'
curl -s localhost:8081/nginx_status
free -m | awk '/Mem/{print "available MB: "$7}'

# T2: one level at a time (raise c, wait for T3 readings, then the next)
c=24
sleep 1.3; OFF=$(stat -c %s deploy/run/access.log)
taskset -c 8-15 wrk -t8 -c$c -d20s --latency -s tests/wrk_predict.lua http://127.0.0.1:8080 | grep RESULT
sleep 1.3; python tests/nginx_log_report.py --since-offset $OFF
```

Suggested levels: 12, 24, 48, 96, 192, 384, 512. **Stop raising the load** if the 5xx rate climbs sharply (apart from intentional 503 shedding above 384 in flight), available memory falls toward a few hundred MB, or p99 stays above several seconds.

Record one row per level:

| c | wrk req/s | p50 | p95 | p99 | 5xx (503 / 504) | instance CPU (avg, max) | Nginx CPU | host busy | `nginx_status` active / writing / waiting | memory available |
|---|---|---|---|---|---|---|---|---|---|---|

Observed (battery; 20 s per level, CPUs 0–7 server, 8–15 `wrk`):

| c | req/s | p50 | p95 | p99 | 5xx | Instances CPU | Nginx CPU | Nginx active |
|---|---|---|---|---|---|---|---|---|
| 12 | 6,281 | 1.2 ms | 2.1 ms | 2.6 ms | 0 | 569% (max 96%) | 83% | 9 |
| 24 | 6,676 | 2.8 ms | 15.1 ms | 24.5 ms | 0 | 583% (max 98%) | 88% | 25 |
| 48 | 6,705 | 6.2 ms | 44.6 ms | 262 ms | 0 | 579% (max 98%) | 89% | 49 |
| 96 | 6,622 | 12.9 ms | 76.8 ms | 424 ms | 0 | 589% (max 99%) | 92% | 97 |
| 192 | 6,652 | 29.9 ms | 112 ms | 485 ms | 0 | 587% (max 99%) | 94% | 193 |
| 384 | 6,185 accepted | 41 ms | 157 ms | 1,132 ms | 68,171 × 503, 3 × 504 | 564% (max 100%) | 127% | 385 |
| 512 | 5,716 accepted | 16 ms | 122 ms | 691 ms | 265,558 × 503 | 594% (max 99%) | 173% | 513 (127 waiting) |

Memory stayed flat: instances 1,172 MB total, Nginx 15 MB, ~5.7 GB available.

**Reading this run:** throughput stopped growing at c≈12–24 while every instance sat at ~97–100% CPU. That's **insufficient capacity (CPU-bound instances)**. After that, extra concurrency only added queueing: p50 roughly doubles with each doubling of c. Above 384 in flight, load shedding returned 503s instead of letting latency explode.

### Telling the causes apart

Compare three latencies for the same run:
1. **`wrk` p95**: what the client saw.
2. **`nginx p95`** in the report: from Nginx reading the request to sending the answer.
3. **`inst p95`**: the instance's own time.

| Symptom | Likely cause | Confirm with |
|---|---|---|
| Throughput flat as c rises; every instance ~100% CPU; latency grows with c | **Insufficient capacity** (instances CPU-bound) | `pidstat` per instance; p50 roughly proportional to c |
| Throughput flat but instances **well under** 80% CPU and host has idle CPU | **Client-side limit** | `top`: `wrk` threads near 100%? Raise `-t`, run a second `wrk`, or test from another machine |
| Host CPUs 0–7 near 100% busy, everything slow including Nginx | **Host CPU saturation** | `mpstat -P 0-7`; also check clocks: `grep MHz /proc/cpuinfo` (power state!) |
| `wrk` p95 ≫ `nginx p95` | Requests waited **before** Nginx handled them: Nginx short of CPU, or client-side queueing | Nginx CPU in `pidstat`; pin Nginx to its own CPU and compare |
| `nginx p95` ≫ `inst p95` (`wait p95` high) | Waiting **in Nginx**: connection setup, retries, slot limits | `new conns`, `retried` columns; `error.log` |
| `inst p95` high on **one** instance only | **Slow upstream** (see A4b) | That instance's `inst p95`; its CPU in `pidstat` |
| `inst p95` high on all, CPU < 100% | Queueing **inside** instances (unfair event-loop scheduling under many connections) | `inst p99` ≫ `inst p50`; fewer connections per instance helps |
| 502/503 with instances idle, `no live upstreams` in `error.log` | **Load balancer marked instances down** (timeouts cascading) or `max_conns` reached | `grep -c "no live upstreams" deploy/run/error.log`; `nginx_status` active ≈ 6 × 64 |
| 504s; `upstream timed out` in `error.log` | Instance **hung or overloaded** beyond 2 s | Which `upstream=` the 504s came from |
| `connection reset by peer`, `prematurely closed`, high `new conns`, many `TIME_WAIT` | **Connection management** (keep-alive mismatch, pool too small) | `ss -tan state time-wait '( dport >= :8001 and dport <= :8006 )' \| wc -l` |

`nginx_status` fields: **active** = open client connections; **writing** = requests currently being processed (sent to an instance, waiting for it); **waiting** = idle keep-alive connections. "writing" stuck at 6 × 64 = 384 means every instance is at its `max_conns` limit.

---

## Cleanup

```bash
kill -CONT $(cat deploy/run/instance_*.pid) 2>/dev/null   # in case an instance is still frozen
pkill -f "while True: pass"                              # in case a busy loop is still running
deploy/stop.sh
```

## What these tests do not show

- **Machine failure.** Everything runs on one host; a crash of the machine stops all six instances and Nginx. Nothing here supports a high-availability claim.
- **Real arrival patterns.** `wrk` is a closed loop (each connection waits for its answer). Real traffic keeps arriving when the service slows, so real tail latency near saturation would be worse.
- **Separate client hardware.** Client and server share the laptop, pinned to separate cores.
