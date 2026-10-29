"""Reliability scenarios against the running Windows stack (not collected by pytest).

Needs the full stack with Kafka (RUN_WINDOWS.md section 6) and KAFKA_BOOTSTRAP_SERVERS set.
Everything goes through Nginx and deploy/windows.py, like an operator would. Each run tags its
transactions with a unique prefix, so it measures only its own events in the live store.

    python tests\\windows_reliability.py consumer-restart [--force]   # stop + start the consumer under load
    python tests\\windows_reliability.py instance-stop [--force]      # stop instance 3 under load, Kafka up
    python tests\\windows_reliability.py instance-stop-kafka-down [--force]   # ... with events pending
    python tests\\windows_reliability.py latency [--kafka-down]       # p95 via load_api.py + Nginx log

--force uses `stop --force` (taskkill /F, the previous behaviour) for the before/after comparison.
Results are printed and appended to deploy/run/reliability-results.jsonl.
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "deploy" / "run"
PY = sys.executable
NGINX = "http://127.0.0.1:8080"
STORE_DIR = Path(os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events")))


def ctl(*args):
    """deploy/windows.py <args>; returns (seconds, output)."""
    t0 = time.monotonic()
    out = subprocess.run([PY, "deploy/windows.py", *args], cwd=ROOT, capture_output=True, text=True)
    return time.monotonic() - t0, (out.stdout + out.stderr).strip()


class StoreWatcher(threading.Thread):
    """Tails the event store; records when each event of this run was first seen (~50 ms resolution)."""

    def __init__(self, tag):
        super().__init__(daemon=True)
        self.tag = tag.encode()
        self.seen = {}           # transaction_id -> first time seen (monotonic)
        self.lines = 0           # tagged lines, duplicates included
        self.event_ids = []
        self.stop_flag = threading.Event()
        files = sorted(STORE_DIR.glob("prediction-events-*.jsonl"))
        self.path = files[-1] if files else None
        self.offset = self.path.stat().st_size if self.path else 0

    def run(self):
        while not self.stop_flag.is_set():
            self.poll()
            time.sleep(0.05)

    def poll(self):
        files = sorted(STORE_DIR.glob("prediction-events-*.jsonl"))
        if files and files[-1] != self.path:
            self.path, self.offset = files[-1], 0
        if not self.path:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.offset += end + 1
        now = time.monotonic()
        for line in chunk[:end].split(b"\n"):
            if self.tag in line:
                e = json.loads(line)
                self.lines += 1
                self.event_ids.append(e["event_id"])
                self.seen.setdefault(e["transaction_id"], now)


class Load(threading.Thread):
    """Paced POST /predict through Nginx: `rate` req/s for `seconds`, transaction_id = <tag>-<i>."""

    def __init__(self, tag, rate, seconds, workers=8):
        super().__init__(daemon=True)
        self.tag, self.rate, self.seconds, self.workers = tag, rate, seconds, workers
        self.results = []        # (transaction_id, sent_at, status, latency_ms)
        self.lock = threading.Lock()

    def run(self):
        start, n = time.monotonic(), int(self.rate * self.seconds)
        slots = iter(range(n))

        def worker():
            with httpx.Client(timeout=5) as c:
                for i in slots:
                    due = start + i / self.rate
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    tx, t0 = f"{self.tag}-{i}", time.monotonic()
                    body = {"transaction_id": tx, "type": "PAYMENT", "amount": 125.0, "sender_id": f"C-{i % 97}",
                            "receiver_id": "M-rel", "sender_balance": 5000.0, "receiver_balance": 0.0}
                    try:
                        status = c.post(NGINX + "/predict", json=body).status_code
                    except httpx.HTTPError:
                        status = None
                    with self.lock:
                        self.results.append((tx, t0, status, (time.monotonic() - t0) * 1000))

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def pct(values, q):
    v = sorted(values)
    return round(v[min(int(q * len(v)), len(v) - 1)], 1) if v else None


def wait_stored(watcher, expected, timeout=90):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not expected <= set(watcher.seen):
        time.sleep(0.2)


def finish(name, result):
    result = {"scenario": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **result}
    print(json.dumps(result, indent=2))
    with open(RUN / "reliability-results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


def http_summary(load):
    ok = [r for r in load.results if r[2] == 200]
    return ok, {"requests": len(load.results), "http_200": len(ok),
                "http_errors": len(load.results) - len(ok),
                "http_error_rate": round(1 - len(ok) / len(load.results), 5) if load.results else None,
                "client_p95_ms": pct([r[3] for r in load.results], 0.95)}


def consumer_restart(force):
    tag = f"rel-cr-{'force' if force else 'graceful'}-{int(time.time())}"
    watcher, load = StoreWatcher(tag), Load(tag, rate=40, seconds=30)
    watcher.start(), load.start()
    time.sleep(6)
    stop_s, stop_out = ctl("stop", "consumer", *(["--force"] if force else []))
    t_start = time.monotonic()
    start_s, start_out = ctl("start", "consumer")
    load.join()
    ok, http = http_summary(load)
    wait_stored(watcher, {r[0] for r in ok})
    watcher.stop_flag.set(), watcher.join()
    sent = {r[0]: r[1] for r in load.results}
    after = [t for tx, t in watcher.seen.items() if t >= t_start]
    resumed = min(after) if after else None
    # End-to-end delay of the events sent while the consumer was stopped or rebalancing
    gap_events = [watcher.seen[tx] - sent[tx] for tx in watcher.seen if sent.get(tx, 0) >= t_start - stop_s]
    finish("consumer-restart", {
        "mode": "force" if force else "graceful", "stop_command_s": round(stop_s, 2), "stop_output": stop_out,
        "start_command_s": round(start_s, 2),
        "restart_to_processing_s": round(resumed - t_start, 2) if resumed else None,
        "max_event_delay_during_restart_s": round(max(gap_events), 2) if gap_events else None,
        **http, "events_stored": len(watcher.seen),
        "missing_events": len({r[0] for r in ok} - set(watcher.seen)),
        **duplicates(watcher),
    })


def duplicates(watcher):
    unique_events = len(set(watcher.event_ids))
    return {"same_event_stored_twice": watcher.lines - unique_events,          # consumer redelivery
            "transactions_with_two_events": unique_events - len(watcher.seen)}  # producer duplicate


def instance_counters():
    out = {}
    for n in range(1, 7):
        try:
            m = httpx.get(f"http://127.0.0.1:800{n}/metrics/events", timeout=2).json()
        except httpx.HTTPError:
            continue
        out[n] = {"enqueued": m["enqueued"], "delivered": m["delivered"],
                  "failed": sum(m["failed"].values()), "dropped": m["buffer_full_drops"]}
    return out


def stop_record(n, pid):
    """The ledger stop line of instance n's process `pid`, or None if it ended without one."""
    path = RUN / f"producer-ledger-instance-{n}.jsonl"
    for line in reversed(path.read_text().splitlines() if path.exists() else []):
        rec = json.loads(line)
        if rec.get("pid") == pid and rec.get("record") == "stop":
            return rec
    return None


def instance_stop(force, kafka_down=False):
    name = "instance-stop-kafka-down" if kafka_down else "instance-stop"
    tag = f"rel-is{'kd' if kafka_down else ''}-{'force' if force else 'graceful'}-{int(time.time())}"
    watcher, load = StoreWatcher(tag), Load(tag, rate=60, seconds=20)
    watcher.start(), load.start()
    time.sleep(6)
    t_down = time.monotonic()
    if kafka_down:                                 # broker down 2 s before the stop: events pile up
        ctl("stop", "kafka")
        time.sleep(2)
    before = httpx.get("http://127.0.0.1:8003/metrics/events").json()
    before["in_flight_max"] = max(httpx.get("http://127.0.0.1:8003/metrics/events").json()["in_flight"]
                                  for _ in range(20))
    counters_before = instance_counters()
    stop_s, stop_out = ctl("stop", "3", *(["--force"] if force else []))
    time.sleep(3)
    ctl("start", "3")
    if kafka_down:
        # Back well inside KAFKA_MESSAGE_TIMEOUT_MS (30 s): the other instances' queued events
        # must all be delivered, so only the stopped instance's pending events are at risk.
        ctl("start", "kafka")
    kafka_down_s = time.monotonic() - t_down
    load.join()
    ok, http = http_summary(load)
    wait_stored(watcher, {r[0] for r in ok}, timeout=120)
    watcher.stop_flag.set(), watcher.join()
    missing = sorted({r[0] for r in ok} - set(watcher.seen))
    time.sleep(1)
    counters_after = instance_counters()
    ledger = stop_record(3, before.get("pid"))
    finish(name, {
        "mode": "force" if force else "graceful", "stop_command_s": round(stop_s, 2), "stop_output": stop_out,
        "instance3_in_flight_max_just_before_stop": before["in_flight_max"],
        "instance3_pid": before.get("pid"),
        "instance3_ledger_stop": ledger or "no stop record (process ended without accounting)",
        **http, "events_stored": len(watcher.seen), "missing_events": len(missing),
        **duplicates(watcher),
        # Other instances (same processes before and after): delivery outcomes during the run.
        "other_instances_delta": {n: {k: counters_after[n][k] - counters_before[n][k] for k in counters_after[n]}
                                  for n in counters_after if n != 3 and n in counters_before},
        "kafka_down_s": round(kafka_down_s, 1) if kafka_down else 0,
    })


def latency(kafka_down):
    if kafka_down:
        ctl("stop", "kafka")
    n0 = sum(1 for _ in open(RUN / "access.log", "rb"))
    out = subprocess.run([PY, "tests/load_api.py", "--url", NGINX, "--requests", "3000", "--concurrency", "10",
                          "--processes", "4"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    time.sleep(2)
    rts, statuses = [], []
    with open(RUN / "access.log", "rb") as f:
        for i, line in enumerate(f):
            if i >= n0 and b"POST /predict" in line:
                parts = line.decode().split()
                statuses.append(parts[parts.index("HTTP/1.1\"") + 1] if "HTTP/1.1\"" in parts else "?")
                rts.append(float(line.decode().split("rt=")[1].split()[0]) * 1000)
    if kafka_down:
        ctl("start", "kafka")
    cols = out[-1].split()                       # conc reqs errors req/s p50 p95 p99 max p95<target
    finish("latency", {
        "kafka": "down" if kafka_down else "up", "client": "load_api.py, 4 processes, concurrency 10",
        "client_req_s": float(cols[3]), "client_p50_ms": float(cols[4]), "client_p95_ms": float(cols[5]),
        "client_p99_ms": float(cols[6]), "client_errors": int(cols[2]),
        "nginx_requests": len(rts), "nginx_p50_ms": pct(rts, 0.5), "nginx_p95_ms": pct(rts, 0.95),
        "nginx_p99_ms": pct(rts, 0.99), "nginx_non_200": sum(1 for s in statuses if s != "200"),
    })


if __name__ == "__main__":
    if not os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        sys.exit("set KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 (the stack must be running with Kafka)")
    force = "--force" in sys.argv
    scenario = (sys.argv[1:] or [""])[0]
    if scenario == "consumer-restart":
        consumer_restart(force)
    elif scenario == "instance-stop":
        instance_stop(force)
    elif scenario == "instance-stop-kafka-down":
        instance_stop(force, kafka_down=True)
    elif scenario == "latency":
        latency("--kafka-down" in sys.argv)
    else:
        sys.exit(__doc__)
