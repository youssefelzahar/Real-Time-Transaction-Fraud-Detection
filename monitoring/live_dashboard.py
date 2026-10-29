"""Live view of the running system: client -> Nginx -> 6 instances -> Kafka -> consumer -> store.

    python -m monitoring.live_dashboard          # then open http://127.0.0.1:8090

Read-only and separate from the API: it polls what the stack already exposes
(Nginx stub_status, each instance's /health, /metrics/decisions, /metrics/events,
Kafka offsets, the consumer's event store) and serves one page that refreshes every second.
Listens on loopback only. Config (env): DASHBOARD_PORT (8090), KAFKA_BOOTSTRAP_SERVERS
(127.0.0.1:9092), KAFKA_TOPIC_PREDICTIONS, KAFKA_CONSUMER_GROUP, EVENT_STORE_DIR.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
NGINX_CONF = ROOT / "deploy" / "nginx" / "nginx.conf"
NGINX_STATUS = "http://127.0.0.1:8081/nginx_status"
NGINX_PUBLIC = "http://127.0.0.1:8080"
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
TOPIC = os.getenv("KAFKA_TOPIC_PREDICTIONS", "nilepay.prediction-events")
GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "nilepay-prediction-monitor")
STORE_DIR = Path(os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events")))
FEED_SIZE = 14

# Sample transactions for the "send one" buttons (no real customer data).
SAMPLES = {
    "payment": {"type": "PAYMENT", "amount": 1250.0, "sender_id": "C-demo-1", "receiver_id": "M-demo-1",
                "sender_balance": 52000.0, "receiver_balance": 0.0},
    "cash_in": {"type": "CASH_IN", "amount": 9000.0, "sender_id": "C-demo-2", "receiver_id": "C-demo-3",
                "sender_balance": 1500.0, "receiver_balance": 30000.0},
    "risky_transfer": {"type": "TRANSFER", "amount": 181.0, "sender_id": "C-demo-4", "receiver_id": "C-demo-5",
                       "sender_balance": 181.0, "receiver_balance": 0.0},
    "risky_cash_out": {"type": "CASH_OUT", "amount": 850002.52, "sender_id": "C-demo-6", "receiver_id": "C-demo-7",
                       "sender_balance": 850002.52, "receiver_balance": 0.0},
}


def instance_ports():
    """Instance N = Nth `server` line of the upstream block (same rule as deploy/start.sh)."""
    return [int(p) for p in re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", NGINX_CONF.read_text(), flags=re.M)]


class StoreTail:
    """Counts lines appended to the newest JSONL file and keeps the last few events."""

    def __init__(self, directory):
        self.dir = directory
        self.path = None
        self.offset = 0
        self.lines = 0
        self.recent = []

    def poll(self):
        files = sorted(self.dir.glob("prediction-events-*.jsonl"))
        if not files:
            return
        if files[-1] != self.path:                      # first poll or a new day's file
            self.path, size = files[-1], files[-1].stat().st_size
            with open(self.path, "rb") as f:            # seed the feed from the file's tail
                f.seek(max(size - 16384, 0))
                self.recent = [ln for ln in f.read().split(b"\n")[1:] if ln][-FEED_SIZE:]
            self.offset = size
            return
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read(32 * 1024 * 1024)
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        chunk = chunk[:end + 1]
        self.offset += len(chunk)
        self.lines += chunk.count(b"\n")
        self.recent = (self.recent + chunk[-16384:].split(b"\n"))
        self.recent = [ln for ln in self.recent if ln][-FEED_SIZE:]

    def feed(self):
        out = []
        for ln in reversed(self.recent):
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            out.append({k: e.get(k) for k in ("occurred_at", "transaction_id", "transaction_type", "fraud_probability",
                                              "decision", "reason_code", "would_decline", "instance_id")})
        return out


class Collector:
    def __init__(self):
        self.http = httpx.Client(timeout=0.5)
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.store = StoreTail(STORE_DIR)
        self.lock = threading.Lock()
        self._kafka = None

    def _get(self, url):
        try:
            r = self.http.get(url)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def nginx(self):
        try:
            text = self.http.get(NGINX_STATUS).text
        except httpx.HTTPError:
            return {"up": False}
        m = re.search(r"Active connections:\s*(\d+).*?\n\s*(\d+)\s+(\d+)\s+(\d+)", text, flags=re.S)
        if not m:
            return {"up": False}
        return {"up": True, "active": int(m[1]), "requests": int(m[4])}

    def instance(self, n, port):
        base = f"http://127.0.0.1:{port}"
        health, decisions, events = (self._get(base + p) for p in ("/health", "/metrics/decisions", "/metrics/events"))
        return {"n": n, "port": port, "up": health is not None,
                "model_version": (health or {}).get("model_version"),
                "scored": (decisions or {}).get("total"),
                "by_decision": (decisions or {}).get("by_decision"),
                "would_decline": (decisions or {}).get("would_decline"),
                "events": events}

    def kafka(self):
        try:
            from confluent_kafka import Consumer, TopicPartition
            if self._kafka is None:
                self._kafka = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": GROUP,
                                        "enable.auto.commit": False, "log_level": 2})
            c = self._kafka
            parts = [TopicPartition(TOPIC, p) for p in sorted(c.list_topics(TOPIC, timeout=1).topics[TOPIC].partitions)]
            end = sum(c.get_watermark_offsets(tp, timeout=1)[1] for tp in parts)
            committed = c.committed(parts, timeout=1)
            done = sum(tp.offset for tp in committed if tp.offset >= 0)
            return {"up": True, "partitions": len(parts), "end_offset": end, "committed": done, "lag": max(end - done, 0)}
        except Exception as e:   # broker down / starting: show it, keep the page alive
            if self._kafka is not None:
                try:
                    self._kafka.close()
                finally:
                    self._kafka = None
            return {"up": False, "error": type(e).__name__}

    def consumer(self):
        pid_file = ROOT / "deploy" / "run" / "consumer.pid"
        alive = False
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), 0)
                alive = True
            except (OSError, ValueError):
                pass
        metrics = None
        try:
            metrics = json.loads((STORE_DIR / "consumer_metrics.json").read_text())
        except (OSError, ValueError):
            pass
        return {"alive": alive, "duplicates": (metrics or {}).get("duplicates"),
                "dead_lettered": (metrics or {}).get("dead_lettered"),
                "latency_ms": (metrics or {}).get("event_latency_ms")}

    def snapshot(self):
        with self.lock:   # one poll at a time (two open tabs share it)
            ports = instance_ports()
            futures = [self.pool.submit(self.instance, i + 1, p) for i, p in enumerate(ports)]
            nginx = self.pool.submit(self.nginx)
            kafka = self.kafka()
            self.store.poll()
            return {"t": time.time(), "nginx": nginx.result(), "instances": [f.result() for f in futures],
                    "kafka": kafka, "consumer": self.consumer(),
                    "store": {"lines": self.store.lines, "file": self.store.path.name if self.store.path else None},
                    "feed": self.store.feed()}


app = FastAPI(title="NilePay live view", docs_url=None, redoc_url=None, openapi_url=None)
collector = Collector()


@app.get("/")
def page():
    return FileResponse(Path(__file__).with_name("live_dashboard.html"))


@app.get("/api/snapshot")
def snapshot():
    return collector.snapshot()


@app.post("/api/send/{sample}")
def send(sample: str):
    """Send one sample transaction through Nginx, like a client would."""
    if sample not in SAMPLES:
        return {"error": f"unknown sample {sample}"}
    body = {**SAMPLES[sample], "transaction_id": f"demo-{sample}-{int(time.time() * 1000) % 10_000_000}"}
    t0 = time.perf_counter()
    try:
        r = collector.http.post(NGINX_PUBLIC + "/predict", json=body, timeout=5)
        out = r.json()
        status = r.status_code
    except (httpx.HTTPError, ValueError) as e:
        out, status = {"error": type(e).__name__}, None
    return {"request": body, "status": status, "ms": round((time.perf_counter() - t0) * 1000, 1), "response": out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("DASHBOARD_PORT", "8090")), log_level="warning")
