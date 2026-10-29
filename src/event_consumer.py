"""Consume prediction events: validate, deduplicate, store, monitor (separate process).

    python -m src.event_consumer            # run until SIGINT/SIGTERM (or SIGBREAK / NILEPAY_STOP_FILE on Windows)
    python -m src.event_consumer --status   # consumer-group lag per partition, then exit

Delivery: at-least-once. Offsets are committed only after an event has been written
(and fsynced) to the event store, or forwarded to the dead-letter topic.
  - invalid JSON / schema   -> dead-letter topic with error headers, then committed (no retry loop)
  - duplicate event_id      -> skipped and counted (bounded in-memory window)
  - transient store error   -> retried with backoff; still failing -> stop WITHOUT committing
                               (exit code 2) so a restart reprocesses the event
  - Kafka unavailable       -> commits are deferred (offsets kept) and retried; the client
                               reconnects by itself, the process keeps running
Configuration comes from environment variables (see KAFKA_EVENTS.md).
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.shutdown import stop_file_from_env, watch_stop_file

log = logging.getLogger("nilepay.consumer")

DEFAULT_TOPIC = "nilepay.prediction-events"
DEFAULT_DLQ = "nilepay.prediction-events.dlq"
DEFAULT_GROUP = "nilepay-prediction-monitor"
EXIT_SINK_FAILURE = 2


class PredictionEvent(BaseModel):
    """Schema v1 of a fraud_prediction event (extra fields are ignored for forward compatibility)."""
    model_config = ConfigDict(extra="ignore")

    event_id: UUID
    event_type: Literal["fraud_prediction"]
    schema_version: Literal[1]
    occurred_at: datetime
    transaction_id: Optional[str]
    transaction_type: Optional[str]
    fraud_probability: float = Field(ge=0, le=1)
    decision: Literal["approve", "review", "decline"]
    reason_code: Literal["LOW_RISK", "REVIEW_THRESHOLD", "HIGH_RISK"]
    would_decline: bool
    threshold: float
    model_version: str
    decision_policy_version: str
    missing_features: list[str]
    balance_source: dict[str, str]
    instance_id: Optional[int] = None
    scoring_ms: Optional[float] = None


class InvalidEvent(Exception):
    pass


class SinkFailure(Exception):
    """The event store kept failing after all retries."""


class JsonlSink:
    """Append-only daily JSONL files. A placeholder for a database: anything with
    write(event_dict) and flush() can replace it."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._day = None
        self._file = None

    def write(self, event):
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day != self._day:
            if self._file:
                self._file.close()
            self._file = open(self.directory / f"prediction-events-{day}.jsonl", "a", encoding="utf-8")
            self._day = day
        self._file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def flush(self):
        """Make written events durable before their offsets are committed."""
        if self._file:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self):
        if self._file:
            self.flush()
            self._file.close()
            self._file = None


class EventProcessor:
    """Kafka-independent processing of one message value; unit-testable."""

    def __init__(self, sink, dedup_size=100_000, max_retries=3, backoff_s=0.2, sleep=time.sleep):
        self.sink = sink
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self._seen = OrderedDict()
        self._dedup_size = dedup_size
        self.counts = Counter()
        self.aggregates = {k: Counter() for k in ("decision", "reason_code", "model_version",
                                                  "decision_policy_version", "transaction_type")}
        self.latency_ms = deque(maxlen=10_000)   # occurred_at -> processed

    @staticmethod
    def parse(value):
        if value is None:
            raise InvalidEvent("empty message")
        try:
            data = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise InvalidEvent(f"invalid JSON: {type(e).__name__}") from e
        if not isinstance(data, dict):
            raise InvalidEvent("event is not a JSON object")
        try:
            return PredictionEvent.model_validate(data)
        except ValidationError as e:
            fields = sorted({".".join(map(str, err["loc"])) for err in e.errors()})
            raise InvalidEvent(f"schema violation: {', '.join(fields)}") from e

    def _remember(self, event_id):
        self._seen[event_id] = None
        if len(self._seen) > self._dedup_size:
            self._seen.popitem(last=False)

    def handle(self, value):
        """Returns "stored" or "duplicate"; raises InvalidEvent or SinkFailure."""
        event = self.parse(value)
        event_id = str(event.event_id)
        if event_id in self._seen:
            self.counts["duplicates"] += 1
            return "duplicate"
        record = event.model_dump(mode="json")
        for attempt in range(self.max_retries + 1):
            try:
                self.sink.write(record)
                break
            except OSError as e:
                if attempt == self.max_retries:
                    self.counts["sink_failures"] += 1
                    raise SinkFailure(f"event store failed {attempt + 1} times: {e}") from e
                self.counts["retries"] += 1
                log.warning("event store error (%s), retry %d/%d", type(e).__name__, attempt + 1, self.max_retries)
                self.sleep(self.backoff_s * 2 ** attempt)
        self._remember(event_id)
        self.counts["stored"] += 1
        self.aggregates["decision"][event.decision] += 1
        self.aggregates["reason_code"][event.reason_code] += 1
        self.aggregates["model_version"][event.model_version] += 1
        self.aggregates["decision_policy_version"][event.decision_policy_version] += 1
        self.aggregates["transaction_type"][event.transaction_type or "unknown"] += 1
        self.counts["would_decline"] += event.would_decline
        self.latency_ms.append((datetime.now(timezone.utc) - event.occurred_at).total_seconds() * 1000)
        return "stored"


class ConsumerRunner:
    """Poll loop: process, dead-letter, commit after durable processing."""

    def __init__(self, consumer, processor, dlq_producer, dlq_topic, *, commit_every=50000,
                 commit_interval_s=5.0, metrics_path=None, metrics_interval_s=10.0, clock=time.monotonic):
        self.consumer = consumer
        self.processor = processor
        self.dlq_producer = dlq_producer
        self.dlq_topic = dlq_topic
        self.commit_every = commit_every
        self.commit_interval_s = commit_interval_s
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.metrics_interval_s = metrics_interval_s
        self.clock = clock
        self.pending = {}          # (topic, partition) -> next offset to commit
        self.pending_count = 0
        self._last_commit = clock()
        self._last_metrics = clock()
        self.counts = Counter()
        self.stopping = False
        self.exit_code = 0

    def stop(self, *_):
        self.stopping = True

    def _dead_letter(self, msg, reason):
        headers = [("error", reason.encode()), ("source_topic", msg.topic().encode()),
                   ("source_partition", str(msg.partition()).encode()),
                   ("source_offset", str(msg.offset()).encode())]
        self.dlq_producer.produce(self.dlq_topic, value=msg.value(), key=msg.key(), headers=headers)
        self.dlq_producer.poll(0)
        self.counts["dead_lettered"] += 1

    def commit(self):
        """Commit processed offsets. Returns False (and keeps them pending) when Kafka can't
        take the commit right now - e.g. broker restarting (NOT_COORDINATOR) or dead letters
        not yet acknowledged. Pending offsets only move forward, so the next successful
        commit covers everything processed meanwhile."""
        if not self.pending:
            return True
        # Durability first: events in the store, dead letters acknowledged by Kafka.
        try:
            self.processor.sink.flush()
        except OSError as e:
            raise SinkFailure(f"event store flush failed: {e}") from e
        self._last_commit = self.clock()        # also paces retries while Kafka is unavailable
        if self.dlq_producer is not None and self.dlq_producer.flush(5) > 0:
            return self._commit_deferred("dead-letter topic did not acknowledge")
        from confluent_kafka import KafkaException, TopicPartition
        offsets = [TopicPartition(t, p, o) for (t, p), o in self.pending.items()]
        try:
            self.consumer.commit(offsets=offsets, asynchronous=False)
        except KafkaException as e:
            return self._commit_deferred(f"commit failed: {e.args[0].name() if e.args else e}")
        if self.counts["commit_deferred_streak"]:
            log.info("commits resumed after %d deferred attempts", self.counts["commit_deferred_streak"])
            self.counts["commit_deferred_streak"] = 0
        self.counts["commits"] += 1
        self.pending.clear()
        self.pending_count = 0
        return True

    def _commit_deferred(self, reason):
        self.counts["commit_deferred"] += 1
        self.counts["commit_deferred_streak"] += 1
        if self.counts["commit_deferred_streak"] in (1, 10, 100) or self.counts["commit_deferred_streak"] % 1000 == 0:
            log.warning("commit deferred (%s); %d partitions pending, will retry", reason, len(self.pending))
        return False

    def handle_message(self, msg):
        if msg.error():
            from confluent_kafka import KafkaError
            if msg.error().code() != KafkaError._PARTITION_EOF:
                self.counts["kafka_errors"] += 1
                log.warning("consumer error: %s", msg.error())
            return
        try:
            self.processor.handle(msg.value())
        except InvalidEvent as e:
            self._dead_letter(msg, str(e))
        self.pending[(msg.topic(), msg.partition())] = msg.offset() + 1
        self.pending_count += 1

    def lag(self):
        # Runs inside the poll loop: one batched position()/committed() call, not one round trip per
        # partition. Watermarks come from the fetch responses (cached), so no broker request either.
        assigned = self.consumer.assignment()
        if not assigned:
            return {}
        positions = {(tp.topic, tp.partition): tp.offset for tp in self.consumer.position(assigned)}
        unknown = [tp for tp in assigned if positions[(tp.topic, tp.partition)] < 0]
        if unknown:   # nothing fetched yet since assignment (e.g. just restarted): resume point
            for tp in self.consumer.committed(unknown, timeout=1.0):
                positions[(tp.topic, tp.partition)] = tp.offset
        lag = {}
        for tp in assigned:
            low, high = self.consumer.get_watermark_offsets(tp, timeout=1.0, cached=True)
            if high < 0:   # no fetch response for this partition yet: ask the broker once
                low, high = self.consumer.get_watermark_offsets(tp, timeout=1.0, cached=False)
            position = positions[(tp.topic, tp.partition)]
            lag[f"{tp.topic}[{tp.partition}]"] = max(high - (position if position >= 0 else low), 0)
        return lag

    def metrics(self):
        p = self.processor
        lat = sorted(p.latency_ms)
        pct = (lambda q: round(lat[min(int(q * len(lat)), len(lat) - 1)], 1)) if lat else (lambda q: None)
        try:
            lag = self.lag()
        except Exception as e:  # lag is best-effort observability
            lag = {"error": type(e).__name__}
        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stored": p.counts["stored"], "duplicates": p.counts["duplicates"],
            "dead_lettered": self.counts["dead_lettered"], "retries": p.counts["retries"],
            "sink_failures": p.counts["sink_failures"], "kafka_errors": self.counts["kafka_errors"],
            "commits": self.counts["commits"], "commit_deferred": self.counts["commit_deferred"],
            "would_decline": p.counts["would_decline"],
            "event_latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1] if lat else None},
            "lag": lag, "lag_total": sum(v for v in lag.values() if isinstance(v, int)),
            "by": {k: dict(v) for k, v in p.aggregates.items()},
        }

    def _maybe_report(self, force=False):
        if not force and self.clock() - self._last_metrics < self.metrics_interval_s:
            return
        m = self.metrics()
        log.info("events stored=%s duplicates=%s dead_lettered=%s lag=%s latency_p95=%sms",
                 m["stored"], m["duplicates"], m["dead_lettered"], m["lag_total"], m["event_latency_ms"]["p95"])
        if self.metrics_path:
            tmp = self.metrics_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(m, indent=2))
            tmp.replace(self.metrics_path)
        self._last_metrics = self.clock()

    def run(self, poll_timeout_s=1.0):
        try:
            while not self.stopping:
                msg = self.consumer.poll(poll_timeout_s)
                if msg is not None:
                    self.handle_message(msg)
                if self.pending_count >= self.commit_every or \
                        (self.pending and self.clock() - self._last_commit >= self.commit_interval_s):
                    self.commit()
                self._maybe_report()
            # Graceful shutdown: the current message was finished; commit what was processed.
            if not self.commit():
                log.warning("final commit deferred: %d partitions will be re-read after restart (at-least-once)",
                            len(self.pending))
        except SinkFailure as e:
            # The failing event is not in `pending`, so it is never committed and a restart
            # reprocesses it (at-least-once). Events stored before it are committed if possible.
            log.error("stopping without committing the failing event: %s", e)
            self.exit_code = EXIT_SINK_FAILURE
            try:
                self.commit()
            except Exception as commit_error:
                log.error("could not commit events processed before the failure: %s", commit_error)
        finally:
            self._maybe_report(force=True)
        return self.exit_code


def kafka_config():
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap:
        raise SystemExit("KAFKA_BOOTSTRAP_SERVERS is not set")
    return bootstrap


def status(bootstrap, topic, group):
    """Committed offset vs end of log, per partition, for the consumer group."""
    from confluent_kafka import Consumer, KafkaException, TopicPartition
    c = Consumer({"bootstrap.servers": bootstrap, "group.id": group, "enable.auto.commit": False,
                  "log_level": 2})   # one-shot command: no per-retry connect errors on stderr
    try:
        meta = c.list_topics(topic, timeout=5)
        parts = [TopicPartition(topic, p) for p in sorted(meta.topics[topic].partitions)]
        committed = c.committed(parts, timeout=5)
        total = 0
        print(f"group {group} on {topic}")
        print(f"{'partition':>9} {'committed':>10} {'end':>10} {'lag':>8}")
        for tp in committed:
            low, high = c.get_watermark_offsets(tp, timeout=5)
            pos = tp.offset if tp.offset >= 0 else low
            total += max(high - pos, 0)
            print(f"{tp.partition:>9} {tp.offset if tp.offset >= 0 else '-':>10} {high:>10} {max(high - pos, 0):>8}")
        print(f"total lag: {total}")
        return 0
    except KafkaException as e:   # broker down or still starting (e.g. NOT_COORDINATOR)
        err = e.args[0] if e.args else e
        print(f"lag unavailable: {err.name() if hasattr(err, 'name') else err}", file=sys.stderr)
        return 1
    finally:
        c.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="print consumer-group lag and exit")
    args = ap.parse_args()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bootstrap = kafka_config()
    topic = os.getenv("KAFKA_TOPIC_PREDICTIONS", DEFAULT_TOPIC)
    group = os.getenv("KAFKA_CONSUMER_GROUP", DEFAULT_GROUP)
    if args.status:
        return status(bootstrap, topic, group)

    from confluent_kafka import Consumer, Producer
    # Default is a Linux filesystem outside the project: fsyncs on the NTFS project drive
    # stalled the API's own log writes on the shared disk (measured, see KAFKA_EVENTS.md).
    store_dir = os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events"))
    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": group, "client.id": "nilepay-event-consumer",
                         "enable.auto.commit": False, "auto.offset.reset": "earliest",
                         "logger": logging.getLogger("nilepay.kafka")})
    dlq = Producer({"bootstrap.servers": bootstrap, "client.id": "nilepay-event-consumer-dlq",
                    "acks": "all", "enable.idempotence": True})
    processor = EventProcessor(JsonlSink(store_dir),
                               dedup_size=int(os.getenv("CONSUMER_DEDUP_SIZE", "100000")),
                               max_retries=int(os.getenv("CONSUMER_MAX_RETRIES", "3")))
    runner = ConsumerRunner(consumer, processor, dlq, os.getenv("KAFKA_TOPIC_DLQ", DEFAULT_DLQ),
                            commit_every=int(os.getenv("CONSUMER_COMMIT_EVERY", "50000")),
                            commit_interval_s=float(os.getenv("CONSUMER_COMMIT_INTERVAL_S", "5")),
                            metrics_path=Path(store_dir) / "consumer_metrics.json")
    signal.signal(signal.SIGTERM, runner.stop)
    signal.signal(signal.SIGINT, runner.stop)
    if hasattr(signal, "SIGBREAK"):           # Windows: Ctrl+Break in the consumer's own console
        signal.signal(signal.SIGBREAK, runner.stop)
    stop_file = stop_file_from_env()        # Windows: deploy/windows.py stop (see src/shutdown.py)
    if stop_file:
        watch_stop_file(stop_file, runner.stop)
    consumer.subscribe([topic])
    log.info("consuming %s as group %s; store %s", topic, group, store_dir)
    try:
        code = runner.run()
    finally:
        started = time.monotonic()
        consumer.close()          # leaves the group cleanly (fast rebalance)
        processor.sink.close()
        log.info("consumer closed and left group %s in %.2fs (exit code %s)", group, time.monotonic() - started,
                 runner.exit_code)
    return code


if __name__ == "__main__":
    sys.exit(main())
