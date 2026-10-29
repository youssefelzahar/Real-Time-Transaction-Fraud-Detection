"""Prediction events published to Kafka, off the request's critical path.

Delivery is best-effort with every loss counted (see KAFKA_EVENTS.md):
  - publish() only serializes the event and appends it to librdkafka's bounded
    in-memory queue (~microseconds). It never waits for the broker and never raises.
  - librdkafka's own threads batch, send, reconnect and retry for up to
    KAFKA_MESSAGE_TIMEOUT_MS; a daemon thread here fires delivery callbacks.
  - A full queue, a serialization error or a delivery failure is counted, not hidden.
Kafka is off unless KAFKA_BOOTSTRAP_SERVERS is set (NullPublisher).

Thread safety: counters written in publish() are only written by the request thread;
counters written in _on_delivery() are only written by the poll thread (or by flush()
at shutdown, after the poll thread has stopped). No counter has two writers.
"""
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

log = logging.getLogger("nilepay.events")

EVENT_TYPE = "fraud_prediction"
SCHEMA_VERSION = 1
DEFAULT_TOPIC = "nilepay.prediction-events"

# Only these response fields leave the API. Account IDs, amounts and balances never do.
RESPONSE_FIELDS = ("transaction_id", "fraud_probability", "decision", "reason_code", "threshold",
                   "decision_policy_version", "model_version", "missing_features", "balance_source")


def build_prediction_event(response, *, transaction_type, would_decline, instance_id=None, scoring_ms=None):
    """Event for one /predict response. `response` is the ScoreResponse as a dict."""
    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": EVENT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "occurred_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "transaction_type": transaction_type,
        "would_decline": bool(would_decline),
        "instance_id": instance_id,
        "scoring_ms": round(scoring_ms, 3) if scoring_ms is not None else None,
    }
    event.update({k: response[k] for k in RESPONSE_FIELDS})
    return event


class RateLimitedLog:
    """Log a recurring warning at most once per `interval` seconds, with a suppressed count."""

    def __init__(self, interval=10.0):
        self.interval = interval
        self._last = {}
        self._suppressed = {}

    def warning(self, key, msg, *args):
        now = time.monotonic()
        if now - self._last.get(key, -self.interval) >= self.interval:
            n = self._suppressed.pop(key, 0)
            log.warning(msg + (f" ({n} similar suppressed)" if n else ""), *args)
            self._last[key] = now
        else:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1


class NullPublisher:
    """Used when Kafka is not configured: publishing is a no-op."""

    enabled = False

    def publish(self, event):
        return False

    def snapshot(self):
        return {"kafka_enabled": False}

    def close(self):
        pass


def _failure_class(err):
    from confluent_kafka import KafkaError
    if err.code() == KafkaError._MSG_TIMED_OUT:
        return "timed_out"
    if err.code() in (KafkaError._TRANSPORT, KafkaError._ALL_BROKERS_DOWN, KafkaError._RESOLVE,
                      KafkaError.NETWORK_EXCEPTION):
        return "broker_unavailable"
    return "other"


class KafkaEventPublisher:
    enabled = True

    def __init__(self, config, topic=DEFAULT_TOPIC, *, producer=None, shutdown_flush_s=5.0, poll_interval_s=0.1):
        if producer is None:
            from confluent_kafka import Producer
            producer = Producer({**config, "logger": logging.getLogger("nilepay.kafka")})
        self._producer = producer
        self.topic = topic
        self.capacity = int(config.get("queue.buffering.max.messages", 100_000))
        self.shutdown_flush_s = shutdown_flush_s
        self._warn = RateLimitedLog()
        # Written by the request thread.
        self.enqueued = 0
        self.buffer_full_drops = 0
        self.serialization_errors = 0
        self.produce_errors = 0
        # Written by the delivery-callback (poll) thread.
        self.delivered = 0
        self.failed = {"timed_out": 0, "broker_unavailable": 0, "other": 0}
        self.last_error = None
        self.last_error_at = None
        self.lost_on_shutdown = 0

        self._stop = threading.Event()
        self._poll_interval_s = poll_interval_s
        self._thread = threading.Thread(target=self._poll_loop, name="kafka-delivery", daemon=True)
        self._thread.start()

    @classmethod
    def from_env(cls, client_id):
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap:
            return NullPublisher()
        config = {
            "bootstrap.servers": bootstrap,
            "client.id": client_id,
            "acks": "all",
            "enable.idempotence": True,
            "linger.ms": int(os.getenv("KAFKA_LINGER_MS", "5")),
            "compression.type": os.getenv("KAFKA_COMPRESSION", "lz4"),
            "queue.buffering.max.messages": int(os.getenv("KAFKA_BUFFER_MAX_MESSAGES", "100000")),
            "message.timeout.ms": int(os.getenv("KAFKA_MESSAGE_TIMEOUT_MS", "30000")),
            "reconnect.backoff.max.ms": 2000,
        }
        return cls(config, topic=os.getenv("KAFKA_TOPIC_PREDICTIONS", DEFAULT_TOPIC),
                   shutdown_flush_s=float(os.getenv("KAFKA_SHUTDOWN_FLUSH_S", "5")))

    def _poll_loop(self):
        while not self._stop.is_set():
            self._producer.poll(self._poll_interval_s)

    def _on_delivery(self, err, msg):
        if err is None:
            self.delivered += 1
            return
        kind = _failure_class(err)
        self.failed[kind] += 1
        self.last_error = f"{kind}: {err.name()}"
        self.last_error_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._warn.warning(kind, "prediction event not delivered: %s", err.name())

    def publish(self, event):
        """Enqueue one event. Never blocks on the broker and never raises."""
        try:
            payload = json.dumps(event, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError) as e:
            self.serialization_errors += 1
            self._warn.warning("serialization", "prediction event not serializable: %s", type(e).__name__)
            return False
        key = (event.get("transaction_id") or event.get("event_id") or "").encode()
        try:
            self._producer.produce(self.topic, value=payload, key=key, on_delivery=self._on_delivery)
        except BufferError:
            self.buffer_full_drops += 1
            self._warn.warning("buffer_full", "event buffer full (%d pending): event dropped", self.capacity)
            return False
        except Exception as e:  # KafkaException and anything else: count, never break /predict
            self.produce_errors += 1
            self._warn.warning("produce", "could not enqueue prediction event: %s", type(e).__name__)
            return False
        self.enqueued += 1
        return True

    def snapshot(self):
        in_flight = len(self._producer)
        return {
            "kafka_enabled": True,
            "topic": self.topic,
            "enqueued": self.enqueued,
            "delivered": self.delivered,
            "failed": dict(self.failed),
            "buffer_full_drops": self.buffer_full_drops,
            "serialization_errors": self.serialization_errors,
            "produce_errors": self.produce_errors,
            "lost_on_shutdown": self.lost_on_shutdown,
            "in_flight": in_flight,
            "queue_capacity": self.capacity,
            "queue_utilization": round(in_flight / self.capacity, 4) if self.capacity else 0.0,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
        }

    def close(self):
        """Stop the poll thread, give pending events a bounded flush, count what is left."""
        self._stop.set()
        self._thread.join(timeout=2)
        remaining = self._producer.flush(self.shutdown_flush_s)
        self.lost_on_shutdown = remaining
        if remaining:
            log.warning("shutdown: %d prediction events still undelivered after %.1fs flush; counted as lost",
                        remaining, self.shutdown_flush_s)
