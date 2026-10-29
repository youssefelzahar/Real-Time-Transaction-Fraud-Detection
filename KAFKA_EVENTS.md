# Prediction events on Kafka

Every `/predict` response is also published as an event to Kafka, so separate consumers can log, monitor and later store predictions **without touching the request path**. Scoring stays in the API instances; Kafka only receives the result.

## Architecture

```
Client ──► Nginx :8080 ──► instance 1..6 (uvicorn, FastAPI)
                              │  validate → features → XGBoost → DecisionEngine (shadow mode)
                              ├──────────────────────────────► HTTP response (contract unchanged)
                              │
                              └─ publisher.publish(event)       non-blocking: serialize + enqueue (~15 µs CPU)
                                   │  librdkafka bounded queue (100k/instance), own I/O thread,
                                   │  batching, lz4, reconnects, retries for up to 30 s;
                                   │  daemon thread fires delivery callbacks → counters
                                   ▼
                 Kafka 4.0 (KRaft, single node, 127.0.0.1:9092 only)
                 topic nilepay.prediction-events   6 partitions, 7 days, key = transaction_id
                                   ▼
                 python -m src.event_consumer      group nilepay-prediction-monitor, manual commits
                    ├─ valid             → JSONL event store (~/.local/share/nilepay-events/) + aggregates
                    ├─ duplicate event_id → skipped, counted (bounded in-memory window)
                    └─ invalid           → nilepay.prediction-events.dlq (1 partition, 30 days) + error headers
```

| Component | File | Responsibility |
|---|---|---|
| Producer | `src/events.py` | Build the event (whitelisted fields), serialize, enqueue without blocking, count every outcome (delivered, timed out, broker unavailable, buffer full, serialization error), bounded flush on shutdown |
| API hook | `api/main.py` | Creates one publisher per instance in the lifespan, publishes after the decision, exposes `GET /metrics/events` (internal) |
| Consumer | `src/event_consumer.py` | Validate (Pydantic schema v1), deduplicate, store with fsync, dead-letter invalid messages, commit offsets only after store/DLQ, retry transient store errors, stop without committing on persistent ones, report lag and latency |
| Broker | `deploy/kafka.sh`, `deploy/kafka/server.properties` | Install (pinned, SHA-512 verified), start/stop, create topics, show status. Native, no Docker |

**Library: `confluent-kafka` (librdkafka).** `produce()` is a C call that appends to a bounded in-memory queue and returns. Networking, batching and retries happen on librdkafka's own threads. It supports the idempotent producer. Rejected: `aiokafka` (network I/O on the API's event loop) and `kafka-python` (pure Python, slower, GIL-bound).

## Event (schema version 1)

```json
{"event_id": "eeaf803b-557e-4f31-bf9f-8f174ae50b86", "event_type": "fraud_prediction", "schema_version": 1,
 "occurred_at": "2026-10-28T18:01:17.360+00:00", "transaction_id": "wrk-2611", "transaction_type": "TRANSFER",
 "fraud_probability": 7.8e-06, "decision": "approve", "reason_code": "LOW_RISK", "would_decline": false,
 "threshold": 0.54, "model_version": "e7_v2", "decision_policy_version": "dp-e7-2026-10-26.1",
 "missing_features": [], "balance_source": {"sender": "request", "receiver": "request"},
 "instance_id": 6, "scoring_ms": 0.18}
```

- **Not published:** sender/receiver account IDs, amounts, balances. Fields come from an explicit whitelist (`RESPONSE_FIELDS`), so a new response field does not leak by default. A test checks this, and a scan of 1.6 million stored events found no such keys.
- `event_id` (uuid4) is the deduplication key. The message key is `transaction_id` (or `event_id` if absent), so events of one transaction stay ordered on one partition.
- `would_decline` keeps the shadow-mode signal: a `review / HIGH_RISK` event with `would_decline: true` is a transaction the policy would have declined.

## Configuration (environment)

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | unset | API, consumer | Broker list. **Unset = publishing off** (`NullPublisher`; API behaves exactly as before). Required by the consumer |
| `KAFKA_TOPIC_PREDICTIONS` | `nilepay.prediction-events` | API, consumer, kafka.sh | Event topic |
| `KAFKA_TOPIC_DLQ` | `nilepay.prediction-events.dlq` | consumer, kafka.sh | Dead-letter topic |
| `KAFKA_MESSAGE_TIMEOUT_MS` | `30000` | API | How long an event is retried before it fails as `timed_out` |
| `KAFKA_BUFFER_MAX_MESSAGES` | `100000` | API | Per-instance queue bound; above it events are dropped and counted |
| `KAFKA_LINGER_MS` | `5` | API | Batching delay |
| `KAFKA_COMPRESSION` | `lz4` | API | Batch compression |
| `KAFKA_SHUTDOWN_FLUSH_S` | `5` | API | Flush budget at instance shutdown |
| `KAFKA_CONSUMER_GROUP` | `nilepay-prediction-monitor` | consumer, kafka.sh | Consumer group |
| `EVENT_STORE_DIR` | `~/.local/share/nilepay-events` | consumer | JSONL store + `consumer_metrics.json`. Keep it on a Linux filesystem (see performance) |
| `CONSUMER_COMMIT_INTERVAL_S` / `CONSUMER_COMMIT_EVERY` | `5` / `50000` | consumer | fsync + offset commit cadence (whichever comes first) |
| `CONSUMER_DEDUP_SIZE` | `100000` | consumer | Dedup window (event IDs kept in memory) |
| `CONSUMER_MAX_RETRIES` | `3` | consumer | Retries of a failing store write before the consumer stops |
| `KAFKA_HOME`, `KAFKA_DATA_DIR`, `KAFKA_HEAP`, `KAFKA_PARTITIONS`, `KAFKA_CPUS` | `~/.local/nilepay-kafka`, `~/.local/share/nilepay-kafka`, `512m`, `6`, none | kafka.sh | Install dir, data dir, JVM heap, partitions, CPU pinning |
| `CPUS_CONSUMER` | none | start.sh | CPU pinning for the consumer |

No secrets are needed: the broker listens on loopback only, without authentication. Nothing is hardcoded in the API. If it is ever exposed beyond loopback it needs SASL/TLS, configured through these variables.

## Commands

```bash
# once: Java + Kafka
sudo apt install openjdk-21-jre-headless
deploy/kafka.sh install                     # Apache Kafka 4.0.0, SHA-512 verified, into ~/.local/nilepay-kafka

# start (Kafka first, then the consumer, then the API with publishing on)
deploy/kafka.sh start                       # formats storage on first run, creates both topics
export KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
deploy/start.sh consumer                    # log: deploy/run/consumer.log
MODEL_DIR=models/e7_v2 deploy/start.sh      # 6 instances + Nginx, publishing events

# stop
deploy/stop.sh; deploy/stop.sh consumer; deploy/kafka.sh stop
```

`deploy/start.sh` (no target) never starts Kafka or the consumer, so the API runs without them.

### Inspect

```bash
deploy/kafka.sh status                                     # topics + consumer-group offsets and lag
python -m src.event_consumer --status                      # lag per partition (exit 1 if the broker is unreachable)
curl -s localhost:8001/metrics/events | python3 -m json.tool   # one instance's producer counters (internal; 404 via :8080)
cat ~/.local/share/nilepay-events/consumer_metrics.json    # stored, duplicates, dead-lettered, lag, latency, aggregates
tail -f deploy/run/consumer.log
tail -1 ~/.local/share/nilepay-events/prediction-events-$(date -u +%F).jsonl
# raw messages / dead letters
~/.local/nilepay-kafka/kafka_2.13-4.0.0/bin/kafka-console-consumer.sh --bootstrap-server 127.0.0.1:9092 \
    --topic nilepay.prediction-events.dlq --from-beginning --property print.headers=true --max-messages 5
```

**Producer metrics** (`/metrics/events`, per instance): `enqueued`, `delivered`, `failed.{timed_out, broker_unavailable, other}`, `buffer_full_drops`, `serialization_errors`, `produce_errors`, `lost_on_shutdown`, `in_flight`, `queue_utilization`, `last_error`. **Consumer metrics:** `stored`, `duplicates`, `dead_lettered`, `retries`, `sink_failures`, `kafka_errors`, `commits`, `commit_deferred`, per-partition `lag`, event latency p50/p95 (`occurred_at` → stored), counts by decision / reason / model / policy / `would_decline`. Labels are low-cardinality only; no transaction IDs.

## Failure behavior and delivery semantics

| Case | What happens | API impact | Counted as |
|---|---|---|---|
| **Kafka healthy** | Delivered in milliseconds (idempotent producer, `acks=all`: producer retries do not create duplicates) | p95 +20-25 ms at saturation on this laptop (see below) | `delivered` |
| **Kafka temporarily down** | Events wait in the queue; librdkafka reconnects by itself. Back within 30 s → delivered. Not back → each event fails after `KAFKA_MESSAGE_TIMEOUT_MS` | None: 0 errors, responses unchanged | `failed.timed_out` (or `broker_unavailable`) + rate-limited warning |
| **Buffer full** (100k pending in one instance) | New events dropped immediately; memory stays bounded; request never blocks (tested: 1,000 drops at < 1 ms per call) | None | `buffer_full_drops` + rate-limited warning |
| **Instance shutdown with pending events** | Up to `KAFKA_SHUTDOWN_FLUSH_S` (5 s) flush; the remainder is logged | None (shutdown) | `lost_on_shutdown` |
| **Event delivered more than once** (consumer restart before commit, rebalance, broker restart) | Consumer skips `event_id`s seen in its window. Measured: 74,566 redeliveries during broker restarts, all skipped; 1.6 M stored lines, all with distinct `event_id` | None | consumer `duplicates` |
| Invalid message on the topic | Dead-lettered with headers `error`, `source_topic`, `source_partition`, `source_offset`, then committed. No retry loop | None | `dead_lettered` |
| Event store write fails | Retried 3 times with backoff; still failing → consumer commits what it already stored and exits with code 2 **without committing the failing event** | None | `sink_failures` |
| Broker down while the consumer runs | Commits deferred and retried; offsets kept; the consumer keeps running and catches up | None | `commit_deferred` |

**Delivery guarantee: at-least-once while the broker is reachable within `KAFKA_MESSAGE_TIMEOUT_MS`; otherwise best-effort, with every lost event counted.** It is **not** exactly-once end to end:

- An event lost before it reaches Kafka (outage > 30 s, buffer full, instance killed with `kill -9`) is gone. Only the counters record it.
- The consumer may receive an event more than once. Its dedup window is in memory and bounded (100k, per process), so a duplicate that arrives after a consumer restart, or after more than 100k other events, can be stored twice. Any database built on these events should use `event_id` as a unique key.

This is acceptable because the event is a **copy for monitoring and analytics**: the payment decision is returned synchronously in the HTTP response. If events become a system of record (audit, chargeback evidence, model retraining labels that must be complete), the upgrade path is a **transactional outbox**: write the event to a local database in the same transaction as the decision, and relay it to Kafka from there. That needs a database the API does not have today, and its write latency would sit inside every request, so it was evaluated and not built.

## Performance (measured 2026-10-28)

Setup: one laptop (Ryzen 7 5800H, AC power, `balanced` profile).
- CPUs 0–7: 6 instances + Nginx.
- CPUs 8–9: Kafka + consumer.
- CPUs 10–15: `wrk -t8 -c100 -d30s` through Nginx with varied payloads (`tests/wrk_predict.lua`).

Server-side figures are p95 from the Nginx log (`tests/nginx_log_report.py`). Each scenario ran on freshly started instances, after a 15 s disk settle. Two rounds were run for (a) and (b).

| Scenario | req/s | client p50 / p95 / p99 (ms) | server p95 (ms) | errors | events |
|---|---|---|---|---|---|
| (a) Kafka off, round 1 / 2 | 9,078 / 8,414 | 9.9 / **46.1** / 227 · 10.5 / **51.7** / 259 | 23.7 / 25.1 | 0 | — |
| (b) Kafka healthy + consumer, round 1 / 2 | 5,986 / 6,112 | 14.0 / **71.1** / 283 · 13.7 / **76.1** / 434 | 34.1 / 32.1 | 0 | 180,269 / 184,048 enqueued = delivered = stored; consumer lag 0 within 0.25 s after the run |
| (c1) broker restart (stop at +10 s, down +16 s, ready +40 s) | 5,183 | 14.8 / **187.1** / 528 | 46.4 | 0 | 207,592 enqueued: 161,537 delivered + 46,055 timed out (counted) |
| (c2) broker down ~63 s (+8.5 s … +71.9 s) | 8,429 | 10.2 / **60.1** / 263 | 27.3 | 0 | 253,819 enqueued: 24,570 delivered + 229,249 timed out (counted), 0 buffer drops |
| (d) 20 s after recovery, no restarts | 6,448 | 13.0 / **62.2** / 202 | 33.0 | 0 | 129,675 enqueued = delivered (reconnect works) |

Resources (averages over the run):
- **(a):** instances 577% CPU / 1,166 MB RSS in total; Nginx 75%.
- **(b):** instances 621% / 1,211 MB; Kafka 96–140% CPU / 620–700 MB; consumer 59–76% / 60–84 MB.
- **Consumer event latency** (`occurred_at` → stored): p95 8.5 ms in round 2 and 1.16 s in round 1, when the consumer was briefly behind during the load.

What this shows:
- **p95 < 100 ms holds with Kafka healthy (71–76 ms) and with Kafka down (60 ms), at saturation load.** It does **not** hold during a broker restart: p95 was 187 ms. Server-side p95 stayed at 46 ms, so the extra time is spent before Nginx handles the request. This matches the disk stalls described below; it was not isolated further.
- **Throughput at saturation drops about 30%** (8.4–9.1k → 6.0–6.1k req/s) while all three components share one machine. Measured contributions:
  - The producer path itself is small: about 15 µs CPU per event in isolation (5.5 µs to build the event, about 10 µs for librdkafka). On a single saturated instance it cost about 20% of that core's throughput.
  - The rest is co-location with the broker and consumer: CPU, cache and the shared disk. Spin loops on CPUs 8–9 alone cost only 0–10%.
  - These numbers would need confirming with Kafka on its own host.
- **The API never failed or blocked because of Kafka.** 0 non-200 responses in every scenario, including both outages.
- **Accounting is exact.** In every scenario, enqueued = delivered + timed out + dropped, and enqueued = requests in the Nginx log.

### Disk finding (why the event store lives outside the project)

This project lives on an NTFS partition of a 5400 rpm HDD (`/dev/sda`). That disk also holds `/`, the Kafka data and `deploy/run/` (Nginx and instance logs).

When the event store was in `deploy/run/events/` (NTFS) with frequent fsyncs, the disk ran at 76–84% utilization. Nginx and the instances then stalled on their own log writes: CPUs sat half idle and p95 rose to 445–623 ms (2.6–3.2k req/s). With the store on tmpfs, the same load gave 6.5–6.9k req/s and p95 60–63 ms.

| Event store, commit cadence | req/s (2 rounds) | p95 ms | disk util |
|---|---|---|---|
| NTFS, every 5 s | 3,537 / 4,006 | 389 / 361 | 59–65% |
| ext4, every 500 events | 5,956 / 7,076 | 177 / 52 | 32–75% |
| **ext4, every 5 s (default)** | **6,907 / 7,030** | **60 / 47** | **11–13%** |

Hence the defaults: store on ext4 (`~/.local/share/nilepay-events`) and fsync + commit every 5 s. Trade-off: a consumer crash replays up to ~5 s of events, which dedup absorbs within one process.

## Tests

```bash
pytest                                                              # unit + API tests (Kafka not needed)
deploy/kafka.sh start
RUN_KAFKA_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 pytest tests/test_kafka_integration.py
```

- `tests/test_events.py`: event fields and no PII; success; broker unavailable; timeout; buffer full (< 1 ms per call); serialization errors; shutdown loss; `NullPublisher`. Plus one real-librdkafka test against a closed port: `publish()` returns immediately and every event fails as counted.
- `tests/test_event_consumer.py`: valid / invalid JSON / schema violations / missing fields; duplicates; bounded dedup; transient and permanent store failures; per-partition offsets committed only after processing; commit batching; DLQ + commit; DLQ not acknowledged → commit deferred; commit `NOT_COORDINATOR` → retried, nothing lost; Kafka errors; metrics.
- `tests/test_api.py`: Kafka off by default; the event matches the response for approve, review and shadow `HIGH_RISK`; response unchanged; a failing, stuck or full publisher never breaks or slows `/predict`; `/metrics/events` hidden from OpenAPI.
- `tests/test_kafka_integration.py` (real broker, temporary topics): API → producer → Kafka → consumer → store, same decisions, offsets committed; invalid and duplicate messages → 1 stored, 1 duplicate, 2 dead-lettered with the right headers.

## Limitations

- **Single broker, replication factor 1, on the same laptop as the API.** No broker redundancy: a broker or disk failure loses events not yet consumed, and an outage over 30 s loses new events (counted). Production needs a ≥3-broker cluster on separate hosts, `min.insync.replicas=2`, and TLS/SASL.
- **Events can be lost by design** (outage > 30 s, buffer full, `kill -9`, instance crash). The loss is counted per instance; nothing re-sends it. Counters reset when an instance restarts.
- **Dedup is per consumer process and bounded** (100k). A duplicate across a consumer restart can be stored twice; downstream storage should key on `event_id`.
- **The JSONL store is not a database:** daily files with no rotation or retention (about 0.8 GB per day at this test load), and no query API. The sink interface (`write`/`flush`) is where a database sink would plug in.
- **One consumer process.** Six partitions allow up to six consumers in the group; only one was run and measured.
- **p95 rose to 187 ms during a broker restart** on this machine (shared HDD). The 100 ms target was met with the broker healthy or down, not while it restarted.
- **All measurements share one machine and one slow disk.** The 30% saturation-throughput cost mostly reflects co-location; it should be re-measured with the broker on separate hardware.
