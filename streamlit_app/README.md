# NilePay fraud detection UI (Streamlit)

A small UI over the running system. It submits a transaction through the public API and shows what the API returned. It also shows the system's health and recent transactions.

It holds **no model and makes no decisions**: every score and decision comes from the existing API (Nginx → FastAPI instances → model `e7_v1` → Decision Engine). It never changes the backend.

## Run

```bash
pip install -r streamlit_app/requirements.txt   # once (Streamlit, requests); the API's requirements are unchanged
deploy/start.sh                                 # the API behind Nginx (see RUN.md); Kafka/consumer optional
python -m monitoring.live_dashboard &           # optional: Kafka, consumer and the event feed (port 8090)
streamlit run streamlit_app/app.py              # http://localhost:8501
```

Run it on the same host as the stack: the instance metrics endpoints only listen on loopback.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `FRAUD_API_BASE_URL` | `http://127.0.0.1:8080` | Public entry point (Nginx). Transactions go to `POST <url>/predict` |
| `FRAUD_MONITOR_URL` | `http://127.0.0.1:8090` | The live monitor (`monitoring/live_dashboard.py`): Kafka state, consumer state, event feed |
| `FRAUD_INSTANCE_URLS` | the `upstream` servers in `deploy/nginx/nginx.conf` | Comma-separated instance URLs, for `/health` and the internal `/metrics/*` |
| `FRAUD_UI_TRAFFIC_SOURCE` | `demo` | `X-Traffic-Source` tag on submitted transactions: `demo`, `test` or `production` |
| `FRAUD_UI_TIMEOUT_S` | `5` | Timeout for a submission. Monitoring reads use at most 2 s |
| `FRAUD_UI_REFRESH_S` | `0` (off) | Dashboard auto-refresh interval. Any value is raised to at least 10 s |

**Why `demo`:** transactions typed into the UI are not real customer traffic. Tagged `demo`, they're stored and monitored in their own series and never count as production in drift detection. In Phase 1, untagged load-test traffic produced false drift alerts. Use `production` only if real transactions are entered here.

## Pages

### Dashboard

| Shows | Source | Notes |
|---|---|---|
| Transactions processed; approve, review and decline counts; would-decline (shadow) | each instance's internal `/metrics/decisions` | Counters since each instance's last start, summed over the instances that answer. Not a lifetime total |
| API availability | public `/health` through Nginx, plus each instance's `/health` | The state right now. An uptime percentage isn't exposed |
| Model version, decision policy and decline mode | `/health`, `/metrics/decisions` | |
| Health of API, model, Kafka pipeline and consumer: Healthy / Degraded / Unavailable / Unknown | the above, plus the live monitor's `/api/snapshot` | Kafka and consumer show **Unknown** if the live monitor isn't running. The UI doesn't guess |
| Decision-step latency (µs) | `/metrics/decisions` histogram | p50/p95/p99 are **bucket bounds** (for example "≤ 25 µs"). Covers only the threshold decision |
| Event-pipeline latency (ms) | consumer metrics via the live monitor | Prediction → stored; p50, p95, max (the consumer reports no p99) |
| End-to-end API latency | **not available** | The running system doesn't expose it. See RELIABILITY_TESTING.md for load-test figures |

### Transaction Analyzer

The form follows the API contract (`TransactionRequest` in `api/main.py`):
- **Type** (required): CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER.
- **Amount** (required): must be finite and ≥ 0.
- **Sender and receiver ID** (required).
- **Transaction ID** (optional).
- **Balances** (optional), finite and ≥ 0. They're fallbacks the API uses only if its account service has no balance. **A balance left empty is not sent and never replaced by 0**; the response's `balance_source` says what the model used.
- **`step`** (optional, under a collapsed section): used only by E5 models; ignored by `e7_v1`.

The result shows the API response as returned:
- the decision (APPROVE / REVIEW / DECLINE);
- the fraud score (not a calibrated probability), the reason code and the approve threshold;
- the model and policy versions, balance sources and missing features;
- the raw JSON.

**Shadow mode.** With the current policy, declines are not enforced: a score at or above the decline threshold returns **REVIEW with reason `HIGH_RISK`** and is counted as a would-decline. The UI says so explicitly and never says a payment was blocked. The decline mode is read from the instances; if they can't be reached, the UI says it's unknown.

**Errors.** Each of these gets a short message, and no prediction is shown:
- connection failure;
- timeout;
- 422 (field errors listed);
- 500, 503 and 504;
- non-JSON or incomplete responses.

**A submission is sent exactly once.** There's no automatic retry, since a retry could score and publish the same transaction twice; after a 504 the transaction may already have been scored. The "UI round-trip" shown under a result is measured by the UI and is not production latency.

### Recent Transactions

The backend has **no transaction-history endpoint**, and none was added. The page shows two things:
1. **"Transactions submitted during this session"**, from Streamlit session state, including failed submissions. It is lost when the browser session ends.
2. **"Latest events in the event store"**: the last 14 stored prediction events from the live monitor's feed, all traffic sources. It isn't shown if the live monitor isn't running.

Neither is a complete history.

## Not changed

- The API, its contract and Nginx.
- Model artifacts, thresholds, the Decision Engine and shadow mode.
- The Kafka producer and consumer, and the live monitor.

The UI reads `deploy/nginx/nginx.conf` only to list the instance ports.

## Tests

`pytest tests/test_streamlit_ui.py` runs without network, using a fake HTTP layer. It covers:
- input validation;
- every error kind, with exactly one POST each;
- status derivation;
- the shadow-mode text;
- smoke tests of each page through Streamlit's `AppTest`.

The tests are skipped if Streamlit isn't installed.
