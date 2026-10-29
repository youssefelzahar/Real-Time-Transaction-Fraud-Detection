# ML monitoring

This shows how the model behaves in production. It covers:
- prediction volume, the fraud-score distribution and decisions, including shadow-mode would-declines;
- input data quality compared with the model's training data;
- model performance, once actual fraud labels exist.

The monitoring runs in the existing Kafka consumer and shows up in the existing dashboard. It adds no work to `/predict` except building a small summary for each event.

**Status (2026-10-29): there is no source of actual fraud labels.** Production precision, recall, PR-AUC and cost therefore **cannot be calculated yet**. The interfaces and the evaluation are implemented and tested, and the dashboard shows "Unavailable: no actual labels ingested". No labels or metrics are made up.

## 1. What existed before

| Already there | Where | Reused? |
|---|---|---|
| Per-instance decision counters (approve/review/decline, reasons, would_decline, invalid scores), per process | `DecisionMetrics`, `/metrics/decisions` | Yes: shown as "current processes" |
| Producer counters (enqueued, delivered, not sent, not confirmed) and the ledger of ended processes | `src/events.py`, `/metrics/events` | Yes: "events not delivered" |
| Consumer counters (stored, duplicates, dead-lettered, lag, latency), process lifetime, in memory | `consumer_metrics.json` | Yes: dead-lettered and lag/freshness |
| Event store of every validated prediction (JSONL) | `~/.local/share/nilepay-events/` | Yes: the join target for labels |
| Training metrics (validation and test P/R/PR-AUC/cost), split boundaries, per-type medians | `models/<v>/metadata.json`, `pipeline.json` | Yes: baseline provenance and the tuning split |
| `confusion_counts`, `business_cost`, `evaluate_at_threshold` | `src/evaluate.py` | Yes: same cost function |

**What was missing:**
- rolling windows;
- persistence across restarts;
- per-version separation in the consumer;
- feature distributions;
- any training baseline;
- any label source.

`monitoring/drift_monitor.py` was an empty placeholder.

### Fields available for monitoring (actual event, schema 1)
- **Identity and time:** `event_id`, `occurred_at` (prediction time, UTC), `transaction_id` (may be null), `transaction_type`.
- **Model output:** `fraud_probability`, `decision`, `reason_code`, `would_decline`, `threshold`, `model_version`, `decision_policy_version`.
- **Input quality:** `missing_features` (features given to the model as NaN) and `balance_source` (where each balance came from: account_service, request or missing).
- **Serving:** `instance_id`, `scoring_ms`.
- **Added by this change** (optional, so older events stay valid): `feature_bins`, `feature_flags`, `non_finite_features` (see §3).
- **Never in events:** account IDs, exact amounts, exact balances.

### Fields that don't exist, so these metrics are unavailable
- **Unknown transaction types:** the API rejects them with 422 (`type` is one of the five training types), so they never reach the model or an event. They are counted as **input rejections** per instance (`/metrics/decisions` → `input_rejections.by_field.type`). The event-level `unknown_type` metric exists for events published by other producers or a future relaxed schema; today it is 0 by construction.
- **Negative amounts or balances:** rejected the same way (counted under `amount`, `sender_balance`, ...).
- **Non-finite inputs do get through (found while verifying this).** Python's JSON parser accepts `Infinity` and `NaN`, and `TransactionRequest` does not set `allow_inf_nan=False`. Probed on the live stack:
  - **`"amount": Infinity`** passes validation, is scored, and is **approved** (score 6e-7, HTTP 200). Monitoring records it as `non_finite_features: ["amount", "amt_vs_type_median"]`, so it is visible on the dashboard.
  - **`"amount": NaN`** fails `ge=0`, but FastAPI's default 422 handler then cannot serialize the NaN back, so the client gets **HTTP 500**. It is counted as an input rejection.
  - Both behaviours predate this change; the request validation was not modified here. **Recommended fix:** `ConfigDict(allow_inf_nan=False)` on `TransactionRequest`.

## 2. Architecture

```
POST /predict ── score ── DecisionEngine ── response
       └─ publish_event: + feature_summary (bins/flags, 7.7 µs) ──► Kafka (non-blocking enqueue)

event consumer (existing process)
  poll ─► validate ─► dedup ─► store JSONL ─► MonitoringAggregator.observe(event, partition, offset)
  every 5 s commit(): fsync store ─► DLQ ack ─► aggregator.save() (atomic) ─► Kafka offset commit

dashboard (127.0.0.1:8090, loopback only)
  /api/ml?window=5m|1h|24h ─► read ml_monitoring_state.json + baseline JSON ─► report
                           └► PerformanceCache (60 s) ─► labels/*.jsonl ⋈ event store
```

| File | Role |
|---|---|
| `src/monitoring_features.py` | `feature_summary`, `log_bin`/`log_bins`: one binning shared by the API, baseline and aggregator |
| `src/ml_monitoring.py` | `MonitoringAggregator` (bounded buckets, offset-guarded recovery), `series_report`, `monitoring_report` |
| `src/monitoring_baseline.py` | Baseline CLI (training data + model artifacts → versioned JSON) |
| `src/ml_performance.py` | Label contract, `add-labels`, `evaluate_performance`, `PerformanceCache`, `report` CLI |
| `src/event_consumer.py` | Optional `aggregator` hook in `ConsumerRunner` (observe after store, save in commit). New optional event fields declared in `PredictionEvent` |
| `api/main.py` | Feature summary in the event; 422 counter (`input_rejections`) in `/metrics/decisions` |
| `monitoring/live_dashboard.py` / `.html` | `/api/ml`, the ML monitoring section |
| `monitoring/baselines/e7_v1/bl-e7_v1-1.json` | The current baseline |

No Redis, Prometheus, Grafana or database is used. The state is one JSON file next to the event store (`ml_monitoring_state.json`), written by the only process that consumes events.

### Bounded by construction
- **Buckets:**
  - 1-minute buckets for the last 60 minutes;
  - 15-minute buckets back to 24 h, so at most 60 + 97 buckets;
  - each bucket holds one stats block per `(model_version, decision_policy_version)`.
- **Series:** at most 20 version pairs; any more share `__other__|__other__`.
- **Counter dicts:** every counter dict (missing features, bins, types, reasons) has at most 64 keys, and the rest go to `__other__`.
- **Score histogram:** 32 fixed bins.
- **Measured size:**
  - 26 hours at one event per minute stays under 600 KB (tested);
  - live, 9265 events across 3 active minutes gave a 4 KB file.
- **Low-cardinality labels:** no transaction IDs, instance IDs or free text are used as aggregation keys.

### Recovery and counting semantics
- The state records the next Kafka offset it has counted for each partition.
- It is saved atomically (tmp file, fsync, replace) after the store fsync and **before** the Kafka offset commit.
- **Crash after save, before commit:** Kafka re-delivers messages below the saved mark. They are skipped and counted as `replay_skipped`, so they aren't counted twice.
- **Crash before save:** the in-memory counts since the last save are lost. Kafka re-delivers exactly those messages (they weren't committed either), so they are counted once.
- **State file lost or older than the committed offsets:** the jump is reported as `offset_gap_not_aggregated`, not hidden.
- **Corrupt state:** moved aside to `*.corrupt-<ts>`, monitoring starts empty, and `state_load_failed` is set. The consumer never stops because of monitoring.
- **Failed save:** logged and counted (`save_failures`), and the commit still goes ahead. Monitoring never blocks event storage.
- Duplicates and dead-lettered messages move the offset mark forward but aren't counted.
- **Upshot:** each stored event is counted once in the windows, even across restarts and crashes. The event store itself stays at-least-once: a crash can store an event twice, and monitoring still counts it once.

## 3. Feature summary in the event (no exact amounts)
`floor(4 * log10(1 + x))`, i.e. quarter-decade bins: 0 → 0, 9 → 4, 1 000 → 12, 1 M → 24, 10 M → 28, capped at 63.
- **Binned:** `amount`, `oldbalanceOrg`, `oldbalanceDest`, `amt_vs_type_median` (and the E5/E6 ratios when the model has them). `null` means missing (NaN given to the model).
- **Flags:** `exceeds_orig_bal`, `orig_bal_zero`, `dest_bal_zero`, `is_max_amount`, kept as 0/1/null.
- **Cost on the request path:** the API recomputes the feature row with the same `pipeline.transform_one` the predictor used, and bins it. This was done rather than passing the row out of `predictor.score()` so that `score()` keeps its contract. Measured cost: **7.7 µs per request** (5.2 µs features + 4.1 µs summary, 20 000 iterations). It only runs when Kafka is enabled, and a failure only drops the summary, never the event or the response.

## 4. Metric definitions

All windows are **rolling, keyed by prediction time** (`occurred_at`), per model and policy version:
- 5 m and 1 h are minute-exact;
- 24 h has 15-minute resolution at its far edge;
- an event older than 24 h when it is processed goes into the cumulative counters only (`older_than_24h`).

**Cumulative** counters cover the period since the state file was created (`cumulative_since`). They are labelled separately and never mixed with windows.

| Metric | Formula / meaning |
|---|---|
| predictions | n = events stored in the window |
| decision rate | count(decision = d) / n, d ∈ {approve, review, decline} |
| actual declines | count(decision = "decline"). Always 0 in shadow mode |
| shadow would-decline | count(would_decline) and its rate. **Hypothetical:** returned to the client as `review / HIGH_RISK`, decline not executed. Never added to actual declines |
| reason distribution | count per reason_code |
| mean score | Σ fraud_probability / n (exact) |
| p50 / p90 / p95 / p99 score | From the 32-bin histogram with linear interpolation inside a bin. **Approximate:** error ≤ one bin width. Edges are fine near 0 and 1 and include the policy thresholds 0.54 and 0.999 |
| events with missing features | count(missing_features ≠ []) / n; per feature: count(feature ∈ missing_features) / n |
| non-finite rate | count(non_finite_features ≠ []) / events that carry the summary |
| unknown type | count(transaction_type ∉ the 5 training types) / n |
| feature summary available | Events carrying `feature_bins` / n (older events: not available) |
| outside training range | Bin < min or > max of that feature's training bins, / observed |
| unexpected amount | amount bin > training p99.9 bin, / observed |
| flag rate | count(flag = 1) / count(flag ∈ {0,1}), shown next to the training rate |
| type mix change | share_t − baseline_share_t per type, and total variation distance TVD = ½ Σ\|share_t − baseline_t\| (0 = same mix, 1 = disjoint). Descriptive only, not a statistical test |
| insufficient data | n < 50 in the window: shown, flagged, rates not meaningful |
| input rejections (422) | Per API process, by first failing request field. **Current processes only** (resets on restart) |
| events not delivered | Producer `not sent + not confirmed`, current processes (see RUN_WINDOWS.md §6) |
| dead-lettered | Consumer process lifetime |
| freshness | State age, newest prediction age, Kafka lag. **Stale** = state older than 30 s *while* lag > 0 (events waiting). An old state with lag 0 just means there's no traffic. Unknown if the lag is unknown |

## 5. Baseline

```bat
venv\Scripts\python -m src.monitoring_baseline --model-dir models\e7_v1 --data data\fraud_data.csv
```

The command takes 28 s and writes `monitoring/baselines/e7_v1/bl-e7_v1-1.json` (10 KB).
- **Reproducible:**
  - It re-creates the chronological split, and **refuses** if it doesn't match the model's `metadata.json` (train steps 1–280, val 281–354) or the row count (6 362 620).
  - It computes features with the model's saved `pipeline.json` (inference-time inputs only; `LEAKY_COLUMNS` never appear).
  - It is deterministic: a test builds it twice and gets byte-identical output, apart from `created_at`.
- **Linked:**
  - the dataset sha256 (`16910f90…e4eea6b`), path, bytes and rows;
  - sha256 of `model.json`, `pipeline.json` and `metadata.json`;
  - `model_version` `e7_v1`, `baseline_version` `bl-e7_v1-1`, the generator command and `created_at`.
- **Contents:**
  - **Per-feature statistics, train split only** (3 792 821 rows): count, missing rate, non-finite rate, mean, std, min, max and p1/p5/p25/p50/p75/p95/p99/p99.9.
  - The bin distribution, plus each feature's bin range and p99.9 bin.
  - Flag rates.
  - Transaction-type proportions.
  - **Validation reference** (val split, 1 276 276 rows), scored once:
    - the score histogram and mean;
    - decision rates under the current policy `dp-e7-2026-10-26.1`: 99.81% approve, 0.19% review, would-decline 0.044% (561 rows, which matches the policy notes).
    - This is a **reference distribution only**; the thresholds were tuned on this split, so it is never used to evaluate the model.
- **Immutable:** an existing version is never overwritten (`--version 2` for a new one).
- **Never fitted on production data.**
- **Version matching:** the dashboard picks the highest `bl-<model_version>-<n>.json` for each series. A model version without a baseline shows "no baseline for this model version", for example the 27 `e6_test` events left in the live store by an earlier test run.

## 6. Model performance (when labels exist)

**Label source:** none exists today; nothing in the repo or the live system produces actual outcomes. Contract for a future source (chargebacks, manual review results):

```jsonl
{"transaction_id": "tx-1", "is_fraud": 1, "label_at": "2026-11-20T10:00:00+00:00", "source": "chargeback"}
```
```bat
venv\Scripts\python -m src.ml_performance add-labels labels.jsonl     :: validates, appends to <store>\labels\
venv\Scripts\python -m src.ml_performance report --model e7_v1 --model-dir models\e7_v1
```

**Rules (all tested):**
- **Join:** label `transaction_id` = stored prediction `transaction_id`, for the chosen model version. Predictions with no `transaction_id` can't be labelled and are counted.
- **Unlabeled predictions are excluded, never treated as legitimate.** Coverage is reported: labelled / all predictions, and labelled / *mature* predictions (older than 30 days, the label-maturity window).
- **Prediction time and label arrival are kept separate.** A label with `label_at` before `occurred_at` is invalid and counted. The label delay (p50/p95 in days) is reported.
- **Conflicting labels** (for example a chargeback later reversed): the latest `label_at` wins, and conflicts are counted.
- **Labels with no matching prediction** are counted as `labels_without_prediction`.
- **Tuning data:** labels whose `dataset_step` falls in the model's train or validation steps are excluded and counted. This matters for offline replays of PaySim data, where the threshold was tuned on steps 281–354.
- **Minimum sample:** fewer than 100 usable labels, or only one class, gives `insufficient_labels` with the counts and no metric values.

**Metrics:**
- TP/FP/FN/TN, precision, recall, F1, FPR, the false-negative count and **cost = 100·FN + 1·FP** (`src/evaluate.business_cost`).
- They are computed at two operating points:
  - the approve threshold, where flagged = the decision actually returned was review or decline;
  - the decline threshold *including shadow would-declines*, which is hypothetical.
- **PR-AUC** from the scores (`average_precision_score`), only when both classes are present.

## 7. Dashboard

Below the existing sections, unchanged, the page has an **ML monitoring** card. It refreshes every 5 s and reads `/api/ml` on loopback only. It contains:
- **Window and series:** a window selector (5 min / 1 h / 24 h), with model and policy version chips showing their counts.
- **Freshness line:**
  - state age, newest prediction age and Kafka lag;
  - resolution and cumulative start;
  - STALE and gap warnings.
- **KPI tiles:**
  - predictions, approve and review rates;
  - actual declines; shadow would-declines labelled "NOT executed";
  - mean score and p95/p99;
  - missing-feature, non-finite and unknown-type rates;
  - amount above training p99.9;
  - feature-summary coverage;
  - 422 rejections, events not delivered, dead-lettered.
- **Volume sparkline.**
- **Score histogram:** bars on a √ scale, colour-coded by decision band, with dashed threshold lines at 0.54 and 0.999 and the validation reference as orange ticks.
- **Decisions:** bars with validation reference rates, a separate shadow would-decline line, the reason counts and the baseline version.
- **Type mix vs training:** share, training share and difference in percentage points, plus TVD.
- **Input data quality vs training:** missing, training missing, outside the training range, above the training p99.9.
- **Model performance panel:** currently "Unavailable: no actual labels ingested (no label source exists yet)", with the rules and the counts. A table appears automatically when there are enough labels.

Live values when this was verified (1 h window, 9265 events, mostly load-test traffic):
- 9212 approve, 33 review, 0 decline;
- 3 shadow would-declines, from the risky demo samples;
- mean score 0.0056;
- 0 missing and 0 non-finite;
- 6 amounts above the training p99.9;
- TVD 0.29.

The type mix differs because the load generator sends a uniform synthetic mix; that's a property of the test traffic, not of the model. The one 422 sent with `type: "WIRE"` was counted under `type` on instance 4.

## 8. Performance validation (Windows, same command before and after)
`python tests\windows_reliability.py latency`: 3000 requests, 4 processes at concurrency 10, through Nginx with Kafka up. Three runs each. CPU and RSS are sampled every 0.5 s with psutil and include the launcher's child processes.

| | Before (10:09) | After (10:22) |
|---|---|---|
| Client throughput | 502 / 523 / 525 req/s | 519 / 517 / 525 req/s |
| Client p50 / p95 / p99 | 9.8 / 12.1–13.7 / 14.4–40.9 ms | 9.6–9.7 / 12.2–12.6 / 14.4–14.8 ms |
| Nginx p50 / p95 / p99 | 2 / 3 / 3–4 ms* | 2 / 3 / 3–4 ms |
| HTTP errors | 0 of 3080 (each run) | 0 of 3080 (each run) |
| 6 instances CPU, mean of the sum (% of one core) | 74–86% | 80–84% |
| Consumer CPU, mean | 11–13% | 15–17% |
| Consumer RSS, max | 38 MB | 60 MB |
| Kafka lag after the run | 0 | 0 |

\* One before-run had a p99 outlier: client 40.9 ms, Nginx 32 ms.

- **p95 stays far below the 100 ms target.** The extra synchronous work in `/predict` is 7.7 µs.
- **Consumer CPU is up about 4 points** at about 520 events/s. That's the aggregation plus the state save every 5 s.
- **Instance memory:** total RSS read 580 MB before and 1771 MB after. That difference comes from **process age, not from this change**: the "before" instances had been running for hours, so Windows had trimmed their working sets, and the "after" instances had just been restarted. Running the old (HEAD) code and the new code under identical conditions (fresh process, model loaded, 3000 requests) gives **272 MB RSS and ~1002 MB private for both**.

## 9. Tests
`tests/test_ml_monitoring.py`: 28 tests. They cover:
- counts and decision rates;
- shadow would-decline kept separate from actual declines;
- score mean (exact) and percentiles within one bin;
- missing, non-finite and unknown types;
- window expiry and folding of old minute buckets into 15-minute buckets;
- version separation and caps;
- the state-size bound;
- recovery:
  - save at commit;
  - replay skipped after a restart;
  - dead letters and duplicates;
  - lost-state gap;
  - corrupt state;
  - save failure;
- the feature summary against the pipeline, and an API event with no exact amounts;
- 422 counting;
- baseline determinism and refusal on a split mismatch;
- the baseline comparison (TVD, ranges);
- labels:
  - none → unavailable;
  - valid → hand-computed P/R/F1/FPR/cost;
  - unlabeled not counted as legitimate;
  - insufficient;
  - early, conflicting, unmatched, tuning-step and invalid labels, and maturity;
- dashboard: missing state, freshness/stale/unknown, insufficient data, window validation.

No test uses a broker. The `no_real_kafka` fixture removes `KAFKA_BOOTSTRAP_SERVERS`.

**Full suite:** 203 passed, 5 skipped (the opt-in broker and Linux tests), both with and without `KAFKA_BOOTSTRAP_SERVERS` exported.

**Live checks:**
- **Graceful consumer restart with 20 events sent while it was down:** the counts continued (9245 → 9265), with no gap and nothing counted twice.
- **Three force-kills under load:** monitoring counted exactly the number of unique events stored (400 / 500 / 500), with 0 gaps. In these runs the kill never left an event that was both stored and re-delivered, so the replay-skip path is proven by the unit test, not by these runs.

## 10. Limitations
- **The API accepts `Infinity` amounts and approves them,** and a `NaN` amount returns 500 (§1). Monitoring detects these; it doesn't prevent them.
- **No labels exist, so production precision, recall, F1, PR-AUC, FPR, false negatives and cost cannot be calculated yet.** Validation and test metrics in `metadata.json` are offline figures on PaySim periods, not production performance.
- Percentiles are approximate (one bin). The mean is exact.
- **Coarse bins:** they can't show changes smaller than a quarter decade, which is the price of not storing exact amounts.
- **Partial history:** monitoring starts when the new consumer is deployed. The 22 k earlier events have no feature summary and weren't backfilled into the windows. The event store still has them.
- **24 h edge resolution:** the window's far edge has 15-minute resolution.
- **Duplicates across a crash:** the per-event duplicate filter in the consumer is in memory, so an event stored twice across a crash is stored twice. Monitoring still counts it once, thanks to the offset guard.
- **Per-process counters:** input rejections and producer counters reset when an instance restarts.
- **No alerts and no statistical tests.** Everything here is descriptive; that is the next stage.
- **Load-test traffic isn't production traffic:** live comparisons with the baseline (the TVD of 0.29) describe the synthetic test traffic, not real users.

## 11. Handoff: Drift Detection
Ready to use:
- `bl-e7_v1-1` holds the per-feature bin distributions (`bins`), quantiles, flag rates, type proportions and a validation score histogram, all on the **same bins** the events and windows use.
- `series_report` already outputs window bin counts for each feature, plus the score histogram, type counts and flag rates.

Next steps:
1. **Tests:** PSI / Jensen–Shannon distance per binned feature and for the score histogram, against `baseline.bins` and `reference_scores`; chi-square or TVD with a bootstrap threshold for the type mix. Implement them as pure functions on `series_report` output and the baseline (put them in `monitoring/drift_monitor.py`, currently empty).
2. **Thresholds and minimum sample:** use a minimum n per window (for example 1 000) and choose thresholds by replaying the PaySim **test** period (steps 355–743) through the aggregator as a stand-in for production. The train and validation periods would be biased.
3. **Alerts:** a `drift` block in `/api/ml` and an alert state on the dashboard. Optionally, log alert transitions to a JSONL file next to the state. Still no new infrastructure.
4. **Model or policy changes:** a new model version gets its own baseline (`bl-e7_v2-1`) before rollout, and series stay separate per version.
5. **Labels:** connect a real outcome source to `add-labels`. Then add label-delay-aware performance trends (per week of prediction time, mature labels only), and concept-drift checks that compare the performance against the offline test metrics.
