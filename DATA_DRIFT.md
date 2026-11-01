# Data drift detection

Compares recent production inputs with the training data the deployed model learned from. It reports evidence per feature, raises deduplicated alerts and shows the results in the live dashboard.

**Drift means the input distribution changed. It does not show that fraud detection got worse.** Measuring that needs actual fraud labels (`src/ml_performance.py`, [ML_MONITORING.md](ML_MONITORING.md)). Nothing here retrains or changes the model.

## 1. What it builds on (inspected, reused)

| Existing piece | Used for |
|---|---|
| Prediction event (`src/events.py`): `feature_bins`, `feature_flags`, `transaction_type`, `non_finite_features`, built by `src/monitoring_features.py` in the API | The production observations. No schema change was needed: it already carries every monitored feature, binned, with no exact amounts or IDs |
| `MonitoringAggregator` (`src/ml_monitoring.py`) inside the existing Kafka consumer | Bounded production counts per minute (last hour) and per 15 minutes (to 24 h), per model and policy version. Persisted atomically; offset marks count each event once across restarts |
| Baseline generator (`src/monitoring_baseline.py`) | Reference distributions from the model's **train** split only. Extended to schema 2 (below) |
| `FeaturePipeline` (`src/data_preprocessing.py`) and `models/<v>/metadata.json` | Feature definitions and names. The baseline is built with the model's own saved pipeline |
| Live dashboard (`monitoring/live_dashboard.py`) | New `GET /api/drift` endpoint and a "Data drift" section |

**Monitored features.** These are all of E7's inputs, all available when the prediction is made:

| Feature | Type | How it is observed |
|---|---|---|
| `amount`, `oldbalanceOrg`, `oldbalanceDest`, `amt_vs_type_median` | numeric | quarter-decade log bin `floor(4·log10(1+x))` |
| `exceeds_orig_bal`, `orig_bal_zero`, `dest_bal_zero`, `is_max_amount` | binary | 0 / 1 |
| `transaction_type` (source of `type_code`) | categorical | the 5 PaySim types; anything else counts as unknown |

`newbalanceOrig`, `newbalanceDest` and `isFraud` are not monitored. They are post-transaction values or the label, and the baseline generator refuses them (`LEAKY_COLUMNS`).

## 2. Architecture and data flow

```
/predict (unchanged) ─► event (feature bins/flags/type) ─► Kafka ─► consumer (unchanged)
                                                           └─► MonitoringAggregator ─► ml_monitoring_state.json
monitoring/drift_monitor.py  (separate process, every DRIFT_INTERVAL_S = 60 s)
   load_state (read-only) + latest valid baseline per model version + config/drift_config.json
   └─► src/drift_detection.py ─► drift_report.json · drift_alerts.jsonl · drift_alert_state.json
live_dashboard.py  GET /api/drift (127.0.0.1 only) ─► "Data drift" section
```

**Why a separate process, rather than the consumer or the dashboard:**
- No work is added to the consumer's loop or to `/predict`.
- One process owns the alert state.
- It can see that the consumer has stopped updating, which the consumer itself can't report.
- The dashboard only displays the report and its age, so a dead drift monitor shows as "not updating".

**Memory is bounded.** Inputs are the aggregator's fixed-size buckets: at most 64 bins per feature and 157 buckets. No raw events are kept.

## Traffic sources

Drift is computed **per traffic source**, and alerts only cover the sources listed in `alerts.traffic_sources` (default `["production"]`). This was added after the [oldbalanceDest investigation](investigations/2026-10-30-oldbalanceDest-drift.md): 99.997% of the "production" events were really synthetic `wrk` load-test traffic.

- **The tag.** Clients may send the header `X-Traffic-Source: production | loadtest | test | demo`.
  - `tests/wrk_predict.lua` (and therefore `wrk_paced.lua` and `stress_suite.sh`) and `tests/load_api.py` send `loadtest`.
  - The dashboard's "send one transaction" buttons send `demo`.
- **Missing or invalid header → `production`.** Untagged traffic is always monitored, and a typo can't hide traffic from drift detection. Invalid values are counted in `/metrics/events` as `traffic_source_invalid` and logged.
- **No effect on scoring.** The tag never changes the prediction or the response. It is only copied into the event as `traffic_source`.
- **Separate series.** The consumer counts non-production traffic in its own series, `model|policy|source`. The report shows them as `e7_v1|loadtest` and so on, with drift results but no alerts. Each window reports `traffic_by_source`, so tagged traffic is never invisible.
- **Trust.** The header is trusted from internal callers. In a real deployment, the public edge (API gateway) must strip `X-Traffic-Source` from external requests, or anyone could keep their traffic out of the production drift checks.
- **Older events.** Events written before the tag existed count as production until they leave the 24 h window.

## 3. Statistical methods (a small, deliberate set)

| Method | Used for | What it measures | Limitations |
|---|---|---|---|
| **PSI** (primary effect size) | every feature | How much the share of observations per group moved, symmetrically weighted: Σ (p−r)·ln(p/r) | A heuristic scale, not a test. It is biased upward for small samples (≈ (K−1)/N), and that bias is what `min_samples` bounds |
| **Chi-square goodness of fit** (statistical evidence) | every feature, same groups as PSI | Whether production counts are plausible under the reference proportions | With large N it flags tiny, irrelevant changes, so it never decides alone. It needs expected counts ≥ 5 |
| **Binned KS distance** (descriptive) | numeric | Largest gap between the cumulative distributions over the fine log bins | Computed on binned data, so it is a lower bound of the true KS statistic. **No p-value**: a KS test on tied, binned data is invalid |
| **Missing-rate change** | every feature | Production missing rate vs training, with an exact binomial test | Counts only explicit missing values (`null` bins) |
| **Unknown-category rate** | `transaction_type` | Share of types the model doesn't know | Threshold only (the training rate is 0) |
| **Prediction drift** (separate) | fraud score | PSI and chi-square of the score histogram vs the baseline's validation-split scores | The score can move because inputs moved or because the model behaves differently. It is not an accuracy measure |

**Not implemented, on purpose:**
- **Wasserstein and Jensen–Shannon distance:** redundant with PSI here.
- **An exact KS test:** invalid on binned data.
- **Multivariate or joint tests:** these would need joint counts that the privacy-preserving event summary doesn't carry.

**Details that keep the numbers safe:**
- **Alignment.** Reference and production go through the same `log_bin` function, so bins match by construction. Categories are compared on the reference category set.
- **Grouping.**
  - For PSI and chi-square, the fine bins are grouped into at most 10 groups of roughly equal **reference** mass.
  - Two open groups catch values below or above the training range, so values never seen in training stay visible.
  - With ~60 raw bins, PSI's no-drift bias at N = 500 would be ~0.12, above the "moderate" band. With 10 groups plus the two out-of-range groups it is about (12−1)/500 = 0.022. The Monte Carlo audit measured a mean of 0.027 (p99 0.048) for `amount` and 0 false alarms in 1,000 no-drift windows ([Phase 1 audit](investigations/2026-10-31-phase1-drift-audit.md)).
- **Smoothing.** Additive pseudo-count α = 0.5 on both sides: p_i = (c_i + α) / (N + α·K). There's no log(0) and no division by zero, and α is negligible against the 3.8M-row reference.
- **Missing values.** PSI and the test use valid values only. Missing values are reported and tested separately (the missing-rate check), and a missing-rate jump can flag a feature even when no valid values are left.
- **Zero-count bins.** These are handled by smoothing for PSI. For chi-square, adjacent groups with expected count < 5 are merged; if fewer than 2 groups remain, the test is `not_applicable`, with the reason stated.
- **Multiple comparisons.** All chi-square and missing-rate p-values in one window are **Holm-adjusted**. Holm controls the family-wise error under any dependence, and these features are dependent (several derive from the same amount and balances).
- **Invalid input.** NaN, inf or negative counts and aggregator overflow make the feature `monitoring_error`. No metric returns NaN or inf.

## 4. Statuses and thresholds (`config/drift_config.json`)

**Configuration.**
- Thresholds come from `config/drift_config.json`, or another file named in `DRIFT_CONFIG`. The monitor reads it on every run, so no code change is needed.
- Invalid values, such as moderate > substantial, make the run `monitoring_error` and keep the previous report's `last_successful_at`.

| Setting | Default | Meaning |
|---|---|---|
| `windows` | `{"1h": 60, "24h": 1440}` | Rolling windows by prediction time. 1 h is minute-exact; 24 h uses 15-minute resolution beyond the last hour, so the report gives both `start` and `effective_start` (up to 14 min earlier) |
| `min_samples` | 500 | Valid production values needed per feature and window before any verdict |
| `psi` | moderate 0.10, substantial 0.20 | Conventional bands, treated as **initial heuristics**. Per-feature overrides go in `psi_overrides` |
| `alpha` | 0.01 | For Holm-adjusted p-values |
| `missing_rate_abs` | 0.05 | Absolute missing-rate change that counts |
| `unknown_rate` | 0.01 | Unknown transaction-type share that counts |
| `non_finite_rate` | 0.001 | Share of non-finite inputs (inf) that counts, reported separately from missing values. The API rejects non-finite amounts and balances with a 422, so this only fires for other producers or a validation regression |
| `critical_features` | amount, oldbalanceOrg, transaction_type | Used by the sustained-drift alert |
| `alerts` | min_severity substantial, sustained 60 min, stale after 120 s, low volume < 100 per hour, traffic_sources [production], expire_without_data true | Alert rules (section 7). Set `low_volume_min_per_hour` to `null` to disable that rule. Only sources in `traffic_sources` raise alerts |

| Status | When |
|---|---|
| `insufficient_data` | Fewer than `min_samples` valid values, no baseline, or an invalid baseline. **Never shown as healthy** |
| `monitoring_error` | Invalid counts, more than 1% of observations in the aggregator's overflow bin, or a failed calculation |
| `drift_detected` | PSI ≥ moderate **and** (Holm p < α, or the test is not applicable); or the missing-rate or unknown-rate check failed. `severity` is moderate or substantial |
| `no_significant_drift` | Enough data and none of the above. A significant p-value with PSI under the threshold is reported as "statistically significant but PSI below threshold" |

**Each result also carries:**
- feature, type, window with start and end;
- reference and production n;
- PSI, chi-square statistic, df, raw and Holm p;
- KS distance;
- missing rates;
- threshold, status and severity;
- baseline, model and pipeline versions;
- `detected_at`.

Different model versions are evaluated separately, each against its own latest **valid** baseline. Policy versions are merged, because input drift doesn't depend on the policy. A model version without a baseline is `insufficient_data` with the reason "no baseline for this model version".

**Summary per window and model:**
- features monitored, with sufficient data, and drifting;
- the three largest changes;
- production n;
- type-mix total variation distance;
- freshness: consumer state age, newest prediction age, Kafka lag, consumer alive;
- versions;
- calculation time and duration.

There is **no combined drift score**: the features are dependent, and these are marginal checks. "Overall input change" is shown as the share of features drifting plus the type-mix TVD. Prediction drift is a separate block.

## 5. Baseline (versioned, reproducible)

```bash
python -m src.monitoring_baseline --model-dir models/e7_v1 --data data/fraud_data.csv --version 2   # ~23 s
python -m src.monitoring_baseline --model-dir models/e7_v2 --data data/fraud_data.csv --version 1
```

**How it's built:**
- It re-creates the model's chronological split from `metadata.json` and **refuses data that doesn't reproduce it**.
- It computes features with the model's saved pipeline, from the **train split only** (steps 1–280, 3,792,821 rows).
- The validation split is used only for the reference score histogram. The test split and production data are never used.
- Baselines are immutable: an existing version is never overwritten.

**Schema 2** (`monitoring/baselines/e7_v1/bl-e7_v1-2.json`, `monitoring/baselines/e7_v2/bl-e7_v2-1.json`) adds the following to schema 1:
- `bin_counts`: raw counts per log bin;
- `flag_counts`;
- `categorical.transaction_type`: counts, proportions, missing and unknown rates;
- `feature_pipeline`: feature set, `pipeline.json` sha256, and id `E7-1fe84390`;
- `monitored_features`.

It keeps every schema-1 key: dataset path, sha256 and rows; model artifact hashes; split; per-feature count, missing rate, mean, std, min, max and percentiles; shares; and `created_at`.

**Schema 1 is still readable.** Its counts are derived as share × valid count and flagged `derived_reference_counts`. For e7_v1, the derived counts equal schema 2's raw counts exactly.

**Validation.** At every run, `validate_baseline` checks:
- required keys;
- the model version;
- feature names against `models/<v>/metadata.json`;
- finite, non-negative counts;
- shares summing to 1;
- the score edges.

An invalid newer baseline is skipped for the newest valid one, and the skipped file and reason appear in the report.

## 6. Example report (synthetic data)

**Input.** 1,200 synthetic predictions whose `amount` is 10× larger than the reference (+4 log bins). The flag and type mix are unchanged. This comes from `tests/test_drift_monitor.py` helpers through `run_once`.

```json
{"summary": {"status": "drift_detected", "features_monitored": 3, "features_with_sufficient_data": 3,
             "features_drifting": 1, "share_of_features_drifting": 0.333333, "production_n": 1200,
             "largest_changes": [{"feature": "amount", "psi": 5.324701}, {"feature": "transaction_type", "psi": 1.7e-05},
                                 {"feature": "orig_bal_zero", "psi": 0.0}], "type_mix_tvd": 0.0005},
 "features": [
  {"feature": "amount", "type": "numeric", "window": "1h", "window_start": "2026-10-30T11:01:00+00:00",
   "window_end": "2026-10-30T12:00:30+00:00", "reference_n": 1000000, "production_n": 1200, "psi": 5.324701,
   "test": {"name": "chi-square goodness of fit", "statistic": 7632.5, "df": 6, "p_value": 0.0, "p_holm": 0.0},
   "ks_distance": 0.7, "missing_rate": {"reference": 0.0, "production": 0.0},
   "threshold": {"moderate": 0.1, "substantial": 0.2}, "status": "drift_detected", "severity": "substantial",
   "reasons": ["PSI 5.324701 >= 0.1 and Holm p 0.00e+00 < 0.01"],
   "baseline_version": "bl-e7_v1-1", "model_version": "e7_v1", "detected_at": "2026-10-30T12:00:30+00:00"},
  {"feature": "orig_bal_zero", "type": "binary", "production_n": 1200, "psi": 0.0,
   "test": {"statistic": 0.0, "df": 1, "p_value": 1.0, "p_holm": 1.0}, "status": "no_significant_drift"}]}
```

The resulting alert record (`drift_alerts.jsonl`):

```json
{"at": "2026-10-30T12:00:30+00:00", "event": "fired", "rule": "feature_drift", "key": "feature_drift|e7_v1|1h|amount",
 "severity": "substantial", "feature": "amount", "model_version": "e7_v1", "baseline_version": "bl-e7_v1-1",
 "window": "1h", "window_start": "2026-10-30T11:01:00+00:00", "window_end": "2026-10-30T12:00:30+00:00",
 "drift_score": 5.324701, "threshold": {"moderate": 0.1, "substantial": 0.2}, "sample_count": 1200,
 "status": "drift_detected"}
```

Prediction drift was also `drift_detected` in this example (PSI 6.67), only because every synthetic event carries the same score. That shows why prediction drift is reported separately and never read as input drift.

**On the live stack, drift is expected.** The `wrk` load-test payloads (`tests/wrk_predict.lua`) aren't PaySim-like, so 7 of 9 features report substantial drift. For example, `oldbalanceDest` has PSI 3.86, and `dest_bal_zero` is 0.1% in the test traffic vs 42% in training. This shows the detector working, not a model problem.

## 7. Alerts

There's no external service. Records are appended to `drift_alerts.jsonl` in `EVENT_STORE_DIR`, logged at WARNING level in `deploy/run/drift.log`, and listed on the dashboard.

| Rule | Fires when |
|---|---|
| `feature_drift` | A feature is `drift_detected` with severity ≥ `alerts.min_severity`, with sufficient data, in a window (per model, window and feature) |
| `sustained_drift` | A critical feature has been drifting in the short window continuously for ≥ `sustained_minutes` **and** is drifting in the long window |
| `monitoring_stale` | The consumer's state is older than `stale_after_s` while Kafka lag > 0, or the consumer process is dead |
| `low_volume` | Fewer than `low_volume_min_per_hour` predictions in the last hour |
| `monitoring_error` | A model's evaluation failed or produced feature errors |

**Each record carries:**
- rule, key, `fired` / `escalated` / `resolved`;
- feature;
- drift score (PSI) and threshold;
- window with start and end;
- sample count;
- baseline and model versions;
- time and reasons.

**Continuity and expiry** (Phase 1 audit fixes D5 and D6):
- `sustained_drift` needs `drift_detected` at **every** short-window evaluation. Any other result, or no data, restarts the clock.
- An active alert whose condition has had no evidence (no data, or insufficient data) for longer than its own window (1 h, 24 h, or the long window for sustained drift) closes with an `expired_no_data` record, giving the last time it had evidence. No verdict is implied. The `low_volume` alert reports the missing traffic.
- Set `alerts.expire_without_data: false` to keep alerts open until data clears them.

**Deduplication.**
- An alert fires when its condition becomes active, and again only if its severity rises. While nothing changes it stays quiet.
- It resolves only when the condition clears **with sufficient data**. `insufficient_data` and errors never resolve it.
- Active alerts and drift-continuity marks are kept in `drift_alert_state.json`. This is written durably only when they change, and it survives restarts.
- Measured: about 15 restarts of the monitor during the benchmarks produced 17 records in total. That's 12 feature-drift alerts fired once each; 3 sustained-drift alerts after 60 minutes; and 1 low-volume alert that fired when the hour emptied and resolved when traffic resumed.

### Acknowledging alerts (operator decision)

```bash
python -m monitoring.drift_monitor --ack "<text in the alert key>" --reason "<why>" [--hours 24]
```

This is step 6 of an investigation. An acknowledged alert stays active, but it's silent until the acknowledgement expires. The record shows who acknowledged it, why, and until when.

- It still **resolves normally** when the condition clears with sufficient data.
- It **fires again** (with a note) if the condition is still present when the acknowledgement expires.
- The running monitor is the only writer of the alert state. `--ack` appends a request to `drift_ack_requests.jsonl`, which the monitor applies on its next run. If no monitor is running, the command applies it directly.
- Acknowledging never changes the model, the thresholds or the data.

## 8. Dashboard

The "Data drift" section is below "ML monitoring" at `http://127.0.0.1:8090` and refreshes every 10 s. It shows:
- the window and model selector;
- calculation time, duration and report age, with "drift monitor not updating" if it's older than 3 intervals;
- the window range and resolution;
- consumer-state freshness and Kafka lag;
- the overall input drift status;
- features monitored, with enough data, and drifting;
- production samples and type-mix TVD;
- baseline and pipeline versions;
- the per-feature table: status, severity, PSI, threshold, Holm p, KS distance, ref/prod n, missing rates, unknown rate, and why;
- reference-vs-production share charts for the three largest changes;
- the separate prediction-drift block;
- active and recent alerts.

**Explicit states.** `insufficient_data`, `monitoring_error`, "no drift report yet" and "no consumer monitoring state" are shown as such.

A banner says that drift is a distribution change, not a measured loss of accuracy. `/api/drift` is served by the dashboard on 127.0.0.1 only, and Nginx (:8080) doesn't expose it.

## 9. Tests

```bash
pytest tests/test_drift_detection.py tests/test_drift_monitor.py    # 33 tests, synthetic data, no broker
pytest                                                              # full suite
```

| Covered | Tests |
|---|---|
| Equivalent distributions → no drift (PSI < 0.01); numeric ×10 shift; categorical shift; binary shift | `test_identical_distributions_no_drift`, `test_numeric_shift_amount_times_ten`, `test_categorical_shift`, `test_binary_flag_shift` |
| Missing-rate change; unknown categories | `test_missing_rate_change_…`, `test_unknown_categories` |
| Empty data; insufficient samples (never healthy) | `test_empty_production_is_insufficient_never_healthy`, `test_below_min_samples_is_insufficient` |
| Zero-count bins; NaN, inf and negative counts; overflow | `test_psi_zero_bins_and_empty`, `test_invalid_and_non_finite_counts_…`, `test_aggregator_overflow_bin_is_an_error` |
| Statistical-test edge cases: too few expected counts, merging, Holm, large-n tiny shift, degenerate binomial | `test_chi_square_not_applicable_and_merging`, `test_holm_adjustment`, `test_large_sample_tiny_change_…`, `test_ks_distance_and_binomial_edges` |
| Baselines: schema 1 = schema 2; wrong model or features; corrupt counts; invalid newer baseline skipped; model without baseline | `test_schema_1_baseline_…`, `test_validate_baseline_…`, `test_invalid_newer_baseline_…`, `test_models_are_separate_…` |
| Windows by prediction time; consumer restart and recovery (same report, re-deliveries not counted) | `test_windows_use_prediction_time`, `test_report_is_identical_after_consumer_restart` |
| Stale monitoring, dead consumer, low volume | `test_stale_monitoring_and_dead_consumer`, `test_low_volume_alert` |
| Alert dedupe, escalation, resolve only with data, survives restart, sustained drift | `test_feature_drift_alert_fires_once_…`, `test_alert_state_survives_restart_…`, `test_sustained_drift_…` |
| Config overrides and validation; failed calculation keeps `last_successful_at`; dashboard endpoint | `test_config_file_overrides_and_validation`, `test_no_state_and_failed_calculation`, `test_dashboard_drift_endpoint` |

Tests never reach a real broker: `tests/conftest.py` removes `KAFKA_BOOTSTRAP_SERVERS`, so the lag is "unknown".

## 10. Performance (measured 2026-10-30)

**Setup.** One laptop (Ryzen 7 5800H, AC power, `balanced` profile). The 6 instances and Nginx run on CPUs 0–7; Kafka, the consumer and the drift monitor on CPUs 8–9; `wrk -t8 -c100 -d30s` on CPUs 10–15, through Nginx. The consumer runs with ML monitoring in all runs. **Comparison:** the drift monitor (60 s interval, running steadily) vs no drift monitor, 7 runs each, alternating, with identical idle gaps before each run.

| | req/s | client p50 | client p95 mean / median (range) | client p99 | server p95 (Nginx log) | errors |
|---|---|---|---|---|---|---|
| without drift monitor | 6,693 | 12.4 ms | 71.7 / 70.0 ms (53.5–93.0) | 359 ms | 31.5 ms | 0 |
| with drift monitor | 6,644 | 12.3 ms | 92.1 / 89.2 ms (68.4–117.8) | 455 ms | 32.0 ms | 0 |

**What the measurements show:**
- **Throughput, p50, server-side p95 and Kafka lag (0 after every run) are unchanged.**
- **Client-side p95 was higher** with the drift monitor running, and **3 of 7 runs exceeded 100 ms** (0 of 7 without).
  - The difference isn't statistically significant at this sample size (Mann-Whitney U = 37, n = 7 + 7, two-sided p > 0.05). Client p95 on this machine varies a lot between identical runs.
  - The cause wasn't isolated. The monitor sleeps almost all the time (one 3–20 ms calculation per minute, 0.1–0.2% CPU), and the extra time happens outside the server (the server-side p95 doesn't move).
  - Two plausible disk-related causes were reduced before the final runs: the report is now written without fsync, the alert state is written only when it changes, and the Kafka lag client is reused.
- **The p95 < 100 ms target** was met in all 7 runs without the monitor and in 4 of 7 with it, at saturation load (100 connections). **It is not shown to hold reliably with the drift monitor at that load on this laptop.** It needs re-measuring with the load generator on separate hardware.

| Drift monitor itself | Value |
|---|---|
| Calculation duration (2 windows, 9 features + prediction) | 3–20 ms (median 4–11 ms) under load |
| CPU / RSS | 0.1–0.2% / 104 MB |
| Restart → first report | ~0.1 s after the process starts (2.0 s through `start.sh`, which waits 2 s) |
| Report / state sizes | `drift_report.json` ~29 KB; `ml_monitoring_state.json` 67 KB after the benchmarks |

| Recovery | Value |
|---|---|
| Consumer monitoring state load (67 KB) | 0.76 ms median, no failures |
| Consumer stopped with 146,801 events waiting → back to lag 0 | 17.9 s (~8,200 events/s), with no re-counting thanks to the offset marks |

## 11. Limitations and operational risks

- **Not accuracy.** Data drift doesn't prove degraded fraud detection. **Concept drift** (a changed relation between features and fraud) needs ground-truth labels, and nothing here infers labels from predictions.
- **Coarse bins.** Quarter-decade bins (×1.78) hide shifts within a bin. The binning trades sensitivity for privacy, since exact amounts never leave the API.
- **Marginal checks only.** A change in how features combine, with unchanged marginals, isn't detected.
- **Windows.** The 24 h window has 15-minute resolution beyond the last hour. Windows are by prediction time, so very late events (> 24 h) count only cumulatively.
- **PSI bands are conventions**, not calibrated for this business. Not done yet: calibrate them by replaying the PaySim **test** period (steps 355–743) through the aggregator as a stand-in for production, as ML_MONITORING.md proposed. Then tune `psi_overrides`.
- **Synthetic traffic drifts.** Tagged load tests (`X-Traffic-Source: loadtest`) are evaluated in their own series and don't alert. Untagged synthetic traffic still counts as production.
- **Single host.** The consumer, monitor, Kafka and API share one machine and one HDD. A dead drift monitor shows as a stale report on the dashboard, but nothing alerts on it: the alerter is the process that stopped.
- **Client-side tail latency** with the monitor running (section 10) is unexplained.

## 12. Commands

```bash
# baselines (once per model version; immutable)
python -m src.monitoring_baseline --model-dir models/e7_v1 --data data/fraud_data.csv --version 2

# run (after Kafka, the consumer and the API are up: see RUN.md)
export KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
deploy/start.sh drift                       # every 60 s; log deploy/run/drift.log; CPUS_DRIFT= to pin
python -m monitoring.drift_monitor --once   # one calculation, summary printed (exit 1 on error)
deploy/stop.sh drift

# inspect
cat ~/.local/share/nilepay-events/drift_report.json | python3 -m json.tool | less
tail ~/.local/share/nilepay-events/drift_alerts.jsonl
curl -s localhost:8090/api/drift | python3 -m json.tool | head -50       # dashboard backend

# tests
pytest tests/test_drift_detection.py tests/test_drift_monitor.py
```

Environment variables:
- `EVENT_STORE_DIR` (`~/.local/share/nilepay-events`);
- `MONITORING_BASELINE_DIR` (`monitoring/baselines`);
- `DRIFT_CONFIG` (`config/drift_config.json`);
- `DRIFT_INTERVAL_S` (60), also read by the dashboard for its staleness check;
- `KAFKA_BOOTSTRAP_SERVERS` (lag; unset means lag unknown);
- `KAFKA_TOPIC_PREDICTIONS`, `KAFKA_CONSUMER_GROUP`.

## 13. Handoff: Retraining Strategy (next phase)

1. **Labels first.** Feed confirmed fraud and chargeback outcomes through `python -m src.ml_performance add-labels`, keyed by `transaction_id`. Track label delay and coverage. Only labelled performance (precision, recall, PR-AUC, business cost 100·FN + FP) can justify a retrain.
2. **Triggers, combined rather than drift alone:**
   - sustained substantial drift on a critical feature (`sustained_drift`), **plus** degraded labelled performance or business cost over a defined window → investigate and retrain;
   - drift without performance loss → review the data and the baseline, not the model;
   - a scheduled retrain (for example monthly) as a floor.
3. **Training data.** The same chronological split method (`chronological_split`) on a newer period, with the leakage checks and `LEAKY_COLUMNS` unchanged. Decide whether to use a sliding or an expanding window from the drift history.
4. **Champion/challenger.**
   - Train `e7_v3` and generate its baseline (`--version 1`).
   - Evaluate on a later held-out period.
   - Run it in shadow next to the champion (the decision engine's shadow mode already separates "would decline" from executed decisions).
   - Compare decisions, cost and calibration before switching. Keep the previous model artifacts and baseline for rollback.
5. **After a switch.** Drift is computed per model version against that model's own baseline. The drift monitor picks up the new model's baseline automatically, and its first hour shows `insufficient_data` until `min_samples` is reached.
6. **Policy.** Thresholds must be re-derived per model (`src/decision_policy_analysis.py`). The policy is validated against the model version (`policy.check_model`).
