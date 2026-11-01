# Phase 1: Data Drift Detection audit (2026-10-31)

Scope: the brief's Phase 1. Verify the reference baseline, investigate the observed drift, validate the drift metrics, fix only confirmed defects, and say whether the alerts can be trusted. **No retraining, no model change.**

## Verdict

- **The drift calculations are correct.** This is established by:
  - independent re-implementation of PSI, chi-square, Holm and KS distance;
  - 0 false alarms in 1,000 no-drift windows;
  - 100% detection of known shifts;
  - an exact recount of 9,782,421 aggregated events;
  - a bit-for-bit regeneration of the baseline.
- **The drift values reported earlier were correct calculations on the wrong population.**
  - The figures in the brief (oldbalanceDest 7.37, dest_bal_zero 6.22, …) are the **load-test series**.
  - The production series was 99.9998% untagged synthetic `wrk` traffic.
  - They are **not evidence of business drift**.
- **Real data does not drift.** 1,000 real PaySim transactions from the later test period (steps 355–743) gave `no_significant_drift` on all 9 features (max PSI 0.019) and on the score distribution (PSI 0.015).
- **Current alerts.**
  - **Can't be trusted as business signals:** the 6 `feature_drift` (24 h) and 3 `sustained_drift` alerts that are still active. They are caused by the untagged synthetic events from 2026-10-30, are acknowledged with that reason, and close automatically once those events leave the 24 h window (`expired_no_data`).
  - **Correct:** the `low_volume` alert. There is no real production traffic.
- **Future alerts can be trusted for what they measure:** a change in the marginal distribution of **production-tagged** inputs relative to the training data. That is subject to the risks at the end. They say nothing about fraud-detection accuracy.

## 1.1 Reference baseline

| Check | Result | Evidence |
|---|---|---|
| Computed from the training data | Yes: train split steps 1–280, 3,792,821 rows (`split.statistics_from = train`) | `bl-e7_v1-2.json`; `build_baseline` uses only `train` for feature statistics |
| Test split not used | Yes: `build_baseline` takes `train, val, _` from `chronological_split`. The test part is discarded; validation is used only for the reference **score** histogram | `src/monitoring_baseline.py` |
| Linked to the served model | Yes: sha256 of `model.json`, `pipeline.json` and `metadata.json` all match `models/e7_v1`; pipeline id `E7-1fe84390` | `baseline_reproducibility.py` |
| Feature definitions, types and semantics match serving | Yes: 0 mismatches on 20,000 train rows (API path vs offline path); 0 mismatches on 200 rows end to end; float64 | 2026-10-30 investigation, `step1_parity.py`, `step1_e2e.py` |
| Enough statistics | Counts per log bin, flag counts, type counts, missing and non-finite rates, mean, std, min, max, 8 percentiles, validation score histogram | `bl-e7_v1-2.json` |
| Historical versions identifiable and reproducible | `bl-e7_v1-1` (schema 1) and `bl-e7_v1-2` (schema 2) are both kept; baselines are immutable. **Regenerating `bl-e7_v1-2` gives an identical file** (only `created_at` differs) | `baseline_reproducibility.py` |
| Dashboard baseline = drift baseline | Same file today (`bl-e7_v1-2`), but chosen by **two different rules**. Defect **D4**, fixed | below |

Split facts, recomputed from the CSV, match the model metadata. They're needed for Phase 2:

| Split | Steps | Rows | Fraud | Fraud rate |
|---|---|---|---|---|
| train | 1–280 | 3,792,821 | 3,175 | 0.0837% |
| validation | 281–354 | 1,276,276 | 780 | 0.0611% |
| test | 355–743 | 1,293,523 | 4,258 | **0.3292%** (about 5× validation) |

## 1.2 The observed drift

The brief's values correspond to the `e7_v1|loadtest` series. In the last saved report, that was the **first series, and the one the dashboard showed by default** (defect **D1**):

| | Brief | loadtest, 1 h | loadtest, 24 h | production, 24 h (untagged `wrk`) |
|---|---|---|---|---|
| oldbalanceDest | 7.3721 | 7.427 | 7.490 | 6.699 |
| dest_bal_zero | 6.2231 | 6.280 | 6.342 | 5.551 |
| amt_vs_type_median | 0.9703 | 0.969 | 0.969 | 0.970 |
| transaction_type | 0.9441 | 0.942 | 0.943 | 0.943 |
| amount | 0.5162 | 0.516 | 0.516 | 0.517 |
| oldbalanceOrg | 0.3660 | 0.366 | 0.366 | 0.366 |
| orig_bal_zero | 0.1212 | 0.121 | 0.121 | 0.121 |
| prediction score | 0.0539 | 0.054 | 0.054 | 0.054 |

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Synthetic load-test traffic differs from training | **Confirmed: the cause** | 10,656,381 of 10,656,403 stored events are `wrk-…`. Its formula `receiver_balance = lognormal(11, 2)` reproduces production `oldbalanceDest` with PSI 0.0006. It never sends a zero receiver balance, while training has 42.2% zeros (100% of PAYMENT, i.e. merchant receivers) |
| Transaction-type proportions changed | Confirmed, **but synthetic**: `wrk` draws types uniformly (20% each vs DEBIT 0.6% in training) | `investigations/2026-10-30-oldbalanceDest-drift.md` |
| Transformation differs between training and serving | **Rejected** | 0/20,000 offline and 0/200 end-to-end mismatches |
| Stale or wrong baseline | **Rejected** for the values shown (hashes match). The selection rule was inconsistent (D4) | `baseline_reproducibility.py` |
| Invalid or missing values | **Rejected** in the stored data: 0 missing, 0 non-finite, 0 dead-lettered. **But** the API accepted `amount: Infinity` and crashed on `NaN` (D2), and drift would have called non-finite values "missing" (D3) | live reproduction below |
| Aggregation or event-processing defect | **Rejected**: an exact recount of the store equals the consumer's state for all 4 series (9,782,421 events), including the `oldbalanceDest` bin counts. 0 replay, gap or observe errors | `aggregation_recount.py` |
| Actual change in the transaction population | **Rejected**: there was no real traffic. Real PaySim test-period rows show no drift | live verification below |

## 1.3 Drift-metric validation

| Item | Result | Evidence (test) |
|---|---|---|
| PSI bins and smoothing | Same `log_bin` on both sides; ≤ 10 reference-mass groups + 2 out-of-range groups; additive α = 0.5. Equals an independent NumPy implementation in 200 random cases; symmetric | `test_psi_matches_an_independent_implementation_and_is_symmetric` |
| Zero-frequency bins | Finite on both sides (smoothing); chi-square merges groups with expected count < 5, or reports `not_applicable` | `test_psi_zero_bins_and_empty`, `test_chi_square_not_applicable_and_merging` |
| Missing and non-finite values | Missing is tested separately (exact binomial + threshold). **Non-finite values were counted as missing: D3, fixed**, now reported and checked on their own | `test_d3_…`, `test_missing_rate_change_…` |
| Categorical comparison | Known categories; unknown rate checked separately; chi-square = `scipy.stats.chisquare` on identical groups | `test_chi_square_matches_scipy_…`, `test_unknown_categories` |
| Significance and multiple testing | Holm across all tests of a window (valid under dependence); equals a brute-force implementation | `test_holm_matches_brute_force` |
| Large-sample significance | Significant p with PSI below threshold → `no_significant_drift`, with the reason stated | `test_large_sample_tiny_change_…` |
| Minimum sample size | 500 per feature and window. Measured no-drift PSI bias for `amount`: mean 0.027, p99 0.048 (the docs said ~0.02; corrected) | `validity_measurements.py`, `test_no_drift_false_alarm_rate_…` |
| False alarms / power at n = 500 | 0 of 1,000 windows flagged (any feature or prediction); 100% detection of amount ×10, amount ×1.8, 20% missing, type-mix shift | `validity_measurements.py`, `test_known_drift_is_detected` |
| Window boundaries and timestamps | Prediction time; the minute before a 1 h window is excluded; **the 24 h effective start was not reported: D7, fixed** | `test_one_hour_window_edges`, `test_long_window_reports_its_effective_start` |
| Sustained drift | **Gaps without data counted as continuous: D5, fixed** | `test_d5_a_gap_without_data_breaks_sustained_drift` |
| Alert deduplication | Fires once, re-fires on escalation, resolves only with data. **Alerts could stay open forever without data: D6, fixed** (`expired_no_data`) | `test_d6_…`, existing alert tests |
| Recovery after consumer restart | Identical report after a restart; re-deliveries are not counted (offset marks) | `test_report_is_identical_after_consumer_restart` |
| Feature drift vs prediction drift | Separate blocks; prediction drift never counts as input drift | `test_prediction_drift_reported_separately` |
| Synthetic traffic separable | `X-Traffic-Source` tag → separate series, no alerts. Historical untagged events can't be re-tagged; they age out (no deletion, no offset change) | `test_load_test_traffic_is_evaluated_but_never_alerts` |

## Confirmed defects and fixes

| # | Defect | Effect | Fix | Test |
|---|---|---|---|---|
| D1 | Dashboard opened on the first series: the largest (ML) or dict order (drift) | The load-test series was shown as if it were the system's status | Production listed first by `/api/ml` and `/api/drift`; the page defaults to production, labels test series "test traffic", and says when production has no data | `test_d1_…`, screenshots checked |
| D2 | `amount: Infinity` accepted and scored; `NaN` → 422 handler crashed → 500 | Non-finite inputs reached the model and monitoring; clients got a 500 | `allow_inf_nan=False` on amount and both balances; the 422 handler renders non-finite inputs as strings | `test_d2_…` (Infinity, −Infinity, NaN, 1e400) |
| D3 | Non-finite inputs counted as "missing" | Wrong diagnosis (missing data vs invalid data) | Separate `non_finite_rate` with its own threshold (`non_finite_rate`, 0.001) | `test_d3_…` |
| D4 | Dashboard ML section: highest-numbered baseline, unvalidated; drift: newest valid | Could show a different baseline than the one used for drift | One function, `drift_detection.select_baseline`, used by both; skipped files listed | `test_d4_…` |
| D5 | `drift_since` survived insufficient or missing data | "Sustained ≥ 60 min" could fire after one evaluation following a gap | Continuity requires `drift_detected` at every short-window evaluation | `test_d5_…` |
| D6 | Insufficient data never resolves, so alerts stayed open forever when traffic stopped | Stale alerts presented as current | `expired_no_data` after a full window without evidence (configurable). Live: the six 1 h alerts expired | `test_d6_…` (2 tests) |
| D7 | 24 h window reported its nominal start; data can start up to 14 min earlier | Window misstated | `effective_start` reported and shown | `test_long_window_reports_its_effective_start` |
| D8 (minor) | Type-mix TVD shown for 5 samples | Small-sample number looked like evidence | Hidden below `min_samples`, like every other statistic | full suite |

## Live verification (stack restarted after the reboot)

- **Traffic sent:**
  - 1,000 real PaySim test-period rows, tagged `test`, all 200 → all 9 features `no_significant_drift` (PSI 0.0001–0.019), prediction PSI 0.015;
  - 5 untagged requests → production (n = 5, `insufficient_data`);
  - a 15 s tagged load test (17,235 requests) → its own series (7 of 9 features drifting), no new alerts.
- **`amount: Infinity`** → 422 with `"input": "inf"`.
- **Both dashboard sections open on production.** The test series are dashed and labelled "test traffic"; the drift summary shows `insufficient_data` honestly.
- **Expiry:** the six 1 h `feature_drift` alerts closed with `expired_no_data`.
  - The 24 h alerts stay open while yesterday's untagged events are in the window (about 13:40–13:55 UTC today).
  - Their acknowledgement expires at 13:46 UTC.
  - They then expire 24 h after their last evidence.
- **Drift calculation:** 12–60 ms per run.

## Tests

- `pytest`: **260 passed**, 4 skipped. That's 17 new tests in `tests/test_drift_audit.py`, plus an updated fixture in `tests/test_ml_monitoring.py`.
- `RUN_KAFKA_TESTS=1 pytest tests/test_kafka_integration.py`: **3 passed**, against the real broker.

## Files

**Inspected:**
- `src/drift_detection.py`, `monitoring/drift_monitor.py`, `src/ml_monitoring.py`, `src/monitoring_features.py`, `src/monitoring_baseline.py`;
- `src/events.py`, `src/event_consumer.py`, `api/main.py`;
- `monitoring/live_dashboard.py`, `monitoring/live_dashboard.html`;
- `config/drift_config.json`, `monitoring/baselines/*`, `models/e7_v1/*`, `data/fraud_data.csv`;
- the saved report, state, alert files and event store.

**Changed:**
- `api/main.py` (D2);
- `src/drift_detection.py` (D3, D4, D8);
- `src/ml_monitoring.py` (D1, D4);
- `monitoring/drift_monitor.py` (D1, D4–D7);
- `monitoring/live_dashboard.py` (D4), `monitoring/live_dashboard.html` (D1, D3, D7);
- `config/drift_config.json` (`non_finite_rate`, `expire_without_data`);
- `DATA_DRIFT.md`;
- `tests/test_drift_audit.py` (new), `tests/test_ml_monitoring.py`;
- `investigations/2026-10-31-phase1-drift-audit/*` (evidence scripts).

**Not changed:** models, feature pipeline, decision policy, Kafka offsets, stored events.

## Remaining risks

- **PSI bands (0.10 / 0.20) are conventions, not calibrated for this business.** Replaying the PaySim test period is the natural calibration source. 1,000 rows gave a max PSI of 0.019.
- **Only one feature is checked at a time.** Coarse quarter-decade bins (×1.78) hide changes within a bin.
- **The traffic tag is trusted from internal callers.** An external gateway must strip it.
- **Semantics:** PaySim records merchant balances as 0 ("not tracked"). Real clients may send real merchant balances, which would raise `oldbalanceDest`/`dest_bal_zero` drift that is a definition difference, not a population change. Agree the definition with the data owners before going live.
- **No labels.** Drift says nothing about accuracy.
- **Cosmetic (pre-existing, not fixed):** in the dashboard ML section, the "Input data quality" table overlaps the "Model performance" text at some widths.

## Reproduce

```bash
D=investigations/2026-10-31-phase1-drift-audit
python $D/baseline_reproducibility.py      # regenerate the baseline, compare, check hashes and split facts (~40 s)
python $D/aggregation_recount.py           # recount the event store vs the consumer state (~2 min)
python $D/validity_measurements.py         # false-alarm and detection rates (Monte Carlo)
python $D/live_traffic.py                  # needs the running stack: replay + untagged + Infinity
pytest tests/test_drift_audit.py tests/test_drift_detection.py tests/test_drift_monitor.py
pytest && RUN_KAFKA_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 pytest tests/test_kafka_integration.py
```

## Next: Phase 2 (model matrix and metrics audit)

**Prerequisites:**
- approval of this phase;
- `data/fraud_data.csv` (present; split facts above);
- the model artifacts in `models/` (`e5_v1`, `e6_v1`, `e7_v1`, `e7_v2`) and their `metadata.json`;
- the notebook `notebooks/01_eda_and_modeling.ipynb`, where E1–E4 were run. E1–E4 have no saved artifacts in `models/`, so their recorded numbers may only be reproducible by re-running the notebook code.

**Phase 2 will:**
- rebuild the E1–E7 matrix from artifacts;
- recompute every metric from raw predictions;
- mark what can't be verified;
- not touch the served model.
