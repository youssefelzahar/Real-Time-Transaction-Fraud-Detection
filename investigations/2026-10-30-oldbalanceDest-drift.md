# Drift investigation: `oldbalanceDest`, 2026-10-30

| | |
|---|---|
| Alert | `feature_drift` on `oldbalanceDest`, model `e7_v1`, baseline `bl-e7_v1-2`. Dashboard PSI 7.37; report at 13:28 UTC: PSI 6.88 (1 h, n = 546,526) and 6.74 (24 h, n = 7,462,559); threshold 0.10 / 0.20 |
| Outcome | **Not production drift.** The "production" data was synthetic load-test traffic whose generator never sends a zero receiver balance. The data pipeline is correct. **No model change.** |
| Actions | Load tests are now tagged and monitored in their own series (no alerts). The 15 drift alerts are acknowledged until 2026-10-31 13:46 UTC, with the reason recorded. Real-traffic drift stays monitored |
| Evidence | Scripts and the scan output are in [`2026-10-30-oldbalanceDest/`](2026-10-30-oldbalanceDest/). The commands are at the end of this report |

## Step 1: verify the data

Does the monitor see what the model sees, computed the way the baseline was computed?

| Check | Result |
|---|---|
| Served model = the baseline's model | All 6 instances run `MODEL_DIR=models/e7_v1`. The sha256 of `model.json`, `pipeline.json` and `metadata.json` all equal the hashes recorded in `bl-e7_v1-2`. Pipeline id `E7-1fe84390` matches |
| Baseline built from the right data | `data/fraud_data.csv`, 6,362,620 rows. Train split steps 1–280, 3,792,821 rows (the split recorded in the model's metadata) |
| Online vs offline features | 20,000 random train rows: API path (`pipeline.transform_one` + `feature_summary`) vs baseline path (`pipeline.transform` + `log_bins`). **0 feature mismatches, 0 bin mismatches, 0 flag mismatches.** Both float64 |
| End to end (API → Kafka → consumer → store) | 200 known train rows sent through Nginx, tagged `test`. 200/200 stored, **0 bin or flag mismatches**. The sample had 78 zero receiver balances, and all 78 were stored as zero (bin 0), so zeros are not lost anywhere |
| Event quality in the store | 0 unparsable lines. Missing features: 0. Non-finite: 0. The dead-letter topic `nilepay.prediction-events.dlq` holds 0 messages (offset 0) |
| Where the balance came from | `balance_source.receiver = request` for 100% of events: the value is what the client sent, with no account-service lookup or default substituted |
| Events without a feature summary | 3,193,833 `e7_v2` events from 2026-10-28, written before monitoring existed. They aren't in any drift statistic |

**Conclusion:** no feature-engineering, dtype or pipeline defect. The monitor faithfully reports what the clients sent.

## Step 2: understand the traffic

Historical events carry no traffic tag, so they were classified by `transaction_id` prefix in one pass over the event store: 10,656,403 events, both daily files.

| Source (by id) | Events | Share | When (UTC) |
|---|---|---|---|
| `wrk-…` (load generator `tests/wrk_predict.lua`) | 10,656,381 | 99.9998% | 7 hours between 2026-10-28 17:00 and 2026-10-30 12:00: the benchmark and stress-test sessions |
| `demo-…` (dashboard buttons) | 20 | — | 2 hours |
| `tx-001` (manual test) | 2 | — | 2026-10-28 |

In the alerting window, the "production" series (`e7_v1`, 24 h, n = 7,462,559) consisted of 7,462,548 `wrk-` events with feature summaries plus 11 dashboard demo events. **There was no real production traffic at all.** Everything counted as production because nothing distinguished test traffic.

## Step 3: investigate the feature

Reference = training split. Production = all `wrk-` events with summaries (n = 7,462,548). Bins are quarter-decade log bins (value ≥ 10^(bin/4) − 1).

| | Reference (train) | Production (`wrk`) |
|---|---|---|
| Share with `oldbalanceDest = 0` | **42.18%** | **0.000%** |
| p25 | bin 0 (zero) | bin 16 (≥ 9,999) |
| p50 | bin 20 (≥ 99,999) | bin 19 (≥ 56,233) |
| p75 / p95 | bin 23 (≥ 562,340) / 26 (≥ 3.16M) | bin 21 (≥ 177,827) / 24 (≥ 1.0M) |
| Range | bins 0–33 | bins 5–31 |
| PSI (10 groups) / binned KS distance | — | **7.98** / 0.42 |
| PSI on non-zero balances only | — | 1.60 (KS 0.53) |

**Why the reference has 42% zeros (PaySim semantics):**

| Type | Train rows | Zero destination balance | Merchant receivers (`M…`) |
|---|---|---|---|
| PAYMENT | 1,277,764 | **100%** | 100% |
| CASH_OUT | 1,347,134 | 14.1% | 0% |
| TRANSFER | 313,391 | 11.8% | 0% |
| CASH_IN | 831,177 | 11.5% | 0% |
| DEBIT | 23,355 | 0% | 0% |

PaySim doesn't track merchant balances, so every PAYMENT has a zero destination balance. Among customer receivers, 12.8% have zero.

**Why production has none.** `tests/wrk_predict.lua` sets `receiver_balance = lognormal(11, 2)` for every transaction type, PAYMENT included. That is always positive, with median e^11 ≈ 59,874 (bin 19). Sampling that formula reproduces the production distribution almost exactly: **PSI 0.0006, KS distance 0.006**. The generator also draws transaction types uniformly (about 20% each, vs DEBIT 0.6% and CASH_OUT 35.5% in training), which explains the `transaction_type` and `dest_bal_zero` alerts raised alongside.

**Two effects, both from the generator:**
1. No zero balances at all. This is most of the PSI: 7.98 drops to 1.60 when zeros are excluded.
2. A narrower and lower distribution of the non-zero balances: median bin 19 vs 23 for non-zero training balances.

## Step 4: prediction drift

Did the input change move the fraud scores?

| | Validation reference | `wrk` traffic |
|---|---|---|
| Score PSI (10 groups) | — | **0.054**, below the 0.10 "moderate" band |
| Chi-square | — | p ≈ 0 at n = 7.46M: significant but practically small. This is the large-sample case the method guards against |
| Decision rates | approve 99.81%, review 0.19% | approve 99.60%, review 0.40% |
| Declines executed | 0 (shadow mode) | 0 (shadow mode) |

The drift monitor reports `no_significant_drift` for predictions in both windows. **The input drift did not translate into a substantial change in the score distribution**, so the model's output is not behaving unusually on this traffic.

## Step 5: model performance

**Unavailable: there are no fraud labels.** `~/.local/share/nilepay-events/labels` doesn't exist, and nothing was submitted through `python -m src.ml_performance add-labels`. Precision, recall, PR-AUC and fraud cost (100·FN + FP) can't be measured, and they were not inferred from predictions. Synthetic `wrk` transactions have no true outcome anyway, so they could never be labelled.

## Step 6: decision

| Question | Answer |
|---|---|
| Data or pipeline defect? | No (step 1) |
| Real-world change in transactions? | No: the window contained no real transactions (step 2) |
| Explained? | Yes, fully: by the load generator's formula (step 3) |
| Model behaviour changed? | No substantial prediction drift (step 4) |
| Performance degraded? | Unknown, and there's nothing to measure (step 5) |

**Decision: monitor, and fix the monitoring input. No model re-evaluation and no retraining.**

Actions taken:
1. **Traffic source tag** (`X-Traffic-Source`, see [DATA_DRIFT.md](../DATA_DRIFT.md#traffic-sources)). Load tests, demos and tests now get their own monitoring series, evaluated but never alerted. Verified live:
   - a 20 s `wrk` run produced 112,085 events in `e7_v1|loadtest` (drift detected, **0 new alert records**);
   - an untagged request was stored as `production`;
   - a dashboard demo was stored as `demo`.
2. **Alerts acknowledged.** 15 alerts (`feature_drift` and `sustained_drift` for `e7_v1`) are acknowledged until 2026-10-31 13:46 UTC, with this report as the reason. The untagged `wrk` events leave the 24 h window around 2026-10-31 13:00 UTC. If drift is still present after the acknowledgement expires, the alerts fire again. The `low_volume` alert stays active, because it is true: there is no production traffic.
3. **Nothing changed:** the model, its thresholds, the decision policy, the baseline and the stored data.

## Follow-ups

- **Make load tests realistic.** Draw request mixes from the PaySim test period (steps 355–743), including merchant receivers with a zero balance, instead of `lognormal`. Then load tests would also exercise realistic score paths, and a drift check on `loadtest` traffic would become a useful self-test.
- **Strip `X-Traffic-Source` at the public edge** when an external gateway is added. Otherwise external callers could keep their traffic out of the production drift checks.
- **Calibrate the PSI bands** by replaying the PaySim test period (DATA_DRIFT.md, limitations).
- **Connect a label source** before any model decision.

## What would change this decision

- **Drift persists in `production`-tagged traffic** after 2026-10-31 13:00 UTC. That would mean real drift, and the same six steps apply from step 1.
- **Real traffic shows no zero receiver balances.** Then check how receiver balances are obtained (`balance_source`): merchants missing from the account service, or a client defaulting balances. That would be a data defect to fix, not a model problem.
- **Substantial prediction drift**, or **degraded labelled performance**. Then re-evaluate the model (DATA_DRIFT.md section 13), still with no automatic retraining.

## Reproduce

```bash
D=investigations/2026-10-30-oldbalanceDest
python $D/step1_parity.py                      # online vs offline features, 20,000 train rows (~40 s)
python $D/step1_e2e.py                         # 200 rows through the live stack, tagged "test" (needs the stack running)
python $D/scan_store.py ~/.local/share/nilepay-events/prediction-events-2026-10-28.jsonl \
                        ~/.local/share/nilepay-events/prediction-events-2026-10-30.jsonl   # ~2 min, writes scan_result.json
python $D/analyze.py                           # steps 1-4 numbers from the scan + baseline
python $D/step3_generator_and_types.py         # generator reproduction + PaySim zero share per type
```

`scan_result.json` (4 KB) contains only counters: no raw events and no identifiers apart from the two manual `tx-001` ids.
