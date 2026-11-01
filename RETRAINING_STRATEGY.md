# Retraining strategy

This document defines:
- when the NilePay fraud model may be retrained;
- which data may be used;
- how a challenger would be judged against the champion;
- how promotion and rollback would work.

It was written in Phase 3 of the ML lifecycle upgrade (2026-11-01). It builds on the Phase 1 drift audit ([investigations/2026-10-31-phase1-drift-audit.md](investigations/2026-10-31-phase1-drift-audit.md)) and the Phase 2 model audit ([investigations/2026-10-31-phase2-model-audit.md](investigations/2026-10-31-phase2-model-audit.md)).

**Status markers used below:**
- ✅ **implemented and tested**;
- 📐 **designed, not implemented**;
- ⛔ **blocked** by missing labels, data or a business decision;
- 🔒 **future work that needs authorization**.

## 1. Executive summary

- **Nothing retrains or promotes automatically, and nothing here changed what is served.**
  - The policy is report-only. `auto_train` and `auto_promote` are `false`, and the code refuses any other value.
  - The champion `models/e7_v1` is unchanged (hashes verified), and so are its thresholds and the API.
- **The evaluator can recommend, never act.** ✅ `python -m src.retraining triggers` reads the drift report and the label store and returns one of seven recommendations. Today it says `insufficient_data`: there are 0 production predictions and no labels.
- **Drift alone can never make a model eligible for retraining.** It can only lead to `investigate_drift`.
- **Retraining is blocked today for three independent reasons:**
  1. ⛔ There are **no production labels**.
  2. ⛔ There is **no point-in-time feature source**. Prediction events deliberately carry no amounts or balances, so production outcomes can be *evaluated* but not *trained on*.
  3. ⛔ There are **no business-approved performance limits**. Every gate is `null` and stays closed.
- **One real defect was fixed.** ✅ `python -m src.train --version v1` would have silently **overwritten the champion** (and e5_v1, e6_v1, comparison_v1.md). Model directories are now immutable, the champion is protected by name, and the check runs before any data is loaded.
- **The strategy is ready for controlled future use** as a policy, an eligibility filter, an evaluator and a set of artifact guards. It is **not** an automatic retraining system, and must not be described as one.

## 2. Existing architecture and current champion

```
 request ─► Nginx :8080 ─► 6 × FastAPI (FraudPredictor e7_v1 + DecisionEngine) ─► response
                                     │  prediction event (score, decision, type, coarse bins; no amounts)
                                     ▼
                              Kafka nilepay.prediction-events
                                     │
                                     ▼
                    event consumer ──► event store (JSONL)  ◄── labels/*.jsonl  (src.ml_performance add-labels)
                          │                     │                     │   ⛔ no label source exists
                          ▼                     │                     │
               ml_monitoring_state.json         │                     │
                          │                     │                     │
                          ▼                     ▼                     ▼
                   drift monitor ─► drift_report.json      evaluate_performance (labels ⋈ predictions)
                                          │                           │
                                          └───────────┬───────────────┘
                                                      ▼
                         ✅ src.retraining triggers  (report-only recommendation; action_taken: "none")
                                                      │  human decision
                                                      ▼
   data snapshot ─► ✅ select_training_records ─► ✅ split ─► src.train (manual) ─► models/<set>_<new version>
   ⛔ no feature source  (eligibility, dedup,       (chronological,     ✅ immutable dirs,      │
                          labels, tags)            label cutoff)       provenance in metadata  ▼
                                                                         ✅ check-model (features, schema)
                                                                                                │
                                                       📐 candidate-vs-champion report ◄────────┘
                                                                     │
                                                       📐 human approval ─► 🔒 shadow ─► canary ─► promote
                                                                     (MODEL_DIR + policy valid_for_models)
```

**Champion `e7_v1`**

| | |
|---|---|
| Artifact | `models/e7_v1/`: model.json sha256 `83f976e0…`, pipeline.json `1fe84390…` (pipeline id `E7-1fe84390`), metadata.json `ad6cdc08…`. The same hashes are in drift baseline `bl-e7_v1-2` |
| Features | amount, oldbalanceOrg, oldbalanceDest, exceeds_orig_bal, orig_bal_zero, dest_bal_zero, is_max_amount, amt_vs_type_median, type_code |
| Training data | PaySim steps 1–280. Threshold 0.54 selected on validation steps 281–354 with cost `100·FN + FP` |
| Decision policy | `dp-e7-2026-10-26.1`: approve < 0.54 ≤ review; decline ≥ 0.999 in **shadow** mode; `valid_for_models: [e7_v1, e7_v2]` |
| Selection | `MODEL_DIR` (deploy default `models/e7_v1`). The API refuses to start if the policy isn't validated for the model |
| Known limits (Phase 2) | Results are PaySim evidence only. `exceeds_orig_bal` is a simulator artifact (R4): FPR is 2.9% on transfers the sender can afford, against 0.13% headline. `e7_v2` is a byte-identical copy, not a challenger |

## 3. Retraining triggers

The evaluator returns exactly one recommendation, chosen in this order:

| Recommendation | When | Meaning |
|---|---|---|
| `investigate_data_quality` | the report is missing, stale (> 900 s) or errored; monitoring is stale; a baseline problem; drift of kind `missing`, `unknown_category` or `non_finite`; `monitoring_error` / `monitoring_stale` alerts | Fix the data or monitoring path. Retraining on broken inputs would learn the defect |
| `eligible_for_candidate_training` | labels sufficient **and** an approved gate is violated with confidence **and** a training-data source exists | A human *may* start candidate training (🔒) |
| `retraining_blocked` | labels sufficient, but a gate is unset, or the model is degraded and there's no feature source | The evidence exists, but a decision or data source is missing |
| `retraining_not_recommended` | labels sufficient and every approved gate is met | Keep the champion |
| `investigate_drift` | labels insufficient **and** production drift: long-window distribution or prediction drift, or a `sustained_drift` alert | Find the cause; obtain labels |
| `insufficient_data` | labels insufficient and fewer than `min_samples` production predictions in every window | Nothing can be concluded |
| `labels_required` | labels insufficient, enough volume, no drift | Performance is unknown |

Every report carries `reasons`, `blockers`, the evidence it used and `action_taken: "none"`.

### A. Data drift (✅ existing detector, ✅ new evaluator)

| Item | Value | Source |
|---|---|---|
| Signals that can start an investigation | feature distribution drift in the long window (24 h); prediction-score drift; an active `sustained_drift` alert | `drift_report.json` (Phase 1) |
| Thresholds | PSI ≥ 0.10 moderate, ≥ 0.20 substantial, **and** Holm-adjusted chi-square p < 0.01 | `config/drift_config.json` |
| Minimum sample | 500 per feature and window | `min_samples` |
| Persistence | `sustained_drift`: drift at **every** short-window evaluation for ≥ 60 min (gaps break it, Phase 1 D5); the long window is 24 h | `alerts.sustained_minutes` |
| Traffic | Only `production`-tagged series (`X-Traffic-Source`). `loadtest`, `test` and `demo` are evaluated but ignored | `triggers.monitored_traffic_sources` |
| Calibration | **The PSI bands are conventions, not calibrated for NilePay.** Calibrate them by replaying the PaySim test period, and later on labelled production weeks | Phase 1 |

**Why drift never triggers training by itself:**
- **Phase 1 is the counter-example.** A PSI of 7.37 came entirely from load-test traffic.
- **Drift measures a change in inputs, not a loss of accuracy.** Phase 1 measured prediction PSI 0.054 on that traffic.
- **A definition change also looks like drift.** For example, real merchant balances where PaySim records 0.
- **The training data couldn't be built anyway.** There are no labels and no point-in-time features.

### B. Performance deterioration (✅ evaluator, ⛔ no labels)

| Item | Policy |
|---|---|
| Source | Confirmed outcomes through the existing label contract: `src.ml_performance add-labels`, with `transaction_id`, `is_fraud`, `label_at` and `source` |
| Metrics | At the served approve threshold: TP, FP, FN, TN, recall, precision, FPR, F1, cost `100·FN + FP`, cost per 1,000 transactions, PR-AUC, fraud prevalence. Accuracy is never used for ranking |
| Minimum | ≥ 100 usable labels (`ml_performance.MIN_LABELS`) **and** ≥ 30 fraud labels. With 30 frauds, a measured recall of 0.90 has a Wilson 95% interval of 0.74–0.97, so only large drops are detectable. This is a placeholder |
| Window | Predictions older than the **30-day label maturity**. Younger ones are pending: a "not fraud" label may still flip |
| Uncertainty | Wilson intervals for recall and FPR. "Degraded" means the **whole** interval is beyond the gate. A point estimate below the gate isn't enough (tested) |
| Gates | `recall_floor`, `fpr_ceiling`, `max_cost_per_1k_transactions` and `min_cost_improvement_vs_champion` are all **`null`** until the business approves values. While they're null, the result is `retraining_blocked`, never a pass |
| Reference | The champion's own labelled performance in earlier mature windows. **Not** the PaySim test metrics: those come from a different prevalence and include simulator artifacts (Phase 2, R4/R6). 📐 Weekly trend of mature labels, not implemented |
| Delayed or insufficient labels | `labels_required` / `insufficient_data`, with counts and no metric values. Unlabeled predictions are never counted as legitimate |

### C. Operational and data-quality signals

| Signal | Detected by | Class | Action |
|---|---|---|---|
| Non-finite input (`Infinity`, `NaN`) | API: 422 since Phase 1 D2. Drift: `non_finite` kind | caller bug | Fix the client. **Not retraining** |
| Missing features | `missing_features` in events; missing-rate drift (`missing` kind) | data pipeline | Repair the balance source or caller |
| Unknown transaction type | `unknown_category` drift; the API `Literal` rejects unknown types | schema change | Contract or schema decision; model work only if the business adds a type |
| Unexpected type mix | `transaction_type` drift, `type_mix_tvd` | investigate | Test traffic? A new product? Compare with the business calendar |
| Feature distribution change | A above | investigate | `investigate_drift` |
| Label delay | maturity window; coverage and delay p50/p95 in `ml_performance` | expected | Wait. Don't evaluate immature windows |
| Missing or duplicated events | producer delivery counters; consumer `duplicates`; dead-letter topic; Kafka lag | operational | Fix delivery. Duplicates are already dropped by `event_id` |
| Monitoring down or stale | report age, `freshness.stale`, `monitoring_stale` alert | operational | `investigate_data_quality`: restart the consumer or monitor |
| Broken preprocessing or schema | `FraudPredictor` refuses a model/pipeline mismatch; `DecisionPolicy.check_model`; `check-model` schema check | **immediate** | Roll back (section 9). Not a retraining case |

## 4. Training-data eligibility

### Categories

| Category | Eligible? | Why |
|---|---|---|
| 1. Confirmed labelled production transactions | **Yes**, if every rule below holds | The only real ground truth. ⛔ None exist, and there's no feature source to train on |
| 2. Awaiting labels (younger than 30 days) | **No** → `label_pending` | Early labels are mostly fraud confirmations, so including them biases prevalence |
| 3. Unlabeled prediction events | **No** → `unlabeled` | A prediction isn't an outcome. Treating unlabeled as legitimate would teach the model its own blind spots |
| 4. Synthetic or load-test traffic (`loadtest`, `test`, `demo`) | **Never** → `synthetic_traffic` | Not a real population (Phase 1: `wrk` sends no zero receiver balances) |
| 4b. Untagged events | **No** → `untagged_traffic` | Before tagging existed, 10.6M untagged events were load tests (Phase 1). Monitoring treats untagged as production; training doesn't |
| 5. Historical PaySim data | **Offline experiments only** (`paysim_offline` label source) | Simulator ground truth with known shortcuts (Phase 2 R2–R5). Fine for development, **not** evidence of production performance |
| 6. Invalid or corrupted records | **No** → `invalid_label` / `invalid_features` / `duplicate` | — |

### Rules

All implemented in `select_training_records` (✅); every row gets exactly one category:
- **Label:** only `0` or `1`, from an allowed source (`chargeback`, `manual_review`, `confirmed_investigation`, `paysim_offline`), with `label_at` at or after the prediction time.
- **Forbidden as the label:** `isFlaggedFraud`, `fraud_probability`, `decision`, `would_decline` and `reason_code`. A model's output never becomes its own ground truth (`check_target`).
- **Duplicates, resolved deterministically and independent of row order:**
  - a re-delivered `event_id` keeps its first copy;
  - a second prediction for the same `transaction_id` keeps the earliest prediction, the same rule as `ml_performance`;
  - per transaction, the latest `label_at` wins, through `read_labels`.
- **Features:** `type` must be one of the five known types. Amount and balances must be present, finite and ≥ 0.
- **Chronology** (`chronological_split_with_label_cutoff`):
  - train is before `train_end`, validation before `val_end`, test after;
  - a training row whose label arrived after `train_end` is dropped, and the same for validation after `val_end`. That label didn't exist when the model was fit or the threshold chosen.
- **PaySim** keeps the existing `chronological_split` (steps 1–280 / 281–354 / 355–743), so results stay comparable with Phases 1–2.

### Prevalence and distribution risk

PaySim fraud prevalence ranges from 0.06% (validation) to 1.35% (last test quarter). Real prevalence is unknown. Precision, F1 and the cost-optimal threshold all depend on it. So:
- every evaluation must report prevalence per period;
- a threshold chosen at one prevalence must be re-checked at another.

### Why a fresh holdout is needed

The PaySim test period (steps 355–743) has scored seven experiments, and E6/E7 were designed after E4/E5 test results (Phase 2 R7). It is no longer an unbiased estimate for choosing between models. A promotion decision needs one of:
- a period that no experiment has seen;
- a realistic replay;
- labelled production weeks.

That data should be split off *before* candidate development starts, and used once.

## 5. Label requirements and limitations

- **Required:** a confirmed-outcome feed (chargebacks, manual-review results, investigations) joined on `transaction_id`. Each label needs its `label_at` and `source`.
- **Contract:** exists and is tested (`src/ml_performance.py`, ML_MONITORING.md §6). Conflicting labels count, and the latest wins.
- **Limitations:**
  - **There is no label source today (⛔).**
  - Fraud labels arrive faster than "legitimate" confirmations, hence the maturity window.
  - Review decisions influence which transactions get investigated, so labels are biased towards flagged ones (selection bias). Measuring recall needs labels on a random sample of approved transactions too.
  - Requests with no `transaction_id` can't be labelled.

## 6. Retraining workflow

| # | Stage | Status | Implementation / design |
|---|---|---|---|
| 1 | Trigger or scheduled evaluation | ✅ report-only | `python -m src.retraining triggers --model e7_v1` (run on demand or by a scheduler; it only prints) |
| 2 | Data eligibility | ✅ filter, ⛔ data | `select_training_records`. ⛔ No production feature source |
| 3 | Label quality | ✅ | `ml_performance.validate_label` / `read_labels`; label categories in step 2 |
| 4 | Schema and feature validation | ✅ | `check_schema`, `check_feature_set`, the `FraudPredictor` schema check, `check-model` CLI |
| 5 | Dataset snapshot | 📐 (fingerprints ✅) | `file_sha256` / `records_fingerprint` / `dataset_manifest` are implemented. Writing an immutable snapshot file next to the candidate is designed, not built |
| 6 | Split | ✅ | `chronological_split` (PaySim), `chronological_split_with_label_cutoff` (timestamped data) |
| 7 | Candidate training | ✅ guarded, 🔒 not run | `python -m src.train --version v3 --feature-sets E7`: manual only, into a **new** directory |
| 8 | Offline evaluation | ✅ metrics, 📐 report | `src.evaluate` (now with accuracy and F1); Phase 2 scripts for bootstrap and sliced FPR |
| 9 | Leakage and shortcut checks | ✅ | `check_feature_set` (rejects post-transaction balances, `isFlaggedFraud`, drain features, `hour`/`step`; `exceeds_orig_bal` needs a written justification) plus the existing `shortcut_checks`. Both are recorded in metadata |
| 10 | Artifact and metadata | ✅ | `models/<set>_<version>/` model, pipeline and metadata, plus a new drift baseline via `src.monitoring_baseline` |
| 11 | Candidate registration | 📐 | An append-only registry (`models/registry.jsonl`: version, hashes, state, approver, date) |
| 12 | Human approval | 📐 process | Named approver signs off on the evaluation report; recorded in the registry |

**Training isolation.** A future candidate goes to a new directory: the next E7 version is `models/e7_v3`, since `v2` is taken by the identical copy. Phase 3 created no model directory.

## 7. Candidate evaluation and promotion gates

Champion and challenger are scored on **the same eligible population**: same rows, same labels, same period. Each uses its own threshold, selected on validation and never on test.

| Gate | Evidence | Limit | Status |
|---|---|---|---|
| No target or post-transaction leakage | `check_feature_set` | none allowed | ✅ |
| No known simulator shortcut without justification | `check_feature_set`; single-feature PR-AUC (`shortcut_checks`) | `exceeds_orig_bal` needs a written justification on realistic data | ✅ |
| Features available at prediction time | feature list vs API inputs | type, amount and old balances only | ✅ by policy |
| Schema and finite-input handling | `check_schema`; API 422 on non-finite | compatible | ✅ |
| Label quality and sample | eligibility counts; ≥ 100 labels and ≥ 30 frauds | placeholders | ✅ / ⛔ data |
| Cost | `100·FN + FP`, per 1,000 transactions, with a paired block-bootstrap 95% CI (Phase 2 method) | `min_cost_improvement_vs_champion` = **null** | 📐 / ⛔ business |
| Recall / FPR | Wilson CI | `recall_floor`, `fpr_ceiling` = **null** | ✅ evaluator / ⛔ business |
| Precision, F1, PR-AUC, confusion matrix, prevalence per period | reported, not gated | — | ✅ metrics |
| Accuracy | reported only | never ranks models | ✅ (tested) |
| Artifact integrity | sha256 in the registry; `FraudPredictor` loads; pipeline round-trip | — | ✅ hashes / 📐 registry |
| Latency | `FraudPredictor.score` p99 (Phase 2: 243 µs for e7_v1) and the end-to-end API SLO (RELIABILITY_TESTING.md: p99 ≤ 100 ms) | no regression beyond an agreed margin (**null**) | 📐 |
| API compatibility | response contract unchanged; `tests/test_api.py` | — | ✅ existing tests |
| Decision policy | a policy version validated for the new model (`valid_for_models`); thresholds re-derived on validation | — | ✅ enforcement / 🔒 new policy |
| Rollback readiness | the previous directory is intact (immutability), its baseline exists, the policy covers it | — | ✅ immutability / 📐 drill |

**What must be decided before any gate can pass:**
- Business owners: acceptable recall floor, FPR ceiling and cost per 1,000 transactions, plus the minimum improvement that justifies a switch.
- Risk/compliance: the cost of a false decline, which is needed before decline leaves shadow mode.

### Promotion states (📐 design)

```
 candidate ──► evaluated ──► approved ──► shadow ──► canary ──► promoted
     │             │            │           │          │            │
     └─────────────┴────────────┴───────────┴──────────┴──► rejected│
                                            │          │            │
                                            └──────────┴────────────┴──► rolled_back (champion restored)
```

| State | Entered when | Serving effect |
|---|---|---|
| candidate | artifact written to a new directory; `check-model` passes | none |
| evaluated | evaluation report produced on the shared population | none |
| approved | a named human signs off on the report and the gates | none |
| shadow | 🔒 challenger scores a copy of traffic; its decisions aren't returned | none for clients (extra CPU) |
| canary | 🔒 a small traffic share is served by the challenger with its own validated policy | partial |
| promoted | 🔒 `MODEL_DIR` points at the challenger on all instances; the old champion stays on disk | full |
| rejected | any gate fails or the approver declines | none |
| rolled_back | post-promotion monitoring or labels show a regression | champion restored |

## 8. Versioning and reproducibility

`src/train.py` writes into `metadata.json` the existing fields:
- version, `trained_at`, data path, rows;
- split boundaries, fraud rate per split, cost function;
- XGBoost parameters, library versions;
- feature set and names, dtype, required inputs, threshold;
- validation and test metrics, `shortcut_checks`, gain importance.

**New in Phase 3, for future artifacts only:**
- `dataset_sha256`;
- `target`, `label_definition`;
- `random_seed`;
- `code_revision` (git commit and dirty flag);
- `retraining_policy_version`;
- `feature_policy_check`.

Existing metadata files are not rewritten. The comparison report title now lists the actual feature sets; it used to be the hard-coded "E5 vs E6".

**Related evidence:**
- **Determinism:** a refit of E7 with the same data and code is bit-identical to `e7_v1` (Phase 2).
- **Drift baselines** are versioned per model (`monitoring/baselines/<model>/bl-<model>-N.json`) and record the artifact hashes.
- **Decision policies** list `valid_for_models`.

## 9. Rollback strategy

Rollback works because model directories are immutable (✅): the previous champion is always on disk, byte-identical.

1. Point `MODEL_DIR` at the previous directory (e.g. `models/e7_v1`). The decision policy must list it in `valid_for_models`, or the API refuses to start, which is intended.
2. Restart one instance at a time: `deploy/stop.sh N && deploy/start.sh N`. Nginx keeps serving from the others.
3. Verify `/health` reports `model_version` on every instance, and the event stream shows the old `model_version`. Monitoring separates series by model version, and the old model's drift baseline still exists.
4. Record the rollback in the registry (📐).

The procedure uses existing mechanisms only. A timed rollback drill is 🔒 future work and was not run in Phase 3.

## 10. Implemented changes (Phase 3)

| File | Change |
|---|---|
| `config/retraining_policy.json` | **New.** Report-only policy: protected champion, label rules, eligibility and feature rules. Business gates are `null` |
| `src/retraining.py` | **New.** `load_policy`, `check_target`, `evaluate_triggers` (+ Wilson CI), `select_training_records`, `chronological_split_with_label_cutoff`, `check_feature_set`, `check_schema`, `guard_new_artifact_dir`, fingerprints / `dataset_manifest` / `code_revision`. CLI: `triggers`, `check-model`. It imports no training, serving or decision code (tested) |
| `src/train.py` | **Fix:** refuses a protected or non-empty output directory and an existing comparison report, checked before loading data. Provenance fields in metadata. Report title fixed. Same training code and parameters |
| `src/drift_detection.py` | Additive `drift_kinds` per feature result (`distribution` / `missing` / `unknown_category` / `non_finite`), so the evaluator doesn't parse reason text. Statuses and thresholds unchanged |
| `tests/test_retraining.py` | **New**, 42 tests |
| `RETRAINING_STRATEGY.md` | **New** (this document) |
| `README.md`, `DATA_DRIFT.md` | Pointers; `drift_kinds` documented |

## 11. Tests executed (2026-11-01)

| Command | Result |
|---|---|
| `pytest tests/test_retraining.py` | **42 passed** |
| `pytest` (full suite) | **327 passed, 7 skipped**, 0 failed. The 7 skips are opt-in data/Kafka tests |
| `RUN_KAFKA_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 pytest tests/test_kafka_integration.py` | **3 passed** (real broker, its own test topics) |
| `RUN_DATA_TESTS=1 pytest tests/test_model_audit.py -k reproduce` | **3 passed**: e5/e6/e7 metrics still reproduce from raw predictions |

**Mutation checks.** Each check deliberately broke one rule in a temporary copy, then restored it:
- removing the synthetic-traffic filter → caught;
- removing champion protection → caught (after a test was added for a missing champion path);
- making drift yield "eligible" → caught.

**Brief requirement → test:**

| # | Requirement | Test(s) |
|---|---|---|
| 1 | Synthetic traffic excluded | `test_1_synthetic_and_untagged_traffic_are_never_eligible`, `test_drift_on_loadtest_traffic_is_ignored` |
| 2 | Unlabeled ≠ fraud or legitimate; predictions aren't labels | `test_2_*` |
| 3 | `isFlaggedFraud` can't be the target | `test_3_isflaggedfraud_cannot_be_the_target` |
| 4 | Missing or invalid labels | `test_4_invalid_early_and_pending_labels` |
| 5 | Too few positives → no claims | `test_5_too_few_fraud_labels_give_no_performance_claim` |
| 6 | Chronological split | `test_6_chronological_split_has_no_future_rows_and_no_late_labels` |
| 7 | Duplicates | `test_7_duplicates_are_resolved_deterministically` |
| 8 | Schema mismatch | `test_8_schema_mismatch_is_reported`, `test_check_model_on_the_champion_flags_exceeds_orig_bal` |
| 9 | Drift without labels doesn't trigger training | `test_9_drift_without_labels_only_asks_for_investigation` |
| 10 | Champion can't be overwritten | `test_10_*` (4 tests, including the `train` CLI refusing before loading data) |
| 11 | No automatic training or promotion | `test_11_*` (policy flags, static import/call check), `test_cli_is_report_only_on_an_empty_store` |
| 12 | Existing tests compatible | full suite above |

**Live checks:**
- **With the stack stopped,** the evaluator returned `investigate_data_quality`: the report was 4 h old.
- **On a fresh report from a copy of the store,** it also returned `investigate_data_quality`: the consumer was down.
- **With the stack running,** it returned `insufficient_data`: 0 production predictions, labels `unavailable`.
- **Side effects:** none. The store listing was unchanged after the CLI runs.

## 12. Remaining risks and prerequisites

**Prerequisites before any production retraining:**
1. **A label source** (chargebacks, review outcomes), including labels on a random sample of approved transactions.
2. **A point-in-time feature source:** the raw request inputs and balances as they were at scoring time, retained under an approved privacy policy. The event store deliberately omits them.
3. **Business-approved gates:** recall floor, FPR ceiling, cost per 1,000 transactions, minimum improvement, and a false-decline cost.
4. **A fresh holdout or realistic replay** for the promotion decision.

**Open risks:**
- **E7's main feature `exceeds_orig_bal` is a PaySim artifact** (R4). An E7 variant without it, or an evaluation on amount ≤ balance traffic, is needed before trusting any FPR figure.
- **PSI bands are uncalibrated.**
- **`min_fraud_labels` = 30 detects only large recall drops.**
- **Label selection bias** from the review process.
- **`X-Traffic-Source` is trusted from internal callers.** An external gateway must strip it.
- **The registry and approval record don't exist yet (📐).** Until they do, promotion state lives in people's heads.

## 13. Actions not performed in Phase 3

- No model trained, retrained or fine-tuned; no `models/e7_v3` or any other new model directory created.
- No model promoted, shadowed or canaried; `MODEL_DIR` unchanged (`e7_v1`).
- `models/*` artifacts unchanged (sha256 before and after identical, and `e7_v1` still matches baseline `bl-e7_v1-2`).
- `config/decision_policy.json` and the thresholds unchanged.
- API code and response contract unchanged:
  - `git diff` on `api/`, `src/decision_engine.py`, `src/data_preprocessing.py`, `src/predict.py`, `src/events.py`, `src/event_consumer.py` and the Nginx config is empty;
  - 1,000 `/predict` responses before and after are byte-identical;
  - 20 live requests through Nginx return identical scores and decisions.
- Kafka consumer-group offsets were identical before and after all Phase 3 work.
  - The final 20 live verification requests then appended **20 `test`-tagged events** through the normal path. The offsets advanced by exactly 20 through normal consumption; there was no reset or seek.
  - The historical event files are unchanged in size, and no stored events were deleted or modified.
- No labels or metrics fabricated. All numbers here are measured, quoted from Phases 1–2, or marked as placeholders.
