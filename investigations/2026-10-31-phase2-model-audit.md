# Phase 2: model matrix and metrics audit (2026-10-31)

Scope: the brief's Phase 2.
- Rebuild the E1–E7 matrix from the evidence that exists.
- Recompute every metric from raw predictions and labels.
- Find discrepancies, leakage and shortcuts.
- Recommend a champion only if the evidence supports one.

**Nothing served was changed:** the model, Decision Engine, thresholds, feature pipeline, Kafka offsets and stored events are all as they were. The served `e7_v1` hashes still equal those recorded in drift baseline `bl-e7_v1-2`. No retraining pipeline was started.

## Verdict

- **Every recorded number is reproduced exactly.** That covers all 7 experiments, on validation and test: the threshold, the confusion matrix, cost, precision, recall, PR-AUC and FPR.
  - **E5, E6, E7:** re-scored from the saved artifacts.
  - **E1–E4:** they have no artifacts. Their numbers were verified by re-executing the notebook code.
  - **Re-execution is trustworthy:**
    - Training is deterministic. A refit of E7 gives scores **bit-identical** to `models/e7_v1`.
    - The notebook's E5 code gives scores bit-identical to `models/e5_v1`.
- **No numeric discrepancies.** The discrepancies found are in labelling and evaluation, not in the arithmetic. See [Recorded vs reproduced](#recorded-vs-reproduced).
- **E4 (test cost 1,002) is a simulator shortcut, not target leakage.**
  - PaySim's fraud agent always transfers exactly the sender's balance, capped at 10M. 97–99% of fraud does this, and essentially **no** legitimate transaction does (1 row in the training period).
  - A one-line rule with no model reaches **test cost 1,746**, far better than any honest model.
  - E6 still contains the same signal (`amt_to_orig_bal`).
- **Champion: keep E7 (`e7_v1`), with a limited claim.**
  - It is the only candidate with no known leak or shortcut feature (post-transaction balances, the drain ratio or `hour`), and it has the lowest validation cost among the admissible candidates.
  - But its headline test metrics (FPR 0.13%, precision 0.72) are inflated by PaySim artifacts. On the realistic subset (transfers and cash-outs the sender can afford), its **FPR is 2.9%**.
  - **The evidence supports E7 as the best of these candidates on PaySim. It is insufficient to state its real-world precision, FPR or cost.**
- **`e7_v2` is not a second model.** Its `model.json` and `pipeline.json` are byte-identical to `e7_v1`. There is no challenger today.

## Evidence inventory

| Exp | Code | Saved artifact | Predictions | Recorded results | Verification |
|---|---|---|---|---|---|
| E1 | notebook cells 26, 33 | none | none | notebook cell 33/34 output | **Reproduced by re-execution** |
| E2 | notebook cells 26, 33 | none | none | notebook cell 33/34 output | **Reproduced by re-execution** |
| E3 | notebook cells 19, 26, 33 | none | none | notebook cell 33/34 output | **Reproduced by re-execution** |
| E4 | notebook cells 31–33 | none | none | notebook cell 33/34 output | **Reproduced by re-execution** |
| E5 | notebook cell 34 + `src/train.py` | `models/e5_v1` | recomputed | metadata.json, comparison_v1.md, notebook | **Reproduced from artifact**; notebook code gives identical scores |
| E6 | `src/train.py` | `models/e6_v1` | recomputed | metadata.json, comparison_v1.md | **Reproduced from artifact** |
| E7 | `src/train.py` | `models/e7_v1` = `models/e7_v2` (byte-identical) | recomputed | metadata.json, comparison_v1/v2.md | **Reproduced from artifact**; refit gives identical scores |

**Not available, and not invented:**
- historical prediction files: none were ever saved, so all predictions were recomputed;
- per-experiment training logs;
- latency for E1–E4 (no artifact to time);
- any production labels.

## Common protocol (all 7 experiments)

| | |
|---|---|
| Data | `data/fraud_data.csv`, 6,362,620 rows, sha256 `16910f90…4eea6b`. One PaySim step = one simulated hour |
| Train | steps 1–280 (hours 1–280, about days 1–12): 3,792,821 rows, 3,175 fraud (0.0837%) |
| Validation | steps 281–354 (about days 12–15): 1,276,276 rows, 780 fraud (0.0611%) |
| Test | steps 355–743 (about days 15–31): 1,293,523 rows, 4,258 fraud (**0.3292%**, 5.4× validation) |
| Split | chronological; cut points between steps; no step in two periods (asserted) |
| Model | XGBoost 3.4.1: 300 trees, depth 6, learning rate 0.1, `hist`, `random_state` 42, `scale_pos_weight` = negatives/positives on train |
| Learned preprocessing | median amount per type, **fitted on train only** (E4–E7) |
| Threshold | grid 0.01…0.99 (step 0.01), minimising validation cost `100·FN + FP`; ties go to the lowest threshold. Applied unchanged to test |
| PR-AUC | `average_precision_score` on raw scores |

## Verified experiment matrix

### Test period (steps 355–743)

Every value below was recomputed from raw scores in this audit. The ranking metric is **cost**, with recall, precision, F1 and PR-AUC as context. **Accuracy is shown, but it is not used for ranking:** "approve everything" scores 99.67% on test.

| Exp | Features (summary) | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | FPR | TP | FP | FN | TN | **Cost** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | raw + **post-tx balances** + hour + type one-hot | 0.36 | 0.99801 | 0.6258 | 0.9861 | 0.7657 | 0.9763 | 0.00195 | 4,199 | 2,511 | 59 | 1,286,754 | 8,411 |
| E2 | raw pre-tx + hour + type one-hot | 0.67 | 0.99652 | 0.4856 | 0.9638 | 0.6458 | 0.9293 | 0.00337 | 4,104 | 4,347 | 154 | 1,284,918 | 19,747 |
| E3 | E2 with log1p | 0.79 | 0.99733 | 0.5557 | 0.9457 | 0.7000 | 0.9312 | 0.00250 | 4,027 | 3,220 | 231 | 1,286,045 | 26,320 |
| E4 | engineered + **`drains_orig`, `amt_to_orig_bal`** + hour | 0.97 | 0.99999 | 0.9995 | 0.9977 | 0.9986 | 0.9998 | 0.0000016 | 4,248 | 2 | 10 | 1,289,263 | 1,002 |
| E5 | E4 without the drain features (keeps hour, one-hot) | 0.53 | 0.99856 | 0.7013 | 0.9814 | 0.8181 | 0.9711 | 0.00138 | 4,179 | 1,780 | 79 | 1,287,485 | 9,680 |
| E6 | engineered + **`amt_to_orig_bal`**, no hour | 0.99 | 0.99999 | 0.9988 | 0.9993 | 0.9991 | 0.9998 | 0.0000039 | 4,255 | 5 | 3 | 1,289,260 | 305 |
| **E7** | E6 without `amt_to_orig_bal`; type as a code | **0.54** | 0.99867 | 0.7163 | 0.9880 | 0.8305 | 0.9682 | 0.00129 | 4,207 | 1,666 | 51 | 1,287,599 | **6,766** |
| *rule* | *TRANSFER/CASH_OUT that empties the sender or is ≥ 10M (no model)* | — | 0.99987 | 0.9667 | 0.9962 | 0.9813 | — | 0.00011 | 4,242 | 146 | 16 | — | *1,746* |

### Validation period (steps 281–354): where the thresholds were selected

| Exp | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | FPR | TP | FP | FN | TN | **Cost** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 0.36 | 0.9982 | 0.2478 | 0.9897 | 0.3964 | 0.9359 | 0.00184 | 772 | 2,343 | 8 | 1,273,153 | 3,143 |
| E2 | 0.67 | 0.9967 | 0.1530 | 0.9692 | 0.2643 | 0.8545 | 0.00328 | 756 | 4,184 | 24 | 1,271,312 | 6,584 |
| E3 | 0.79 | 0.9975 | 0.1940 | 0.9551 | 0.3224 | 0.8538 | 0.00243 | 745 | 3,096 | 35 | 1,272,400 | 6,596 |
| E4 | 0.97 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 780 | 0 | 0 | 1,275,496 | 0 |
| E5 | 0.53 | 0.9987 | 0.3138 | 0.9833 | 0.4758 | 0.9201 | 0.00131 | 767 | 1,677 | 13 | 1,273,819 | 2,977 |
| E6 | 0.99 | 0.99999 | 0.9911 | 1.0000 | 0.9955 | 1.0000 | 0.0000055 | 780 | 7 | 0 | 1,275,489 | 7 |
| **E7** | **0.54** | 0.9987 | 0.3178 | 0.9859 | 0.4806 | 0.8916 | 0.00129 | 769 | 1,651 | 11 | 1,273,845 | **2,751** |

The same threshold has very different precision on validation (E7: 0.32) and test (0.72). That's the prevalence shift (fraud is 5.4× denser on test), not a model change. See risk R6.

### Uncertainty of the cost comparisons

These use a paired block bootstrap over time steps (2,000 resamples). Fraud arrives in per-step bursts, so rows are not independent.

| Comparison | Validation: Δcost, 95% CI | Test: Δcost, 95% CI |
|---|---|---|
| E7 − E5 | −226, [−882, +383]: **not significant** (E7 cheaper in 77% of resamples) | −2,914, [−4,704, −1,214] |
| E7 − E2 | −3,833, [−4,966, −2,769] | −12,981, [−15,986, −10,228] |
| E7 − E1 | −392, [−1,173, +323] | −1,645, [−3,522, +219] |
| E5 − E2 | −3,607, [−4,590, −2,688] | −10,067, [−12,412, −7,945] |

E7 is stable across the test period:

| Steps | Rows | Fraud rate | FN | FP |
|---|---|---|---|---|
| 355–451 | 949,371 | 0.12% | 11 | 1,169 |
| 452–548 | 142,105 | 0.69% | 14 | 187 |
| 549–645 | 120,893 | 0.89% | 14 | 195 |
| 646–743 | 81,154 | 1.35% | 12 | 115 |

## Per-experiment details

| Exp | Artifact / pipeline | Inputs needed at prediction time | Leakage / shortcut | Threshold selection |
|---|---|---|---|---|
| E1 | none (notebook) | amount, both old balances, **`newbalanceOrig`, `newbalanceDest`**, step, type | **Target leakage (confirmed):** new balances exist only *after* the transaction executes. `newbalanceOrig` is its top feature (gain 54,264). Also `hour` (R3) | val cost grid → 0.36 (unrounded 0.36000000000000004) |
| E2 | none (notebook) | amount, `oldbalanceOrg`, `oldbalanceDest`, **step**, type | `hour` (R3) | → 0.67 |
| E3 | none (notebook) | same as E2 | `hour` (R3). log1p doesn't change tree splits; the difference from E2 is noise | → 0.79 |
| E4 | none (notebook) | same as E2 | **Simulator shortcut (confirmed):** `drains_orig` (gain 191,093; alone PR-AUC 0.991 on validation) and `amt_to_orig_bal` (alone 0.992). Also `hour` | validation cost 0 at 0.97, 0.98 **and** 0.99 (a tie). The lowest was taken; test cost is 1,002 or 1,202 depending on the tie |
| E5 | `e5_v1`, pipeline sha `6e1c03d7…` | as E2: **requires `step`** | `hour`; `exceeds_orig_bal` (R4) | → 0.53 (unique) |
| E6 | `e6_v1`, pipeline sha `75a31eb3…` | type, amount, both old balances | **Simulator shortcut (confirmed):** `amt_to_orig_bal` (alone PR-AUC 0.992). Its own metadata already warns about it | **→ 0.99, the grid's upper edge.** Validation cost keeps falling (to 0 at 0.995), so the optimum lies outside the grid (R8) |
| **E7** | `e7_v1` = `e7_v2`, model sha `83f976e0…`, pipeline `E7-1fe84390` | type, amount, `oldbalanceOrg`, `oldbalanceDest` | no post-transaction values, no drain ratio, no hour. **But** `exceeds_orig_bal` (top feature, gain 44,008) is a PaySim artifact (R4) | → 0.54 (unique). This equals `approve_threshold` in `config/decision_policy.json` (tested) |

**Feature availability at prediction time.** All inputs except E1's new balances are known when the request arrives:
- type and amount come from the request;
- the sender's and receiver's current balances come from the account system;
- `hour` comes from the clock.

But "available" is not "valid". `hour` predicts fraud here only because of how PaySim schedules agents. And `oldbalanceDest` for merchant receivers is a placeholder 0 in PaySim (see the Phase 1 report).

## E4 investigation

| Question | Finding | Evidence |
|---|---|---|
| Target leakage? | **No.** Every E4 input is known before the transaction. Medians are fitted on train only, and the threshold on validation only | code (cells 31–33) |
| Dataset-specific shortcut? | **Yes, confirmed.** In PaySim, fraud = take over an account, then move `min(balance, 10M)` by TRANSFER and CASH_OUT | below |
| — fraud types | 100% of fraud in every split is TRANSFER (50%) or CASH_OUT (50%) | `shortcut_results.json` |
| — empties the sender | 97.8% / 99.1% / 97.2% of fraud (train/val/test) vs **1 legitimate row in 3.79M** in train, and none among legitimate TRANSFER/CASH_OUT | same |
| — empties the sender or is ≥ 10M | 98.9% / 100% / 99.6% of fraud | same |
| How much is the shortcut worth? | The **rule** "TRANSFER/CASH_OUT and (empties sender or ≥ 10M)", with no model, gives test TP 4,242, FP 146, FN 16, **cost 1,746**: better than every model without the shortcut. On validation it costs 5,049 (5,049 FP, from legitimate ≥ 10M transfers) | same |
| Is the score stable? | The validation cost is 0 for thresholds 0.97–0.99; the chosen one depends on the tie rule. Validation PR-AUC is exactly 1.0 | same |
| Real-time feasible? | Yes, mechanically: it needs only the request and the sender's balance | — |
| Would it generalise? | **No evidence it would.** Real account takeovers don't always move exactly the full balance. Any fraudster leaving 1 unit behind defeats `drains_orig`. The near-perfect score measures how well the model learned PaySim's agent, not fraud | — |

**Conclusion:** E4 and E6 are an upper bound on PaySim, not production candidates. The existing `shortcut_checks` flagged E6 correctly. The notebook's own analysis already called E4 an upper bound, and this audit confirms it with numbers.

## Confirmed leakage and evaluation risks

| # | Risk | Affects | Evidence |
|---|---|---|---|
| R1 | **Target leakage:** post-transaction balances | E1 | definition. Top gain is `newbalanceOrig` |
| R2 | **Simulator shortcut:** "drains the sender" | E4, E6 | see above. Single-feature PR-AUC 0.991–0.992 |
| R3 | **Simulator shortcut:** `hour` | E1–E5 | In train, 1.8% of legitimate transactions but **26.1% of fraud** fall in hours 0–6 (fraud rate 1.19% vs 0.063% in hours 7–23, 19×). PaySim's fraud agents ignore the time of day. `hour` alone has PR-AUC 0.232. E5 also requires the client to send `step` |
| R4 | **Simulator artifact:** "amount exceeds sender balance" | E5, E7 (top feature), E2/E3 implicitly | 0% of validation/test fraud vs **90% of legitimate TRANSFER/CASH_OUT** (97% of legitimate TRANSFER) exceed the sender's balance. PaySim doesn't enforce balances for legitimate agents; a real bank declines most of these before scoring. **Effect on E7 (test, TRANSFER/CASH_OUT):** on the 59,845 rows the sender can afford, which contain all 4,258 frauds, FP is 1,619 and **FPR 2.9%**. On the 492,659 rows that exceed the balance, FP is 47 (FPR 0.010%). The headline FPR of 0.13% comes mostly from rows that wouldn't exist in production |
| R5 | **Fraud only in two types** | all | The models have never seen fraud in PAYMENT, CASH_IN or DEBIT and will score those near zero. Real fraud there would be missed |
| R6 | **Prevalence shift** | all | Fraud rate: train 0.084%, validation 0.061%, test 0.329%, and 0.12% → 1.35% across the test quarters. Precision and F1 change with prevalence (E7 precision 0.32 on validation vs 0.72 on test), so neither is a production estimate |
| R7 | **Test-set reuse** | E5–E7 | All 7 experiments were scored on the same test period, and E6/E7 were designed after E4/E5 test results. The test set is no longer a clean holdout for choosing between them. Mitigation in this audit: selection on validation only, test as confirmation |
| R8 | **Threshold grid limits** | E6, E4 | E6 sits at the 0.99 edge (validation optimum ≥ 0.995). E4 has a 3-way tie. The 0.01 step is coarse at the top. Not a problem for E7 (unique optimum at 0.54) |
| R9 | **Shortcut detector blind spot** | E5, E7 | `shortcut_checks` flags features that *find* fraud alone (single-feature PR-AUC). `exceeds_orig_bal` works by *clearing* legitimate transactions (alone PR-AUC 0.0018), so it was never flagged. E7 metadata says "warnings: none" |
| R10 | **One synthetic dataset, no labels from production** | all | No external validation is possible. Scores are not calibrated probabilities (`scale_pos_weight`) |

## Recorded vs reproduced

| Item | Recorded | Reproduced | Status |
|---|---|---|---|
| E1–E5 test metrics and validation cost (notebook outputs) | cell 33/34 | identical (counts exact, ratios to the 4 printed decimals) | **match** |
| E5, E6, E7 validation and test metrics (metadata.json) | 11 fields × 2 splits | identical (ratios to ≤ 1e-12) | **match** |
| E5, E6, E7 thresholds | 0.53 / 0.99 / 0.54 | re-selected on validation: 0.53 / 0.99 / 0.54 | **match** |
| `comparison_v1.md`, `comparison_v2.md` | | | numbers match. **Both are titled "E5 vs E6 comparison"**, but v1 contains E7 and v2 contains only E7. A hard-coded title in `src/train.py` (cosmetic; not changed in this phase) |
| `e7_v2` "v2" | trained_at 2026-10-24 | `model.json` and `pipeline.json` byte-identical to `e7_v1`; only `trained_at` differs | **not a different model**. The version label implies a change that doesn't exist |
| Notebook cell 26 | execution count is empty (the cell was edited after its last run) | the committed code reproduces the recorded outputs exactly | the code as committed is the code that produced the results |
| E7 metadata "Shortcut warnings: none" | | true for the drain check; misses R4 | incomplete (R9) |
| Notebook remark "test fraud rate about 4× train" | | 3.9× train, 5.4× validation | correct |

## Inference latency

| Model | p50 | p95 | p99 |
|---|---|---|---|
| e5_v1 | 127 µs | 150 µs | 246 µs |
| e6_v1 | 123 µs | 164 µs | 249 µs |
| e7_v1 | 123 µs | 164 µs | 243 µs |

How this was measured:
- **What:** offline `FraudPredictor.score()`, the API hot path (`transform_one` plus the booster), on one thread pinned to CPU 12, over 5,000 real test rows.
- **Not included:** HTTP, Nginx, the decision engine and Kafka.

Latency doesn't distinguish the candidates. End-to-end API latency for e7 is documented in `RELIABILITY_TESTING.md` (for example p50 1.8 ms at 3,812 req/s). It was **not re-measured** in this phase. E1–E4 can't be timed (no artifacts).

## Champion recommendation

**Admissibility filter**, applied before any metric: the candidate must have no target leakage (R1), no drain shortcut (R2), no `hour` (R3), and only inputs a real request provides.

| Exp | Admissible? | Reason |
|---|---|---|
| E1 | no | R1, R3 |
| E2, E3 | no | R3 (`hour`). Also 2.4× E7's validation cost |
| E4, E6 | no | R2 |
| E5 | no | R3; requires `step` |
| **E7** | **yes** | with risk R4 |

**Recommendation: keep `e7_v1` as champion.** This is the status quo, so nothing changes.

**Supported by the evidence:**
- E7 is the only admissible candidate. Its validation cost (2,751) is the lowest of all non-shortcut models, and it beats E5 on test (−2,914, CI excluding 0).
- Its results are exactly reproducible from the artifact and from a retrain.

**Not supported:**
- Any production estimate of precision, FPR or cost.
- The claim that E7 is *significantly* better than E5 on validation: CI [−882, +383]. E5 is excluded for `hour`, not on cost.

**Before a promotion decision can rest on metrics (later phases):**
1. labels from production or a realistic replay;
2. evaluation on a population where amount ≤ balance, or an E7 variant without `exceeds_orig_bal`, to measure how much of E7 is R4;
3. a challenger that is actually different, since `e7_v2` = `e7_v1`.

## Tests executed

| Command | Result |
|---|---|
| `pytest` (full suite) | **285 passed**, 7 skipped (Phase 1 was 260 passed, 4 skipped; +25 new tests, +3 opt-in skipped) |
| `RUN_DATA_TESTS=1 pytest tests/test_model_audit.py -k reproduce` | **3 passed** (e5_v1, e6_v1, e7_v1: thresholds re-selected on validation and all recorded metrics reproduced from raw predictions on the full CSV, 53 s) |
| `RUN_KAFKA_TESTS=1 pytest tests/test_kafka_integration.py` | **3 passed** (real broker) |

**New tests (`tests/test_model_audit.py`):**
- metrics against scikit-learn on 20 random cases, including score ties at the threshold;
- why accuracy can't rank models;
- threshold tie rule;
- grid-edge clipping;
- served policy threshold = artifact threshold = validation-selected threshold;
- `e7_v1` and `e7_v2` identical;
- the opt-in full-data reproduction.

## Files

**Changed:** `src/evaluate.py`. `evaluate_at_threshold` now also returns `accuracy` and `f1`; the change is additive. Existing callers and metadata files are unaffected, and Phase 4 will need these metrics.

**New:**
- `tests/test_model_audit.py`;
- this report;
- `investigations/2026-10-31-phase2-model-audit/`:
  - `common.py`;
  - `score_artifacts.py`;
  - `rerun_notebook.py`;
  - `shortcut_analysis.py`;
  - `latency.py`;
  - results: `artifact_results.json`, `notebook_results.json`, `shortcut_results.json`, `latency_results.json`.

Raw scores (71 MB) are cached outside the repository in `~/.cache/nilepay-model-audit/`.

**Not changed:**
- `models/*` (hashes unchanged);
- `config/decision_policy.json`;
- `src/data_preprocessing.py`, `src/decision_engine.py`, `api/`;
- Kafka offsets, stored events;
- all Phase 1 fixes.

## Remaining risks

- **R4 is the largest open question for E7.** Its measured FPR depends on PaySim's unrealistic legitimate traffic. The realistic slice suggests about 2.9% FPR on transfers and cash-outs the sender can afford, but that slice is also synthetic.
- **No production labels**, so no production performance can be stated.
- **Precision and F1 depend on prevalence**, which varies 20× across PaySim periods.
- **The test period has been looked at many times (R7).** A fresh holdout, or labelled production data, is needed for an unbiased promotion decision.
- **`hour`-free and drain-free variants of E5 weren't trained**, by design: no retraining in this phase.

## Reproduce

```bash
D=investigations/2026-10-31-phase2-model-audit
taskset -c 8-15 python $D/score_artifacts.py      # re-score e5/e6/e7 artifacts, compare with metadata (~1 min)
taskset -c 8-15 python $D/rerun_notebook.py       # re-execute notebook E1-E5 + E7 refit control (~6 min, in memory only)
python $D/shortcut_analysis.py                    # E4/shortcut evidence, sliced FPR, bootstrap (needs the two above)
TIMING_LOG=0 taskset -c 12 python $D/latency.py   # offline scoring latency
pytest tests/test_model_audit.py
RUN_DATA_TESTS=1 pytest tests/test_model_audit.py -k reproduce
pytest && RUN_KAFKA_TESTS=1 KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092 pytest tests/test_kafka_integration.py
```

`AUDIT_SCORES_DIR` overrides the score cache location.
