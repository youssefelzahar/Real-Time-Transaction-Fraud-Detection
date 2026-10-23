# Real-Time-Transaction-Fraud-Detection
The system receives a transaction, estimates its fraud risk, and helps the business decide whether to approve it, reject it, or send it for manual review.  The important business trade-off is that missing fraudulent transactions costs money, but incorrectly blocking legitimate customers also costs money.  Your system should balance both.


## How to run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train E5 (baseline), E6 and E7 on the same chronological split; writes
# models/e5_v1/, models/e6_v1/, models/e7_v1/ and models/comparison_v1.md
python -m src.train --data data/fraud_data.csv --version v1
#   --max-amount 10000000   transaction cap used by is_max_amount
#   --cost-fn 100 --cost-fp 1   business cost used to pick the threshold

pytest                                        # unit + API tests (no dataset needed)

MODEL_DIR=models/e6_v1 ACCOUNT_SNAPSHOT=balances.json uvicorn api.main:app --reload
```

`ACCOUNT_SNAPSHOT` is optional: a JSON file `{"account_id": balance}` used by the local `InMemoryAccountService` stand-in.

## Pipeline

| File | Role |
|---|---|
| `src/data_preprocessing.py` | `chronological_split` and `FeaturePipeline`: the single feature function used by training **and** the API |
| `src/train.py` | Trains E5/E6/E7, picks the threshold on validation, evaluates once on test, runs shortcut checks, saves artifacts + report |
| `src/evaluate.py` | Cost function, threshold selection, metrics, leakage/shortcut checks |
| `src/predict.py` | `FraudPredictor`: loads artifacts, checks the model and pipeline schemas match, scores records |
| `src/decision_engine.py` | Probability to `approve` / `review` |
| `api/main.py` | FastAPI `/predict` and `/health` |

Chronological split: train steps 1-280, validation 281-354, test 355-743. Business cost = `100 x FN + 1 x FP`.

### Features (no `isFraud`, `isFlaggedFraud`, `newbalanceOrig`, `newbalanceDest`)

| Feature | Definition | E5 | E6 | E7 |
|---|---|---|---|---|
| `amount`, `oldbalanceOrg`, `oldbalanceDest` | raw pre-transaction values | ✓ | ✓ | ✓ |
| `amt_to_orig_bal` | amount / oldbalanceOrg, NaN if balance is 0 or missing | | ✓ | |
| `exceeds_orig_bal` | amount > oldbalanceOrg | ✓ | ✓ | ✓ |
| `orig_bal_zero` / `dest_bal_zero` | balance == 0 | ✓ | ✓ | ✓ |
| `amt_to_dest_bal` | amount / oldbalanceDest | ✓ | | |
| `is_max_amount` | amount >= `max_amount` (configurable, default 10,000,000) | ✓ | ✓ | ✓ |
| `amt_vs_type_median` | amount / train-period median amount of its type | ✓ | ✓ | ✓ |
| `hour`, `type_*` | step % 24, one-hot type | ✓ | | |
| `type_code` | fixed mapping CASH_IN=0, CASH_OUT=1, DEBIT=2, PAYMENT=3, TRANSFER=4 (unseen -> NaN), saved in `pipeline.json` | | | ✓ |

Missing inputs stay NaN (a flag is NaN when its input is unknown, not 0), and XGBoost routes NaN natively. An unseen type gives a NaN median ratio and all-zero `type_*` columns. All features are `float64` in a fixed order, which is saved in the artifacts.

### Artifacts (`models/<feature_set>_<version>/`)

- `model.json`: the XGBoost model
- `pipeline.json`: type medians, `max_amount`, feature names/order, required inputs
- `metadata.json`: threshold, cost function, split boundaries, validation/test metrics, shortcut checks, feature importance, library versions

## E5 vs E6 vs E7 results (test period, threshold chosen on validation)

| Exp | Threshold | Precision | Recall | PR-AUC | FPR | FP | FN | Cost |
|---|---|---|---|---|---|---|---|---|
| E5 | 0.53 | 0.7013 | 0.9814 | 0.9711 | 0.00138 | 1,780 | 79 | 9,680 |
| E6 | 0.99 | 0.9988 | 0.9993 | 0.9998 | 0.00000 | 5 | 3 | 305 |
| **E7** | 0.54 | 0.7163 | 0.9880 | 0.9682 | 0.00129 | 1,666 | 51 | **6,766** |

**E6's score is a simulator shortcut, not a real-world estimate.** `amt_to_orig_bal == 1` (sender's balance fully drained) holds for 99.1% of validation fraud and 0.00% of non-fraud. That ratio alone reaches PR-AUC 0.992, and validation PR-AUC is 1.0000. This is how PaySim's fraud agent behaves, and real fraud will not follow it this cleanly. E5 deliberately excludes this signal. Use E6 for PaySim scoring only, and don't quote its numbers as expected production performance.

**E7** = E6 without `amt_to_orig_bal`, plus `type_code`. It has no shortcut warnings and is the best realistic model: test cost is 30% lower than E5 (6,766 vs 9,680), with fewer missed frauds (51 vs 79) and fewer false alerts (1,666 vs 1,780). Its PR-AUC is slightly lower (0.968 vs 0.971 on test, 0.892 vs 0.920 on validation), so the gain comes mainly at the chosen operating threshold rather than in ranking overall.

## API contract

The client sends only transaction fields. Pre-transaction balances are **not** accepted from the client: the server fetches them through an `AccountService` (`get_balance(account_id)`). In production that is the core-banking/account service; locally it is `InMemoryAccountService`.

| Field | Source |
|---|---|
| `transaction_id` (optional), `type`, `amount`, `sender_id`, `receiver_id` | request |
| `step` | request, only for E5 models (`hour`) |
| `oldbalanceOrg`, `oldbalanceDest` | account service, looked up by sender/receiver ID (source of truth) |
| `sender_balance`, `receiver_balance` (optional) | request: fallback only when the account service has no balance for that account |
| type medians, `max_amount`, threshold | model artifacts |

Unknown accounts are still scored, and the affected features are listed in `missing_features`. `balance_source` in the response shows where each balance came from (`account_service`, `request` or `missing`). Swagger (`/docs`) has ready-made request examples in the **Examples** dropdown of `POST /predict`.

```http
POST /predict
{"transaction_id": "tx-001", "type": "TRANSFER", "amount": 181.0,
 "sender_id": "C1305486145", "receiver_id": "C553264065"}
```

```json
{"transaction_id": "tx-001", "fraud_probability": 0.99999988, "decision": "review",
 "threshold": 0.99, "model_version": "e6_v1", "missing_features": []}
```

A merchant receiver with no balance on record:

```json
{"transaction_id": "tx-002", "fraud_probability": 1.45e-08, "decision": "approve",
 "threshold": 0.99, "model_version": "e6_v1", "missing_features": ["oldbalanceDest", "dest_bal_zero"]}
```
