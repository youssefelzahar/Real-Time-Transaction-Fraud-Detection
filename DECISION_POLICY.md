# Decision policy

The model produces a **fraud score**. The Decision Engine (`src/decision_engine.py`) turns that score into **approve**, **review** or **decline** using a policy file. The policy is business configuration: it lives in `config/decision_policy.json`, separate from training and feature preprocessing, and can change without retraining.

## Current policy: `dp-e7-2026-10-26.1`

```json
{
  "policy_version": "dp-e7-2026-10-26.1",
  "approve_threshold": 0.54,
  "decline_threshold": 0.999,
  "decline_mode": "shadow",
  "valid_for_models": ["e7_v1", "e7_v2"]
}
```

| Score | Decision | `reason_code` |
|---|---|---|
| `score < 0.54` | approve | `LOW_RISK` |
| `0.54 ≤ score < 0.999` | review | `REVIEW_THRESHOLD` |
| `score ≥ 0.999`, `decline_mode = "enforce"` | decline | `HIGH_RISK` |
| `score ≥ 0.999`, `decline_mode = "shadow"` (current) | **review** | `HIGH_RISK` (counted as `would_decline`) |

**Boundaries:** a score exactly at a threshold falls in the **higher** band. 0.54 → review, 0.999 → decline (or review in shadow). That's the same `>=` behavior the API had before.

**Rules enforced by `DecisionPolicy`:** `0 ≤ approve_threshold < decline_threshold ≤ 1`, both finite; `decline_mode` is `shadow` or `enforce`; a non-empty `policy_version`. An invalid policy fails at startup.

**Model compatibility:** an instance refuses to start if its model version isn't in `valid_for_models`. Thresholds only mean something for the score distribution they were measured on. A retrained model needs its own validation and a new policy version.

## Where the thresholds come from

All numbers are from the **validation period only** (steps 281–354, 1,276,276 transactions, 780 fraud, model `e7_v2`). The test period was not used. Reproduce with:

```bash
python -m src.decision_policy_analysis --model-dir models/e7_v2
```

### approve_threshold = 0.54 (validated)

Chosen during training on validation data with the project's cost function `100 × missed fraud + 1 × flagged legit` (see `models/e7_v2/metadata.json`). At 0.54: 2,420 not auto-approved, 769 of 780 frauds caught, 1,651 legitimate transactions flagged, 11 frauds approved.

### decline_threshold = 0.999 (PROVISIONAL, shadow mode)

The binary cost function above says nothing about **declining**. It prices a missed fraud against a review, and the project has **no cost for wrongly declining a legitimate customer**. So no "optimal" decline threshold can be derived yet. Candidates, with approve fixed at 0.54:

| decline_threshold | declined | legit declined | fraud declined | left in review | fraud in review | decline precision |
|---|---|---|---|---|---|---|
| 0.9 | 1,704 | 957 | 747 | 716 | 22 | 0.438 |
| 0.99 | 1,089 | 407 | 682 | 1,331 | 87 | 0.626 |
| 0.995 | 739 | 131 | 608 | 1,681 | 161 | 0.823 |
| **0.999** | **561** | **24** | **537** | **1,859** | **232** | **0.957** |

0.999 is the most conservative candidate: 24 legitimate customers declined out of ~1.28 million legitimate transactions (0.0019%), and it takes 537 frauds out of the review queue. Until NilePay sets a false-decline cost, it runs in **shadow mode**. The engine records what it *would* decline (`would_decline` in the metrics), but the customer gets `review`, so no one is declined by a provisional number.

### Scores are not calibrated probabilities

Training reweights fraud heavily (`scale_pos_weight`), which pushes scores up. Observed fraud rate per score band on validation:

| Score band | Transactions | Mean score | Actual fraud rate |
|---|---|---|---|
| [0.54, 0.7) | 367 | 0.61 | 1.4% |
| [0.7, 0.9) | 349 | 0.80 | 4.9% |
| [0.9, 0.99) | 615 | 0.97 | 10.6% |
| [0.99, 0.999) | 528 | 0.99 | 27.5% |
| [0.999, 1.0] | 561 | 0.9998 | 95.7% |

Read `fraud_probability` as a **risk score**, not a probability: a 0.99 is "27% of these were fraud", not "99% fraud".

## Trade-offs

| Outcome | Cost if wrong | Who pays |
|---|---|---|
| approve | Missed fraud: the loss (valued at 100 in the cost function) | NilePay |
| review | Analyst time (valued at 1), plus a delay for the customer | Operations, customer |
| decline | A legitimate customer refused: lost revenue, complaint, churn. **Not yet priced** | Customer, NilePay |

- Lowering `decline_threshold` shrinks the review queue but declines more legitimate customers.
- Raising `approve_threshold` lets more fraud through untouched.
- The review band is the safety margin between the two, and each one needs its own decision.

## When fraud prevalence changes

The thresholds apply to scores, not to rates. If the fraud share rises (the test period has ~4× the training rate):
- More transactions land in review and decline.
- The decline band's precision **rises**, because more of the high scores are real fraud.
- Legitimate declines follow **legitimate volume**, not the fraud rate.

If prevalence falls, the opposite happens: the review queue's precision drops and analysts see more false alerts. Watch `by_decision` and `would_decline` over time. A shift in their shares without a policy change is a drift signal.

## Turning decline on

1. NilePay defines a false-decline cost (or a maximum false-decline rate).
2. Re-run `src.decision_policy_analysis` on recent **validation** data and pick `decline_threshold` with that cost.
3. Set `"decline_mode": "enforce"`, bump `policy_version`, and restart the instances (`deploy/stop.sh && deploy/start.sh`).
4. Compare the `would_decline` counts collected in shadow mode with the expected decline volume before switching.

## Technical errors vs decisions

- A score that's NaN, infinite or outside [0, 1], or any failure inside model serving, returns **HTTP 500 `{"detail": "internal scoring error"}`**. It is not a decline.
- The error is logged with the `transaction_id` only, with no amounts or account data.
- It is **not retried**, since a POST may already have been processed.
- Missing balances are **not** treated as evidence of fraud: the model scores them as missing, and they're listed in `missing_features`.

## Observability

Each instance exposes counters at `GET /metrics/decisions` (internal: `127.0.0.1:800N`, blocked on the public `:8080`):

```bash
curl -s localhost:8001/metrics/decisions | python3 -m json.tool   # one instance
python tests/decision_metrics.py                                # all six summed
```

The counters are: total decisions, per decision, per `reason_code`, `would_decline`, invalid scores, decision latency (count, mean, max, µs buckets), policy version and decline mode. They are per process and reset when an instance restarts. Measured: ~7 µs mean per decision over 217,595 decisions under load.

## Future rules (not implemented)

`DecisionEngine(policy, rules=[...])` runs extra rules after the thresholds. Each rule gets the request context and the threshold result and can return a different result. No rules are configured, because none of these have defined NilePay requirements yet:

- **Amount limit:** review or decline above a configured amount. Needs the limit and the action.
- **Duplicate transactions:** needs idempotency keys and a store shared across instances.
- **Account-service unavailable:** the balance fallback already handles this. Whether to force review instead of scoring with missing features is a business choice. Never decline just because a service is down.
- **Manual-review lists:** accounts or merchants that always go to review. Needs an owner and a data source.
