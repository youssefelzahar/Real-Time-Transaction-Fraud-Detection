# Demo test cases (video walkthrough)

A scripted set of test cases for recording a demo of NilePay real-time fraud detection. Every expected
result below was measured against the running stack (model `e7_v1`, policy `dp-e7-2026-10-26.1`,
decline mode **shadow**) on 2026-11-03. Scores are rounded; the decision and reason code are what to check.

Commands are PowerShell, from the project root, with the venv active. For Linux, use the equivalents in [RUN.md](RUN.md).

## 0. Before recording

Start everything (see [RUN_WINDOWS.md](RUN_WINDOWS.md), sections 5 and 6):

```powershell
python deploy\windows.py start kafka
$env:KAFKA_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
$env:MODEL_DIR = "models/e7_v1"
python deploy\windows.py start consumer
python deploy\windows.py start                      # 6 instances + Nginx
python -m monitoring.live_dashboard                 # separate terminal, http://127.0.0.1:8090
streamlit run streamlit_app/app.py                  # separate terminal, http://127.0.0.1:8501
```

Check before you start recording:

| Check | Command | Expected |
|---|---|---|
| All parts up | `python deploy\windows.py status` | instances 1–6, nginx, consumer and kafka all `running`/`ready` |
| Public health | `Invoke-RestMethod http://127.0.0.1:8080/health` | `status: ok`, `model_version: e7_v1` |

Open these browser tabs: Streamlit (`:8501`), live monitor (`:8090`), Swagger (`http://127.0.0.1:8080/docs`).

**Tip:** restart the instances right before recording (`python deploy\windows.py stop; python deploy\windows.py start`)
so the Dashboard counters start at 0 and each case visibly adds one.

---

## Scene 1 — Automated tests (≈30 s)

| ID | Step | Expected |
|---|---|---|
| T-01 | `pytest -q` | `383 passed, 10 skipped` in about 25 s. The skipped tests are the Linux/Kafka-integration ones |

---

## Scene 2 — Decisions in the Streamlit UI (main part)

Page: **Transaction Analyzer**. Fill the form and submit. Leave **Step** empty (model E7 doesn't use it).

### Approve (low risk)

| ID | Story | Type | Amount | Sender ID | Receiver ID | Sender bal. | Receiver bal. | Score | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-01 | Normal merchant payment | PAYMENT | 9839.64 | C1231006815 | M1979787155 | 170136 | 0 | 0.000000 | **APPROVE** | LOW_RISK |
| TC-02 | Transfer of part of a balance to a funded account | TRANSFER | 25000 | C1111111111 | C2222222222 | 300000 | 150000 | 0.000000 | **APPROVE** | LOW_RISK |
| TC-03 | Cash-in deposit | CASH_IN | 5000 | C3333333333 | C4444444444 | 1200 | 80000 | 0.000002 | **APPROVE** | LOW_RISK |
| TC-04 | Card debit | DEBIT | 2500 | C5555555555 | C6666666666 | 40000 | 10000 | 0.000000 | **APPROVE** | LOW_RISK |

Say: the green box, the score and `balance_source = request` (the in-memory account service has no balances, so the request's balances are used).

### Review (between the two thresholds)

| ID | Story | Type | Amount | Sender ID | Receiver ID | Sender bal. | Receiver bal. | Score | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-05 | Cash-out that empties the sender's account | CASH_OUT | 50000 | C7200000001 | C7200000002 | 50000 | 300000 | 0.970 | **REVIEW** | REVIEW_THRESHOLD |

Say: the score is between `approve_threshold` 0.54 and `decline_threshold` 0.999, so an analyst checks it. It's a risk score, not a probability (see [DECISION_POLICY.md](DECISION_POLICY.md)).

### High risk (would decline, shadow mode)

| ID | Story | Type | Amount | Sender ID | Receiver ID | Sender bal. | Receiver bal. | Score | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-06 | Classic fraud: TRANSFER drains the sender into an empty account | TRANSFER | 181 | C1305486145 | C553264065 | 181 | 0 | 0.999897 | **REVIEW** | HIGH_RISK |
| TC-07 | Large account takeover: drains 850k into an empty account | TRANSFER | 850000 | C7000000001 | C7000000002 | 850000 | 0 | 0.999763 | **REVIEW** | HIGH_RISK |
| TC-08 | Cash-out at the 10M cap | CASH_OUT | 10000000 | C1000000001 | C2000000002 | 10000000 | 0 | 0.999997 | **REVIEW** | HIGH_RISK |

Expected on screen: a yellow REVIEW box with `HIGH_RISK`, and the note *"The score reached the decline threshold, but
declines run in shadow mode … It was not blocked."* Say: the policy would decline this, but the decline threshold is
provisional, so in shadow mode the customer goes to review and the case is counted as `would_decline`.

### Borderline (good for explaining the threshold)

| ID | Story | Type | Amount | Sender ID | Receiver ID | Sender bal. | Receiver bal. | Score | Decision | Reason |
|---|---|---|---|---|---|---|---|---|---|---|
| TC-09 | Cash-out from a zero balance | CASH_OUT | 120000 | C7500000001 | C7500000002 | 0 | 0 | 0.506 | **APPROVE** | LOW_RISK |
| TC-10 | Transfer of 90% of the balance to an empty account | TRANSFER | 90000 | C7300000001 | C7300000002 | 100000 | 0 | 0.293 | **APPROVE** | LOW_RISK |

Say: TC-09 is just under 0.54, so it's approved. Moving the threshold is a business decision in
`config/decision_policy.json` and doesn't require retraining the model.

### Missing data

| ID | Story | Type | Amount | Sender ID | Receiver ID | Sender bal. | Receiver bal. | Expected |
|---|---|---|---|---|---|---|---|---|
| TC-11 | No balances known anywhere | TRANSFER | 5000 | C9999999999 | C8888888888 | *(empty)* | *(empty)* | **APPROVE**, `balance_source` = `missing` for both, `missing_features` lists `oldbalanceOrg`, `oldbalanceDest`, `exceeds_orig_bal`, `orig_bal_zero`, `dest_bal_zero` |

Say: the API still answers, but it reports which features it was missing, so nothing is silently guessed.

---

## Scene 3 — Input validation

In the UI, the form rejects bad input before anything is sent:

| ID | Input | Expected in the UI |
|---|---|---|
| TC-12 | Amount left empty | "Please fix the form: Amount is required." Nothing is sent |
| TC-13 | Sender ID left empty | "Please fix the form: Sender ID is required." Nothing is sent |

(A negative amount can't even be typed: the amount and balance fields have a minimum of 0.)

At the API (Swagger "Try it out", or PowerShell), each case returns **HTTP 422**, never reaches the model, and is counted in `input_rejections`:

| ID | Body | Expected 422 message |
|---|---|---|
| TC-14 | `{"type":"TRANSFER","amount":-50,"sender_id":"C1","receiver_id":"C2"}` | `amount`: Input should be greater than or equal to 0 |
| TC-15 | `{"type":"WIRE","amount":50,"sender_id":"C1","receiver_id":"C2"}` | `type`: Input should be 'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT' or 'TRANSFER' |
| TC-16 | `{"type":"TRANSFER","amount":50,"receiver_id":"C2"}` | `sender_id`: Field required |
| TC-17 | `{"type":"TRANSFER","amount":50,"sender_id":"C1","receiver_id":"C2","sender_balance":-1}` | `sender_balance`: Input should be greater than or equal to 0 |

PowerShell for an API case (example: TC-15):

```powershell
Invoke-WebRequest -Method Post http://127.0.0.1:8080/predict -ContentType application/json `
  -Body '{"type":"WIRE","amount":50,"sender_id":"C1","receiver_id":"C2"}' -SkipHttpErrorCheck
```

(`-SkipHttpErrorCheck` needs PowerShell 7; in Windows PowerShell 5.1 the 422 is shown as an exception, which works for the demo too.)

---

## Scene 4 — Dashboard and recent transactions

| ID | Step | Expected |
|---|---|---|
| TC-18 | Streamlit **Dashboard** → Refresh | The approve/review counts match what you submitted in Scene 2. `would_decline` = number of HIGH_RISK cases (3 if you ran TC-06–08). API, model, Kafka pipeline and consumer show **Healthy**. Model `e7_v1`, decline mode `shadow` |
| TC-19 | **Recent Transactions** page | *Transactions submitted during this session* lists each case. *Latest events in the event store* shows the same transactions, which came through Kafka and the consumer |
| TC-20 | Live monitor `http://127.0.0.1:8090` | Nginx up, 6 instances up, events delivered > 0, Kafka lag returns to 0, consumer `running` |
| TC-21 | `Invoke-RestMethod http://127.0.0.1:8080/metrics/decisions` | **404**: internal metrics aren't exposed on the public Nginx port. `http://127.0.0.1:8001/metrics/decisions` (instance, loopback only) returns them |

---

## Scene 5 — Load balancing and resilience

| ID | Step | Expected |
|---|---|---|
| TC-22 | `Get-Content deploy\run\access.log -Wait -Tail 20`, then submit a few transactions | `upstream=127.0.0.1:800N` changes from request to request: Nginx spreads load over the 6 instances |
| TC-23 | `python tests\load_api.py --url http://127.0.0.1:8080 --requests 4000 --concurrency 50 --processes 8` | Every request succeeds (0 errors); throughput and p95 latency are printed. Dashboard totals jump by 4000 |
| TC-24 | `python deploy\windows.py stop 3`, then submit TC-01 again | Still **APPROVE**. Instance 3 shows down on the Dashboard and the live monitor; Nginx routes around it. The first few requests after the stop can take about 1 s longer (see the Windows differences in RUN_WINDOWS.md) |
| TC-25 | `python deploy\windows.py start 3` | Instance 3 back to `ready`; the Dashboard shows 6/6 healthy |
| TC-26 | `python deploy\windows.py stop consumer`, submit TC-06 and TC-07, then `python deploy\windows.py start consumer` | While the consumer is stopped, predictions keep working and the live monitor's **Kafka lag** grows. After the restart, lag drains back to 0 and the events appear in *Latest events*: no event is lost |

---

## Clean-up after recording

```powershell
python deploy\windows.py stop; python deploy\windows.py stop consumer; python deploy\windows.py stop kafka
Remove-Item Env:KAFKA_BOOTSTRAP_SERVERS
```

Then close the Streamlit and live-monitor terminals (Ctrl+C).

## Suggested video order (≈8–10 min)

1. Architecture in one sentence: Streamlit → Nginx → 6 FastAPI instances (model E7 + Decision Engine) → Kafka → consumer → event store and monitoring.
2. Scene 1: tests pass.
3. Scene 2: TC-01, TC-05, TC-06, TC-09, TC-11 (one of each kind; the rest are spares).
4. Scene 3: TC-12 in the UI, TC-15 in Swagger.
5. Scene 4: Dashboard, Recent Transactions, live monitor.
6. Scene 5: TC-22, TC-24/25, TC-26.
