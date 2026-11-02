"""Transaction Analyzer: submit one transaction to the existing /predict endpoint and show the answer."""
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from streamlit_app import api_client, config, logic  # noqa: E402

DECISION_BOX = {"approve": st.success, "review": st.warning, "decline": st.error}
REASONS = {"LOW_RISK": "score below the approve threshold",
           "REVIEW_THRESHOLD": "score between the approve and decline thresholds",
           "HIGH_RISK": "score at or above the decline threshold"}


def submit(form):
    payload, errors = logic.build_payload(form)
    if errors:
        st.error("Please fix the form:\n\n" + "\n".join(f"- {e}" for e in errors))
        return
    t0 = perf_counter()
    with st.spinner("Scoring..."):
        result = api_client.submit_transaction(config.API_BASE_URL, payload, config.TRAFFIC_SOURCE, config.TIMEOUT_S)
    round_trip_ms = (perf_counter() - t0) * 1000
    modes = logic.decline_modes(api_client.instances_status(config.INSTANCE_URLS, min(config.TIMEOUT_S, 2.0))) \
        if result.ok else []
    entry = {"submitted_at": datetime.now().astimezone(), "payload": payload, "ok": result.ok,
             "data": result.data, "error_kind": result.error_kind, "message": result.message,
             "status": result.status, "round_trip_ms": round_trip_ms, "decline_modes": modes}
    st.session_state.setdefault("submissions", []).append(entry)
    st.session_state["last_submission"] = entry


def show(entry):
    st.divider()
    st.caption(f"Submitted at {entry['submitted_at']:%H:%M:%S} (traffic tag `{config.TRAFFIC_SOURCE}`)")
    if not entry["ok"]:
        st.error(entry["message"])
        st.caption("No prediction is shown because the API call did not succeed. Nothing was retried.")
        return
    r = entry["data"]
    DECISION_BOX[r["decision"]](f"### {r['decision'].upper()}\n{r['reason_code']}: {REASONS[r['reason_code']]}")
    for note in logic.shadow_note(r, entry["decline_modes"]):
        st.info(note)

    cols = st.columns(4)
    p = r["fraud_probability"]
    cols[0].metric("Fraud score", "< 0.0001" if 0 < p < 0.0001 else f"{p:.4f}",
                   help="Model risk score in [0, 1]. Not a calibrated probability (see DECISION_POLICY.md).")
    cols[1].metric("Approve threshold", f"{r['threshold']:g}")
    cols[2].metric("Model version", r["model_version"])
    cols[3].metric("Policy version", r["decision_policy_version"])

    st.markdown(f"**Transaction ID:** {r['transaction_id'] or 'not provided'}  \n"
                f"**Balance source:** sender `{r['balance_source'].get('sender')}`, "
                f"receiver `{r['balance_source'].get('receiver')}`")
    for side, source in r["balance_source"].items():
        if source == "missing":
            st.warning(f"No {side} balance was available, so the model scored with that feature missing "
                       "(it was not replaced by 0).")
    if r["missing_features"]:
        st.warning("Missing features: " + ", ".join(r["missing_features"]))
    st.caption(f"UI round-trip {entry['round_trip_ms']:.0f} ms, measured by this UI including network; "
               "not a production latency figure.")
    with st.expander("Raw API response"):
        st.json(r)


st.title("Transaction Analyzer")
st.caption(f"Submits one transaction to `POST {config.API_BASE_URL}/predict` (the public Nginx entry point). "
           "Leave a balance empty if it is unknown: the API then uses the account service, and nothing is "
           "replaced by 0.")

with st.form("transaction", clear_on_submit=False):
    c1, c2 = st.columns(2)
    tx_id = c1.text_input("Transaction ID (optional)", key="transaction_id", max_chars=100)
    tx_type = c2.selectbox("Transaction type", logic.TRANSACTION_TYPES, index=None,
                           placeholder="Choose a type", key="type")
    amount = c1.number_input("Amount", min_value=0.0, value=None, step=100.0, format="%.2f", key="amount",
                             placeholder="Required")
    sender = c1.text_input("Sender ID", key="sender_id", placeholder="Required, e.g. C123456789")
    receiver = c2.text_input("Receiver ID", key="receiver_id", placeholder="Required, e.g. C987654321 or M123")
    sender_balance = c1.number_input("Sender balance (optional)", min_value=0.0, value=None, step=100.0,
                                     format="%.2f", key="sender_balance", placeholder="Unknown")
    receiver_balance = c2.number_input("Receiver balance (optional)", min_value=0.0, value=None, step=100.0,
                                       format="%.2f", key="receiver_balance", placeholder="Unknown")
    with st.expander("Optional: step"):
        step = st.number_input("Step (hours since start)", min_value=0, value=None, step=1, key="step",
                               placeholder="Not sent", help="Part of the API contract, used only by E5 models. "
                                                             "The current e7 model ignores it.")
    submitted = st.form_submit_button("Submit Transaction", type="primary")

if submitted:
    submit({"transaction_id": tx_id, "type": tx_type, "amount": amount, "sender_id": sender,
            "receiver_id": receiver, "sender_balance": sender_balance, "receiver_balance": receiver_balance,
            "step": step})

if "last_submission" in st.session_state:
    show(st.session_state["last_submission"])
