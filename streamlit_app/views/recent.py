"""Recent Transactions: this session's submissions, plus the live monitor's feed of stored events.
The backend has no transaction-history endpoint, so neither view is a complete history."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from streamlit_app import api_client, config  # noqa: E402


def session_rows(entries):
    rows = []
    for e in reversed(entries):
        p, r = e["payload"], e["data"] or {}
        rows.append({"submitted at": e["submitted_at"].strftime("%H:%M:%S"),
                     "transaction ID": p.get("transaction_id", "—"), "type": p["type"], "amount": p["amount"],
                     "result": r["decision"].upper() if e["ok"] else f"error: {e['error_kind']}",
                     "reason": r.get("reason_code"), "fraud score": r.get("fraud_probability"),
                     "model": r.get("model_version"), "policy": r.get("decision_policy_version")})
    return pd.DataFrame(rows)


st.title("Recent Transactions")

st.subheader("Transactions submitted during this session")
st.caption("Only what this browser session submitted from the Transaction Analyzer, including failed "
           "submissions. It is not a transaction history and is lost when the session ends.")
entries = st.session_state.get("submissions", [])
if entries:
    st.dataframe(session_rows(entries), hide_index=True, width="stretch",
                 column_config={"fraud score": st.column_config.NumberColumn(format="%.4f"),
                                "amount": st.column_config.NumberColumn(format="%.2f")})
    if st.button("Clear this session's list"):
        st.session_state.pop("submissions", None)
        st.session_state.pop("last_submission", None)
        st.rerun()
else:
    st.info("No transactions submitted in this session yet.")

st.subheader("Latest events in the event store")
st.caption("The last prediction events stored by the Kafka consumer, as shown by the live monitor "
           "(at most 14, all traffic sources including load tests and demos). The system has no "
           "transaction-history query endpoint, so this is not a complete history.")
st.button("Refresh feed", icon=":material/refresh:")   # the click reruns the page
snap = api_client.monitor_snapshot(config.MONITOR_URL, min(config.TIMEOUT_S, 2.0))
if not snap.ok:
    st.info(f"Not available: the live monitor at `{config.MONITOR_URL}` is not reachable "
            "(start it with `python -m monitoring.live_dashboard`).")
elif not (snap.data or {}).get("feed"):
    st.info("The event store has no recent events.")
else:
    feed = pd.DataFrame(snap.data["feed"])
    st.dataframe(feed, hide_index=True, width="stretch",
                 column_config={"fraud_probability": st.column_config.NumberColumn("fraud score", format="%.4f")})
