"""NilePay fraud detection UI.

    streamlit run streamlit_app/app.py

A thin client over the existing system: it submits transactions through the public Nginx entry
point and reads the existing health and monitoring endpoints. It holds no model and makes no
decisions itself (streamlit_app/README.md).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from streamlit_app import config  # noqa: E402

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="NilePay Fraud Detection", page_icon=":material/shield:", layout="wide")

nav = st.navigation([
    st.Page(HERE / "views" / "dashboard.py", title="Dashboard", icon=":material/monitoring:", default=True),
    st.Page(HERE / "views" / "analyzer.py", title="Transaction Analyzer", icon=":material/fact_check:"),
    st.Page(HERE / "views" / "recent.py", title="Recent Transactions", icon=":material/history:"),
])

with st.sidebar:
    st.caption("NilePay real-time fraud detection")
    st.caption(f"API: `{config.API_BASE_URL}`  \nMonitor: `{config.MONITOR_URL}`  \n"
               f"UI traffic tag: `{config.TRAFFIC_SOURCE}`")

nav.run()
