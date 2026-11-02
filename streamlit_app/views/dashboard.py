"""Dashboard: decision counters, component health and the latency the system actually exposes."""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from streamlit_app import api_client, config, logic  # noqa: E402

BADGE = {logic.HEALTHY: "green", logic.DEGRADED: "orange", logic.UNAVAILABLE: "red", logic.UNKNOWN: "gray"}
READ_TIMEOUT_S = min(config.TIMEOUT_S, 2.0)   # monitoring reads must not stall the page


def badge(status):
    return f":{BADGE[status]}-badge[{status}]"


def render():
    health = api_client.health(config.API_BASE_URL, READ_TIMEOUT_S)
    instances = api_client.instances_status(config.INSTANCE_URLS, READ_TIMEOUT_S)
    snap = api_client.monitor_snapshot(config.MONITOR_URL, READ_TIMEOUT_S)
    s = logic.summarize_decisions(instances)
    st.caption(f"Last updated {datetime.now():%H:%M:%S}")

    st.subheader("Decisions")
    cols = st.columns(5)
    cols[0].metric("Transactions processed", logic.fmt_count(s["total"]))
    cols[1].metric("Approve", logic.fmt_count(s["approve"]))
    cols[2].metric("Review", logic.fmt_count(s["review"]))
    cols[3].metric("Decline", logic.fmt_count(s["decline"]),
                   help="Executed declines. In shadow mode the API never returns decline.")
    cols[4].metric("Would decline (shadow)", logic.fmt_count(s["would_decline"]),
                   help="Scores at or above the decline threshold that were sent to review because "
                        "declines are not enforced.")
    if s["reporting"]:
        st.caption(f"Decision Engine counters since each instance's last start, summed over {s['reporting']} of "
                   f"{s['instances']} instances (internal `/metrics/decisions`). Not a lifetime total.")
    else:
        st.caption("Decision counters not available: the instances' internal metrics endpoints are not "
                   "reachable from this host.")

    cols = st.columns(3)
    up = sum(1 for i in instances if i["up"])
    availability = "Up" if health.ok else "Down"
    cols[0].metric("API availability", availability,
                   delta=f"{up}/{len(instances)} instances up" if instances else None, delta_color="off")
    versions = logic.model_versions(health, instances)
    cols[1].metric("Current model version", ", ".join(versions) if versions else "Not available")
    policy = ", ".join(s["policy_versions"]) or "Not available"
    mode = ", ".join(s["decline_modes"]) or "not available"
    cols[2].metric("Decision policy", policy, delta=f"decline mode: {mode}", delta_color="off")
    st.caption("Availability is the state right now (public `/health` through Nginx). An uptime percentage "
               "is not exposed by the system.")

    st.subheader("System health")
    rows = [("API (Nginx + FastAPI)", logic.status_api(health, instances)),
            ("Model", logic.status_model(health, instances)),
            ("Kafka event pipeline", logic.status_kafka(snap, instances)),
            ("Event consumer (monitoring)", logic.status_consumer(snap))]
    for name, (status, detail) in rows:
        st.markdown(f"{badge(status)} **{name}**: {detail}")
    if not snap.ok:
        st.caption(f"Kafka and consumer state come from the live monitor at `{config.MONITOR_URL}` "
                   "(`python -m monitoring.live_dashboard`), which is not reachable.")

    st.subheader("Latency")
    lat = logic.latency_from_buckets(instances)
    st.markdown("**Decision step** (Decision Engine inside each API instance; source: `/metrics/decisions`)")
    if lat:
        cols = st.columns(5)
        cols[0].metric("p50", lat["p50"])
        cols[1].metric("p95", lat["p95"])
        cols[2].metric("p99", lat["p99"])
        cols[3].metric("Mean", f"{lat['mean_us']:.1f} µs")
        cols[4].metric("Max", f"{lat['max_us']:.1f} µs")
        st.caption(f"{lat['count']:,} decisions. Percentiles are histogram bucket bounds, not exact values. "
                   "Covers the threshold decision only, not feature building, model scoring or HTTP.")
    else:
        st.info("Not available: no instance reported decision latency.")

    st.markdown("**Event pipeline** (prediction time → stored by the consumer; source: consumer metrics via "
                "the live monitor)")
    c_lat = ((snap.data or {}).get("consumer") or {}).get("latency_ms") if snap.ok else None
    if c_lat and c_lat.get("p50") is not None:
        cols = st.columns(3)
        cols[0].metric("p50", f"{c_lat['p50']:.1f} ms")
        cols[1].metric("p95", f"{c_lat['p95']:.1f} ms")
        cols[2].metric("Max", f"{c_lat['max']:.1f} ms")
        st.caption("Asynchronous monitoring path, off the request path. p99 is not reported by the consumer.")
    else:
        st.info("Not available: the consumer has not reported event latency, or the live monitor is not running.")
    st.info("End-to-end API latency (client → Nginx → instance → response) is not exposed by the running "
            "system. Load-test measurements are in RELIABILITY_TESTING.md.")


st.title("Dashboard")
st.button("Refresh", icon=":material/refresh:")   # any click reruns the page and re-reads every source
if config.REFRESH_S:
    st.caption(f"Auto-refresh every {config.REFRESH_S:g} s (FRAUD_UI_REFRESH_S).")
    st.fragment(run_every=config.REFRESH_S)(render)()
else:
    render()
