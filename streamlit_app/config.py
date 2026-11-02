"""UI settings, all from environment variables (see streamlit_app/README.md)."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGINX_CONF = ROOT / "deploy" / "nginx" / "nginx.conf"

# Values accepted by the API's X-Traffic-Source header (src/events.py TRAFFIC_SOURCES).
UI_TRAFFIC_SOURCES = ("demo", "test", "production")


def _instance_urls():
    """Instance base URLs: FRAUD_INSTANCE_URLS (comma-separated), else the upstream servers in
    deploy/nginx/nginx.conf, the same list Nginx balances over (monitoring/live_dashboard.py does the same)."""
    raw = os.getenv("FRAUD_INSTANCE_URLS", "").strip()
    if raw:
        return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    try:
        text = NGINX_CONF.read_text(encoding="utf-8")
    except OSError:
        return []
    return [f"http://127.0.0.1:{p}" for p in re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", text, flags=re.M)]


def _float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


# Public entry point: Nginx, which balances over the FastAPI instances.
API_BASE_URL = os.getenv("FRAUD_API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
# Read-only live view (python -m monitoring.live_dashboard): Kafka, consumer and the recent-event feed.
MONITOR_URL = os.getenv("FRAUD_MONITOR_URL", "http://127.0.0.1:8090").rstrip("/")
# Internal per-instance endpoints (/metrics/* is blocked on the public Nginx port by design).
INSTANCE_URLS = _instance_urls()
# Tag for transactions submitted from the UI, so they never count as production traffic in
# drift monitoring (DATA_DRIFT.md). "production" only if real transactions are entered here.
TRAFFIC_SOURCE = os.getenv("FRAUD_UI_TRAFFIC_SOURCE", "demo")
if TRAFFIC_SOURCE not in UI_TRAFFIC_SOURCES:
    raise ValueError(f"FRAUD_UI_TRAFFIC_SOURCE must be one of {UI_TRAFFIC_SOURCES}, got {TRAFFIC_SOURCE!r}")
TIMEOUT_S = _float("FRAUD_UI_TIMEOUT_S", 5)
# Dashboard auto-refresh in seconds; 0 = manual only. Never faster than every 10 s.
_refresh = _float("FRAUD_UI_REFRESH_S", 0)
REFRESH_S = 0 if _refresh <= 0 else max(10.0, _refresh)
