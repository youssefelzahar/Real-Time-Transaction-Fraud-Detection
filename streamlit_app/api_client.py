"""HTTP calls to the existing NilePay interfaces. No Streamlit here, nothing raises: every call
returns a Result, so the UI can always show what happened.

A transaction is submitted exactly once (one POST, no retries): a retry could score and publish
the same transaction twice, and after a 504 it may already have been scored.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import requests

# api/main.py ScoreResponse
RESPONSE_FIELDS = ("transaction_id", "fraud_probability", "decision", "reason_code", "threshold",
                   "decision_policy_version", "model_version", "missing_features", "balance_source")
DECISIONS = ("approve", "review", "decline")
TRAFFIC_HEADER = "X-Traffic-Source"


@dataclass
class Result:
    ok: bool
    data: Any = None
    error_kind: Optional[str] = None   # connection | timeout | client_error | server_error | invalid_response
    message: str = ""
    status: Optional[int] = None


def _detail(resp):
    """The API's error text: {"detail": str} or the 422 list of field errors."""
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        return resp.text[:200] or resp.reason
    if isinstance(detail, list):
        parts = []
        for err in detail:
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
            parts.append(f"{loc}: {err.get('msg', 'invalid')}" if loc else str(err.get("msg", "invalid")))
        return "; ".join(parts)
    return str(detail)


def _http_error(resp):
    status = resp.status_code
    if status == 504:
        msg = ("The fraud API timed out (504). The transaction may already have been scored: "
               "do not resubmit it automatically.")
    elif status == 503:
        msg = "The fraud API is temporarily unavailable (503: no healthy instance or overloaded)."
    elif status >= 500:
        msg = f"The fraud API failed ({status}): {_detail(resp)}"
    else:
        msg = f"The request was rejected ({status}): {_detail(resp)}"
    return Result(False, error_kind="server_error" if status >= 500 else "client_error", message=msg, status=status)


def validate_score_response(data):
    """Checks the fields the UI relies on; returns an error message or None."""
    if not isinstance(data, dict):
        return "response is not a JSON object"
    missing = [f for f in RESPONSE_FIELDS if f not in data]
    if missing:
        return f"response is missing {', '.join(missing)}"
    p = data["fraud_probability"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
        return "fraud_probability is not a number in [0, 1]"
    if data["decision"] not in DECISIONS:
        return f"unknown decision {data['decision']!r}"
    return None


def submit_transaction(base_url, payload, traffic_source, timeout_s, session=requests):
    """POST /predict once. Returns Result(ok, data=ScoreResponse dict, status=200)."""
    try:
        resp = session.post(f"{base_url}/predict", json=payload, timeout=timeout_s,
                            headers={TRAFFIC_HEADER: traffic_source})
    except requests.Timeout:
        return Result(False, error_kind="timeout",
                      message=f"No answer from the fraud API within {timeout_s:g} s. The transaction may "
                              "have been scored: check Recent Transactions before resubmitting.")
    except requests.ConnectionError:
        return Result(False, error_kind="connection",
                      message=f"Cannot connect to the fraud API at {base_url}. Is Nginx running?")
    except requests.RequestException as e:
        return Result(False, error_kind="connection", message=f"Request failed: {type(e).__name__}")
    if resp.status_code != 200:
        return _http_error(resp)
    try:
        data = resp.json()
    except ValueError:
        return Result(False, error_kind="invalid_response", status=200,
                      message="The fraud API returned a response that is not JSON.")
    problem = validate_score_response(data)
    if problem:
        return Result(False, error_kind="invalid_response", status=200,
                      message=f"Unexpected response from the fraud API: {problem}.")
    return Result(True, data=data, status=200)


def get_json(url, timeout_s, session=requests):
    try:
        resp = session.get(url, timeout=timeout_s)
    except requests.Timeout:
        return Result(False, error_kind="timeout", message=f"{url}: timed out")
    except requests.RequestException:
        return Result(False, error_kind="connection", message=f"{url}: not reachable")
    if resp.status_code != 200:
        return _http_error(resp)
    try:
        return Result(True, data=resp.json(), status=200)
    except ValueError:
        return Result(False, error_kind="invalid_response", status=200, message=f"{url}: not JSON")


def health(base_url, timeout_s, session=requests):
    """GET /health through the public entry point."""
    return get_json(f"{base_url}/health", timeout_s, session)


def instance_status(url, timeout_s, session=requests):
    """One instance's internal endpoints: /health, /metrics/decisions, /metrics/events."""
    out = {"url": url}
    for key, path in (("health", "/health"), ("decisions", "/metrics/decisions"), ("events", "/metrics/events")):
        r = get_json(url + path, timeout_s, session)
        out[key] = r.data if r.ok else None
    out["up"] = out["health"] is not None
    return out


def instances_status(urls, timeout_s, session=requests):
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as pool:
        return list(pool.map(lambda u: instance_status(u, timeout_s, session), urls))


def monitor_snapshot(monitor_url, timeout_s, session=requests):
    """The live dashboard's aggregated view (Kafka, consumer, recent-event feed)."""
    return get_json(f"{monitor_url}/api/snapshot", timeout_s, session)
