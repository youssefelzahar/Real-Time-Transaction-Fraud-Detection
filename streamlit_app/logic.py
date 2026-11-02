"""Pure functions behind the pages: input validation, status derivation, aggregation.
Every value the backend doesn't expose stays None; the UI shows it as "Not available"."""
import math
import numbers

# api/main.py TransactionRequest
TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
REQUEST_FIELDS = ("transaction_id", "type", "amount", "sender_id", "receiver_id", "step",
                  "sender_balance", "receiver_balance")

HEALTHY, DEGRADED, UNAVAILABLE, UNKNOWN = "Healthy", "Degraded", "Unavailable", "Unknown"


# --- transaction form -------------------------------------------------------------------------------

def _number(value, label, errors, required=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            errors.append(f"{label} is required.")
        return None
    if isinstance(value, bool):
        errors.append(f"{label} must be a number.")
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be a number.")
        return None
    if not math.isfinite(x):
        errors.append(f"{label} must be a finite number.")
        return None
    if x < 0:
        errors.append(f"{label} cannot be negative.")
        return None
    return x


def build_payload(form):
    """Validate the form and build the /predict body. Returns (payload, errors).
    Optional fields left empty are omitted (never replaced by 0); nothing outside the API contract is sent."""
    errors, payload = [], {}
    tx_id = (form.get("transaction_id") or "").strip()
    if tx_id:
        payload["transaction_id"] = tx_id
    if form.get("type") not in TRANSACTION_TYPES:
        errors.append(f"Transaction type must be one of {', '.join(TRANSACTION_TYPES)}.")
    else:
        payload["type"] = form["type"]
    amount = _number(form.get("amount"), "Amount", errors, required=True)
    if amount is not None:
        payload["amount"] = amount
    for key, label in (("sender_id", "Sender ID"), ("receiver_id", "Receiver ID")):
        value = (form.get(key) or "").strip()
        if not value:
            errors.append(f"{label} is required.")
        else:
            payload[key] = value
    for key, label in (("sender_balance", "Sender balance"), ("receiver_balance", "Receiver balance")):
        value = _number(form.get(key), label, errors)
        if value is not None:
            payload[key] = value
    step = form.get("step")
    if step is not None and not (isinstance(step, str) and not step.strip()):
        if isinstance(step, bool) or not isinstance(step, numbers.Number) or step != int(step) or step < 0:
            errors.append("Step must be a whole number >= 0.")
        else:
            payload["step"] = int(step)
    assert set(payload) <= set(REQUEST_FIELDS)
    return (None if errors else payload), errors


# --- decision display ---------------------------------------------------------------------------------

def decline_modes(instances):
    return sorted({(i.get("decisions") or {}).get("decline_mode") for i in instances} - {None})


def shadow_note(response, modes):
    """What the returned decision means for the payment. Only states what the backend established."""
    notes = []
    if response["decision"] == "review" and response["reason_code"] == "HIGH_RISK":
        notes.append("The score reached the decline threshold, but declines run in shadow mode: the "
                     "transaction was sent to review and counted as a would-decline. It was not blocked.")
    elif response["decision"] == "decline":
        notes.append("The API returned DECLINE under an enforced decline policy.")
    if modes == ["shadow"]:
        notes.append("Decline mode: shadow. The API never returns DECLINE in this mode; this result is a "
                     "model and policy evaluation, and this UI does not execute or block payments.")
    elif modes == ["enforce"]:
        notes.append("Decline mode: enforce.")
    elif len(modes) > 1:
        notes.append(f"Instances report different decline modes: {', '.join(modes)}.")
    else:
        notes.append("Decline mode: not available (instance metrics not reachable). "
                     "This result is a model and policy evaluation; this UI does not execute payments.")
    return notes


# --- dashboard ----------------------------------------------------------------------------------------

def summarize_decisions(instances):
    """Decision counters summed over the instances that answered /metrics/decisions."""
    reporting = [i["decisions"] for i in instances if i.get("decisions")]
    out = {"instances": len(instances), "reporting": len(reporting), "total": None, "approve": None,
           "review": None, "decline": None, "would_decline": None,
           "policy_versions": sorted({d.get("policy_version") for d in reporting} - {None}),
           "decline_modes": sorted({d.get("decline_mode") for d in reporting} - {None})}
    if reporting:
        out["total"] = sum(d.get("total", 0) for d in reporting)
        for k in ("approve", "review", "decline"):
            out[k] = sum((d.get("by_decision") or {}).get(k, 0) for d in reporting)
        out["would_decline"] = sum(d.get("would_decline", 0) for d in reporting)
    return out


def model_versions(health_result, instances):
    versions = {(i.get("health") or {}).get("model_version") for i in instances if i.get("up")}
    if health_result.ok:
        versions.add((health_result.data or {}).get("model_version"))
    return sorted(versions - {None})


def status_api(health_result, instances):
    n, up = len(instances), sum(1 for i in instances if i.get("up"))
    if not health_result.ok:
        if up:
            return DEGRADED, f"{up}/{n} instances answer, but the public endpoint does not ({health_result.message})"
        return UNAVAILABLE, health_result.message or "public endpoint not reachable"
    if not n:
        return HEALTHY, "public endpoint answers; no instance list configured"
    if up == n:
        return HEALTHY, f"public endpoint answers; {up}/{n} instances up"
    if up == 0:
        return HEALTHY, "public endpoint answers; instance endpoints not reachable from this host"
    return DEGRADED, f"public endpoint answers; only {up}/{n} instances up"


def status_model(health_result, instances):
    versions = model_versions(health_result, instances)
    if not versions:
        return (UNAVAILABLE if not health_result.ok else UNKNOWN), "no instance reports a loaded model"
    if len(versions) > 1:
        return DEGRADED, f"instances serve different models: {', '.join(versions)}"
    loaded = sum(1 for i in instances if i.get("up"))
    return HEALTHY, f"{versions[0]} loaded" + (f" on {loaded}/{len(instances)} instances" if instances else "")


def _publish_failures(events):
    return (sum((events.get("failed") or {}).values()) + events.get("buffer_full_drops", 0)
            + events.get("produce_errors", 0) + events.get("serialization_errors", 0))


def status_kafka(snapshot_result, instances):
    events = [i["events"] for i in instances if i.get("events")]
    enabled = [e for e in events if e.get("kafka_enabled")]
    failures = sum(_publish_failures(e) for e in enabled)
    delivered = sum(e.get("delivered", 0) for e in enabled)
    if events and not enabled:
        return UNAVAILABLE, "event publishing is disabled on the instances (KAFKA_BOOTSTRAP_SERVERS not set)"
    pub = f"producers: {delivered:,} delivered, {failures:,} failed since instance start" if enabled else \
        "producer counters not reachable"
    if not snapshot_result.ok:
        return UNKNOWN, f"broker state unknown (live monitor not reachable); {pub}"
    kafka = (snapshot_result.data or {}).get("kafka") or {}
    if not kafka.get("up"):
        return UNAVAILABLE, f"broker not reachable ({kafka.get('error', 'down')}); {pub}"
    detail = f"broker up, consumer lag {kafka.get('lag', 'n/a'):,} events; {pub}" if isinstance(kafka.get("lag"), int) \
        else f"broker up; {pub}"
    return (DEGRADED if failures else HEALTHY), detail


CONSUMER_STATES = {"running": HEALTHY, "starting": DEGRADED, "stalled": DEGRADED, "stopped": UNAVAILABLE}


def status_consumer(snapshot_result):
    if not snapshot_result.ok:
        return UNKNOWN, "live monitor not reachable (python -m monitoring.live_dashboard)"
    c = (snapshot_result.data or {}).get("consumer") or {}
    state = c.get("state")
    age = c.get("heartbeat_age_s")
    detail = f"{state}" + (f", heartbeat {age:g} s ago" if isinstance(age, (int, float)) else "")
    return CONSUMER_STATES.get(state, UNKNOWN), detail


def latency_from_buckets(instances):
    """Decision-engine latency (decision step only) merged over instances: count, mean, max and the
    histogram bucket that holds p50/p95/p99 (an upper bound, not an exact percentile)."""
    merged, count, weighted, mx = {}, 0, 0.0, None
    for i in instances:
        lat = (i.get("decisions") or {}).get("decision_latency_us")
        if not lat or not lat.get("count"):
            continue
        for label, c in (lat.get("buckets") or {}).items():
            merged[label] = merged.get(label, 0) + c
        count += lat["count"]
        weighted += lat.get("mean", 0) * lat["count"]
        mx = lat.get("max") if mx is None else max(mx, lat.get("max", 0))
    if not count:
        return None
    labels = list(merged)            # bucket order as reported: <=5us ... <=1000us, >1000us

    def bucket(q):
        cum = 0
        for label in labels:
            cum += merged[label]
            if cum >= q * count:
                return label.replace("<=", "≤ ").replace(">", "> ").replace("us", " µs")
        return labels[-1]
    return {"count": count, "mean_us": weighted / count, "max_us": mx,
            "p50": bucket(0.50), "p95": bucket(0.95), "p99": bucket(0.99)}


def fmt_count(value):
    return "Not available" if value is None else f"{value:,}"
