"""ML monitoring aggregates: prediction behaviour and input data quality (see ML_MONITORING.md).

Runs inside the existing event consumer, after an event is stored, never in /predict.

State, bounded by construction:
  - 1-minute buckets for the last 60 minutes, 15-minute buckets back to 24 h (<= 60 + 97 buckets),
    keyed by prediction time (occurred_at), one stats block per (model_version, policy_version);
  - at most MAX_SERIES version pairs (the rest share "__other__"), at most MAX_KEYS keys in any
    counter dict (the rest share "__other__"), fixed score-histogram edges;
  - cumulative stats per version pair since the state file was created.
Windows of <= 60 min are minute-exact; longer windows use 15-minute resolution at the far edge.

Recovery: the state remembers the next Kafka offset it has counted per partition. It is saved
(atomically) in ConsumerRunner.commit() after the store fsync and before the offset commit, so
after a crash or restart every re-delivered message has an offset below that mark and is skipped:
each stored event is counted once. If the state file is lost or older than the committed offsets,
the skipped range is reported as `offset_gap_not_aggregated` instead of being silently missing.
"""
import bisect
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("nilepay.ml_monitoring")

STATE_VERSION = 1
KNOWN_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
DECISIONS = ("approve", "review", "decline")
MINUTE_HORIZON = 60           # minutes kept at 1-minute resolution
COARSE_MINUTES = 15
HORIZON_MINUTES = 24 * 60
MAX_SERIES = 20
MAX_KEYS = 64
OTHER = "__other__"
UNKNOWN_TYPE = "__unknown__"
DEFAULT_WINDOWS = {"5m": 5, "1h": 60, "24h": 1440}
MIN_EVENTS = 50               # below this a window is reported as "insufficient data"

# Fixed score-histogram edges: fine at both ends, with the policy thresholds (0.54, 0.999) as edges.
SCORE_EDGES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
               0.45, 0.5, 0.54, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 1.0)
N_SCORE_BINS = len(SCORE_EDGES) - 1


def score_bin(p):
    return min(max(bisect.bisect_right(SCORE_EDGES, p) - 1, 0), N_SCORE_BINS - 1)


def _inc(d, key, n=1):
    key = str(key)
    if key not in d and len(d) >= MAX_KEYS:
        key = OTHER
    d[key] = d.get(key, 0) + n


def new_stats():
    return {"n": 0, "decision": {}, "reason": {}, "would_decline": 0, "score_sum": 0.0, "score_hist": {},
            "type": {}, "missing": {}, "events_with_missing": 0, "non_finite": {}, "events_with_non_finite": 0,
            "summary_available": 0, "bins": {}, "flags": {}}


def merge(into, other):
    """Add counters of `other` into `into` (same nested shape, numbers at the leaves)."""
    for k, v in other.items():
        if isinstance(v, dict):
            merge(into.setdefault(k, {}), v)
        else:
            into[k] = into.get(k, 0) + v
    return into


def add_event(stats, e):
    """Count one stored event (dict as written to the event store) into a stats block."""
    stats["n"] += 1
    _inc(stats["decision"], e["decision"])
    _inc(stats["reason"], e["reason_code"])
    stats["would_decline"] += bool(e.get("would_decline"))
    p = float(e["fraud_probability"])
    stats["score_sum"] += p
    _inc(stats["score_hist"], score_bin(p))
    t = e.get("transaction_type")
    _inc(stats["type"], t if t in KNOWN_TYPES else UNKNOWN_TYPE)
    missing = e.get("missing_features") or []
    for name in missing:
        _inc(stats["missing"], name)
    stats["events_with_missing"] += bool(missing)
    bins, flags = e.get("feature_bins"), e.get("feature_flags")
    if bins is None and flags is None:
        return                                  # older event / summary not built: "not available"
    stats["summary_available"] += 1
    non_finite = e.get("non_finite_features") or []
    for name in non_finite:
        _inc(stats["non_finite"], name)
    stats["events_with_non_finite"] += bool(non_finite)
    for name, b in (bins or {}).items():
        _inc(stats["bins"].setdefault(name, {}), "null" if b is None else b)
    for name, f in (flags or {}).items():
        _inc(stats["flags"].setdefault(name, {}), "null" if f is None else f)


def _minute(ts):
    return int(ts // 60)


def _iso_minute(m):
    return datetime.fromtimestamp(m * 60, timezone.utc).isoformat(timespec="minutes")


def _parse_time(value):
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class MonitoringAggregator:
    def __init__(self, path=None, clock=time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.reset()
        if self.path:
            self.load()

    def reset(self):
        self.created_at = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(timespec="seconds")
        self.minutes = {}          # minute -> {series: stats}
        self.coarse = {}           # 15-minute slot start (in minutes) -> {series: stats}
        self.cumulative = {}       # series -> stats (since created_at)
        self.offsets = {}          # "topic:partition" -> next offset already counted
        self.counts = {"observed": 0, "replay_skipped": 0, "offset_gap_not_aggregated": 0,
                       "older_than_24h": 0, "future_clock_skew": 0, "observe_errors": 0,
                       "save_failures": 0, "state_load_failed": 0}
        self.last_event_at = None
        self.last_observed_at = None
        self.saved_at = None
        self.dirty = False

    # --- recovery ---------------------------------------------------------------------------
    def load(self):
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as e:
            log.warning("monitoring state not readable (%s): starting empty", e)
            self.counts["state_load_failed"] = 1
            return
        try:
            s = json.loads(raw)
            if s.get("state_version") != STATE_VERSION:
                raise ValueError(f"state_version {s.get('state_version')!r}")
            self.created_at = s["created_at"]
            self.minutes = {int(k): v for k, v in s["minutes"].items()}
            self.coarse = {int(k): v for k, v in s["coarse"].items()}
            self.cumulative = s["cumulative"]
            self.offsets = {k: int(v) for k, v in s["offsets"].items()}
            self.counts.update(s["counts"])
            self.last_event_at, self.last_observed_at = s.get("last_event_at"), s.get("last_observed_at")
            self.saved_at = s.get("saved_at")
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            bad = self.path.with_name(self.path.name + f".corrupt-{int(self.clock())}")
            log.warning("monitoring state %s is invalid (%s): moved to %s, starting empty", self.path, e, bad.name)
            try:
                self.path.replace(bad)
            except OSError:
                pass
            self.reset()
            self.counts["state_load_failed"] = 1

    def to_dict(self):
        return {"state_version": STATE_VERSION, "created_at": self.created_at, "saved_at": self.saved_at,
                "last_event_at": self.last_event_at, "last_observed_at": self.last_observed_at,
                "offsets": self.offsets, "counts": self.counts,
                "minutes": {str(k): v for k, v in self.minutes.items()},
                "coarse": {str(k): v for k, v in self.coarse.items()}, "cumulative": self.cumulative}

    def save(self):
        """Atomic write (tmp + fsync + replace). Never raises: monitoring must not block storage."""
        if not self.path or not self.dirty:
            return True
        self.compact()
        self.saved_at = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(timespec="seconds")
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(self.path)
        except OSError as e:
            self.counts["save_failures"] += 1
            log.warning("monitoring state not saved (%s); the next commit retries", e)
            return False
        self.dirty = False
        return True

    # --- ingestion --------------------------------------------------------------------------
    def _series(self, e):
        # Production keeps the original "model|policy" key (older state stays valid); tagged
        # non-production traffic (load tests, demos, tests) is counted in its own series.
        key = f"{e.get('model_version')}|{e.get('decision_policy_version')}"
        source = e.get("traffic_source") or "production"
        if source != "production":
            key = f"{key}|{source}"
        if key not in self.cumulative and len(self.cumulative) >= MAX_SERIES:
            key = f"{OTHER}|{OTHER}"
        return key

    def observe(self, event, topic=None, partition=None, offset=None):
        """Count a stored event once. event=None only advances the offset mark (dead-lettered or
        duplicate message). Returns True if the event was counted."""
        if offset is not None:
            tp = f"{topic}:{partition}"
            mark = self.offsets.get(tp)
            if mark is not None and offset < mark:
                self.counts["replay_skipped"] += event is not None
                return False
            if mark is not None and offset > mark:
                self.counts["offset_gap_not_aggregated"] += offset - mark
            self.offsets[tp] = offset + 1
            self.dirty = True
        if event is None:
            return False
        try:
            now = self.clock()
            ts = _parse_time(event["occurred_at"])
            if ts > now + 60:
                self.counts["future_clock_skew"] += 1
                ts = now
            key = self._series(event)
            add_event(self.cumulative.setdefault(key, new_stats()), event)
            m, now_m = _minute(ts), _minute(now)
            if m > now_m - MINUTE_HORIZON:
                add_event(self.minutes.setdefault(m, {}).setdefault(key, new_stats()), event)
            elif m > now_m - HORIZON_MINUTES - COARSE_MINUTES:
                slot = m - m % COARSE_MINUTES
                add_event(self.coarse.setdefault(slot, {}).setdefault(key, new_stats()), event)
            else:
                self.counts["older_than_24h"] += 1        # in cumulative, not in any window
        except (KeyError, TypeError, ValueError) as e:
            self.counts["observe_errors"] += 1
            log.warning("event not counted by monitoring: %s", type(e).__name__)
            return False
        self.counts["observed"] += 1
        self.last_event_at = max(self.last_event_at or "", datetime.fromtimestamp(ts, timezone.utc).isoformat(
            timespec="seconds"))
        self.last_observed_at = datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds")
        self.dirty = True
        return True

    def compact(self):
        """Fold minutes older than the minute horizon into 15-minute slots; drop slots beyond 24 h."""
        now_m = _minute(self.clock())
        for m in [m for m in self.minutes if m <= now_m - MINUTE_HORIZON]:
            slot = self.coarse.setdefault(m - m % COARSE_MINUTES, {})
            for key, stats in self.minutes.pop(m).items():
                merge(slot.setdefault(key, new_stats()), stats)
        for slot in [s for s in self.coarse if s + COARSE_MINUTES <= now_m - HORIZON_MINUTES]:
            del self.coarse[slot]

    # --- queries ----------------------------------------------------------------------------
    def window(self, minutes):
        """Merged stats per series for predictions made in the last `minutes`, plus volume points."""
        self.compact()
        now_m = _minute(self.clock())
        start = now_m - minutes + 1
        merged, volume = {}, {}
        for m, series in self.minutes.items():
            if m >= start:
                for key, stats in series.items():
                    merge(merged.setdefault(key, new_stats()), stats)
                    volume.setdefault(key, {})[m] = volume.get(key, {}).get(m, 0) + stats["n"]
        if minutes > MINUTE_HORIZON:
            for slot, series in self.coarse.items():
                if slot + COARSE_MINUTES > start:
                    for key, stats in series.items():
                        merge(merged.setdefault(key, new_stats()), stats)
                        volume.setdefault(key, {})[slot] = volume.get(key, {}).get(slot, 0) + stats["n"]
        return merged, {k: sorted(v.items()) for k, v in volume.items()}


# --- reporting (dashboard side, read-only) --------------------------------------------------

def _rate(n, d):
    return round(n / d, 6) if d else None


def score_percentile(hist, q):
    """Approximate percentile from the fixed histogram (linear inside a bin; error <= bin width)."""
    total = sum(hist)
    if not total:
        return None
    target, cum = q * total, 0
    for i, c in enumerate(hist):
        if c and cum + c >= target:
            lo, hi = SCORE_EDGES[i], SCORE_EDGES[i + 1]
            return lo + (hi - lo) * ((target - cum) / c)
        cum += c
    return SCORE_EDGES[-1]


def dense_hist(sparse):
    h = [0] * N_SCORE_BINS
    for k, v in sparse.items():
        if k != OTHER:
            h[int(k)] += v
    return h


def series_report(stats, baseline=None, min_events=MIN_EVENTS):
    n = stats["n"]
    hist = dense_hist(stats["score_hist"])
    dec = {d: stats["decision"].get(d, 0) for d in DECISIONS}
    types = {t: stats["type"].get(t, 0) for t in (*KNOWN_TYPES, UNKNOWN_TYPE)}
    known_n = sum(types[t] for t in KNOWN_TYPES)
    avail = stats["summary_available"]
    report = {
        "n": n, "insufficient_data": n < min_events, "min_events": min_events,
        "decisions": {d: {"n": c, "rate": _rate(c, n)} for d, c in dec.items()},
        # Shadow mode: these were returned as "review"; the decline was NOT executed.
        "shadow_would_decline": {"n": stats["would_decline"], "rate": _rate(stats["would_decline"], n)},
        "actual_declines": {"n": dec["decline"], "rate": _rate(dec["decline"], n)},
        "reasons": dict(stats["reason"]),
        "score": {"mean": round(stats["score_sum"] / n, 6) if n else None,
                  **{f"p{int(q * 100)}": (round(score_percentile(hist, q), 6) if n else None)
                     for q in (0.5, 0.9, 0.95, 0.99)},
                  "percentiles_are_approximate": True, "hist": hist, "edges": list(SCORE_EDGES)},
        "types": {t: {"n": c, "share": _rate(c, n)} for t, c in types.items()},
        "quality": {
            "events_with_missing_features": {"n": stats["events_with_missing"],
                                             "rate": _rate(stats["events_with_missing"], n)},
            "missing_by_feature": {f: {"n": c, "rate": _rate(c, n)} for f, c in sorted(stats["missing"].items())},
            "unknown_type": {"n": types[UNKNOWN_TYPE], "rate": _rate(types[UNKNOWN_TYPE], n)},
            "feature_summary_available": {"n": avail, "rate": _rate(avail, n)},
            "events_with_non_finite": {"n": stats["events_with_non_finite"],
                                       "rate": _rate(stats["events_with_non_finite"], avail)},
            "non_finite_by_feature": dict(stats["non_finite"]),
        },
        "flags": {f: {"rate_1": _rate(c.get("1", 0), c.get("0", 0) + c.get("1", 0)), "missing": c.get("null", 0)}
                  for f, c in stats["flags"].items()},
    }
    if baseline is None:
        report["baseline"] = None
        return report
    base_types = baseline["transaction_types"]
    diffs = {t: round((types[t] / known_n if known_n else 0) - base_types.get(t, 0.0), 6) for t in KNOWN_TYPES}
    report["types_vs_baseline"] = {
        "baseline_share": base_types, "share_diff": diffs if known_n else None,
        # Total variation distance between the type mixes: 0 = identical, 1 = disjoint. Descriptive only.
        "tvd": round(0.5 * sum(abs(v) for v in diffs.values()), 6) if known_n else None}
    ranges = {}
    for f, counts in stats["bins"].items():
        rng = baseline.get("bin_range", {}).get(f)
        p999 = baseline.get("bin_p999", {}).get(f)
        observed = sum(c for b, c in counts.items() if b not in ("null", OTHER))
        out = sum(c for b, c in counts.items() if b not in ("null", OTHER) and rng and not rng[0] <= int(b) <= rng[1])
        above = sum(c for b, c in counts.items() if b not in ("null", OTHER) and p999 is not None and int(b) > p999)
        ranges[f] = {"observed": observed, "outside_training_range": {"n": out, "rate": _rate(out, observed)},
                     "above_training_p99_9": {"n": above, "rate": _rate(above, observed)},
                     "missing_rate": _rate(counts.get("null", 0), observed + counts.get("null", 0)),
                     "baseline_missing_rate": baseline["features"].get(f, {}).get("missing_rate")}
    report["feature_ranges"] = ranges
    amount = ranges.get("amount")
    report["quality"]["unexpected_amount"] = amount["above_training_p99_9"] if amount else None
    for f, r in report["flags"].items():
        r["baseline_rate_1"] = baseline.get("flags", {}).get(f)
    ref = baseline.get("reference_scores")
    if ref:
        total = sum(ref["score_hist"])
        report["score"]["baseline_hist_share"] = [round(c / total, 6) for c in ref["score_hist"]] if total else None
        report["baseline_reference"] = {k: ref.get(k) for k in ("split", "n", "mean", "decision_rates",
                                                                "would_decline_rate", "policy_version")}
    report["baseline_missing_by_feature"] = {f: v.get("missing_rate") for f, v in baseline["features"].items()}
    return report


def series_parts(key):
    """"model|policy" (production) or "model|policy|source" -> (model, policy, source)."""
    parts = key.split("|")
    return parts[0], parts[1] if len(parts) > 1 else OTHER, parts[2] if len(parts) > 2 else "production"


def load_state(path):
    """Read-only view of the consumer's saved state (the dashboard never writes it)."""
    agg = MonitoringAggregator(None)
    try:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        agg.created_at, agg.saved_at = s["created_at"], s.get("saved_at")
        agg.minutes = {int(k): v for k, v in s["minutes"].items()}
        agg.coarse = {int(k): v for k, v in s["coarse"].items()}
        agg.cumulative, agg.offsets, agg.counts = s["cumulative"], s["offsets"], {**agg.counts, **s["counts"]}
        agg.last_event_at, agg.last_observed_at = s.get("last_event_at"), s.get("last_observed_at")
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return agg


def series_order(key, n):
    """Production series first (the business population), then by size: what dashboards show first."""
    return (series_parts(key)[2] != "production", -n)


def monitoring_report(agg, window_minutes, baseline_dir, now=None, min_events=MIN_EVENTS, stale_after_s=30,
                      lag=None, models_dir=None):
    """Everything the dashboard's ML section shows for one window.

    The consumer saves the state only when it commits new events, so an old state is normal when
    there is no traffic. It is stale when it is older than `stale_after_s` while events are
    waiting in Kafka (lag > 0); unknown (None) when the lag is unknown."""
    now = now if now is not None else time.time()
    if agg is None:
        return {"available": False, "reason": "no monitoring state yet (consumer not started with monitoring, "
                                              "or no event processed since)"}
    from src.drift_detection import select_baseline   # same validated choice as the drift monitor
    merged, volume = agg.window(window_minutes)
    series, chosen = {}, {}
    for key, stats in sorted(merged.items(), key=lambda kv: series_order(kv[0], kv[1]["n"])):
        model_version, policy_version, source = series_parts(key)
        if model_version not in chosen:
            chosen[model_version] = select_baseline(baseline_dir, model_version, models_dir) \
                if model_version != OTHER else (None, ["aggregated overflow series"], [])
        baseline, problems, skipped = chosen[model_version]
        rep = series_report(stats, baseline, min_events)
        rep["model_version"], rep["policy_version"], rep["traffic_source"] = model_version, policy_version, source
        rep["baseline_version"] = baseline["baseline_version"] if baseline else None
        rep["baseline_status"] = "ok" if baseline else "; ".join(problems)
        rep["baselines_skipped"] = skipped
        rep["volume_per_bucket"] = [[_iso_minute(m), c] for m, c in volume.get(key, [])]
        series[key] = rep

    def age(iso):
        return round(now - _parse_time(iso), 1) if iso else None

    saved_age = age(agg.saved_at)
    return {
        "available": True, "window_minutes": window_minutes,
        "resolution": "1 minute" if window_minutes <= MINUTE_HORIZON else "15 minutes beyond the last hour",
        "series": series,
        "cumulative": {k: {"n": v["n"], "would_decline": v["would_decline"],
                           "decisions": {d: v["decision"].get(d, 0) for d in DECISIONS}}
                       for k, v in agg.cumulative.items()},
        "cumulative_since": agg.created_at,
        "freshness": {"state_saved_at": agg.saved_at, "state_age_s": saved_age,
                      "stale": (None if lag is None else bool(lag > 0 and (saved_age is None
                                                                            or saved_age > stale_after_s))),
                      "kafka_lag": lag,
                      "newest_prediction_at": agg.last_event_at, "newest_prediction_age_s": age(agg.last_event_at),
                      "last_processed_at": agg.last_observed_at},
        "counters": dict(agg.counts),
    }
