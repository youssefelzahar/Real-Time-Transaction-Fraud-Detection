"""Data drift monitor: a separate, read-only process next to the event consumer (DATA_DRIFT.md).

    python -m monitoring.drift_monitor            # every DRIFT_INTERVAL_S (60 s) until SIGINT/SIGTERM
    python -m monitoring.drift_monitor --once     # one calculation, print the summary, exit

Each run reads the consumer's monitoring state (ml_monitoring_state.json, never written here),
the latest valid baseline per model version and config/drift_config.json, computes drift per
window with src/drift_detection.py and writes, in EVENT_STORE_DIR:
  drift_report.json        latest report (atomic replace)
  drift_alerts.jsonl       alert records: fired / escalated / resolved (append-only)
  drift_alert_state.json   active alerts + drift-continuity marks (atomic), survives restarts
Nothing here touches /predict, Kafka offsets or the event store. Kafka is only read for the lag.
Config (env): EVENT_STORE_DIR, MONITORING_BASELINE_DIR, DRIFT_CONFIG, DRIFT_INTERVAL_S,
KAFKA_BOOTSTRAP_SERVERS (lag; unset = lag unknown), KAFKA_TOPIC_PREDICTIONS, KAFKA_CONSUMER_GROUP.
"""
import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.drift_detection import (DRIFT, ERROR, NO_DRIFT, SEVERITY_RANK, evaluate_window, load_config,
                                 select_baseline)
from src.ml_monitoring import COARSE_MINUTES, MINUTE_HORIZON, OTHER, load_state, merge, new_stats, series_parts

ROOT = Path(__file__).resolve().parent.parent
STORE_DIR = Path(os.path.expanduser(os.getenv("EVENT_STORE_DIR", "~/.local/share/nilepay-events")))
BASELINE_DIR = Path(os.getenv("MONITORING_BASELINE_DIR") or ROOT / "monitoring" / "baselines")
CONFIG_PATH = Path(os.getenv("DRIFT_CONFIG") or ROOT / "config" / "drift_config.json")
MODELS_DIR = ROOT / "models"
CONSUMER_PID = ROOT / "deploy" / "run" / "consumer.pid"
REPORT_FILE, ALERTS_FILE, ALERT_STATE_FILE = "drift_report.json", "drift_alerts.jsonl", "drift_alert_state.json"
ACK_REQUESTS_FILE = "drift_ack_requests.jsonl"   # appended by --ack; applied by the monitor (single writer)
DRIFT_PID = ROOT / "deploy" / "run" / "drift.pid"
REPORT_VERSION = 1

log = logging.getLogger("nilepay.drift")


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def parse_iso(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def atomic_write_json(path, obj, durable=True):
    """tmp + rename, so readers never see a half-written file. durable=False skips the fsync: for
    files that are regenerated every run (the report), an fsync would only add disk flushes on
    the shared disk (ext4 can flush other files' dirty data with it)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
        if durable:
            f.flush()
            os.fsync(f.fileno())
    tmp.replace(path)


# --- inputs ---------------------------------------------------------------------------------------

class LagReader:
    """Consumer-group lag from committed offsets and watermarks. One long-lived client (it never
    subscribes, so it never joins the group); recreated after an error."""

    def __init__(self, timeout=2.0):
        self.timeout, self.client = timeout, None

    def __call__(self):
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap:
            return None
        topic = os.getenv("KAFKA_TOPIC_PREDICTIONS", "nilepay.prediction-events")
        try:
            from confluent_kafka import Consumer, TopicPartition
            if self.client is None:
                self.client = Consumer({"bootstrap.servers": bootstrap, "enable.auto.commit": False, "log_level": 2,
                                        "group.id": os.getenv("KAFKA_CONSUMER_GROUP", "nilepay-prediction-monitor")})
            c, t = self.client, self.timeout
            parts = [TopicPartition(topic, p) for p in c.list_topics(topic, timeout=t).topics[topic].partitions]
            lag = 0
            for tp in c.committed(parts, timeout=t):
                low, high = c.get_watermark_offsets(tp, timeout=t)
                lag += max(high - (tp.offset if tp.offset >= 0 else low), 0)
            return lag
        except Exception:   # broker down / starting: lag unknown, not zero
            self.close()
            return None

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


def consumer_alive(pid_file=CONSUMER_PID):
    try:
        pid = int(Path(pid_file).read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def merge_by_model(series):
    """{"model|policy[|source]": stats} -> {series_id: (model, source, stats)}. Policies are merged
    (input drift does not depend on the policy); traffic sources are kept apart, so load tests and
    demos never mix with production. series_id is the model for production, "model|source" otherwise."""
    out = {}
    for key, stats in series.items():
        model, _, source = series_parts(key)
        sid = model if source == "production" else f"{model}|{source}"
        merge(out.setdefault(sid, (model, source, new_stats()))[2], stats)
    return out


# --- report ---------------------------------------------------------------------------------------

def compute(agg, cfg, *, baseline_dir, models_dir, now, lag=None, alive=None):
    """Drift report for every configured window and model version in the consumer's state."""
    t0 = time.perf_counter()
    agg.clock = lambda: now
    a = cfg["alerts"]
    state_age = round(now - parse_iso(agg.saved_at), 1) if agg.saved_at else None
    newest_age = round(now - parse_iso(agg.last_event_at), 1) if agg.last_event_at else None
    stale = None
    if alive is False:
        stale = True
    elif lag is not None:
        stale = bool(lag > 0 and (state_age is None or state_age > a["stale_after_s"]))
    report = {"report_version": REPORT_VERSION, "status": "ok", "computed_at": iso(now),
              "freshness": {"state_saved_at": agg.saved_at, "state_age_s": state_age,
                            "newest_prediction_at": agg.last_event_at, "newest_prediction_age_s": newest_age,
                            "kafka_lag": lag, "consumer_alive": alive, "stale": stale},
              "config": {k: cfg[k] for k in ("windows", "min_samples", "psi", "psi_overrides", "alpha",
                                             "missing_rate_abs", "unknown_rate", "critical_features", "alerts")},
              "windows": {}}
    baselines = {}
    now_m = int(now // 60)
    for label, minutes in cfg["windows"].items():
        merged, volume = agg.window(int(minutes))
        models = merge_by_model(merged)
        nominal_m = now_m - int(minutes) + 1
        # Beyond the last hour data is kept in whole 15-minute slots, so a long window can include up
        # to 14 minutes before its nominal start. Report where the data really starts.
        starts = [m for pts in volume.values() for m, _ in pts]
        effective_m = min([nominal_m, *starts]) if starts else nominal_m
        by_source = {}
        for _, source, stats in models.values():
            by_source[source] = by_source.get(source, 0) + stats["n"]
        win = {"minutes": int(minutes), "start": iso(nominal_m * 60), "effective_start": iso(effective_m * 60),
               "end": iso(now),
               "resolution": "1 minute" if minutes <= MINUTE_HORIZON else
               f"1 minute for the last hour, {COARSE_MINUTES} minutes beyond", "traffic_by_source": by_source,
               "models": {}}
        # Production first (the business population), then by size: dashboards open on index 0.
        for sid, (model, source, stats) in sorted(models.items(),
                                                  key=lambda kv: (kv[1][1] != "production", -kv[1][2]["n"])):
            if model == OTHER:
                continue
            if model not in baselines:
                baselines[model] = select_baseline(baseline_dir, model, models_dir)
            baseline, problems, skipped = baselines[model]
            try:
                ev = evaluate_window(stats, baseline, cfg, baseline_problems=problems)
            except Exception as e:      # a bug or bad data must show as an error, not as "healthy"
                log.exception("drift evaluation failed for %s/%s", model, label)
                ev = {"features": [], "prediction": None,
                      "summary": {"status": ERROR, "reason": f"{type(e).__name__}: {e}", "production_n": stats["n"]}}
            bv = baseline["baseline_version"] if baseline else None
            for r in ev["features"]:
                r.update({"window": label, "window_minutes": int(minutes), "window_start": win["start"],
                          "window_end": win["end"], "model_version": model, "traffic_source": source,
                          "baseline_version": bv, "detected_at": iso(now)})
            win["models"][sid] = {
                "model_version": model, "traffic_source": source, "baseline_version": bv, "baseline_schema": baseline.get("baseline_schema", 1) if baseline else None,
                "feature_pipeline": baseline.get("feature_pipeline") if baseline else None,
                "baseline_problems": problems, "baselines_skipped": skipped, **ev}
        report["windows"][label] = win
    report["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report


# --- alerts ---------------------------------------------------------------------------------------

def _short_long(cfg):
    w = sorted(cfg["windows"].items(), key=lambda kv: kv[1])
    return w[0][0], w[-1][0]


def conditions(report, cfg, drift_since, now):
    """Alert conditions from a report: [{key, rule, state: active|clear|unknown, severity, details}].
    Also updates `drift_since` (continuity marks for sustained drift)."""
    a, out = cfg["alerts"], []
    min_rank = SEVERITY_RANK[a["min_severity"]]
    short, long_ = _short_long(cfg)
    by_key = {}
    alert_sources = set(a.get("traffic_sources", ["production"]))
    for label, win in report["windows"].items():
        for model, m in win["models"].items():
            if m.get("traffic_source", "production") not in alert_sources:
                continue                       # evaluated and shown, never alerted (e.g. load tests)
            for r in m["features"]:
                by_key[(label, model, r["feature"])] = r
                if r["status"] == DRIFT and SEVERITY_RANK[r["severity"]] >= min_rank:
                    state = "active"
                elif r["status"] in (NO_DRIFT, DRIFT):
                    state = "clear"
                else:
                    state = "unknown"          # insufficient data or error never resolves an alert
                out.append({"key": f"feature_drift|{model}|{label}|{r['feature']}", "rule": "feature_drift",
                            "state": state, "severity": r["severity"], "details": _details(r)})
            err = m["summary"].get("status") == ERROR
            out.append({"key": f"monitoring_error|{model}|{label}", "rule": "monitoring_error",
                        "state": "active" if err else "clear", "severity": "substantial",
                        "details": {"model_version": model, "window": label,
                                    "message": m["summary"].get("reason") or "feature-level monitoring errors"}})
    # Continuity: a critical feature is "drifting since" only while EVERY short-window evaluation
    # says drift_detected. Insufficient data, an error, or no data at all break the run.
    drifting_now = {f"{model}|{feature}" for (label, model, feature), r in by_key.items()
                    if label == short and feature in cfg["critical_features"] and r["status"] == DRIFT}
    for mark in list(drift_since):
        if mark not in drifting_now:
            del drift_since[mark]
    for mark in drifting_now:
        drift_since.setdefault(mark, now)
    for (label, model, feature), r in by_key.items():
        if label != short or feature not in cfg["critical_features"]:
            continue
        mark = f"{model}|{feature}"
        lr = by_key.get((long_, model, feature))
        since = drift_since.get(mark)
        if r["status"] == NO_DRIFT or (lr and lr["status"] == NO_DRIFT):
            state = "clear"
        elif since is not None and lr and lr["status"] == DRIFT and now - since >= a["sustained_minutes"] * 60:
            state = "active"
        else:
            state = "unknown"
        out.append({"key": f"sustained_drift|{model}|{feature}", "rule": "sustained_drift", "state": state,
                    "severity": "substantial",
                    "details": {**_details(r), "drifting_since": iso(since) if since else None,
                                "long_window": long_, "long_window_psi": lr["psi"] if lr else None}})
    f = report["freshness"]
    if f["stale"] is not None:
        out.append({"key": "monitoring_stale", "rule": "monitoring_stale", "state": "active" if f["stale"] else "clear",
                    "severity": "substantial",
                    "details": {"state_age_s": f["state_age_s"], "kafka_lag": f["kafka_lag"],
                                "consumer_alive": f["consumer_alive"], "threshold_s": a["stale_after_s"],
                                "message": "consumer monitoring state is not updating"}})
    low = a.get("low_volume_min_per_hour")
    hour = next((lbl for lbl, mins in cfg["windows"].items() if int(mins) == 60), None)
    if low is not None and hour is not None:
        n = sum(m["summary"].get("production_n", 0) for m in report["windows"][hour]["models"].values()
                if m.get("traffic_source", "production") in alert_sources)
        out.append({"key": f"low_volume|{hour}", "rule": "low_volume", "state": "active" if n < low else "clear",
                    "severity": "moderate", "details": {"window": hour, "sample_count": n, "threshold": low,
                                                        "message": "fewer predictions than expected"}})
    return out


def expiry_rule(cfg):
    """Seconds without evidence after which an alert expires, derived from its key (so alerts
    created before this rule existed expire too). None = never."""
    if not cfg["alerts"].get("expire_without_data", True):
        return None
    windows = {label: int(m) * 60 for label, m in cfg["windows"].items()}
    long_s = max(windows.values())

    def rule(key):
        parts = key.split("|")
        if parts[0] == "feature_drift" and len(parts) >= 4:
            return windows.get(parts[2])
        if parts[0] == "monitoring_error" and len(parts) >= 3:
            return windows.get(parts[2])
        if parts[0] == "sustained_drift":
            return long_s
        return None              # monitoring_stale, low_volume: they always have evidence
    return rule


def _details(r):
    return {"feature": r["feature"], "model_version": r["model_version"], "baseline_version": r["baseline_version"],
            "window": r["window"], "window_start": r["window_start"], "window_end": r["window_end"],
            "drift_score": r["psi"], "threshold": r["threshold"], "sample_count": r["production_n"],
            "status": r["status"], "reasons": r["reasons"][-2:]}


class AlertManager:
    """Fires an alert when a condition becomes active or its severity rises, resolves it when the
    condition clears with sufficient data, and stays quiet while nothing changes."""

    def __init__(self, store_dir):
        self.state_path = Path(store_dir) / ALERT_STATE_FILE
        self.log_path = Path(store_dir) / ALERTS_FILE
        self.ack_path = Path(store_dir) / ACK_REQUESTS_FILE
        self.active, self.drift_since, self.ack_offset, self._saved = {}, {}, 0, None
        try:
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.active, self.drift_since = dict(s["active"]), {k: float(v) for k, v in s["drift_since"].items()}
            self.ack_offset = int(s.get("ack_offset", 0))
            self._saved = self._key()
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log.warning("alert state unreadable (%s): starting without active alerts", e)

    def apply_ack_requests(self, now):
        """Operator acknowledgements (--ack): an acknowledged alert stays active but is silent until
        `until`, with who/why recorded. It still resolves normally when the condition clears, and it
        fires again if the condition is still there when the acknowledgement expires."""
        records = []
        try:
            with open(self.ack_path, "rb") as f:
                f.seek(self.ack_offset)
                data = f.read()
        except OSError:
            return records
        end = data.rfind(b"\n") + 1
        for line in data[:end].splitlines():
            try:
                req = json.loads(line)
                pattern, reason = str(req["pattern"]), str(req["reason"])
                until = parse_iso(req["at"]) + float(req.get("hours", 24)) * 3600
            except (ValueError, KeyError, TypeError):
                log.warning("ignoring malformed acknowledgement request")
                continue
            matched = [k for k in self.active if pattern in k]
            if not matched:
                log.warning("acknowledgement %r matched no active alert", pattern)
            for k in matched:
                a = self.active[k]
                a["ack"] = {"by": req.get("by"), "reason": reason, "at": req["at"], "until": iso(until)}
                records.append({"at": iso(now), "event": "acknowledged", "rule": a["rule"], "key": k,
                                "severity": a["severity"], "by": req.get("by"), "reason": reason,
                                "acknowledged_until": iso(until), **a.get("details", {})})
        self.ack_offset += end
        return records

    def update(self, conds, now, expire_after=None):
        """expire_after(key) -> seconds or None: an active alert whose condition has had no evidence
        (no data, insufficient data, or missing from the report) for longer than that closes with
        an `expired_no_data` record (D6). None = never expires (e.g. monitoring_stale, low_volume)."""
        records = self.apply_ack_requests(now)
        evidence = set()
        for c in conds:
            if c["state"] in ("active", "clear"):
                evidence.add(c["key"])
            prev = self.active.get(c["key"])
            rec = {"at": iso(now), "rule": c["rule"], "key": c["key"], "severity": c["severity"], **c["details"]}
            if c["state"] == "active":
                ack = (prev or {}).get("ack")
                if prev is None:
                    rec["event"] = "fired"
                elif SEVERITY_RANK.get(c["severity"], 0) > SEVERITY_RANK.get(prev["severity"], 0):
                    rec["event"] = "escalated"
                elif ack and now >= parse_iso(ack["until"]):
                    rec["event"] = "fired"
                    rec["note"] = f"acknowledgement by {ack.get('by')} expired; condition still present"
                else:
                    prev["last_seen"] = iso(now)
                    prev["details"] = c["details"]
                    continue
                self.active[c["key"]] = {"rule": c["rule"], "severity": c["severity"], "since": (prev or {}).get(
                    "since", iso(now)), "last_seen": iso(now), "details": c["details"]}
                records.append(rec)
            elif c["state"] == "clear" and prev is not None:
                rec["event"] = "resolved"
                rec["active_since"] = prev["since"]
                del self.active[c["key"]]
                records.append(rec)
        for key, a in list(self.active.items()):
            limit = expire_after(key) if expire_after else None
            if key in evidence or limit is None:
                continue
            last = parse_iso(a.get("last_seen") or a["since"])
            if now - last > limit:
                records.append({"at": iso(now), "event": "expired_no_data", "rule": a["rule"], "key": key,
                                "severity": a["severity"], "active_since": a["since"], "last_evidence": iso(last),
                                "message": f"no data for this condition for more than {limit / 60:.0f} min; "
                                           "closed without a verdict", **a.get("details", {})})
                del self.active[key]
        if records:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            for r in records:
                (log.warning if r["event"] in ("fired", "escalated") else log.info)(
                    "drift alert %s: %s %s score=%s threshold=%s window=%s n=%s baseline=%s", r["event"], r["rule"],
                    r.get("feature", ""), r.get("drift_score"), r.get("threshold"), r.get("window"),
                    r.get("sample_count"), r.get("baseline_version"))
        if self._key() != self._saved:        # durable write only when what matters changed
            atomic_write_json(self.state_path, {"active": self.active, "drift_since": self.drift_since,
                                                "ack_offset": self.ack_offset})
            self._saved = self._key()
        return records

    def _key(self):
        """What must survive a restart: active alerts (severity, acknowledgement), drift-continuity
        marks and how far the acknowledgement requests were applied. Refreshed details/last_seen
        are not worth a disk flush."""
        return json.dumps([{k: [v["severity"], v.get("ack")] for k, v in self.active.items()}, self.drift_since,
                           self.ack_offset], sort_keys=True)


def request_ack(store_dir, pattern, reason, hours=24.0, by=None, now=None):
    """Queue an acknowledgement for the monitor (it is the only writer of the alert state)."""
    req = {"at": iso(time.time() if now is None else now), "pattern": pattern, "reason": reason, "hours": hours,
           "by": by or os.getenv("USER")}
    with open(Path(store_dir) / ACK_REQUESTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(req) + "\n")
    return req


# --- run ------------------------------------------------------------------------------------------

def run_once(store_dir=STORE_DIR, *, baseline_dir=BASELINE_DIR, models_dir=MODELS_DIR, config_path=CONFIG_PATH,
             now=None, lag=None, alive=None, alerts=None):
    """One calculation. Writes the report and alerts; returns the report. Never raises."""
    store_dir = Path(store_dir)
    now = time.time() if now is None else now
    report_path = store_dir / REPORT_FILE
    previous = {}
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    alerts = alerts or AlertManager(store_dir)
    try:
        cfg = load_config(config_path)
        agg = load_state(store_dir / "ml_monitoring_state.json")
        if agg is None:
            report = {"report_version": REPORT_VERSION, "status": "no_state", "computed_at": iso(now),
                      "reason": "no consumer monitoring state yet (consumer not running or no events processed)",
                      "windows": {}, "freshness": {"consumer_alive": alive, "kafka_lag": lag}}
            conds = [] if alive is not False else [
                {"key": "monitoring_stale", "rule": "monitoring_stale", "state": "active", "severity": "substantial",
                 "details": {"consumer_alive": False, "message": "consumer not running"}}]
        else:
            report = compute(agg, cfg, baseline_dir=baseline_dir, models_dir=models_dir, now=now, lag=lag,
                             alive=alive)
            conds = conditions(report, cfg, alerts.drift_since, now)
        report["last_successful_at"] = report["computed_at"]
        alerts.update(conds, now, expire_after=expiry_rule(cfg))
    except Exception as e:
        log.exception("drift calculation failed")
        report = {"report_version": REPORT_VERSION, "status": ERROR, "computed_at": iso(now),
                  "reason": f"{type(e).__name__}: {e}", "windows": {},
                  "last_successful_at": previous.get("last_successful_at")}
    report["active_alerts"] = [{"key": k, **v} for k, v in sorted(alerts.active.items())]
    try:
        atomic_write_json(report_path, report, durable=False)
    except OSError as e:
        log.warning("drift report not written: %s", e)
    return report


def summary_lines(report):
    yield f"drift report {report['computed_at']}  status={report['status']}  duration={report.get('duration_ms')} ms"
    if report.get("reason"):
        yield f"  {report['reason']}"
    for label, win in report.get("windows", {}).items():
        for model, m in win["models"].items():
            s = m["summary"]
            yield (f"  [{label}] {model} baseline={m['baseline_version']} n={s.get('production_n')} "
                   f"status={s['status']} drifting={s.get('features_drifting')}/{s.get('features_monitored')}")
            for r in m["features"]:
                yield f"      {r['feature']:<20} {r['status']:<22} psi={r['psi']}  n={r['production_n']}"
    for a in report.get("active_alerts", []):
        yield f"  ALERT {a['rule']} {a['key']} since {a['since']}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=float(os.getenv("DRIFT_INTERVAL_S", "60")))
    ap.add_argument("--ack", metavar="KEY_SUBSTRING",
                    help="acknowledge active alerts whose key contains this text (needs --reason)")
    ap.add_argument("--reason", help="why the alerts are acknowledged (recorded)")
    ap.add_argument("--hours", type=float, default=24.0, help="acknowledgement duration (default 24)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if args.ack:
        if not args.reason:
            ap.error("--ack needs --reason")
        req = request_ack(STORE_DIR, args.ack, args.reason, args.hours)
        if consumer_alive(DRIFT_PID):
            print(f"queued: the running drift monitor applies it within {args.interval:.0f} s: {req}")
        else:                       # no monitor running: apply now (it is the only writer otherwise)
            m = AlertManager(STORE_DIR)
            m.update([], time.time())
            print(f"applied: {sum('ack' in a for a in m.active.values())} acknowledged active alert(s)")
        return 0
    alerts = AlertManager(STORE_DIR)
    kafka_lag = LagReader()
    if args.once:
        report = run_once(lag=kafka_lag(), alive=consumer_alive(), alerts=alerts)
        kafka_lag.close()
        print("\n".join(summary_lines(report)))
        return 0 if report["status"] != ERROR else 1
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    log.info("drift monitor: every %.0f s, store %s, config %s", args.interval, STORE_DIR, CONFIG_PATH)
    while not stop.is_set():
        report = run_once(lag=kafka_lag(), alive=consumer_alive(), alerts=alerts)
        log.info("drift %s in %s ms, %d active alerts", report["status"], report.get("duration_ms"),
                 len(report.get("active_alerts", [])))
        stop.wait(args.interval)
    kafka_lag.close()
    log.info("drift monitor stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
