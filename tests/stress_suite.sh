#!/usr/bin/env bash
# Automated high-traffic test through Nginx: capacity ramp, sustained maximum throughput,
# and availability while an instance and the Kafka broker are stopped and restarted under load.
# Watch it live: python -m monitoring.live_dashboard  ->  http://127.0.0.1:8090
#
#   tests/stress_suite.sh                  # all phases (~5 min)
#   PHASES="ramp" tests/stress_suite.sh    # only some phases: ramp sustain avail
#
# Needs the stack running (deploy/start.sh), and Kafka + consumer for the broker part of
# "avail" (skipped automatically when the instances run without Kafka).
# Config (env): RAMP="25 50 100 200 400 800" (wrk connections per stage), STAGE_S=15,
#   SUSTAIN_S=60, AVAIL_S=75, CONC (fixed concurrency for sustain/avail; default = best ramp stage),
#   STOP_INSTANCE=3, WRK_CPUS=10-15 (taskset for the load generator; empty = no pinning),
#   P95_TARGET_MS=100.
# Report: deploy/run/stress-<timestamp>.txt
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN="$ROOT/deploy/run"
LOG="$RUN/access.log"
PY="${PYTHON:-$ROOT/venv/bin/python}"
PHASES="${PHASES:-ramp sustain avail}"
RAMP="${RAMP:-25 50 100 200 400 800}"
STAGE_S="${STAGE_S:-15}"
SUSTAIN_S="${SUSTAIN_S:-60}"
AVAIL_S="${AVAIL_S:-75}"
STOP_INSTANCE="${STOP_INSTANCE:-3}"
WRK_CPUS="${WRK_CPUS-10-15}"
P95_TARGET_MS="${P95_TARGET_MS:-100}"
URL="http://127.0.0.1:8080"
REPORT="$RUN/stress-$(date +%Y%m%d-%H%M%S).txt"

say() { echo "$@" | tee -a "$REPORT"; }
pin() { [[ -n "$WRK_CPUS" ]] && echo "taskset -c $WRK_CPUS"; }
offset() { stat -c %s "$LOG"; }
field() { sed -n "s|.*$1=\([0-9.]*\).*|\1|p" <<< "$2"; }

# wrk run -> RESULT line (req/s, p50, p95, p99, max, requests, non_200, socket_errors)
load() {  # <connections> <seconds>
    local c="$1" threads=8
    (( c < threads )) && threads=$c
    $(pin) wrk -t"$threads" -c"$c" -d"$2"s --timeout 10s --latency -s tests/wrk_predict.lua "$URL" \
        | grep RESULT | sed 's/RESULT //'
}

# Nginx log since a byte offset: status counts, and per-second throughput min/median/max
statuses() {  # <offset>
    tail -c +"$(( $1 + 1 ))" "$LOG" | awk '$5 == "/predict" {s[$7]++; n++} END {for (k in s) printf "%s=%d ", k, s[k]; printf "| total=%d\n", n}'
}
per_second() {  # <offset>
    tail -c +"$(( $1 + 1 ))" "$LOG" | awk '$5 == "/predict" {c[$2]++} END {for (k in c) print c[k]}' | sort -n \
        | awk '{v[NR]=$1} END {if (NR > 2) printf "per-second req/s (full seconds): min %d, median %d, max %d over %d s\n", v[2], v[int((NR+1)/2)], v[NR-1], NR-2}'
}
error_seconds() {  # <offset>: seconds that had non-200 responses
    tail -c +"$(( $1 + 1 ))" "$LOG" | awk '$5 == "/predict" && $7 != 200 && $7 != 499 {split($2, t, ":"); e[t[2]":"t[3]":"t[4]" "$7]++} END {for (k in e) print "    " k " x" e[k]}' | sort
}

events() {  # summed producer counters; EXCLUDE_INSTANCE=n leaves out a restarted instance (its counters reset)
    EXCLUDE_INSTANCE="${EXCLUDE_INSTANCE:-}" "$PY" - <<'EOF'
import httpx, os, re
ports = re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", open("deploy/nginx/nginx.conf").read(), flags=re.M)
skip = os.environ.get("EXCLUDE_INSTANCE")
if skip:
    ports = [p for i, p in enumerate(ports, 1) if str(i) != skip]
tot = {}
for p in ports:
    try:
        m = httpx.get(f"http://127.0.0.1:{p}/metrics/events", timeout=1).json()
    except Exception:
        continue
    if not m.get("kafka_enabled"):
        continue
    for k in ("enqueued", "delivered", "buffer_full_drops", "in_flight"):
        tot[k] = tot.get(k, 0) + m.get(k, 0)
    tot["failed"] = tot.get("failed", 0) + sum(m.get("failed", {}).values())
print(" ".join(f"{k}={v}" for k, v in tot.items()) if tot else "kafka_disabled")
EOF
}

preflight() {
    curl -sf "$URL/health" > /dev/null || { echo "Nginx not answering on :8080. Start the stack: deploy/start.sh" >&2; exit 1; }
    command -v wrk > /dev/null || { echo "wrk not installed" >&2; exit 1; }
    local up=0 n=0
    for port in $(grep -oP '^\s*server\s+127\.0\.0\.1:\K\d+' deploy/nginx/nginx.conf); do
        n=$((n + 1)); curl -sf "http://127.0.0.1:$port/health" > /dev/null && up=$((up + 1))
    done
    say "NilePay stress suite  $(date '+%F %T')"
    say "instances healthy: $up/$n | model: $(curl -s "$URL/health" | sed 's/.*"model_version":"\([^"]*\)".*/\1/')" \
        "| AC online: $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1) | power profile: $(powerprofilesctl get 2>/dev/null || echo ?)"
    say "load generator: wrk -t8 on CPUs ${WRK_CPUS:-any}, payloads tests/wrk_predict.lua, through Nginx $URL"
    say "events (before): $(events)"
    $(pin) wrk -t4 -c50 -d3s -s tests/wrk_predict.lua "$URL" > /dev/null   # warm-up
}

BEST_C=""
ramp() {
    say ""
    say "=== 1. Capacity ramp (${STAGE_S}s per stage) ==="
    printf "%-6s %9s %9s %8s %8s %8s %8s %9s  %s\n" conns req/s "ok/s" p50 p95 p99 max errors "nginx statuses" | tee -a "$REPORT"
    local best_rps=0 peak_rps=0 peak_c=""
    for c in $RAMP; do
        local off r; off=$(offset)
        r=$(load "$c" "$STAGE_S"); sleep 1.2      # Nginx flushes its log buffer every second
        local rps p50 p95 p99 mx non200 sock
        rps=$(field req/s "$r"); p50=$(field p50 "$r"); p95=$(field p95 "$r"); p99=$(field p99 "$r")
        mx=$(field max "$r"); non200=$(field non_200 "$r"); sock=$(field socket_errors "$r")
        local st ok; st=$(statuses "$off")
        ok=$(( $(sed -n 's/.*\b200=\([0-9]*\).*/\1/p' <<< "$st" | head -1)0 / 10 / STAGE_S ))   # successful req/s
        printf "%-6s %9s %9s %7sms %7sms %7sms %7sms %9s  %s\n" "$c" "$rps" "$ok" "$p50" "$p95" "$p99" "$mx" "$((non200 + sock))" "$st" | tee -a "$REPORT"
        if (( ok > peak_rps )); then peak_rps=$ok; peak_c=$c; fi
        if [[ "$((non200 + sock))" == 0 ]] && awk -v p="$p95" -v t="$P95_TARGET_MS" -v a="$ok" -v b="$best_rps" 'BEGIN{exit !(p < t && a > b)}'; then
            best_rps=$ok; BEST_C=$c
        fi
    done
    say "peak successful throughput: $peak_rps req/s at $peak_c connections"
    if [[ -n "$BEST_C" ]]; then
        say "capacity within target (p95 < ${P95_TARGET_MS} ms, 0 errors): $best_rps req/s at $BEST_C connections"
    else
        say "no stage met p95 < ${P95_TARGET_MS} ms with 0 errors"
    fi
    say "(503 = Nginx load shedding: all instances at max_conns=64; it answers fast instead of queueing)"
}

sustain() {
    local c="${CONC:-${BEST_C:-100}}" off r
    say ""
    say "=== 2. Sustained throughput: $c connections for ${SUSTAIN_S}s ==="
    off=$(offset)
    r=$(load "$c" "$SUSTAIN_S"); sleep 1.2
    say "client: $r"
    say "nginx:  $(statuses "$off")"
    say "        $(per_second "$off")"
}

restart_env() {  # environment to restart instance N exactly as it was started
    local pid="$1" vars="" v
    for v in MODEL_DIR KAFKA_BOOTSTRAP_SERVERS KAFKA_TOPIC_PREDICTIONS DECISION_POLICY ACCOUNT_SNAPSHOT LOG_LEVEL WORKERS_PER_INSTANCE; do
        local val; val=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n "s/^$v=//p")
        [[ -n "$val" ]] && vars+="$v=$val "
    done
    echo "$vars CPUS_INSTANCES=$(taskset -cp "$pid" 2>/dev/null | awk '{print $NF}')"
}

avail() {
    local c="${CONC:-${BEST_C:-100}}" off before r pidf="$RUN/instance_$STOP_INSTANCE.pid"
    [[ -f "$pidf" ]] || { say "instance $STOP_INSTANCE not running, skipping availability test"; return; }
    local env; env=$(restart_env "$(cat "$pidf")")
    local kafka=0
    [[ "$(events)" != kafka_disabled ]] && [[ -f "$HOME/.local/share/nilepay-kafka/kafka.pid" ]] && kafka=1
    say ""
    say "=== 3. Availability under load: $c connections for ${AVAIL_S}s ==="
    say "timeline: +10s stop instance $STOP_INSTANCE, +25s start it again$([[ $kafka == 1 ]] && echo ", +40s stop Kafka broker, +50s start it again")"
    before=$(EXCLUDE_INSTANCE=$STOP_INSTANCE events)
    off=$(offset)
    (
        T0=$(date +%s.%N)
        at() { local d; d=$(echo "$1 - ($(date +%s.%N) - $T0)" | bc); [[ "$d" != -* ]] && sleep "$d"; }
        stamp() { printf "  [+%4.1fs] %s\n" "$(echo "$(date +%s.%N) - $T0" | bc)" "$1"; }
        at 10; deploy/stop.sh "$STOP_INSTANCE" > /dev/null; stamp "instance $STOP_INSTANCE stopped"
        at 25; env $env deploy/start.sh "$STOP_INSTANCE" > /dev/null; stamp "instance $STOP_INSTANCE started and healthy"
        if [[ $kafka == 1 ]]; then
            at 40; deploy/kafka.sh stop > /dev/null; stamp "Kafka broker stopped"
            at 50; KAFKA_CPUS="${KAFKA_CPUS:-8-9}" deploy/kafka.sh start > /dev/null; stamp "Kafka broker ready"
        fi
    ) | tee -a "$REPORT" &
    local ctl=$!
    r=$(load "$c" "$AVAIL_S")
    wait "$ctl"; sleep 1.2
    say "client: $r"
    local st; st=$(statuses "$off")
    say "nginx:  $st"
    local ok total closed
    ok=$(sed -n 's/.*\b200=\([0-9]*\).*/\1/p' <<< "$st"); total=$(sed -n 's/.*total=\([0-9]*\).*/\1/p' <<< "$st")
    closed=$(sed -n 's/.*\b499=\([0-9]*\).*/\1/p' <<< "$st")   # 499 = the client closed the connection (wrk stopping), not a server failure
    [[ -n "$total" && "$total" -gt 0 ]] && say "availability (200 / all /predict answered, 499 excluded): $(awk -v a="${ok:-0}" -v b="$total" -v c="${closed:-0}" 'BEGIN{printf "%.4f%% (%d failed of %d)", 100*a/(b-c), b-c-a, b-c}')"
    say "        $(per_second "$off")"
    local errs; errs=$(error_seconds "$off")
    if [[ -n "$errs" ]]; then
        say "  seconds with non-200 responses (time status xcount):"; say "$errs"
        say "  (502/503 right when an instance stops = requests already sent to it; Nginx does not retry a POST that may have been processed)"
    else
        say "  no non-200 responses"
    fi
    if [[ $kafka == 1 ]]; then
        sleep 3
        say "Kafka events on the 5 instances that were not restarted (instance $STOP_INSTANCE's counters reset on restart):"
        say "  before: $before"
        say "  after:  $(EXCLUDE_INSTANCE=$STOP_INSTANCE events)"
        say "  (events not delivered within KAFKA_MESSAGE_TIMEOUT_MS while the broker was down are counted as failed, not hidden)"
    fi
}

preflight
for p in $PHASES; do "$p"; done
say ""
say "report saved: $REPORT"
