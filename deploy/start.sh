#!/usr/bin/env bash
# Start the FastAPI instances and Nginx (:8080).
# The instance list comes from the upstream block in deploy/nginx/nginx.conf
# (instance N = Nth `server` line), so Nginx and the started instances always match.
#   deploy/start.sh            # every instance, then Nginx
#   deploy/start.sh 3          # only instance 3 (e.g. restart after a failure)
#   deploy/start.sh nginx
#   deploy/start.sh consumer   # Kafka event consumer (needs KAFKA_BOOTSTRAP_SERVERS; not part of `all`)
#   deploy/start.sh drift      # data drift monitor (reads the consumer's state; not part of `all`)
# Config (env): MODEL_DIR (default models/e7_v1), WORKERS_PER_INSTANCE (default 1),
#   ACCOUNT_SNAPSHOT, LOG_LEVEL (default warning; `info` logs every request),
#   TIMING_LOG (default 0; 1 prints timings), DECISION_POLICY (default config/decision_policy.json),
#   CPUS_INSTANCES (taskset CPU list for all instances, e.g. 0-7; or one list per
#   instance separated by spaces, e.g. "0 1 2 3 4 5"), CPUS_NGINX, CPUS_CONSUMER, CPUS_DRIFT. Empty = no pinning.
#   KAFKA_BOOTSTRAP_SERVERS (unset = no event publishing) and the other KAFKA_* settings
#   in KAFKA_EVENTS.md are passed through to the instances and the consumer.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/deploy/run"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
export MODEL_DIR="${MODEL_DIR:-models/e7_v1}"
export TIMING_LOG="${TIMING_LOG:-0}"
export EVENT_LEDGER_DIR="${EVENT_LEDGER_DIR:-$RUN}"   # producer ledger: final counters per instance process
WORKERS="${WORKERS_PER_INSTANCE:-1}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
read -r -a CPU_LISTS <<< "${CPUS_INSTANCES:-}"
mkdir -p "$RUN/tmp"
cd "$ROOT"

# Ports of the upstream servers, in order.
mapfile -t PORTS < <(awk '/^[[:space:]]*server[[:space:]]+127\.0\.0\.1:[0-9]+/ {split($2, a, ":"); print a[2]}' \
    deploy/nginx/nginx.conf)
[[ ${#PORTS[@]} -gt 0 ]] || { echo "no upstream servers found in deploy/nginx/nginx.conf" >&2; exit 1; }

pin_prefix() {  # prints a taskset prefix for <cpu-list>, or nothing
    if [[ -n "$1" ]] && command -v taskset >/dev/null; then echo "taskset -c $1"; fi
}

cpus_for() {  # cpus_for <instance number>
    if [[ ${#CPU_LISTS[@]} -eq 1 ]]; then echo "${CPU_LISTS[0]}"
    elif [[ ${#CPU_LISTS[@]} -ge $1 ]]; then echo "${CPU_LISTS[$(($1 - 1))]}"
    fi
}

wait_for() {  # wait_for <url> <name>
    for _ in $(seq 100); do
        curl -fs "$1" >/dev/null && { echo "  $2 ready"; return 0; }
        sleep 0.2
    done
    echo "  $2 did not become healthy: $1" >&2; return 1
}

running() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

launch_instance() {  # launch_instance <n>  (does not wait)
    local n="$1" port="${PORTS[$(($1 - 1))]}"
    if running "$RUN/instance_$n.pid"; then echo "  instance $n already running"; return 0; fi
    # Launched directly (no wrapper function/subshell) so $! is the uvicorn process
    # itself (taskset execs in place); stop.sh relies on that PID.
    INSTANCE_ID="$n" $(pin_prefix "$(cpus_for "$n")") "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$port" \
        --workers "$WORKERS" --log-level "$LOG_LEVEL" >> "$RUN/instance_$n.log" 2>&1 &
    echo $! > "$RUN/instance_$n.pid"
}

start_instances() {  # start_instances <n...>: launch all, then wait for each
    for n in "$@"; do launch_instance "$n"; done
    for n in "$@"; do wait_for "http://127.0.0.1:${PORTS[$((n - 1))]}/health" "instance $n (:${PORTS[$((n - 1))]})"; done
}

start_nginx() {
    if running "$RUN/nginx.pid"; then echo "  nginx already running"; return 0; fi
    nginx -t -q -p "$ROOT/deploy/" -c nginx/nginx.conf -e run/error.log   # validate before start
    $(pin_prefix "${CPUS_NGINX:-}") nginx -p "$ROOT/deploy/" -c nginx/nginx.conf -e run/error.log
    wait_for "http://127.0.0.1:8080/health" "nginx (:8080)"
}

start_consumer() {
    [[ -n "${KAFKA_BOOTSTRAP_SERVERS:-}" ]] || { echo "  set KAFKA_BOOTSTRAP_SERVERS (e.g. 127.0.0.1:9092)" >&2; return 1; }
    if running "$RUN/consumer.pid"; then echo "  consumer already running"; return 0; fi
    $(pin_prefix "${CPUS_CONSUMER:-}") "$PYTHON" -m src.event_consumer >> "$RUN/consumer.log" 2>&1 &
    echo $! > "$RUN/consumer.pid"
    sleep 2
    running "$RUN/consumer.pid" && echo "  consumer running (log: deploy/run/consumer.log)" \
        || { echo "  consumer exited, see deploy/run/consumer.log" >&2; return 1; }
}

start_drift() {
    if running "$RUN/drift.pid"; then echo "  drift monitor already running"; return 0; fi
    $(pin_prefix "${CPUS_DRIFT:-}") "$PYTHON" -m monitoring.drift_monitor >> "$RUN/drift.log" 2>&1 &
    echo $! > "$RUN/drift.pid"
    sleep 2
    running "$RUN/drift.pid" && echo "  drift monitor running (log: deploy/run/drift.log)" \
        || { echo "  drift monitor exited, see deploy/run/drift.log" >&2; return 1; }
}

case "${1:-all}" in
    nginx) start_nginx ;;
    consumer) start_consumer ;;
    drift) start_drift ;;
    all)   start_instances $(seq "${#PORTS[@]}")
           start_nginx ;;
    *)     if [[ "$1" =~ ^[0-9]+$ ]] && (( $1 >= 1 && $1 <= ${#PORTS[@]} )); then
               start_instances "$1"
           else
               echo "usage: $0 [all|nginx|consumer|drift|1..${#PORTS[@]}]" >&2; exit 2
           fi ;;
esac
