#!/usr/bin/env bash
# Start the FastAPI instances and Nginx (:8080).
# The instance list comes from the upstream block in deploy/nginx/nginx.conf
# (instance N = Nth `server` line), so Nginx and the started instances always match.
#   deploy/start.sh            # every instance, then Nginx
#   deploy/start.sh 3          # only instance 3 (e.g. restart after a failure)
#   deploy/start.sh nginx
# Config (env): MODEL_DIR (default models/e7_v1), WORKERS_PER_INSTANCE (default 1),
#   ACCOUNT_SNAPSHOT, LOG_LEVEL (default warning; `info` logs every request),
#   TIMING_LOG (default 0; 1 prints timings),
#   CPUS_INSTANCES (taskset CPU list for all instances, e.g. 0-7; or one list per
#   instance separated by spaces, e.g. "0 1 2 3 4 5"), CPUS_NGINX. Empty = no pinning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/deploy/run"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
export MODEL_DIR="${MODEL_DIR:-models/e7_v1}"
export TIMING_LOG="${TIMING_LOG:-0}"
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
    $(pin_prefix "$(cpus_for "$n")") "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$port" \
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

case "${1:-all}" in
    nginx) start_nginx ;;
    all)   start_instances $(seq "${#PORTS[@]}")
           start_nginx ;;
    *)     if [[ "$1" =~ ^[0-9]+$ ]] && (( $1 >= 1 && $1 <= ${#PORTS[@]} )); then
               start_instances "$1"
           else
               echo "usage: $0 [all|nginx|1..${#PORTS[@]}]" >&2; exit 2
           fi ;;
esac
