#!/usr/bin/env bash
# Start FastAPI instance A (:8001), instance B (:8002) and Nginx (:8080).
#   deploy/start.sh            # everything
#   deploy/start.sh a          # only instance A (e.g. restart after a failure)
#   deploy/start.sh b | nginx
# Config (env): MODEL_DIR (default models/e7_v1), WORKERS_PER_INSTANCE (default 1),
#   ACCOUNT_SNAPSHOT, CPUS_A / CPUS_B / CPUS_NGINX (taskset CPU lists; empty = no pinning),
#   LOG_LEVEL (default warning; `info` logs every request), TIMING_LOG (default 0; 1 prints timings).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/deploy/run"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
export MODEL_DIR="${MODEL_DIR:-models/e7_v1}"
export TIMING_LOG="${TIMING_LOG:-0}"
WORKERS="${WORKERS_PER_INSTANCE:-1}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
mkdir -p "$RUN/tmp"
cd "$ROOT"

pin_prefix() {  # prints a taskset prefix for <cpu-list>, or nothing
    if [[ -n "$1" ]] && command -v taskset >/dev/null; then echo "taskset -c $1"; fi
}

wait_for() {  # wait_for <url> <name>
    for _ in $(seq 50); do
        curl -fs "$1" >/dev/null && { echo "  $2 ready"; return 0; }
        sleep 0.2
    done
    echo "  $2 did not become healthy: $1" >&2; return 1
}

start_instance() {  # start_instance <a|b> <port> <cpus>
    local name="$1" port="$2" cpus="$3"
    if [[ -f "$RUN/instance_$name.pid" ]] && kill -0 "$(cat "$RUN/instance_$name.pid")" 2>/dev/null; then
        echo "  instance $name already running"; return 0
    fi
    # Launched directly (no wrapper function/subshell) so $! is the uvicorn process
    # itself (taskset execs in place); stop.sh relies on that PID.
    $(pin_prefix "$cpus") "$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port "$port" \
        --workers "$WORKERS" --log-level "$LOG_LEVEL" >> "$RUN/instance_$name.log" 2>&1 &
    echo $! > "$RUN/instance_$name.pid"
    wait_for "http://127.0.0.1:$port/health" "instance $name (:$port)"
}

start_nginx() {
    if [[ -f "$RUN/nginx.pid" ]] && kill -0 "$(cat "$RUN/nginx.pid")" 2>/dev/null; then
        echo "  nginx already running"; return 0
    fi
    nginx -t -q -p "$ROOT/deploy/" -c nginx/nginx.conf -e run/error.log   # validate before start
    $(pin_prefix "${CPUS_NGINX:-}") nginx -p "$ROOT/deploy/" -c nginx/nginx.conf -e run/error.log
    wait_for "http://127.0.0.1:8080/health" "nginx (:8080)"
}

case "${1:-all}" in
    a)     start_instance a 8001 "${CPUS_A:-}" ;;
    b)     start_instance b 8002 "${CPUS_B:-}" ;;
    nginx) start_nginx ;;
    all)   start_instance a 8001 "${CPUS_A:-}"
           start_instance b 8002 "${CPUS_B:-}"
           start_nginx ;;
    *)     echo "usage: $0 [all|a|b|nginx]" >&2; exit 2 ;;
esac
