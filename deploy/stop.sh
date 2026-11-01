#!/usr/bin/env bash
# Stop Nginx and/or the FastAPI instances started by deploy/start.sh.
#   deploy/stop.sh             # Nginx, then every instance
#   deploy/stop.sh 3           # only instance 3 (failure testing)
#   deploy/stop.sh nginx
#   deploy/stop.sh consumer    # Kafka event consumer (SIGTERM: commits processed events, then exits)
#   deploy/stop.sh drift       # data drift monitor
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/deploy/run"

stop_pidfile() {  # stop_pidfile <file> <name>
    local file="$1" name="$2"
    if [[ ! -f "$file" ]]; then echo "  $name not running"; return 0; fi
    local pid; pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        echo "  $name stopped"
    else
        echo "  $name not running"
    fi
    rm -f "$file"
}

stop_nginx() {
    if [[ -f "$RUN/nginx.pid" ]]; then
        nginx -p "$ROOT/deploy/" -c nginx/nginx.conf -e run/error.log -s quit 2>/dev/null || true
        for _ in $(seq 50); do [[ -f "$RUN/nginx.pid" ]] || break; sleep 0.1; done
        echo "  nginx stopped"
    else
        echo "  nginx not running"
    fi
}

case "${1:-all}" in
    nginx) stop_nginx ;;
    consumer) stop_pidfile "$RUN/consumer.pid" "consumer" ;;
    drift) stop_pidfile "$RUN/drift.pid" "drift monitor" ;;
    all)   stop_nginx
           shopt -s nullglob
           for f in "$RUN"/instance_*.pid; do
               n="${f##*/instance_}"; stop_pidfile "$f" "instance ${n%.pid}"
           done ;;
    *)     if [[ "$1" =~ ^[0-9]+$ ]]; then
               stop_pidfile "$RUN/instance_$1.pid" "instance $1"
           else
               echo "usage: $0 [all|nginx|consumer|drift|<instance number>]" >&2; exit 2
           fi ;;
esac
