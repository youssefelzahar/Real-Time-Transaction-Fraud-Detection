#!/usr/bin/env bash
# Stop Nginx and/or the FastAPI instances started by deploy/start.sh.
#   deploy/stop.sh             # everything
#   deploy/stop.sh a           # only instance A (failure testing)
#   deploy/stop.sh b | nginx
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
    a)     stop_pidfile "$RUN/instance_a.pid" "instance a" ;;
    b)     stop_pidfile "$RUN/instance_b.pid" "instance b" ;;
    nginx) stop_nginx ;;
    all)   stop_nginx
           stop_pidfile "$RUN/instance_a.pid" "instance a"
           stop_pidfile "$RUN/instance_b.pid" "instance b" ;;
    *)     echo "usage: $0 [all|a|b|nginx]" >&2; exit 2 ;;
esac
