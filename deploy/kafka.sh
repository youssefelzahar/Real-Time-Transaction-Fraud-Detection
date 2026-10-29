#!/usr/bin/env bash
# Native single-node Kafka for NilePay (no Docker). Needs Java 17+:
#   sudo apt install openjdk-21-jre-headless
#
#   deploy/kafka.sh install   # download + verify Apache Kafka into KAFKA_HOME (once)
#   deploy/kafka.sh start     # format storage (first run), start broker, create topics
#   deploy/kafka.sh stop
#   deploy/kafka.sh status    # broker state, topics, consumer-group lag
#
# Config (env): KAFKA_HOME (default ~/.local/nilepay-kafka), KAFKA_DATA_DIR (default
#   ~/.local/share/nilepay-kafka; keep it on a Linux filesystem), KAFKA_HEAP (default 512m),
#   KAFKA_TOPIC_PREDICTIONS, KAFKA_TOPIC_DLQ, KAFKA_PARTITIONS (6), KAFKA_CONSUMER_GROUP,
#   KAFKA_CPUS (taskset CPU list for the broker, e.g. 8-9; empty = no pinning).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAFKA_VERSION="4.0.0"
SCALA_VERSION="2.13"
DIST="kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
URL="https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${DIST}.tgz"

KAFKA_HOME="${KAFKA_HOME:-$HOME/.local/nilepay-kafka}"
KAFKA_DATA_DIR="${KAFKA_DATA_DIR:-$HOME/.local/share/nilepay-kafka}"
BIN="$KAFKA_HOME/$DIST/bin"
CONFIG="$KAFKA_DATA_DIR/server.properties"
PIDFILE="$KAFKA_DATA_DIR/kafka.pid"
BOOTSTRAP="127.0.0.1:9092"
TOPIC="${KAFKA_TOPIC_PREDICTIONS:-nilepay.prediction-events}"
DLQ="${KAFKA_TOPIC_DLQ:-nilepay.prediction-events.dlq}"
PARTITIONS="${KAFKA_PARTITIONS:-6}"
GROUP="${KAFKA_CONSUMER_GROUP:-nilepay-prediction-monitor}"

need_java() {
    command -v java >/dev/null || { echo "Java not found: sudo apt install openjdk-21-jre-headless" >&2; exit 1; }
}

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

install() {
    if [[ -x "$BIN/kafka-server-start.sh" ]]; then echo "  Kafka $KAFKA_VERSION already installed in $KAFKA_HOME"; return; fi
    mkdir -p "$KAFKA_HOME"
    local tmp; tmp="$(mktemp -d)"
    echo "  downloading $URL"
    curl -fsSL -o "$tmp/$DIST.tgz" "$URL"
    curl -fsSL -o "$tmp/$DIST.tgz.sha512" "$URL.sha512"
    # Apache .sha512 files are "name: HEX HEX ..." over several lines; compare hex only.
    local expected actual
    expected="$(sed 's/^[^:]*://' "$tmp/$DIST.tgz.sha512" | tr -d ' \n\t' | tr 'A-F' 'a-f')"
    actual="$(sha512sum "$tmp/$DIST.tgz" | cut -d' ' -f1)"
    [[ "$expected" == "$actual" ]] || { echo "  SHA-512 mismatch, not installing" >&2; rm -rf "$tmp"; exit 1; }
    echo "  SHA-512 verified"
    tar -xzf "$tmp/$DIST.tgz" -C "$KAFKA_HOME"
    rm -rf "$tmp"
    echo "  installed to $KAFKA_HOME/$DIST"
}

wait_ready() {
    for _ in $(seq 60); do
        "$BIN/kafka-broker-api-versions.sh" --bootstrap-server "$BOOTSTRAP" >/dev/null 2>&1 && return 0
        running || { echo "  broker exited, see $KAFKA_DATA_DIR/server.log" >&2; return 1; }
        sleep 1
    done
    echo "  broker not ready after 60s, see $KAFKA_DATA_DIR/server.log" >&2; return 1
}

create_topics() {
    "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists --topic "$TOPIC" \
        --partitions "$PARTITIONS" --replication-factor 1 --config retention.ms=604800000 >/dev/null
    "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists --topic "$DLQ" \
        --partitions 1 --replication-factor 1 --config retention.ms=2592000000 >/dev/null
    echo "  topics: $TOPIC ($PARTITIONS partitions, 7 days), $DLQ (1 partition, 30 days)"
}

start() {
    need_java
    [[ -x "$BIN/kafka-server-start.sh" ]] || { echo "  not installed: deploy/kafka.sh install" >&2; exit 1; }
    if running; then echo "  kafka already running (pid $(cat "$PIDFILE"))"; return; fi
    mkdir -p "$KAFKA_DATA_DIR"
    sed "s|@DATA_DIR@|$KAFKA_DATA_DIR|" "$ROOT/deploy/kafka/server.properties" > "$CONFIG"
    if [[ ! -f "$KAFKA_DATA_DIR/kraft-logs/meta.properties" ]]; then
        "$BIN/kafka-storage.sh" format --standalone -t "$("$BIN/kafka-storage.sh" random-uuid)" -c "$CONFIG" >/dev/null
        echo "  storage formatted in $KAFKA_DATA_DIR/kraft-logs"
    fi
    # kafka-server-start.sh execs java, so $! is the broker JVM.
    local pin=""
    [[ -n "${KAFKA_CPUS:-}" ]] && pin="taskset -c $KAFKA_CPUS"
    KAFKA_HEAP_OPTS="-Xms${KAFKA_HEAP:-512m} -Xmx${KAFKA_HEAP:-512m}" LOG_DIR="$KAFKA_DATA_DIR/logs" \
        nohup $pin "$BIN/kafka-server-start.sh" "$CONFIG" >> "$KAFKA_DATA_DIR/server.log" 2>&1 &
    echo $! > "$PIDFILE"
    wait_ready
    echo "  kafka ready on $BOOTSTRAP (pid $(cat "$PIDFILE"))"
    create_topics
}

stop() {
    if ! running; then echo "  kafka not running"; rm -f "$PIDFILE"; return; fi
    local pid; pid="$(cat "$PIDFILE")"
    kill "$pid"
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && { echo "  kafka did not stop in 30s, killing" >&2; kill -9 "$pid"; }
    rm -f "$PIDFILE"
    echo "  kafka stopped"
}

status() {
    if ! running; then echo "  kafka not running"; return 1; fi
    echo "  kafka running (pid $(cat "$PIDFILE")) on $BOOTSTRAP"
    "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --describe --topic "$TOPIC" | head -1
    "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --describe --topic "$DLQ" | head -1
    echo
    "$BIN/kafka-consumer-groups.sh" --bootstrap-server "$BOOTSTRAP" --describe --group "$GROUP" 2>&1 \
        | grep -v "^$" || true
}

case "${1:-}" in
    install) install ;;
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    *)       echo "usage: $0 install|start|stop|status" >&2; exit 2 ;;
esac
