"""Windows equivalent of deploy/start.sh, deploy/stop.sh and deploy/kafka.sh (no WSL, no Docker, no admin).

Run from the project root with the venv's Python (PowerShell, cmd or Git Bash):
    python deploy\\windows.py install          # once: nginx, a portable Java 21 JRE and Kafka into NILEPAY_TOOLS
    python deploy\\windows.py start            # instances 1-6 on :8001-:8006, then Nginx :8080
    python deploy\\windows.py start 3          # one part only: 1..6 | nginx | consumer | kafka
    python deploy\\windows.py stop             # Nginx + every instance (same targets as start), graceful
    python deploy\\windows.py stop 3 --force   # skip the graceful stop: taskkill /T /F at once
    python deploy\\windows.py status           # what is running, Kafka topics + consumer lag

The instance list comes from the upstream block in deploy/nginx/nginx.conf, as in start.sh.
Config (env), same meaning as start.sh / kafka.sh: MODEL_DIR (default models/e7_v1),
  WORKERS_PER_INSTANCE (1), LOG_LEVEL (warning), TIMING_LOG (0), DECISION_POLICY, ACCOUNT_SNAPSHOT,
  KAFKA_BOOTSTRAP_SERVERS and the other KAFKA_* settings (passed through), KAFKA_HEAP (512m),
  KAFKA_TOPIC_PREDICTIONS, KAFKA_TOPIC_DLQ, KAFKA_PARTITIONS (6).
Windows only: NILEPAY_TOOLS (default %LOCALAPPDATA%\\nilepay), KAFKA_DATA_DIR (default
  NILEPAY_TOOLS\\kafka-data), KAFKA_VERSION (for `install`; default 4.0.0, later commands use the
  installed one). CPU pinning (CPUS_*) is not supported.

Graceful stop: Windows has no SIGTERM, and Ctrl+Break can't reach these detached processes, so
`stop` creates deploy/run/<name>.stop and the process shuts itself down (src/shutdown.py):
  - instance: uvicorn stops accepting, finishes in-flight requests (GRACEFUL_SHUTDOWN_S, 5),
    the Kafka producer flushes (KAFKA_SHUTDOWN_FLUSH_S, 5), counters go to the producer ledger.
  - consumer: finishes the current message, commits what is stored or dead-lettered,
    closes the Kafka consumer (leaves the group, so a restart is assigned partitions at once).
A process still running after STOP_TIMEOUT_S (20) is force-killed with `taskkill /T /F`, as is
everything with --force. Graceful instance stop needs WORKERS_PER_INSTANCE=1 (the default).
"""
import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "deploy" / "run"
NGINX_CONF = ROOT / "deploy" / "nginx" / "nginx.conf"
TOOLS = Path(os.getenv("NILEPAY_TOOLS") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "nilepay")

NGINX_VERSION = "1.30.6"
NGINX_URL = f"https://nginx.org/download/nginx-{NGINX_VERSION}.zip"
NGINX_EXE = TOOLS / f"nginx-{NGINX_VERSION}" / "nginx.exe"

JRE_URL = "https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jre/hotspot/normal/eclipse"
JRE_DIR = TOOLS / "jre"

SCALA_VERSION = "2.13"


def kafka_version():
    """KAFKA_VERSION if set, else the version `install` put in TOOLS, else 4.0.0 (as deploy/kafka.sh)."""
    if os.getenv("KAFKA_VERSION"):
        return os.environ["KAFKA_VERSION"]
    installed = sorted(TOOLS.glob(f"kafka_{SCALA_VERSION}-*/libs"))
    return installed[-1].parent.name.split("-", 1)[1] if installed else "4.0.0"


KAFKA_VERSION = kafka_version()
KAFKA_DIST = f"kafka_{SCALA_VERSION}-{KAFKA_VERSION}"
# archive.apache.org has every release but is often rate-limited; dlcdn has only current ones.
KAFKA_URLS = [f"https://archive.apache.org/dist/kafka/{KAFKA_VERSION}/{KAFKA_DIST}.tgz",
              f"https://dlcdn.apache.org/kafka/{KAFKA_VERSION}/{KAFKA_DIST}.tgz"]
KAFKA_LIBS = TOOLS / KAFKA_DIST / "libs"
KAFKA_DATA = Path(os.getenv("KAFKA_DATA_DIR") or TOOLS / "kafka-data")
BOOTSTRAP = "127.0.0.1:9092"
TOPIC = os.getenv("KAFKA_TOPIC_PREDICTIONS", "nilepay.prediction-events")
DLQ = os.getenv("KAFKA_TOPIC_DLQ", "nilepay.prediction-events.dlq")
PARTITIONS = int(os.getenv("KAFKA_PARTITIONS", "6"))

PYTHON = os.getenv("PYTHON") or sys.executable
GRACEFUL_SHUTDOWN_S = os.getenv("GRACEFUL_SHUTDOWN_S", "5")   # uvicorn: in-flight requests at shutdown
STOP_TIMEOUT_S = float(os.getenv("STOP_TIMEOUT_S", "20"))     # graceful stop budget before force-kill

# Detached from this console: processes keep running after this script (or its terminal) exits.
CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_BREAKAWAY_FROM_JOB = 0x200, 0x08000000, 0x01000000


def ports():
    """Ports of the upstream servers in nginx.conf, in order (instance N = Nth `server` line)."""
    found = re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", NGINX_CONF.read_text(), re.M)
    if not found:
        sys.exit(f"no upstream servers found in {NGINX_CONF}")
    return found


# ---------------------------------------------------------------- processes

def alive(pid):
    """True if a process with this PID is running (os.kill(pid, 0) would terminate it on Windows)."""
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return bool(ok) and code.value == 259                                # STILL_ACTIVE


def pid_from(pidfile):
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


def spawn(cmd, log, env=None, cwd=ROOT):
    """Start cmd in the background, appending stdout+stderr to log; returns the Popen."""
    out = open(log, "ab")
    kwargs = dict(cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    try:
        return subprocess.Popen(cmd, creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
                                | CREATE_BREAKAWAY_FROM_JOB, **kwargs)
    except OSError:   # the parent job (some terminals/IDEs) does not allow breakaway
        return subprocess.Popen(cmd, creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, **kwargs)
    finally:
        out.close()


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if not alive(pid):
            return
        time.sleep(0.1)


def stop_pidfile(pidfile, name):
    pid = pid_from(pidfile)
    if pid:
        kill_tree(pid)
        print(f"  {name} stopped")
    else:
        print(f"  {name} not running")
    pidfile.unlink(missing_ok=True)


def stop_file_for(pidfile):
    return pidfile.with_suffix(".stop")


def stop_gracefully(targets, force=False):
    """targets: [(pidfile, name)]. Ask every running process to stop (stop file), wait for all of
    them together up to STOP_TIMEOUT_S, then force-kill whatever is still running."""
    waiting = []
    for pidfile, name in targets:
        pid = pid_from(pidfile)
        if not pid:
            print(f"  {name} not running")
        elif force:
            kill_tree(pid)
            print(f"  {name} force-stopped")
        else:
            stop_file_for(pidfile).write_text(str(time.time()))
            waiting.append((pidfile, name, pid))
            continue
        pidfile.unlink(missing_ok=True)
        stop_file_for(pidfile).unlink(missing_ok=True)
    started = time.monotonic()
    while waiting and time.monotonic() - started < STOP_TIMEOUT_S:
        for item in list(waiting):
            pidfile, name, pid = item
            if not alive(pid):
                print(f"  {name} stopped gracefully in {time.monotonic() - started:.1f}s")
                waiting.remove(item)
                pidfile.unlink(missing_ok=True)
                stop_file_for(pidfile).unlink(missing_ok=True)
        time.sleep(0.05)
    for pidfile, name, pid in waiting:
        kill_tree(pid)
        print(f"  {name} did not stop within {STOP_TIMEOUT_S:.0f}s: force-killed", file=sys.stderr)
        pidfile.unlink(missing_ok=True)
        stop_file_for(pidfile).unlink(missing_ok=True)


def wait_for(url, name, proc=None):
    for _ in range(100):
        try:
            with urllib.request.urlopen(url, timeout=1):
                print(f"  {name} ready")
                return True
        except OSError:
            pass
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(0.2)
    print(f"  {name} did not become healthy: {url}", file=sys.stderr)
    return False


# ---------------------------------------------------------------- API instances

def start_instances(numbers):
    env = dict(os.environ, MODEL_DIR=os.getenv("MODEL_DIR", "models/e7_v1"), TIMING_LOG=os.getenv("TIMING_LOG", "0"))
    all_ports, started = ports(), {}
    for n in numbers:
        pidfile = RUN / f"instance_{n}.pid"
        if pid_from(pidfile):
            print(f"  instance {n} already running")
            continue
        cmd = [PYTHON, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", all_ports[n - 1],
               "--workers", os.getenv("WORKERS_PER_INSTANCE", "1"), "--log-level", os.getenv("LOG_LEVEL", "warning"),
               "--timeout-graceful-shutdown", GRACEFUL_SHUTDOWN_S]
        stop_file = stop_file_for(pidfile)
        stop_file.unlink(missing_ok=True)                  # stale request from an earlier stop
        proc = spawn(cmd, RUN / f"instance_{n}.log",
                     env=dict(env, INSTANCE_ID=str(n), NILEPAY_STOP_FILE=str(stop_file), EVENT_LEDGER_DIR=str(RUN)))
        pidfile.write_text(str(proc.pid))
        started[n] = proc
    ok = True
    for n, proc in started.items():
        port = all_ports[n - 1]
        if not wait_for(f"http://127.0.0.1:{port}/health", f"instance {n} (:{port})", proc):
            print(f"    see deploy/run/instance_{n}.log", file=sys.stderr)
            ok = False
    return ok


# ---------------------------------------------------------------- nginx

def nginx_args():
    # Prefix deploy/ as on Linux, so every relative path in nginx.conf (run/...) is unchanged.
    return [str(NGINX_EXE), "-p", f"{(ROOT / 'deploy').as_posix()}/", "-c", "run/nginx.windows.conf",
            "-e", "run/error.log"]


def write_windows_nginx_conf():
    """nginx.conf adapted to Windows:
    - nginx for Windows runs one worker that does all the work, with at most 1024 connections (select()).
    - Windows answers a connect to a closed loopback port only after ~2 s of SYN retries instead
      of an immediate refusal, so a stopped instance shows up as a connect timeout (504), not an
      error. Retrying on `timeout` too keeps failover error-free. Still never resends a POST that
      reached an instance: without `non_idempotent`, nginx only retries a POST it could not send."""
    conf = NGINX_CONF.read_text()
    conf = re.sub(r"^worker_processes\s+\d+;", "worker_processes 1;", conf, flags=re.M)
    conf = re.sub(r"worker_connections\s+\d+;", "worker_connections 1024;", conf)
    conf = re.sub(r"proxy_next_upstream\s+error;", "proxy_next_upstream         error timeout;", conf)
    (RUN / "nginx.windows.conf").write_text(conf)


def start_nginx():
    if not NGINX_EXE.exists():
        sys.exit("  nginx not installed: python deploy\\windows.py install")
    if pid_from(RUN / "nginx.pid"):
        print("  nginx already running")
        return True
    write_windows_nginx_conf()
    check = subprocess.run(nginx_args() + ["-t", "-q"], cwd=NGINX_EXE.parent, capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"  nginx -t failed:\n{check.stderr}")
    # nginx for Windows never daemonizes, so it is started in the background here.
    proc = spawn(nginx_args(), RUN / "nginx.stdout.log", cwd=NGINX_EXE.parent)
    return wait_for("http://127.0.0.1:8080/health", "nginx (:8080)", proc)


def stop_nginx():
    pid = pid_from(RUN / "nginx.pid")
    if not pid:
        print("  nginx not running")
        (RUN / "nginx.pid").unlink(missing_ok=True)
        return
    subprocess.run(nginx_args() + ["-s", "quit"], cwd=NGINX_EXE.parent, capture_output=True)
    for _ in range(50):
        if not alive(pid):
            break
        time.sleep(0.1)
    else:
        kill_tree(pid)
    (RUN / "nginx.pid").unlink(missing_ok=True)
    print("  nginx stopped")


# ---------------------------------------------------------------- Kafka

def java_home():
    """The JRE installed by `install`, else JAVA_HOME / java on PATH."""
    bundled = next(JRE_DIR.glob("*/bin/java.exe"), None)
    if bundled:
        return bundled.parent.parent
    if os.getenv("JAVA_HOME"):
        return Path(os.environ["JAVA_HOME"])
    java = shutil.which("java")
    if java:
        return Path(java).resolve().parent.parent
    sys.exit("  Java not found: python deploy\\windows.py install")


def kafka_java(main_class, *args, server=False):
    """The java command bin\\windows\\kafka-run-class.bat would run. Called directly because that
    .bat lists every jar in libs\\ on one line and overflows cmd's 8191-character limit."""
    home = TOOLS / KAFKA_DIST
    heap = os.getenv("KAFKA_HEAP", "512m") if server else "256m"
    log4j = "log4j2.yaml" if server else "tools-log4j2.yaml"
    (KAFKA_DATA / "logs").mkdir(parents=True, exist_ok=True)
    return [str(java_home() / "bin" / "java.exe"), f"-Xms{heap}", f"-Xmx{heap}", "-server", "-XX:+UseG1GC",
            "-XX:MaxGCPauseMillis=20", "-XX:InitiatingHeapOccupancyPercent=35", "-XX:+ExplicitGCInvokesConcurrent",
            f"-Dkafka.logs.dir={(KAFKA_DATA / 'logs').as_posix()}",
            f"-Dlog4j2.configurationFile=file:{(home / 'config' / log4j).as_posix()}",
            "-cp", str(home / "libs" / "*"), main_class, *args]


def kafka_tool(main_class, *args):
    return subprocess.run(kafka_java(main_class, *args), capture_output=True, text=True)


def admin():
    from confluent_kafka.admin import AdminClient
    return AdminClient({"bootstrap.servers": BOOTSTRAP, "log_level": 2})


def broker_ready(timeout=1.0):
    try:
        admin().list_topics(timeout=timeout)
        return True
    except Exception:
        return False


def create_topics():
    from confluent_kafka.admin import NewTopic
    client = admin()
    existing = client.list_topics(timeout=10).topics
    wanted = [NewTopic(TOPIC, PARTITIONS, 1, config={"retention.ms": "604800000"}),
              NewTopic(DLQ, 1, 1, config={"retention.ms": "2592000000"})]
    new = [t for t in wanted if t.topic not in existing]
    for topic, future in (client.create_topics(new) if new else {}).items():
        future.result(timeout=30)
    print(f"  topics: {TOPIC} ({PARTITIONS} partitions, 7 days), {DLQ} (1 partition, 30 days)")


def start_kafka():
    if not KAFKA_LIBS.is_dir():
        sys.exit("  Kafka not installed: python deploy\\windows.py install")
    pidfile = KAFKA_DATA / "kafka.pid"
    if pid_from(pidfile):
        print(f"  kafka already running (pid {pidfile.read_text().strip()})")
        return True
    KAFKA_DATA.mkdir(parents=True, exist_ok=True)
    config = KAFKA_DATA / "server.properties"
    # Forward slashes: backslashes are escape characters in .properties files.
    template = (ROOT / "deploy" / "kafka" / "server.properties").read_text()
    config.write_text(template.replace("@DATA_DIR@", KAFKA_DATA.as_posix()))
    if not (KAFKA_DATA / "kraft-logs" / "meta.properties").exists():
        uuid = kafka_tool("kafka.tools.StorageTool", "random-uuid")
        if uuid.returncode != 0:
            sys.exit(f"  kafka-storage random-uuid failed:\n{uuid.stdout}{uuid.stderr}")
        fmt = kafka_tool("kafka.tools.StorageTool", "format", "--standalone", "-t",
                         uuid.stdout.strip().splitlines()[-1], "-c", str(config))
        if fmt.returncode != 0:
            sys.exit(f"  storage format failed:\n{fmt.stdout}{fmt.stderr}")
        print(f"  storage formatted in {KAFKA_DATA / 'kraft-logs'}")
    proc = spawn(kafka_java("kafka.Kafka", str(config), server=True), KAFKA_DATA / "server.log",
                 cwd=TOOLS / KAFKA_DIST)
    pidfile.write_text(str(proc.pid))
    for _ in range(60):
        if broker_ready():
            print(f"  kafka ready on {BOOTSTRAP} (pid {proc.pid})")
            create_topics()
            return True
        if proc.poll() is not None:
            break
        time.sleep(1)
    print(f"  broker not ready, see {KAFKA_DATA / 'server.log'}", file=sys.stderr)
    return False


def stop_kafka():
    stop_pidfile(KAFKA_DATA / "kafka.pid", "kafka")


def start_consumer():
    if not os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        print("  set KAFKA_BOOTSTRAP_SERVERS (e.g. 127.0.0.1:9092)", file=sys.stderr)
        return False
    pidfile = RUN / "consumer.pid"
    if pid_from(pidfile):
        print("  consumer already running")
        return True
    stop_file = stop_file_for(pidfile)
    stop_file.unlink(missing_ok=True)
    proc = spawn([PYTHON, "-m", "src.event_consumer"], RUN / "consumer.log",
                 env=dict(os.environ, NILEPAY_STOP_FILE=str(stop_file)))
    pidfile.write_text(str(proc.pid))
    time.sleep(2)
    if proc.poll() is None:
        print("  consumer running (log: deploy/run/consumer.log)")
        return True
    print("  consumer exited, see deploy/run/consumer.log", file=sys.stderr)
    return False


# ---------------------------------------------------------------- install / status

def download(url, dest):
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "nilepay-deploy"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def install():
    TOOLS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if NGINX_EXE.exists():
            print(f"  nginx {NGINX_VERSION} already installed")
        else:
            download(NGINX_URL, tmp / "nginx.zip")
            zipfile.ZipFile(tmp / "nginx.zip").extractall(TOOLS)
            print(f"  installed {NGINX_EXE}")

        if next(JRE_DIR.glob("*/bin/java.exe"), None):
            print("  Java 21 JRE already installed")
        else:
            download(JRE_URL, tmp / "jre.zip")
            zipfile.ZipFile(tmp / "jre.zip").extractall(JRE_DIR)
            print(f"  installed Java to {java_home()}")

        if KAFKA_LIBS.is_dir():
            print(f"  Kafka {KAFKA_VERSION} already installed")
        else:
            archive = tmp / f"{KAFKA_DIST}.tgz"
            for url in KAFKA_URLS:
                try:
                    download(url, archive)
                    download(url + ".sha512", tmp / "kafka.sha512")
                    break
                except OSError as e:
                    print(f"  failed: {e}")
            else:
                sys.exit(f"  could not download Kafka {KAFKA_VERSION}. Pick a version listed on "
                         "https://dlcdn.apache.org/kafka/ and rerun, e.g. KAFKA_VERSION=4.3.2")
            # Apache .sha512 files are "name: HEX HEX ..." over several lines; compare hex only.
            expected = re.sub(r"\s", "", (tmp / "kafka.sha512").read_text().split(":", 1)[1]).lower()
            if hashlib.sha512(archive.read_bytes()).hexdigest() != expected:
                sys.exit("  SHA-512 mismatch, not installing")
            print("  SHA-512 verified")
            with tarfile.open(archive) as tar:
                tar.extractall(TOOLS, filter="data")
            print(f"  installed to {TOOLS / KAFKA_DIST}")


def status():
    for n, port in enumerate(ports(), 1):
        pid = pid_from(RUN / f"instance_{n}.pid")
        print(f"  instance {n} (:{port}): {'running, pid ' + str(pid) if pid else 'stopped'}")
    pid = pid_from(RUN / "nginx.pid")
    print(f"  nginx (:8080): {'running, pid ' + str(pid) if pid else 'stopped'}")
    pid = pid_from(RUN / "consumer.pid")
    print(f"  consumer: {'running, pid ' + str(pid) if pid else 'stopped'}")
    pid = pid_from(KAFKA_DATA / "kafka.pid")
    if not pid:
        print("  kafka: stopped")
        return
    print(f"  kafka ({BOOTSTRAP}): running, pid {pid}")
    if broker_ready(timeout=5):
        topics = admin().list_topics(timeout=5).topics
        for name in (TOPIC, DLQ):
            print(f"    {name}: {len(topics[name].partitions) if name in topics else 'missing'} partitions")
        subprocess.run([PYTHON, "-m", "src.event_consumer", "--status"], cwd=ROOT,
                       env=dict(os.environ, KAFKA_BOOTSTRAP_SERVERS=BOOTSTRAP))


def main():
    if os.name != "nt":
        sys.exit("Windows only; on Linux use deploy/start.sh, deploy/stop.sh and deploy/kafka.sh")
    sys.stdout.reconfigure(line_buffering=True)   # keep order with child-process output (status)
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--force"]
    action = args[0] if args else ""
    target = args[1] if len(args) > 1 else "all"
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "tmp").mkdir(exist_ok=True)
    os.chdir(ROOT)
    count = len(ports())
    numeric = target.isdigit() and 1 <= int(target) <= count

    if action == "install":
        install()
    elif action == "status":
        status()
    elif action == "start":
        if target == "all":
            ok = start_instances(range(1, count + 1)) and start_nginx()
        elif target in ("nginx", "consumer", "kafka"):
            ok = {"nginx": start_nginx, "consumer": start_consumer, "kafka": start_kafka}[target]()
        elif numeric:
            ok = start_instances([int(target)])
        else:
            sys.exit(f"usage: {sys.argv[0]} start [all|nginx|consumer|kafka|1..{count}]")
        sys.exit(0 if ok else 1)
    elif action == "stop":
        if target == "all":
            stop_nginx()
            stop_gracefully([(f, f"instance {f.stem.split('_')[1]}") for f in sorted(RUN.glob("instance_*.pid"))],
                            force)
        elif target == "nginx":
            stop_nginx()
        elif target == "consumer":
            stop_gracefully([(RUN / "consumer.pid", "consumer")], force)
        elif target == "kafka":
            stop_kafka()
        elif target.isdigit():
            stop_gracefully([(RUN / f"instance_{target}.pid", f"instance {target}")], force)
        else:
            sys.exit(f"usage: {sys.argv[0]} stop [all|nginx|consumer|kafka|<instance number>] [--force]")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
