"""Graceful stop requests that behave the same on Windows and Linux.

Windows has no SIGTERM, and a Ctrl+Break event only reaches processes attached to the sender's
console. The API instances and the consumer run detached (own hidden console, behind the venv
launcher), so deploy/windows.py cannot signal them. Instead it creates a stop file, and the
process watches for it and then takes its normal graceful path (the same one SIGTERM takes on
Linux). On Linux, deploy/stop.sh keeps using SIGTERM; the stop file is only used if
NILEPAY_STOP_FILE is set.
"""
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("nilepay.shutdown")

STOP_FILE_ENV = "NILEPAY_STOP_FILE"


def watch_stop_file(path, on_stop, interval_s=0.2):
    """Call on_stop() once, from a daemon thread, as soon as `path` exists. Returns the thread."""
    path = Path(path)

    def loop():
        while not path.exists():
            if stop_watching.wait(interval_s):
                return
        log.warning("stop requested (%s): shutting down gracefully", path.name)
        on_stop()

    stop_watching = threading.Event()
    thread = threading.Thread(target=loop, name="stop-file-watcher", daemon=True)
    thread.stop_watching = stop_watching   # set() to end the watcher without stopping the process
    thread.start()
    return thread


def stop_file_from_env():
    path = os.getenv(STOP_FILE_ENV)
    return Path(path) if path else None
