"""Static checks of the Nginx load-balancer config and deployment scripts.

The live end-to-end check (both instances behind Nginx) is opt-in because it binds
ports 8001/8002/8080:  RUN_DEPLOY_TESTS=1 pytest tests/test_deploy_config.py
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
CONF = (DEPLOY / "nginx" / "nginx.conf").read_text()
DIRECTIVES = "\n".join(line.split("#", 1)[0] for line in CONF.splitlines())  # comments stripped


def directive(name):
    return re.findall(rf"^[ \t]*{name}[ \t]+([^;\n]+);", DIRECTIVES, flags=re.M)


def test_upstream_has_both_instances_with_load_shedding():
    servers = directive("server")
    assert [s.split()[0] for s in servers] == ["127.0.0.1:8001", "127.0.0.1:8002"]
    for s in servers:
        assert "max_conns=" in s and "max_fails=" in s and "fail_timeout=" in s
    assert directive("zone"), "max_conns needs a shared zone"
    assert directive("keepalive")


def test_listens_on_single_public_port():
    assert directive("listen") == ["127.0.0.1:8080"]


def test_forwarding_headers():
    headers = {h.split()[0]: h.split(None, 1)[1] for h in directive("proxy_set_header")}
    assert headers["Host"] == "$host"
    assert headers["X-Real-IP"] == "$remote_addr"
    assert headers["X-Forwarded-For"] == "$proxy_add_x_forwarded_for"
    assert headers["Connection"] == '""'
    assert directive("proxy_http_version") == ["1.1"]


def test_timeouts_are_explicit():
    for name in ("proxy_connect_timeout", "proxy_send_timeout", "proxy_read_timeout"):
        assert directive(name), f"{name} not set"


def test_transaction_requests_are_never_resent():
    """POST /predict must not be retried once delivered: no non_idempotent,
    and no retry on timeout (the instance may already have scored it)."""
    (next_upstream,) = directive("proxy_next_upstream")
    assert next_upstream.split() == ["error"]
    assert "non_idempotent" not in DIRECTIVES


def test_unavailable_returns_503_json():
    assert re.search(r"error_page\s+502 503\s*=\s*@unavailable", DIRECTIVES)
    assert re.search(r"return 503 '\{\"detail\"", DIRECTIVES)


@pytest.mark.skipif(shutil.which("nginx") is None, reason="nginx not installed")
def test_nginx_config_is_valid(tmp_path):
    shutil.copytree(DEPLOY / "nginx", tmp_path / "nginx")
    (tmp_path / "run" / "tmp").mkdir(parents=True)
    r = subprocess.run(["nginx", "-t", "-p", f"{tmp_path}/", "-c", "nginx/nginx.conf", "-e", "run/error.log"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", ["start.sh", "stop.sh"])
def test_scripts_are_executable_and_parse(script):
    path = DEPLOY / script
    assert os.access(path, os.X_OK)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


@pytest.mark.skipif(os.getenv("RUN_DEPLOY_TESTS") != "1", reason="set RUN_DEPLOY_TESTS=1 (binds 8001/8002/8080)")
def test_requests_through_nginx_reach_both_instances():
    import httpx

    subprocess.run([str(DEPLOY / "start.sh")], check=True, cwd=ROOT)
    try:
        log = DEPLOY / "run" / "access.log"
        offset = log.stat().st_size if log.exists() else 0
        body = {"type": "TRANSFER", "amount": 181.0, "sender_id": "C1", "receiver_id": "C2",
                "sender_balance": 181.0, "receiver_balance": 0.0}
        via_nginx = [httpx.post("http://127.0.0.1:8080/predict", json=body).json() for _ in range(20)]
        direct = [httpx.post(f"http://127.0.0.1:{p}/predict", json=body).json() for p in (8001, 8002)]
        time.sleep(1.5)  # access log is flushed every second
        with log.open() as f:
            f.seek(offset)
            upstreams = re.findall(r"POST /predict.*upstream=(\S+)", f.read())

        assert {"127.0.0.1:8001", "127.0.0.1:8002"} <= set(upstreams)
        # Same answer from A, B and through Nginx; response schema unchanged.
        assert len({r["fraud_probability"] for r in via_nginx + direct}) == 1
        assert {r["model_version"] for r in via_nginx + direct} == {"e7_v1"}
        assert set(via_nginx[0]) == {"transaction_id", "fraud_probability", "decision", "threshold",
                                     "model_version", "missing_features", "balance_source"}
    finally:
        subprocess.run([str(DEPLOY / "stop.sh")], cwd=ROOT)
