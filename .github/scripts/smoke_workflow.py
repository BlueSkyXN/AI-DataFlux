"""临时 CSV + 本地假上游，验证真实 CLI / Control / Worker / Gateway 闭环。

只监听 loopback 临时端口；所有配置、数据和日志均在 TemporaryDirectory。
--serve 保持环境供浏览器联调，按 Enter 后验证 GUI 创建的 Job 并关闭子进程。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "local-workflow-test-token"


def stop_process(process):
    # onefile bootloader 会再启动子进程；清理测试拥有的进程组，不能只杀父进程。
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
    else:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        process.wait(timeout=5)


class Provider(BaseHTTPRequestHandler):
    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps(
            {
                "id": "chatcmpl-local",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":"local-smoke"}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request(url, *, method="GET", payload=None, authorized=True, headers=None):
    actual_headers = {"Authorization": f"Bearer {TOKEN}"} if authorized else {}
    if payload is not None:
        actual_headers["Content-Type"] = "application/json"
    actual_headers.update(headers or {})
    req = Request(
        url,
        method=method,
        headers=actual_headers,
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    with build_opener(ProxyHandler({})).open(req, timeout=5) as response:
        return json.loads(response.read())


@contextmanager
def environment(binary=None, *, separate_worker=False):
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    processes = []
    logs = []
    with tempfile.TemporaryDirectory(prefix="dataflux-workflow-") as folder:
        workspace = Path(folder)
        gateway_port, control_port = free_port(), free_port()
        while control_port == gateway_port:
            control_port = free_port()
        config = {
            "schema_version": 4,
            "control": {},
            "runtime": {
                "auth": {"token": TOKEN},
                "log": {"level": "error"},
                "workspace": {
                    "roots": {"project": str(workspace)},
                    "state_dir": ".dataflux/jobs",
                },
                "scheduler": {"max_active_jobs": 1, "sample_interval_seconds": 0.1},
            },
            "job": {
                "gateway_url": f"http://127.0.0.1:{gateway_port}",
                "datasource": {
                    "type": "csv",
                    "input_path": str(workspace / "input.csv"),
                    "output_path": str(workspace / "input.csv"),
                    "engine": "pandas",
                },
                "columns": {"extract": ["input"], "write": {"answer": "result"}},
                "prompt": {"template": "{record_json}", "required_fields": ["answer"]},
                "model_selection": {"mode": "strict", "route_id": "local"},
                "concurrency": {"batch_size": 1, "max_in_flight": 1},
            },
            "gateway": {
                "channels": {
                    "local": {
                        "base_url": f"http://127.0.0.1:{provider.server_port}",
                        "endpoints": {"chat_completions": "/v1/chat/completions"},
                    }
                },
                "routes": [
                    {
                        "id": "local",
                        "display_name": "Local",
                        "channel_id": "local",
                        "upstream_model": "local",
                        "capabilities": ["chat_completions"],
                        "safe_rps": 100,
                    }
                ],
            },
        }
        (workspace / "input.csv").write_text("input,result\nhello,\n", encoding="utf-8")
        path = workspace / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        command = (
            [str(Path(binary).resolve())]
            if binary
            else [sys.executable, str(ROOT / "cli.py")]
        )
        try:
            if separate_worker:
                log = (workspace / "worker.log").open("w+")
                logs.append(log)
                processes.append(
                    subprocess.Popen(
                        [*command, "worker", "--config", str(path)],
                        cwd=ROOT,
                        env={**os.environ, "DATAFLUX_TOKEN": TOKEN},
                        stdout=log,
                        stderr=log,
                        start_new_session=os.name == "posix",
                    )
                )
            for kind, port in (("gateway", gateway_port), ("gui", control_port)):
                log = (workspace / f"{kind}.log").open("w+")
                logs.append(log)
                args = [*command, kind, "--config", str(path), "--port", str(port)]
                if kind == "gui":
                    args.append("--no-browser")
                    if separate_worker:
                        args.append("--no-worker")
                process = subprocess.Popen(
                    args,
                    cwd=ROOT,
                    env={**os.environ, "DATAFLUX_TOKEN": TOKEN},
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name == "posix",
                )
                processes.append(process)
                url = f"http://127.0.0.1:{port}" + (
                    "/admin/health" if kind == "gateway" else "/health"
                )
                deadline = time.monotonic() + 50
                while True:
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(f"{kind} exited: {log.read()}")
                    try:
                        request(url)
                        break
                    except (URLError, TimeoutError):
                        if time.monotonic() >= deadline:
                            log.seek(0)
                            raise RuntimeError(
                                f"{kind} startup timed out: {log.read()[-4000:]}"
                            )
                        time.sleep(0.1)
            yield f"http://127.0.0.1:{control_port}", workspace, command
        except Exception:
            # CI 的任务提交/回读失败也要保留服务端诊断，而不只是启动失败。
            for log in logs:
                log.seek(0)
                print(f"--- {Path(log.name).name} ---", file=sys.stderr)
                print(log.read()[-4000:], file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                stop_process(process)
            for log in logs:
                log.close()
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)


def verify(base, workspace, command, *, submit=True):
    try:
        request(base + "/api/v1/jobs", authorized=False)
    except HTTPError as error:
        assert error.code == 401
    else:
        raise AssertionError("unauthenticated Job API must be rejected")
    selection = {"root_id": "project", "relative_path": "config.yaml"}
    if submit:
        created = request(base + "/api/v1/jobs", method="POST", payload=selection)
        job_id = created["job_id"]
    else:
        jobs = request(base + "/api/v1/jobs")["jobs"]
        assert jobs, "browser did not submit a job"
        job_id = jobs[-1]["job_id"]
    deadline = time.monotonic() + 25
    while True:
        state = request(base + f"/api/v1/jobs/{job_id}")
        if state["status"] == "completed":
            break
        assert state["status"] in {"queued", "running"}, {
            "status": state["status"],
            "last_error": state.get("last_error"),
            "counts": state.get("counts"),
        }
        if time.monotonic() >= deadline:
            raise AssertionError("job did not complete")
        time.sleep(0.1)
    assert state["counts"]["persisted"] == 1
    assert "checkpoints" not in state
    with (workspace / "input.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["result"] == "local-smoke"
    result = subprocess.run(
        [
            *command,
            "job",
            "status",
            job_id,
            "--config",
            str(workspace / "config.yaml"),
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout)["counts"]["persisted"] == 1
    events = request(base + f"/api/v1/jobs/{job_id}/events?limit=100")["events"]
    assert any(event["type"] == "record_persisted" for event in events)
    print(
        json.dumps(
            {
                "result": "PASS",
                "status": state["status"],
                "persisted": 1,
                "layers": [
                    "Control HTTP",
                    "Worker",
                    "Gateway HTTP",
                    "local provider",
                    "CSV readback",
                    "CLI JSON",
                ],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--separate-worker", action="store_true")
    args = parser.parse_args()
    with environment(args.binary, separate_worker=args.separate_worker) as (
        base,
        workspace,
        command,
    ):
        if args.serve:
            print(
                json.dumps({"control_url": base, "workspace": str(workspace)}),
                flush=True,
            )
            input("Browser fixture ready; press Enter after submitting a Job: ")
        verify(base, workspace, command, submit=not args.serve)


if __name__ == "__main__":
    main()
