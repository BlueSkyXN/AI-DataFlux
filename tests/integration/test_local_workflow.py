"""真实 loopback HTTP 与临时 CSV；无需外部凭据，必须保持 integration 标记。"""

from pathlib import Path
import runpy
import subprocess
import sys

import pytest


def test_smoke_cleanup_signals_children_even_after_bootloader_exits(monkeypatch):
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[2]
    module = runpy.run_path(str(root / ".github/scripts/smoke_workflow.py"))
    signals = []
    fake_os = SimpleNamespace(
        name="posix", killpg=lambda pid, sig: signals.append((pid, sig))
    )
    stop = module["stop_process"]
    monkeypatch.setitem(stop.__globals__, "os", fake_os)
    process = SimpleNamespace(pid=1234, wait=lambda timeout: 0)
    stop(process)
    signal = module["signal"]
    assert signals == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


@pytest.mark.integration
@pytest.mark.parametrize("separate", [False, True])
def test_control_worker_gateway_csv_and_cli_round_trip(separate):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / ".github/scripts/smoke_workflow.py"),
            *(["--separate-worker"] if separate else []),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"result": "PASS"' in result.stdout
