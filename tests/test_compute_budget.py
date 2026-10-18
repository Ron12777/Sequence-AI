import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.compute_budget_experiments import exclusive_window, process_identity


def state(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def wait_for(predicate, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Process state did not change")


def test_exclusive_window_resumes_controller_after_exception(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "scripts/overnight.py"]
    )
    try:
        with pytest.raises(RuntimeError, match="deliberate"):
            with exclusive_window(process.pid, 10, tmp_path):
                wait_for(lambda: state(process.pid) == "T")
                raise RuntimeError("deliberate")
        wait_for(lambda: state(process.pid) != "T")
        assert process_identity(process.pid) is not None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_independent_guard_resumes_after_supervisor_is_killed(tmp_path):
    controller = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "scripts/overnight.py"]
    )
    source = """
from pathlib import Path
import sys,time
from scripts.compute_budget_experiments import exclusive_window
with exclusive_window(int(sys.argv[1]), 10, Path(sys.argv[2])):
    Path(sys.argv[2], 'ready').touch()
    time.sleep(60)
"""
    supervisor = subprocess.Popen(
        [sys.executable, "-c", source, str(controller.pid), str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    try:
        wait_for(lambda: (tmp_path / "ready").exists(), 15)
        wait_for(lambda: state(controller.pid) == "T")
        supervisor.kill()
        supervisor.wait(timeout=5)
        wait_for(lambda: state(controller.pid) != "T")
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=5)
        os.kill(controller.pid, 18)  # Ensure cleanup can terminate a stopped child.
        controller.terminate()
        controller.wait(timeout=5)
