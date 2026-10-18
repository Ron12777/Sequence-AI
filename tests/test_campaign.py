import json
import subprocess
import sys
import pytest
from scripts.overnight import atomic_copy, atomic_json, promotion_allowed, run


def test_promotion_needs_independent_positive_confirmation():
    assert not promotion_allowed(
        {"score": 0.60}, {"score": 0.55, "paired_score_95ci": [0.49, 0.61]}
    )
    assert not promotion_allowed(
        {"score": 0.51}, {"score": 0.60, "paired_score_95ci": [0.55, 0.65]}
    )
    assert promotion_allowed(
        {"score": 0.55}, {"score": 0.56, "paired_score_95ci": [0.52, 0.60]}
    )


def test_atomic_outputs_preserve_selected_bytes(tmp_path):
    original = tmp_path / "original.pt"
    original.write_bytes(b"validated checkpoint")
    selected = tmp_path / "champion.pt"
    atomic_copy(original, selected)
    assert selected.read_bytes() == original.read_bytes()
    state = tmp_path / "state.json"
    atomic_json(state, {"status": "running"})
    assert json.loads(state.read_text()) == {"status": "running"}


def test_child_failure_and_timeout_do_not_look_successful(tmp_path):
    with pytest.raises(RuntimeError):
        run([sys.executable, "-c", "raise SystemExit(3)"], tmp_path / "failure.log", 5)
    with pytest.raises(subprocess.TimeoutExpired):
        run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path / "timeout.log",
            0.1,
        )
