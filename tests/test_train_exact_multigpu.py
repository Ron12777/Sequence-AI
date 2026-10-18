"""CPU-only scheduling, lease, checkpoint-authority and shutdown regressions."""

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import torch

from scripts.train_exact_multigpu import (
    Scheduler,
    exclusive_lock,
    parse_devices,
    paused_slots,
    read_states,
    selected_contenders,
    scheduler_devices,
    shared_devices,
    validate_cuda_rng_count,
    validate_shared_selection,
    worker_command,
    worker_leases,
)


class Process:
    def __init__(self):
        self.status = None
        self.terminations = 0
        self.closed = False

    def poll(self):
        return self.status

    def terminate(self):
        self.terminations += 1

    def close(self):
        self.closed = True


def scheduler(devices=("0", "1")):
    launched = []

    def launch(slot, device, contender):
        process = Process()
        launched.append((slot, device, contender, process))
        return process

    return Scheduler(devices, ["cnn", "transformer", "rnad"], launch, 1000), launched


def states(**values):
    return {name: {"training_seconds": value} for name, value in values.items()}


def test_assigns_minimum_budget_and_never_duplicates_an_active_contender():
    controller, launched = scheduler()
    progress = states(cnn=100, transformer=50, rnad=0)
    controller.tick(progress, 10)
    assert [row[2] for row in launched] == ["rnad", "transformer"]
    # Progress remains stale while running, but an active contender is excluded.
    controller.tick(progress, 11)
    assert len(launched) == 2
    launched[0][3].status = 0
    progress["rnad"]["training_seconds"] = 600
    controller.tick(progress, 12)
    assert len(launched) == 2  # Reap, then refresh checkpoint budgets before dispatch.
    controller.tick(progress, 12.25)
    assert [row[2] for row in launched] == ["rnad", "transformer", "cnn"]
    assert launched[0][3].closed
    assert len({job["contender"] for job in controller.jobs.values()}) == 2


def test_pause_signals_only_requested_device_and_waits_for_its_checkpoint():
    controller, launched = scheduler()
    progress = states(cnn=0, transformer=0, rnad=0)
    controller.tick(progress, 10)
    controller.tick(progress, 11, paused={0})
    assert launched[0][3].terminations == 1
    assert launched[1][3].terminations == 0
    assert len(controller.jobs) == 2  # A signal does not imply checkpoint completion.
    launched[0][3].status = 0
    progress["cnn"]["training_seconds"] = 20
    controller.tick(progress, 12, paused={0})
    assert len(launched) == 2
    controller.tick(progress, 13)
    assert launched[-1][2] == "rnad"


@pytest.mark.parametrize("reason", ["signal", "deadline", "failure"])
def test_shutdown_waits_for_running_workers_and_does_not_launch_replacements(reason):
    controller, launched = scheduler()
    progress = states(cnn=0, transformer=0, rnad=0)
    controller.tick(progress, 10)
    if reason == "signal":
        controller.request_stop()
    elif reason == "failure":
        launched[0][3].status = 17
    controller.tick(progress, 990 if reason == "deadline" else 11)
    assert controller.stopping
    assert launched[1][3].terminations == 1
    assert controller.jobs  # Root owner must keep its lock while these remain.
    controller.tick(progress, 991)
    assert launched[1][3].terminations == 1  # No repeated signals.
    for row in launched:
        if row[3].status is None:
            row[3].status = 0
    controller.tick(progress, 992)
    assert not controller.jobs and len(launched) == 2
    assert bool(controller.failure) == (reason == "failure")


@pytest.mark.parametrize("name", ["controller.lock", "supervisor.lock", "locks/gpu-0.lock", "cnn/worker.lock"])
def test_exclusive_leases_block_duplicates_and_release_after_exit(tmp_path, name):
    path = tmp_path / name
    with exclusive_lock(path), pytest.raises(RuntimeError, match="Another process owns"), exclusive_lock(path):
        pytest.fail("Duplicate lease")
    with exclusive_lock(path):
        pass


def test_scheduler_reads_authoritative_checkpoint_not_stale_progress(tmp_path):
    directory = tmp_path / "cnn"
    directory.mkdir()
    (directory / "progress.json").write_text(json.dumps({"training_seconds": 999999}))
    torch.save({"metadata": {"training_seconds": 2}}, directory / "latest.pt")
    assert read_states(tmp_path, ["cnn"])["cnn"]["training_seconds"] == 2
    temporary = directory / "next.pt"
    torch.save({"metadata": {"training_seconds": 5}}, temporary)
    temporary.replace(directory / "latest.pt")
    assert read_states(tmp_path, ["cnn"])["cnn"]["training_seconds"] == 5


def test_selection_cannot_skip_minimum_training_budget(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"contenders": ["cnn"]}))
    with pytest.raises(ValueError, match="matched training hour"):
        selected_contenders(tmp_path, ["cnn", "rnad"], states(cnn=4000, rnad=3599))
    assert selected_contenders(tmp_path, ["cnn", "rnad"], states(cnn=4000, rnad=3600)) == ["cnn"]


def test_device_identity_and_fresh_exec_worker_command(tmp_path):
    assert parse_devices("0,1") == ["0", "1"]
    assert parse_devices("cpu,cpu") == ["cpu", "cpu"]
    with pytest.raises(ValueError, match="only once"):
        parse_devices("0,00")
    with pytest.raises(ValueError):
        parse_devices("cuda:0,1")
    args = SimpleNamespace(output_dir=tmp_path, reference=tmp_path / "reference.pt", stop_at="2026-10-17T21:00:00Z",
                           block_seconds=600, evaluation_games=256)
    command = worker_command(args, 1, "1", "cnn")
    assert "--worker" in command
    assert command[command.index("--worker-device") + 1] == "1"
    assert command[command.index("--contender") + 1] == "cnn"
    assert "--init-from" not in command  # Continuation only, never reset the model.


def test_shared_worker_is_one_slot_with_explicit_device_command_and_rng_transition(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, reference=tmp_path / 'reference.pt',
                           stop_at='2026-10-17T21:30:00Z', block_seconds=600,
                           evaluation_games=256, devices='0,1', shared_update_devices='0,1')
    assert shared_devices(args) == ['0', '1']
    assert scheduler_devices(args) == ['0,1']
    command = worker_command(args, 0, '0,1', 'cnn_ppo')
    assert command[command.index('--worker-device') + 1] == '0,1'
    assert command[command.index('--shared-update-devices') + 1] == '0,1'
    assert validate_cuda_rng_count(1, 2, True)
    assert not validate_cuda_rng_count(2, 2, True)
    assert not validate_cuda_rng_count(1, 1, False)
    for count, visible, shared in [(2, 1, False), (0, 2, True), (3, 2, True)]:
        with pytest.raises(ValueError):
            validate_cuda_rng_count(count, visible, shared)
    validate_shared_selection(True, ['cnn_ppo'])
    for selected in [['cnn_ppo', 'cnn_rnad'], ['transformer_ppo'], []]:
        with pytest.raises(ValueError, match='only cnn_ppo'):
            validate_shared_selection(True, selected)
    for invalid in ['0', 'cpu,cpu', '0,0']:
        args.shared_update_devices = invalid
        with pytest.raises(ValueError):
            shared_devices(args)


@pytest.mark.parametrize('paused_gpu', ['0', '1'])
def test_shared_reservation_owns_both_gpus_and_either_pause_stops_sole_worker(tmp_path, paused_gpu):
    with worker_leases(tmp_path, ['0', '1'], 'cnn_ppo') as acquired:
        assert acquired
        for key in ['locks/gpu-0.lock', 'locks/gpu-1.lock', 'cnn_ppo/worker.lock']:
            with pytest.raises(RuntimeError), exclusive_lock(tmp_path / key):
                pytest.fail('Shared worker did not own all required leases')
    pause = tmp_path / f'gpu-{paused_gpu}.pause'
    pause.touch()
    with worker_leases(tmp_path, ['0', '1'], 'cnn_ppo') as acquired:
        assert not acquired
    controller, launched = scheduler(devices=('0,1',))
    progress = states(cnn=0, transformer=0, rnad=0)
    controller.tick(progress, 1, allowed=['cnn'])
    assert len(launched) == 1
    paused = paused_slots(tmp_path, ['0,1'])
    assert paused == [0]
    controller.tick(progress, 2, allowed=['cnn'], paused=paused)
    assert launched[0][3].terminations == 1
    assert len(launched) == 1
    pause.unlink()
    with worker_leases(tmp_path, ['0', '1'], 'cnn_ppo') as acquired:
        assert acquired


def test_shared_secondary_lease_conflict_releases_primary_lease(tmp_path):
    with exclusive_lock(tmp_path / 'locks/gpu-1.lock'):
        with pytest.raises(RuntimeError), worker_leases(tmp_path, ['0', '1'], 'cnn_ppo'):
            pytest.fail('Must not overlap secondary GPU owner')
        with exclusive_lock(tmp_path / 'locks/gpu-0.lock'):
            pass


def test_shared_pause_race_on_secondary_gpu_exits_cleanly(tmp_path, monkeypatch):
    import scripts.train_exact_multigpu as module

    original = module.exclusive_lock

    @contextlib.contextmanager
    def secondary_reservation(path):
        if path.name == 'gpu-1.lock':
            (tmp_path / 'gpu-1.pause').touch()
        with original(path):
            yield

    with original(tmp_path / 'locks/gpu-1.lock'):
        monkeypatch.setattr(module, 'exclusive_lock', secondary_reservation)
        with worker_leases(tmp_path, ['0', '1'], 'cnn_ppo') as acquired:
            assert not acquired
        with original(tmp_path / 'locks/gpu-0.lock'):
            pass  # Failed shared acquisition must release the first GPU.


def test_pause_appearing_during_gpu_lease_acquisition_is_clean_exit(tmp_path, monkeypatch):
    import scripts.train_exact_multigpu as module

    original = module.exclusive_lock
    pause = tmp_path / "gpu-0.pause"

    @contextlib.contextmanager
    def racing_lock(path):
        pause.touch()  # External reservation wins after the initial pause check.
        with original(path):
            yield

    with original(tmp_path / "locks/gpu-0.lock"):
        monkeypatch.setattr(module, "exclusive_lock", racing_lock)
        with worker_leases(tmp_path, "0", "cnn") as acquired:
            assert not acquired
    monkeypatch.setattr(module, "exclusive_lock", original)
    pause.unlink()
    with original(tmp_path / "locks/gpu-0.lock"), pytest.raises(RuntimeError, match="Another process owns"), worker_leases(tmp_path, "0", "cnn"):
        pytest.fail("Unpaused duplicate ownership must remain an error")


def test_actual_two_cpu_workers_checkpoint_resume_and_gracefully_stop(tmp_path):
    native = pytest.importorskip("sequence_native")
    if not hasattr(native, "BatchEnv"):
        pytest.skip("Native BatchEnv is needed for real continuation integration")
    from scripts.train_exact_policy import RECIPE, initialize
    from src.exact_policy import ExactPolicy
    from src.history_policy import HistoryPolicy, save_policy
    from src.model import SequenceNet

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        teacher = HistoryPolicy(SequenceNet(num_filters=8, num_res_blocks=1), "board", width=8).eval()
        reference = tmp_path / "reference.pt"
        save_policy(teacher, reference, {})
        recipe = {**RECIPE, "games_per_iteration": 4, "batch_size": 32, "epochs": 1,
                  "rnad_games_per_minibatch": 2, "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        initial = ExactPolicy.from_history_policy(teacher)
        for name in recipe["contenders"]:
            initialize(initial, name, tmp_path / name, recipe, "cpu")
        (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    finally:
        torch.set_num_threads(previous_threads)

    def run_until_every_contender_advances(before, suffix):
        command = [sys.executable, "-u", "scripts/train_exact_multigpu.py", "--output-dir", str(tmp_path),
                   "--reference", str(reference), "--stop-at", (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat(),
                   "--devices", "cpu,cpu", "--block-seconds", "0.1", "--evaluation-games", "2"]
        with (tmp_path / f"supervisor-{suffix}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    assert process.poll() is None, (tmp_path / f"supervisor-{suffix}.log").read_text()
                    progress = {name: tmp_path / name / "progress.json" for name in recipe["contenders"]}
                    if all(path.exists() and json.loads(path.read_text())["iterations"] > before[name]
                           for name, path in progress.items()):
                        break
                    time.sleep(0.1)
                else:
                    pytest.fail("Workers did not checkpoint every contender")
                with pytest.raises(RuntimeError, match="Another process owns"), exclusive_lock(tmp_path / "controller.lock"):
                    pytest.fail("Root lease lost while workers were active")
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=30)
                assert process.returncode == 0, (tmp_path / f"supervisor-{suffix}.log").read_text()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=30)
        final = json.loads((tmp_path / "state.json").read_text())
        assert final["status"] == "stopped" and final["workers"] == {} and final["failure"] is None
        with exclusive_lock(tmp_path / "controller.lock"):
            pass
        after = {}
        for name in recipe["contenders"]:
            payload = torch.load(tmp_path / name / "latest.pt", map_location="cpu", weights_only=True)
            assert payload["metadata"]["recipe"] == recipe
            assert payload["training_state"]["optimizer"]["state"]
            if name == "cnn_rnad":
                assert payload["training_state"]["rnad"] is not None
            after[name] = payload["metadata"]["iterations"]
            assert after[name] > before[name]
        return after

    first = run_until_every_contender_advances(dict.fromkeys(recipe["contenders"], 0), "initial")
    run_until_every_contender_advances(first, "resume")
