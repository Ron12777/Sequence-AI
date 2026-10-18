"""Resume an existing exact-policy campaign with one worker per visible GPU.

The supervisor owns controller.lock, excluding the original single-GPU trainer.
Workers expose one GPU each as cuda:0, retain the saved recipe and optimizer/RNG,
and lock both their physical GPU and contender until checkpointing and monitoring
finish. No checkpoint initialization or recipe migration happens here.

Optional shared-update mode runs one CNN worker holding every selected GPU.
Checkpoint training_seconds remains wall time; the dispatch ledger separately
records physical GPU count and allocated GPU-seconds (wall time times count).
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_STATE_CACHE = {}


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def parse_devices(value):
    devices = value.split(",")
    if not devices or any(device != "cpu" and not device.isdecimal() for device in devices):
        raise ValueError("Devices must be comma-separated GPU indices, or cpu for local checks")
    gpu_ids = [int(device) for device in devices if device != "cpu"]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("Each physical GPU may appear only once")
    return [str(int(device)) if device != "cpu" else device for device in devices]


def shared_devices(args):
    value = getattr(args, "shared_update_devices", None)
    if value is None:
        return None
    devices = parse_devices(value)
    if len(devices) < 2 or "cpu" in devices:
        raise ValueError("Shared updates require at least two distinct physical GPU indices")
    return devices


def scheduler_devices(args):
    shared = shared_devices(args)
    return [",".join(shared)] if shared else parse_devices(args.devices)


def gpu_keys(device, slot):
    return [f"cpu-{slot}"] if device == "cpu" else device.split(",")


def paused_slots(output, devices):
    return [slot for slot, device in enumerate(devices)
            if any((output / f"gpu-{key}.pause").exists() for key in gpu_keys(device, slot))]


def validate_shared_selection(shared, allowed):
    if shared and allowed != ["cnn_ppo"]:
        raise ValueError("Shared updates require selection.json to allow only cnn_ppo")


def validate_cuda_rng_count(saved_count, visible_count, shared):
    allowed = {1, visible_count} if shared else {1}
    if saved_count not in allowed:
        raise ValueError("Saved CUDA RNG state count is incompatible with worker GPU ownership")
    return bool(shared and saved_count == 1)  # Explicit single-to-shared transition.


@contextlib.contextmanager
def worker_leases(output, gpu_key, contender):
    keys = [gpu_key] if isinstance(gpu_key, str) else list(gpu_key)
    pauses = [output / f"gpu-{key}.pause" for key in keys]
    if any(pause.exists() for pause in pauses):
        yield False
        return
    with contextlib.ExitStack() as stack:
        try:
            for key in sorted(keys):
                stack.enter_context(exclusive_lock(output / "locks" / f"gpu-{key}.lock"))
        except RuntimeError:
            # An external reservation may win the race after child dispatch.
            if any(pause.exists() for pause in pauses):
                yield False
                return
            raise
        stack.enter_context(exclusive_lock(output / contender / "worker.lock"))
        yield not any(pause.exists() for pause in pauses)


class Scheduler:
    """Process lifecycle independent of CUDA, allowing deterministic CPU tests."""

    def __init__(self, devices, contenders, launch, deadline, reserve_seconds=15):
        self.devices = list(devices)
        self.contenders = list(contenders)
        self.launch = launch
        self.deadline = deadline
        self.reserve_seconds = reserve_seconds
        self.jobs = {}
        self.stopping = False
        self.reason = None
        self.failure = None
        self.completed = []

    def request_stop(self, reason="signal"):
        self.stopping = True
        self.reason = self.reason or reason

    def tick(self, states, now, allowed=None, paused=()):
        reaped = False
        for slot, job in list(self.jobs.items()):
            status = job["process"].poll()
            if status is None:
                continue
            close = getattr(job["process"], "close", None)
            if close:
                close()
            self.completed.append({"slot": slot, "contender": job["contender"], "exit_code": status})
            del self.jobs[slot]
            reaped = True
            if status != 0:
                self.failure = f"Worker {slot} ({job['contender']}) exited with status {status}"
                self.request_stop("worker_failure")
        if now >= self.deadline - self.reserve_seconds:
            self.request_stop("training_deadline")
        if self.stopping:
            for job in self.jobs.values():
                if not job["stop_sent"]:
                    job["process"].terminate()
                    job["stop_sent"] = True
            return
        allowed = self.contenders if allowed is None else list(allowed)
        paused = set(paused)
        for slot, job in self.jobs.items():
            if slot in paused and not job["stop_sent"]:
                job["process"].terminate()
                job["stop_sent"] = True
        if reaped:
            return  # Reload authoritative checkpoints before making a new assignment.
        active = {job["contender"] for job in self.jobs.values()}
        for slot, device in enumerate(self.devices):
            if slot in self.jobs or slot in paused:
                continue
            candidates = [name for name in allowed if name not in active]
            if not candidates:
                break
            contender = min(candidates, key=lambda name: (states[name]["training_seconds"], self.contenders.index(name)))
            process = self.launch(slot, device, contender)
            self.jobs[slot] = {"process": process, "contender": contender, "device": device,
                               "started_epoch": now, "stop_sent": False}
            active.add(contender)

    def snapshot(self):
        return {"status": "stopping" if self.stopping and self.jobs else "stopped" if self.stopping else "training",
                "reason": self.reason, "failure": self.failure,
                "workers": {str(slot): {key: value for key, value in job.items() if key != "process"}
                            | {"pid": getattr(job["process"], "pid", None)} for slot, job in self.jobs.items()},
                "completed_blocks": len(self.completed), "last_completed": self.completed[-len(self.devices):]}


class WorkerProcess:
    def __init__(self, process, log, stop_file):
        self.process, self.log, self.pid = process, log, process.pid
        self.stop_file = stop_file

    def poll(self):
        return self.process.poll()

    def terminate(self):
        # A per-dispatch request also works during Python startup/teardown,
        # when SIGTERM could otherwise bypass the worker's checkpoint handler.
        self.stop_file.touch()

    def close(self):
        self.log.close()
        self.stop_file.unlink(missing_ok=True)


def worker_command(args, slot, device, contender):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--output-dir", str(args.output_dir),
            "--reference", str(args.reference), "--stop-at", args.stop_at,
            "--block-seconds", str(args.block_seconds), "--evaluation-games", str(args.evaluation_games),
            "--worker-slot", str(slot), "--worker-device", device, "--contender", contender,
            "--parent-pid", str(os.getpid())]
    if shared_devices(args):
        command += ["--shared-update-devices", ",".join(shared_devices(args))]
    return command


def read_states(output, contenders):
    states = {}
    for contender in contenders:
        path = output / contender / "latest.pt"
        stat = path.stat()
        identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        cached = _STATE_CACHE.get(str(path))
        if cached is None or cached[0] != identity:
            import torch
            metadata = torch.load(path, map_location="cpu", weights_only=True)["metadata"]
            _STATE_CACHE[str(path)] = (identity, metadata)
        states[contender] = _STATE_CACHE[str(path)][1]
    return states


def selected_contenders(output, contenders, states):
    path = output / "selection.json"
    if not path.exists():
        return contenders
    selected = json.loads(path.read_text())["contenders"]
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(contenders):
        raise ValueError("Invalid contender selection")
    if len(selected) < len(contenders) and any(state["training_seconds"] < 3600 for state in states.values()):
        raise ValueError("All contenders require one matched training hour before narrowing")
    return selected


def validate_existing_campaign(args):
    recipe = json.loads((args.output_dir / "recipe.json").read_text())
    contenders = recipe["contenders"]
    if not contenders or len(contenders) != len(set(contenders)):
        raise ValueError("Invalid saved contender list")
    for contender in contenders:
        if Path(contender).name != contender or contender in (".", ".."):
            raise ValueError("Invalid contender path")
        directory = args.output_dir / contender
        if not (directory / "latest.pt").is_file() or not list(directory.glob("historical_*.pt")):
            raise ValueError(f"Missing continuation checkpoint/history for {contender}")
    digest = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    if digest != recipe["reference_sha256"]:
        raise ValueError("Reference differs from the preserved campaign recipe")
    return recipe


def run_worker(args, deadline):
    shared = shared_devices(args)
    if shared and (args.contender != "cnn_ppo" or args.worker_device != ",".join(shared) or args.worker_slot != 0):
        raise ValueError("Shared worker must be the sole slot0 cnn_ppo assignment")
    keys = gpu_keys(args.worker_device, args.worker_slot)
    with worker_leases(args.output_dir, keys, args.contender) as acquired:
        if not acquired:
            return
        stopped = threading.Event()
        trainer = None

        def stop(*_):
            stopped.set()
            if trainer is not None:
                trainer.STOP_REQUESTED = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if args.stop_file.exists():
            return  # A stop during startup leaves the previous checkpoint intact.
        # Default workers expose one GPU; explicit shared workers own all visible GPUs.
        expected_visible = "" if args.worker_device == "cpu" else args.worker_device
        if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
            raise ValueError("Worker CUDA visibility does not match its exclusively assigned device")
        import torch

        import scripts.train_exact_policy as trainer_module
        from scripts.overnight import atomic_copy, digest
        from src.history_policy import load_policy

        trainer = trainer_module
        torch.set_num_threads(1)
        trainer.STOP_REQUESTED = stopped.is_set()
        recipe = validate_existing_campaign(args)
        if args.contender not in recipe["contenders"]:
            raise ValueError("Worker contender is not in the saved recipe")
        device = "cpu" if args.worker_device == "cpu" else "cuda:0"
        gpu_count = len(shared) if shared else 1
        if device != "cpu" and torch.cuda.device_count() != gpu_count:
            raise RuntimeError("Visible CUDA count does not match exclusively owned devices")
        payload = torch.load(args.output_dir / args.contender / "latest.pt", map_location="cpu", weights_only=True)
        cuda_rng = payload["training_state"]["cuda_rng"]
        rng_transition = validate_cuda_rng_count(len(cuda_rng), gpu_count, shared) if device != "cpu" else False
        if shared and (payload['metadata'].get('variant') != 'cnn' or payload['metadata'].get('algorithm') != 'ppo'):
            raise ValueError("Shared update checkpoint must be CNN/PPO")
        before = payload["metadata"]["training_seconds"]
        del payload, cuda_rng
        dispatch_started = time.time()
        ledger_dir = args.output_dir / "hardware_dispatches"
        ledger_dir.mkdir(exist_ok=True)
        ledger_path = ledger_dir / f"{time.time_ns()}-{args.contender}.json"
        ledger = {"contender": args.contender, "physical_device": args.worker_device, "worker_slot": args.worker_slot,
                  "device_name": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
                  "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "pid": os.getpid(),
                  "started_utc": datetime.now(timezone.utc).isoformat(), "training_seconds_before": before,
                  "block_seconds_requested": args.block_seconds, "status": "loading",
                  "accounting": "Single-GPU allocated training seconds; overlapping node billing is separate. Hardware changes are not bitwise reproducible."}
        if shared:
            ledger.update(physical_devices=shared, gpu_count=gpu_count, shared_update=True,
                          single_to_shared_rng_transition=rng_transition,
                          accounting="Checkpoint training_seconds and training_wall_seconds are wall time; allocated_gpu_seconds equals wall time times exclusively owned GPU count. This is allocated cost, not measured kernel activity or equivalent FLOPs.")
        atomic_json(ledger_path, ledger)

        def watch_parent():
            while not stopped.wait(0.25):
                if args.stop_file.exists() or os.getppid() != args.parent_pid:
                    stop()
                    return

        watcher = threading.Thread(target=watch_parent, daemon=True)
        watcher.start()
        if stopped.is_set() or time.time() >= deadline - 15:
            ledger.update(status="stopped_before_training", worker_wall_seconds=time.time() - dispatch_started)
            atomic_json(ledger_path, ledger)
            return
        reference, _ = load_policy(args.reference, device)
        reference.requires_grad_(False)
        directory = args.output_dir / args.contender
        options = {"parallel_update_devices": list(range(gpu_count))} if shared else {}
        model, state = trainer.train_block(directory, reference, device, recipe, args.block_seconds, deadline, **options)
        if shared:
            saved = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
            if len(saved['training_state']['cuda_rng']) != gpu_count:
                raise RuntimeError("Shared continuation did not save every visible CUDA RNG state")
            del saved
        monitor_started = time.monotonic()
        if not stopped.is_set() and time.time() < deadline:
            seed = recipe["monitor_seed"] + state["blocks"] * 1000000
            monitor = trainer.evaluate(model, reference, device, args.evaluation_games, seed)
            report = dict(state, result=monitor, seed=seed, monitoring_only=True, production_promoted=False,
                          checkpoint_sha256=digest(directory / "latest.pt"))
            atomic_json(directory / f"evaluation_{state['blocks']:03d}.json", report)
            best_file = directory / "best_monitor.json"
            best = json.loads(best_file.read_text()) if best_file.exists() else None
            if best is None or monitor["score"] > best["score"]:
                atomic_copy(directory / "latest.pt", directory / "best_monitor.pt")
                atomic_json(best_file, {"score": monitor["score"], "block": state["blocks"], "monitoring_only": True})
            print(json.dumps({"contender": args.contender, "training_seconds": state["training_seconds"],
                              "monitor_score": monitor["score"]}), flush=True)
        ledger.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      training_seconds_after=state["training_seconds"],
                      training_gpu_seconds=state["training_seconds"] - before,
                      monitor_seconds=time.monotonic() - monitor_started,
                      worker_wall_seconds=time.time() - dispatch_started, stopped_by_signal=stopped.is_set())
        if shared:
            wall = state['training_seconds'] - before
            ledger.update(training_wall_seconds=wall, allocated_gpu_seconds=wall * gpu_count,
                          training_gpu_seconds=wall * gpu_count)
        atomic_json(ledger_path, ledger)
        stopped.set()


def run_supervisor(args, deadline):
    output = args.output_dir
    with exclusive_lock(output / "controller.lock"), exclusive_lock(output / "supervisor.lock"):
        recipe = validate_existing_campaign(args)
        shared = shared_devices(args)
        devices = scheduler_devices(args)
        if shared:
            validate_shared_selection(shared, selected_contenders(output, recipe['contenders'], read_states(output, recipe['contenders'])))
        # A dead supervisor may leave a worker finishing its final save. Refuse
        # to overlap it even though its controller.lock was released on death.
        with contextlib.ExitStack() as stack:
            for contender in recipe["contenders"]:
                stack.enter_context(exclusive_lock(output / contender / "worker.lock"))
            for slot, device in enumerate(devices):
                for key in gpu_keys(device, slot):
                    if not (output / f"gpu-{key}.pause").exists():
                        stack.enter_context(exclusive_lock(output / "locks" / f"gpu-{key}.lock"))
        logs = output / "worker_logs"
        logs.mkdir(exist_ok=True)
        controls = output / "worker_control"
        controls.mkdir(exist_ok=True)

        def launch(slot, device, contender):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if device == "cpu" else device,
                               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            log = (logs / f"gpu-{slot}.log").open("a", buffering=1)
            stop_file = controls / f"{slot}-{time.time_ns()}.stop"
            try:
                command = worker_command(args, slot, device, contender) + ["--stop-file", str(stop_file)]
                process = subprocess.Popen(command, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            return WorkerProcess(process, log, stop_file)

        scheduler = Scheduler(devices, recipe["contenders"], launch, deadline)
        signal.signal(signal.SIGTERM, lambda *_: scheduler.request_stop())
        signal.signal(signal.SIGINT, lambda *_: scheduler.request_stop())
        previous = None
        try:
            while True:
                states = read_states(output, recipe["contenders"])
                allowed = selected_contenders(output, recipe["contenders"], states)
                validate_shared_selection(shared, allowed)
                paused = paused_slots(output, devices)
                scheduler.tick(states, time.time(), allowed, paused)
                snapshot = scheduler.snapshot() | {"paused_slots": paused}
                if snapshot != previous:
                    atomic_json(output / "state.json", snapshot | {"controller": "multi_gpu", "stop_at": args.stop_at,
                                                                  "updated_utc": datetime.now(timezone.utc).isoformat()})
                    print(json.dumps(snapshot), flush=True)
                    previous = snapshot
                if scheduler.stopping and not scheduler.jobs:
                    break
                time.sleep(0.25)
        except BaseException as error:
            scheduler.failure = str(error)
            scheduler.request_stop("supervisor_failure")
            raise
        finally:
            scheduler.request_stop("supervisor_exit")
            # Preserve continuation state: SIGTERM requests the normal final save.
            # Do not kill workers midway through an optimizer step/checkpoint.
            while scheduler.jobs:
                scheduler.tick({}, time.time())
                time.sleep(0.25)
            atomic_json(output / "state.json", scheduler.snapshot() | {"controller": "multi_gpu", "stop_at": args.stop_at,
                                                                       "updated_utc": datetime.now(timezone.utc).isoformat()})
        if scheduler.failure:
            raise RuntimeError(scheduler.failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--shared-update-devices", help="Opt-in one CNN worker owning these GPUs for shared-model updates, e.g.0,1")
    parser.add_argument("--block-seconds", type=float, default=600)
    parser.add_argument("--evaluation-games", type=int, default=256)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-slot", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-device", help=argparse.SUPPRESS)
    parser.add_argument("--contender", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--stop-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    shared_devices(args)  # Validate explicit transition before acquiring any leases.
    if args.block_seconds <= 0 or args.evaluation_games < 2 or args.evaluation_games % 2:
        parser.error("Positive block duration and a positive even evaluation count are required")
    if args.worker and (args.worker_slot is None or args.worker_device is None or args.contender is None or args.parent_pid is None or args.stop_file is None):
        parser.error("Incomplete internal worker assignment")
    args.output_dir, args.reference = args.output_dir.resolve(), args.reference.resolve()
    deadline = datetime.fromisoformat(args.stop_at.replace("Z", "+00:00")).timestamp()
    if time.time() >= deadline:
        raise SystemExit("Training deadline has passed")
    (run_worker if args.worker else run_supervisor)(args, deadline)


if __name__ == "__main__":
    main()
