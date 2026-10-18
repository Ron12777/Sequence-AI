"""Compare incremental learning per rented GPU-hour, with exclusive GPU use.

All candidates inherit the same champion CNN. Pilot throughput tuning has the
same configuration search for each architecture and is reported separately.
This is a conditional warm-start comparison, not a from-scratch scaling claim.
"""

import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.fast_game import FastSequenceGame
from src.history_policy import (
    HistoryPolicy,
    apply_public_move,
    load_policy,
    save_policy,
)
from src.model import load_model
from scripts.architecture_experiments import direct_moves, evaluate_policy
from scripts.arena import choose_moves
from scripts.benchmark import gpu_sample
from scripts.direct_selfplay import rollout, update
from scripts.overnight import atomic_copy, atomic_json, digest


def process_identity(pid):
    """Start ticks disambiguate a live process from a reused PID."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]
    except FileNotFoundError:
        return None


def descendants(pid):
    result = []
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except FileNotFoundError:
        return result
    for child in map(int, children):
        result.append(child)
        result.extend(descendants(child))
    return result


def local_gpu_users():
    """NVML reports host PIDs on Vast; inspect driver handles in this namespace."""
    users = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().split(b"\0")[0]
            if Path(os.fsdecode(command)).name == "nvidia-smi":
                continue  # Read-only telemetry also opens driver handles.
            handles = {os.readlink(fd) for fd in (proc / "fd").iterdir()}
            if "/dev/nvidia-uvm" in handles:
                users.add(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return users


@contextmanager
def exclusive_window(controller_pid, maximum_seconds, output):
    """Pause one verified campaign tree; an independent guard always resumes it."""
    command = Path(f"/proc/{controller_pid}/cmdline").read_bytes()
    if b"scripts/overnight.py" not in command:
        raise ValueError("Refusing to pause an unrelated process")
    records = [(controller_pid, process_identity(controller_pid))]
    record_path = output / "paused_processes.json"
    atomic_json(record_path, records)
    # Guard survives SIGKILL of this supervisor. Check both PID and start ticks.
    guard_code = """
import json,os,pathlib,signal,sys,time
parent,start,path,limit=sys.argv[1:]
def identity(pid):
 try: return pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError: return None
until=time.monotonic()+float(limit)
while identity(parent)==start and time.monotonic()<until: time.sleep(1)
for pid,born in json.loads(pathlib.Path(path).read_text()):
 if identity(pid)==born:
  try: os.kill(pid,signal.SIGCONT)
  except ProcessLookupError: pass
"""
    guard = subprocess.Popen(
        [
            sys.executable,
            "-c",
            guard_code,
            str(os.getpid()),
            process_identity(os.getpid()),
            str(record_path),
            str(maximum_seconds),
        ],
        start_new_session=True,
    )
    try:
        os.kill(controller_pid, signal.SIGSTOP)
        # Stop descendants individually and rescan to capture in-flight spawns.
        for _ in range(3):
            known = {pid for pid, _ in records}
            new = [p for p in descendants(controller_pid) if p not in known]
            for pid in new:
                born = process_identity(pid)
                if born is not None:
                    records.append((pid, born))
            atomic_json(record_path, records)
            for pid, born in records:
                if process_identity(pid) == born:
                    os.kill(pid, signal.SIGSTOP)
        yield
    finally:
        # Resume children before the controller can inspect their status.
        for pid, born in reversed(records):
            if process_identity(pid) == born:
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        guard.terminate()
        guard.wait(timeout=10)


class Telemetry:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                self.samples.append({"unix": time.time(), **gpu_sample()})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def fresh(base, architecture, seed):
    torch.manual_seed(seed)
    return HistoryPolicy(base, architecture).to("cuda")


def tune(base, architecture, seed):
    rows = []
    # Same search space and number of trials for every architecture. Pilot
    # weights are discarded, so tuning cannot add hidden training to a trial.
    for games, batch in ((128, 256), (512, 1024)):
        model = fresh(base, architecture, seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        torch.cuda.synchronize()
        start = time.monotonic()
        data, info = rollout(model, "cuda", games, seed)
        info.update(update(model, optimizer, data, "cuda", batch_size=batch))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        rows.append(
            {
                "games": games,
                "batch_size": batch,
                "seconds": elapsed,
                "positions_per_second": info["positions"] / elapsed,
                "updates": info["updates"],
            }
        )
        del data, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    return max(rows, key=lambda x: x["positions_per_second"]), rows


def train_budget(base, architecture, seed, config, seconds, output):
    output.mkdir()
    # Initialization, self-play, updates, and checkpoint I/O all consume budget.
    start = time.monotonic()
    model = fresh(base, architecture, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    latest = output / "latest.pt"
    save_policy(model, latest, {"seed": seed, "iteration": 0})
    rows = []
    accepted = 0
    with Telemetry() as telemetry:
        while time.monotonic() - start < seconds:
            iteration = len(rows)
            if seed + iteration * 10000 + config["games"] >= 3000000000:
                raise ValueError(
                    "Training seed range would reach reserved evaluation deals"
                )
            data, row = rollout(
                model, "cuda", config["games"], seed + iteration * 10000
            )
            row.update(
                update(model, optimizer, data, "cuda", batch_size=config["batch_size"])
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            row.update(
                iteration=iteration + 1, seconds=elapsed, eligible=elapsed <= seconds
            )
            rows.append(row)
            if row["eligible"]:
                accepted += 1
                save_policy(
                    model,
                    latest,
                    {
                        "seed": seed,
                        "iteration": accepted,
                        "training_seconds": elapsed,
                        "budget_seconds": seconds,
                    },
                )
            # A last iteration may cross the boundary; its weights are discarded.
            # Report the overrun as actual cost, never credit it to the model.
            atomic_json(output / "metrics.json", rows)
            del data
        actual = time.monotonic() - start
    report = {
        "architecture": architecture,
        "seed": seed,
        "config": config,
        "allocated_gpu_hours": seconds / 3600,
        "actual_gpu_hours": actual / 3600,
        "actual_seconds": actual,
        "budget_seconds": seconds,
        "accepted_iterations": accepted,
        "training_games": sum(r["games"] for r in rows if r["eligible"]),
        "training_positions": sum(r["positions"] for r in rows if r["eligible"]),
        "optimizer_updates": sum(r["updates"] for r in rows if r["eligible"]),
        "gpu_util_mean": statistics.mean(s["utilization"] for s in telemetry.samples)
        if telemetry.samples
        else None,
        "telemetry": telemetry.samples,
        "checkpoint_sha256": digest(latest),
    }
    atomic_json(output / "training.json", report)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def latency(model, teacher):
    # Fixed real positions at several game stages; both agents see exactly the
    # same positions. Batch=1 measures UI latency; batch=128 measures throughput.
    games, events = [], []
    for i in range(128):
        random.seed(82190001 + i)
        game, history = FastSequenceGame(), []
        for _ in range((i % 4) * 8):
            moves = game.get_legal_moves()
            if game.game_over or not moves:
                break
            apply_public_move(game, history, random.choice(moves))
        if game.game_over or not game.get_legal_moves():
            game, history = FastSequenceGame(), []
        games.append(game)
        events.append(history)
    result = []
    for batch in (1, 128):
        ids = list(range(batch))
        counts = [len(h) for h in events]
        for method in (
            "candidate_direct",
            "champion_direct",
            "champion_4",
            "champion_16",
            "champion_128",
        ):

            def run():
                if method == "candidate_direct":
                    return direct_moves(model, "cuda", games, events, ids)
                sims = 0 if method == "champion_direct" else int(method.split("_")[-1])
                return choose_moves(
                    teacher,
                    torch.device("cuda"),
                    games,
                    ids,
                    sims,
                    912719,
                    counts,
                    "ensemble",
                )

            run()
            samples = []
            for repeat in range(8):
                if batch == 1:
                    ids = [repeat % 4]
                torch.cuda.synchronize()
                start = time.monotonic()
                run()
                torch.cuda.synchronize()
                samples.append(time.monotonic() - start)
            result.append(
                {
                    "batch": batch,
                    "method": method,
                    "median_batch_ms": statistics.median(samples) * 1000,
                    "amortized_ms_per_move": statistics.median(samples) * 1000 / batch,
                }
            )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--controller-pid", type=int, required=True)
    p.add_argument("--seconds", type=float, default=180)
    p.add_argument("--seeds", default="1200000001,1400000001")
    p.add_argument("--evaluation-games", type=int, default=1024)
    p.add_argument("--window-seconds", type=int, default=1800)
    a = p.parse_args()
    if (
        a.seconds <= 0
        or a.evaluation_games < 4
        or a.evaluation_games % 4
        or a.window_seconds <= 30
    ):
        p.error("Positive time and paired evaluation games required")
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock = (output.parent / "compute_experiments.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupt(*_):
        raise KeyboardInterrupt("Bounded experiment interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.alarm(a.window_seconds - 20)
    torch.set_num_threads(1)
    manifest = {
        "config": vars(a),
        "started_unix": time.time(),
        "status": "running",
        "training_comparison": "Equal incremental rented GPU-time, common pretrained backbone",
        "not_claimed": "Equal from-scratch compute or equal inference cost",
        "runs": [],
    }
    try:
        with exclusive_window(a.controller_pid, a.window_seconds, output):
            gpu_pids = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            paused = json.loads((output / "paused_processes.json").read_text())
            local_users = local_gpu_users()
            unexpected = local_users - {pid for pid, _ in paused} - {os.getpid()}
            if unexpected:
                raise RuntimeError(
                    f"GPU is used by other processes: {sorted(unexpected)}"
                )
            host_users = {int(p) for p in gpu_pids.split() if p.isdigit()}
            if len(host_users) > len(local_users):
                raise RuntimeError(
                    "GPU compute users exist outside the visible process namespace"
                )
            manifest["isolation"] = {
                "nvml_host_pids": sorted(host_users),
                "container_gpu_pids": sorted(local_users),
                "paused_pids": [pid for pid, _ in paused],
                "note": "Host and container PIDs differ; all visible CUDA users are paused and their count covers NVML compute users.",
            }
            if torch.cuda.device_count() != 1:
                raise RuntimeError("This comparison requires exactly one GPU")
            frozen = output / "teacher.pt"
            atomic_copy(a.champion, frozen)
            manifest["teacher_sha256"] = digest(frozen)
            base, _, _ = load_model(frozen, device="cuda", compile_model=False)
            control = fresh(base, "board", 910000001).eval()
            manifest["initial_control"] = evaluate_policy(
                control,
                base,
                torch.device("cuda"),
                a.evaluation_games,
                3100100001,
                "champion",
                128,
            )
            del control
            architectures = ["board", "gru", "joint_attention"]
            configs = {}
            manifest["pilots"] = {}
            for architecture in architectures:
                configs[architecture], rows = tune(base, architecture, 910000001)
                manifest["pilots"][architecture] = rows
                atomic_json(output / "manifest.json", manifest)
            for replica, seed in enumerate(map(int, a.seeds.split(","))):
                # Reverse order for the second replicate to reduce order effects.
                for architecture in architectures[:: 1 if replica % 2 == 0 else -1]:
                    directory = output / f"{architecture}_seed{replica}"
                    report = train_budget(
                        base,
                        architecture,
                        seed,
                        configs[architecture],
                        a.seconds,
                        directory,
                    )
                    evaluation_start = time.monotonic()
                    model, _ = load_policy(directory / "latest.pt", "cuda")
                    report["latency"] = latency(model, base)
                    report["evaluations"] = []
                    # Training seeds stay below 2e9 with the bounded iteration
                    # count. Fixed >3e9 evaluation seeds are identical across models.
                    for opponent, sims, offset in (
                        ("champion", 0, 0),
                        ("champion", 128, 100000),
                        ("greedy", 0, 200000),
                    ):
                        result = evaluate_policy(
                            model,
                            base,
                            torch.device("cuda"),
                            a.evaluation_games,
                            3100000001 + offset,
                            opponent,
                            sims,
                        )
                        report["evaluations"].append(result)
                        atomic_json(directory / "results.json", report)
                    report["evaluation_and_latency_seconds"] = (
                        time.monotonic() - evaluation_start
                    )
                    atomic_json(directory / "results.json", report)
                    manifest["runs"].append(
                        {
                            "architecture": architecture,
                            "replica": replica,
                            "directory": str(directory),
                            "actual_gpu_hours": report["actual_gpu_hours"],
                            "scores": [e["score"] for e in report["evaluations"]],
                        }
                    )
                    atomic_json(output / "manifest.json", manifest)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                # Compare architecture changes directly with the equally trained
                # CNN control, not only with a much stronger search opponent.
                board_control, _ = load_policy(
                    output / f"board_seed{replica}" / "latest.pt", "cuda"
                )
                for architecture in ("gru", "joint_attention"):
                    model, _ = load_policy(
                        output / f"{architecture}_seed{replica}" / "latest.pt", "cuda"
                    )
                    pair = evaluate_policy(
                        model,
                        board_control.base,
                        torch.device("cuda"),
                        a.evaluation_games,
                        3200000001,
                        "champion",
                        0,
                    )
                    pair.update(
                        architecture=architecture,
                        replica=replica,
                        opponent="equal_training_budget_board",
                        inference_compute_equal=False,
                    )
                    manifest.setdefault("equal_training_budget_pairs", []).append(pair)
                    atomic_json(output / "manifest.json", manifest)
                    del model
                del board_control
                gc.collect()
                torch.cuda.empty_cache()
            manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="interrupted", error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        manifest["finished_unix"] = time.time()
        atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
