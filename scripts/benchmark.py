"""Measure end-to-end self-play throughput on one GPU; emit machine-readable evidence."""

import argparse
import json
import multiprocessing as mp
import queue
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import create_model
from src.train import worker_process, InferenceServer
from src.inference_transport import InferenceSlot


def gpu_sample():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    util, memory, watts = map(float, result.stdout.strip().splitlines()[0].split(","))
    return {"utilization": util, "memory_mib": memory, "watts": watts}


def benchmark(
    workers,
    games,
    filters,
    blocks,
    seconds,
    simulations=16,
    compact_observations=False,
    shared_inference=False,
):
    torch.manual_seed(42)
    model, device = create_model(
        "cuda", compile_model=False, num_filters=filters, num_res_blocks=blocks
    )
    model.to(memory_format=torch.channels_last)
    requests, completed = mp.Queue(), mp.Queue()
    responses = {i: mp.Queue() for i in range(workers)}
    ready = [mp.Event() for _ in range(workers)]
    processes = []
    slots = (
        {i: InferenceSlot(games) for i in range(workers)} if shared_inference else None
    )
    try:
        for i in range(workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    i,
                    requests,
                    responses[i],
                    completed,
                    ready[i],
                    {
                        "games_per_worker": games,
                        "simulations": simulations,
                        "seed": 42,
                        "compact_observations": compact_observations,
                        "search_mode": "ensemble",
                        "inference_slot": slots[i] if slots else None,
                    },
                ),
            )
            p.start()
            processes.append(p)
        for event in ready:
            if not event.wait(90):
                raise RuntimeError("worker startup timed out")
        server = InferenceServer(
            model, device, responses, batch_size=4096, timeout=0.002, shared_slots=slots
        )
        # Warm up kernels and workers before measuring.
        warmup = time.monotonic() + 5
        while time.monotonic() < warmup:
            server.step(requests)
            # Do not count games completed before the measurement window.
            while True:
                try:
                    completed.get_nowait()
                except queue.Empty:
                    break
        initial = server.total_processed
        initial_batches = server.total_batches
        start = time.monotonic()
        last_sample = 0.0
        games_done = positions = 0
        samples = []
        while time.monotonic() - start < seconds:
            if any(not p.is_alive() for p in processes):
                raise RuntimeError("worker exited")
            server.step(requests)
            while True:
                try:
                    history = completed.get_nowait()
                    games_done += 1
                    positions += len(history)
                except queue.Empty:
                    break
            if time.monotonic() - last_sample > 1:
                samples.append(gpu_sample())
                last_sample = time.monotonic()
        elapsed = time.monotonic() - start
        count = server.total_processed - initial
        return {
            "workers": workers,
            "games_per_worker": games,
            "filters": filters,
            "blocks": blocks,
            "simulations": simulations,
            "transport": "shared" if shared_inference else "queue",
            "search_mode": "ensemble",
            "observation_dtype": "uint8" if compact_observations else "float32",
            "seconds": elapsed,
            "leaf_evaluations_per_second": count / elapsed,
            "mean_inference_batch": count
            / max(1, server.total_batches - initial_batches),
            "completed_games": games_done,
            "completed_positions": positions,
            "gpu_util_mean": sum(s["utilization"] for s in samples) / len(samples),
            "gpu_memory_peak_mib": max(s["memory_mib"] for s in samples),
            "gpu_power_mean_w": sum(s["watts"] for s in samples) / len(samples),
        }
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join(10)
        for q in [requests, completed, *responses.values()]:
            q.cancel_join_thread()
            q.close()
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--compact-observations", action="store_true")
    parser.add_argument("--shared-inference", action="store_true")
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--output", default="artifacts/benchmark.jsonl")
    parser.add_argument(
        "--configs", default="4:64:64:3,8:64:64:3,12:64:64:3,8:128:64:3,8:128:128:6"
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.simulations <= 0:
        parser.error("--seconds and --simulations must be positive")
    if args.shared_inference and args.compact_observations:
        parser.error(
            "--shared-inference uses float32 slots; omit --compact-observations"
        )
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for config in args.configs.split(","):
        workers, games, filters, blocks = map(int, config.split(":"))
        if min(workers, games, filters, blocks) <= 0:
            parser.error("config values must be positive")
        result = benchmark(
            workers,
            games,
            filters,
            blocks,
            args.seconds,
            compact_observations=args.compact_observations,
            shared_inference=args.shared_inference,
            simulations=args.simulations,
        )
        print(json.dumps(result), flush=True)
        with output.open("a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
