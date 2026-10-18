"""Continuous, resumable public-information PPO with matched training-time blocks.

Alternate two architectures without resetting their optimizers. Keep current
training state, historical opponents, and monitoring-selected weights separate.
Monitoring is not independent promotion evidence; final confirmation is external.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.history_policy import HistoryPolicy, load_policy, save_policy
from src.model import load_model
from scripts.architecture_experiments import evaluate_policy
from scripts.compute_budget_experiments import Telemetry
from scripts.direct_selfplay import rollout, update
from scripts.overnight import atomic_copy, atomic_json, digest


RECIPE = {
    "architectures": ["board", "gru"],
    "games_per_iteration": 512,
    "batch_size": 1024,
    "epochs_per_update": 2,
    "initial_lr": 1e-5,
    "minimum_lr": 2e-6,
    "schedule_training_hours": 8,
    "ppo_clip": 0.1,
    "target_kl": 0.015,
    "initial_entropy_coefficient": 0.003,
    "final_entropy_coefficient": 0.0005,
    "opponents": "50% current self-play,25% frozen champion direct,25% historical policy; both learner seats balanced",
    "historical_pool_size": 3,
    "checkpoint_seconds": 60,
    "seed": 4100000001,
    "monitor_seed": 1000000000001,
    "final_confirmation_seed_reserved": 9000000000001,
}


def schedule(seconds, recipe=RECIPE):
    fraction = min(1.0, seconds / (recipe["schedule_training_hours"] * 3600))
    lr = (
        recipe["minimum_lr"]
        + (recipe["initial_lr"] - recipe["minimum_lr"])
        * (1 + math.cos(math.pi * fraction))
        / 2
    )
    entropy = (
        recipe["initial_entropy_coefficient"] * (1 - fraction)
        + recipe["final_entropy_coefficient"] * fraction
    )
    return lr, entropy


def checkpoint(model, optimizer, path, state, recipe):
    save_policy(
        model,
        path,
        {"recipe": recipe, **state},
        training_state={
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_rng": random.getstate(),
        },
    )


def restore(path, device):
    model, metadata = load_policy(path, device)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    training = payload["training_state"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=metadata["recipe"]["initial_lr"]
    )
    optimizer.load_state_dict(training["optimizer"])
    # Constructors and model loading consume RNG: restore only after all of them.
    torch.set_rng_state(training["torch_rng"])
    if training["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(training["cuda_rng"])
    random.setstate(training["python_rng"])
    state = {k: v for k, v in metadata.items() if k != "recipe"}
    return model, optimizer, state


def initialize(base, architecture, directory, recipe):
    directory.mkdir(exist_ok=True)
    if (directory / "latest.pt").exists():
        return
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    model = HistoryPolicy(base, architecture).to(next(base.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["initial_lr"])
    checkpoint(
        model,
        optimizer,
        directory / "latest.pt",
        {
            "architecture": architecture,
            "iterations": 0,
            "games": 0,
            "positions": 0,
            "updates": 0,
            "training_seconds": 0.0,
            "blocks": 0,
        },
        recipe,
    )
    save_policy(model, directory / "historical_0.pt", {"iteration": 0})
    atomic_copy(directory / "latest.pt", directory / "best_monitor.pt")


def train_block(directory, reference, recipe, seconds, deadline, device):
    # Load historical models before restore so their constructors cannot perturb
    # the resumed learner's RNG stream.
    historical = [
        load_policy(p, device)[0] for p in sorted(directory.glob("historical_*.pt"))
    ]
    model, optimizer, state = restore(directory / "latest.pt", device)
    previous = state["training_seconds"]
    start = time.monotonic()
    last_checkpoint = start
    with Telemetry() as telemetry:
        while time.monotonic() - start < seconds and time.time() < deadline - 30:
            step_start = time.monotonic()
            lr, entropy = schedule(state["training_seconds"], recipe)
            for group in optimizer.param_groups:
                group["lr"] = lr
            seed = recipe["seed"] + state["iterations"] * 10000
            if seed + recipe["games_per_iteration"] >= recipe["monitor_seed"]:
                raise ValueError("Training seeds reached the evaluation reservation")
            data, row = rollout(
                model,
                device,
                recipe["games_per_iteration"],
                seed,
                opponents=[reference, random.choice(historical)],
            )
            row.update(
                update(
                    model,
                    optimizer,
                    data,
                    device,
                    epochs=recipe["epochs_per_update"],
                    batch_size=recipe["batch_size"],
                    entropy_coefficient=entropy,
                    clip=recipe["ppo_clip"],
                    target_kl=recipe["target_kl"],
                )
            )
            torch.cuda.synchronize()
            state["iterations"] += 1
            state["games"] += row["games"]
            state["positions"] += row["positions"]
            state["updates"] += row["updates"]
            state["training_seconds"] = previous + time.monotonic() - start
            row.update(
                **state,
                learning_rate=lr,
                entropy_coefficient=entropy,
                iteration_seconds=time.monotonic() - step_start,
            )
            with (directory / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if time.monotonic() - last_checkpoint >= recipe["checkpoint_seconds"]:
                checkpoint(model, optimizer, directory / "latest.pt", state, recipe)
                atomic_json(directory / "progress.json", state)
                last_checkpoint = time.monotonic()
            del data
        state["blocks"] += 1
        state["training_seconds"] = previous + time.monotonic() - start
        checkpoint(model, optimizer, directory / "latest.pt", state, recipe)
        atomic_json(directory / "progress.json", state)
    slot = state["blocks"] % recipe["historical_pool_size"]
    save_policy(
        model,
        directory / f"historical_{slot}.pt",
        {
            "iterations": state["iterations"],
            "training_seconds": state["training_seconds"],
        },
    )
    with (directory / "telemetry.jsonl").open("a") as f:
        f.write(
            json.dumps({"block": state["blocks"], "samples": telemetry.samples}) + "\n"
        )
    del optimizer, historical
    return model.eval(), state


def evaluate_checkpoint(
    model, teacher, baseline, directory, state, recipe, games, device
):
    started = time.monotonic()
    evaluations = []
    # Fixed monitoring seeds make learning curves comparable. Fresh untouched
    # seeds are reserved for final confirmation, never used for checkpoint choice.
    matchups = [
        ("champion_direct", teacher, "champion", 0, games, 0),
        ("greedy", teacher, "greedy", 0, games, 100000),
    ]
    if state["blocks"] % 2 == 0:
        matchups += [
            ("champion_search", teacher, "champion", 128, games, 200000),
            ("heldout_baseline", baseline, "champion", 32, games, 300000),
        ]
    for label, opponent, kind, sims, count, offset in matchups:
        result = evaluate_policy(
            model, opponent, device, count, recipe["monitor_seed"] + offset, kind, sims
        )
        result["label"] = label
        evaluations.append(result)
    report = {
        **state,
        "evaluations": evaluations,
        "evaluation_seconds": time.monotonic() - started,
        "checkpoint_sha256": digest(directory / "latest.pt"),
        "monitoring_only": True,
        "production_promoted": False,
    }
    atomic_json(directory / f"evaluation_{state['blocks']:03d}.json", report)
    # Use one selection metric consistently, not whichever opponent looks best.
    search = next((e for e in evaluations if e["label"] == "champion_search"), None)
    best_path = directory / "best_monitor.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if search and (best is None or search["score"] > best["score"]):
        atomic_copy(directory / "latest.pt", directory / "best_monitor.pt")
        atomic_json(
            best_path,
            {
                "score": search["score"],
                "block": state["blocks"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "monitoring_only": True,
            },
        )
    return {k: v for k, v in report.items() if k != "evaluations"} | {
        "scores": {e["label"]: e["score"] for e in evaluations}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--controller-lock", default="campaign/controller.lock")
    parser.add_argument("--block-seconds", type=float, default=1800)
    parser.add_argument("--evaluation-games", type=int, default=512)
    parser.add_argument("--max-blocks", type=int, default=1000)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    a = parser.parse_args()
    deadline = datetime.fromisoformat(a.stop_at.replace("Z", "+00:00")).timestamp()
    if (
        deadline <= time.time()
        or a.block_seconds <= 0
        or a.evaluation_games < 4
        or a.evaluation_games % 4
        or a.games_per_iteration <= 0
        or a.max_blocks <= 0
    ):
        parser.error("Valid future deadline and positive paired run sizes required")
    torch.set_num_threads(1)
    lock = Path(a.controller_lock).open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(*_):
        raise KeyboardInterrupt("Training deadline or shutdown signal")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(max(1, int(deadline - time.time())))
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recipe = {**RECIPE, "games_per_iteration": a.games_per_iteration}
    recipe_path = output / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError(
            "Resume requires the original recipe; do not silently reset optimizer settings"
        )
    atomic_json(recipe_path, recipe)
    frozen = output / "reference.pt"
    if not frozen.exists():
        atomic_copy(a.champion, frozen)
    device = torch.device("cuda")
    teacher, _, _ = load_model(frozen, device=device, compile_model=False)
    baseline, _, _ = load_model(a.baseline, device=device, compile_model=False)
    teacher.eval()
    baseline.eval()
    reference = HistoryPolicy(teacher, "board").to(device).eval()
    manifest_path = output / "state.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"history": []}
    )
    manifest.update(
        status="running",
        pid=os.getpid(),
        stop_at=a.stop_at,
        reference_sha256=digest(frozen),
        recipe=recipe,
    )
    for architecture in recipe["architectures"]:
        initialize(teacher, architecture, output / architecture, recipe)
    if "initial_monitor" not in manifest:
        initial = evaluate_policy(
            reference,
            teacher,
            device,
            a.evaluation_games,
            recipe["monitor_seed"] + 200000,
            "champion",
            128,
        )
        manifest["initial_monitor"] = initial
        for architecture in recipe["architectures"]:
            directory = output / architecture
            atomic_json(
                directory / "best_monitor.json",
                {
                    "score": initial["score"],
                    "block": 0,
                    "checkpoint_sha256": digest(directory / "best_monitor.pt"),
                    "monitoring_only": True,
                },
            )
    manifest["hardware"] = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    }
    atomic_json(manifest_path, manifest)
    try:
        for _ in range(a.max_blocks):
            if time.time() >= deadline - 300:
                break
            active = recipe["architectures"]
            selection = output / "selection.json"
            if selection.exists():
                chosen = json.loads(selection.read_text())["architectures"]
                if not chosen or not set(chosen) <= set(active):
                    raise ValueError("Invalid sustained-training selection")
                for architecture in active:
                    _, info = load_policy(output / architecture / "latest.pt", "cpu")
                    if info["training_seconds"] < 3600:
                        raise ValueError(
                            "Every serious contender requires an hour before narrowing"
                        )
                active = chosen
            totals = {}
            for architecture in active:
                _, info = load_policy(output / architecture / "latest.pt", "cpu")
                totals[architecture] = info["training_seconds"]
            architecture = min(totals, key=totals.get)
            manifest.update(
                stage="training",
                active_architecture=architecture,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(manifest_path, manifest)
            model, state = train_block(
                output / architecture,
                reference,
                recipe,
                min(a.block_seconds, deadline - time.time() - 300),
                deadline,
                device,
            )
            manifest["stage"] = "monitoring"
            atomic_json(manifest_path, manifest)
            summary = evaluate_checkpoint(
                model,
                teacher,
                baseline,
                output / architecture,
                state,
                recipe,
                a.evaluation_games,
                device,
            )
            manifest["history"].append(summary)
            atomic_json(manifest_path, manifest)
            print(json.dumps(summary), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        manifest.update(
            status="completed", stage="awaiting_independent_final_confirmation"
        )
    except BaseException as exc:
        manifest.update(status="interrupted", error=repr(exc))
        raise
    finally:
        signal.alarm(0)
        atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
