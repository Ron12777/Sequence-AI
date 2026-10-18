"""Deadline-bounded training tournament. Keeps the champion across failed trials."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = [
    dict(
        name="finetune32", epochs=64, simulations=32, lr=0.0003, noise=0.25, fresh=False
    ),
    dict(
        name="finetune64", epochs=64, simulations=64, lr=0.0001, noise=0.25, fresh=False
    ),
    dict(
        name="fresh96",
        epochs=128,
        simulations=32,
        lr=0.001,
        noise=0.25,
        fresh=True,
        filters=96,
        blocks=4,
    ),
    dict(
        name="finetune_low_noise",
        epochs=64,
        simulations=32,
        lr=0.0001,
        noise=0.1,
        fresh=False,
    ),
    dict(
        name="fresh_small64",
        epochs=96,
        simulations=64,
        lr=0.0007,
        noise=0.25,
        fresh=True,
        filters=64,
        blocks=3,
    ),
    dict(
        name="fresh128",
        epochs=128,
        simulations=32,
        lr=0.001,
        noise=0.25,
        fresh=True,
        filters=128,
        blocks=6,
    ),
]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log, timeout):
    if timeout <= 0:
        raise TimeoutError("Deadline reached")
    with Path(log).open("w") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Command exited with {code}; inspect {log}")
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def promotion_allowed(screen, confirmation):
    return (
        screen["score"] >= 0.52
        and confirmation["score"] >= 0.53
        and confirmation["paired_score_95ci"][0] > 0.5
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--stop-at", required=True)
    p.add_argument("--output-dir", default="campaign")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-rounds", type=int, default=1000)
    a = p.parse_args()
    work = Path(a.output_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.fromisoformat(a.stop_at.replace("Z", "+00:00")).timestamp()
    if deadline <= time.time():
        raise ValueError("Training deadline has passed")
    champion, baseline = work / "champion.pt", work / "baseline.pt"
    if not champion.exists():
        atomic_copy(a.baseline, champion)
    if not baseline.exists():
        atomic_copy(a.baseline, baseline)
    plan_file = work / "plan.json"
    if not plan_file.exists():
        atomic_json(plan_file, DEFAULT_PLAN)
    state_file = work / "state.json"
    state = (
        json.loads(state_file.read_text())
        if state_file.exists()
        else {"round": -1, "promotions": 0}
    )
    failures = 0
    state.update(
        status="running",
        pid=os.getpid(),
        training_stop_utc=a.stop_at,
        champion_sha256=digest(champion),
    )
    atomic_json(state_file, state)
    try:
        for index in range(state["round"] + 1, a.max_rounds):
            if deadline - time.time() < 900:
                break
            plan = json.loads(plan_file.read_text())
            config = plan[index % len(plan)]
            trial = work / f"round_{index:04d}"
            trial.mkdir(exist_ok=True)
            state.update(
                round=index,
                stage="training",
                config=config,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(state_file, state)
            parent_sha = digest(champion)
            command = [
                sys.executable,
                "-m",
                "src.train",
                "train",
                "--fresh",
                "--no-compile",
                "--epochs",
                str(config["epochs"]),
                "--games-per-epoch",
                str(config.get("games_per_epoch", 1024)),
                "--simulations",
                str(config["simulations"]),
                "--search-mode",
                config.get("search_mode", "ensemble"),
                "--workers",
                str(a.workers),
                "--games-per-worker",
                "128",
                "--batch-size",
                "2048",
                "--train-steps",
                str(config.get("train_steps", 96)),
                "--buffer-size",
                "200000",
                "--lr",
                str(config["lr"]),
                "--seed",
                str(1009 + index * 97),
                "--root-noise-fraction",
                str(config["noise"]),
                "--output-dir",
                str(trial),
            ]
            if config["fresh"]:
                command += [
                    "--num-filters",
                    str(config["filters"]),
                    "--num-res-blocks",
                    str(config["blocks"]),
                ]
            else:
                command += ["--init-from", str(champion)]
            started = time.time()
            outcome = {
                "round": index,
                "config": config,
                "parent_sha256": parent_sha,
                "seed": 1009 + index * 97,
            }
            candidate = trial / "latest.pt"
            try:
                run(
                    command,
                    trial / "train.log",
                    min(3600, deadline - time.time() - 300),
                )
                state.update(stage="screening")
                atomic_json(state_file, state)
                screen_path = trial / "screen.json"
                candidate_search = config.get(
                    "evaluation_search", config.get("search_mode", "ensemble")
                )
                common = [
                    sys.executable,
                    "scripts/arena.py",
                    "--candidate",
                    str(candidate),
                    "--champion",
                    str(champion),
                    "--simulations",
                    "128",
                    "--candidate-search",
                    candidate_search,
                    "--champion-search",
                    state.get("champion_search", "ensemble"),
                ]
                run(
                    common
                    + [
                        "--games",
                        "512",
                        "--seed",
                        str(10000019 + index * 10000),
                        "--output",
                        str(screen_path),
                    ],
                    trial / "screen.log",
                    600,
                )
                screen = json.loads(screen_path.read_text())
                outcome["screen"] = screen
                promoted = False
                if screen["score"] >= 0.52 and time.time() < deadline:
                    state.update(stage="confirming")
                    atomic_json(state_file, state)
                    confirm_path = trial / "confirmation.json"
                    run(
                        common
                        + [
                            "--games",
                            "1024",
                            "--seed",
                            str(50000017 + index * 10000),
                            "--output",
                            str(confirm_path),
                        ],
                        trial / "confirmation.log",
                        900,
                    )
                    confirmation = json.loads(confirm_path.read_text())
                    outcome["confirmation"] = confirmation
                    eligible = promotion_allowed(screen, confirmation)
                    if eligible and state["promotions"]:
                        baseline_path = trial / "baseline_check.json"
                        baseline_command = common.copy()
                        baseline_command[baseline_command.index("--champion") + 1] = (
                            str(baseline)
                        )
                        baseline_command[
                            baseline_command.index("--champion-search") + 1
                        ] = "single"
                        run(
                            baseline_command
                            + [
                                "--games",
                                "512",
                                "--seed",
                                str(90000049 + index * 10000),
                                "--output",
                                str(baseline_path),
                            ],
                            trial / "baseline_check.log",
                            600,
                        )
                        outcome["baseline_check"] = json.loads(
                            baseline_path.read_text()
                        )
                        eligible = outcome["baseline_check"]["score"] >= 0.52
                    if eligible:
                        greedy_path = trial / "greedy_check.jsonl"
                        run(
                            [
                                sys.executable,
                                "scripts/evaluate.py",
                                "--checkpoint",
                                str(candidate),
                                "--games",
                                "512",
                                "--simulations",
                                "128",
                                "--search-mode",
                                candidate_search,
                                "--seed",
                                str(130000007 + index * 10000),
                                "--output",
                                str(greedy_path),
                            ],
                            trial / "greedy_check.log",
                            600,
                        )
                        outcome["greedy_check"] = json.loads(
                            greedy_path.read_text().splitlines()[-1]
                        )
                        eligible = outcome["greedy_check"]["score"] >= 0.70
                    if eligible:
                        atomic_copy(champion, work / "previous.pt")
                        atomic_copy(candidate, champion)
                        promoted = True
                        state["promotions"] += 1
                        state["champion_search"] = candidate_search
                        state["champion_sha256"] = digest(champion)
                        atomic_json(
                            work / "champion.json",
                            {
                                "round": index,
                                "config": config,
                                "search_mode": candidate_search,
                                "evaluation_simulations": 128,
                                "sha256": state["champion_sha256"],
                                "confirmation": confirmation,
                                "parent_sha256": parent_sha,
                            },
                        )
                outcome.update(promoted=promoted, seconds=time.time() - started)
                failures = 0
            except Exception as exc:
                outcome.update(error=str(exc), seconds=time.time() - started)
                failures += 1
            finally:
                atomic_json(trial / "result.json", outcome)
                # Keep the selected champion/rollback/baseline, not every trial's weights.
                if candidate.exists():
                    candidate.unlink()
                state.update(
                    last_result={
                        k: v
                        for k, v in outcome.items()
                        if k not in ("screen", "confirmation")
                    },
                    stage="between_rounds",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                atomic_json(state_file, state)
                print(json.dumps(state), flush=True)
            if failures >= 3:
                raise RuntimeError(
                    "Three trials failed consecutively; stopping for inspection"
                )
        state.update(stage="final_evaluation")
        atomic_json(state_file, state)
        run(
            [
                sys.executable,
                "scripts/arena.py",
                "--candidate",
                str(champion),
                "--champion",
                str(baseline),
                "--games",
                "1024",
                "--simulations",
                "128",
                "--candidate-search",
                state.get("champion_search", "ensemble"),
                "--champion-search",
                "single",
                "--seed",
                "982451653",
                "--output",
                str(work / "final_vs_baseline.json"),
            ],
            work / "final_arena.log",
            1200,
        )
        run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint",
                str(champion),
                "--games",
                "1024",
                "--simulations",
                "0,32,128",
                "--search-mode",
                state.get("champion_search", "ensemble"),
                "--seed",
                "961748941",
                "--output",
                str(work / "final_greedy.jsonl"),
            ],
            work / "final_greedy.log",
            1200,
        )
        state.update(
            status="complete",
            stage="awaiting_download",
            champion_sha256=digest(champion),
        )
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_file, state)


if __name__ == "__main__":
    main()
