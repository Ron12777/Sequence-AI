"""Evaluate the official pretrained ReBeL solver and its zero-leaf ablation.

Build the official source with compatibility.patch and CMakeLists.txt first.
The external binary writes diagnostic strategies into each isolated run directory.
"""

import argparse
import datetime
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="/tmp/sequence-rebel-build/recursive_eval")
    p.add_argument(
        "--checkpoint", default="/tmp/sequence-rebel-reference/1x4f_cfr_1000.ckpt"
    )
    p.add_argument("--iterations", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    result_path = a.output / f"results-i{a.iterations}-r{a.repeats}.json"
    results = {
        "source": "https://github.com/facebookresearch/rebel",
        "source_commit": "7960a42750f3407ea9eb2c3333d4c2a7961f6df4",
        "game": "two-player zero-sum Liars Dice, one four-sided die per player, four wild",
        "binary_sha256": sha(a.binary),
        "checkpoint_sha256": sha(a.checkpoint),
        "subgame_iterations": a.iterations,
        "sampled_policy_repeats": a.repeats,
        "worker_threads": a.threads,
        "torch_threads_env": 1,
        "runs": [],
        "status": "running",
    }
    for variant, net in [("learned", a.checkpoint), ("zero", "zero")]:
        run_dir = a.output / f"{variant}-i{a.iterations}-r{a.repeats}"
        run_dir.mkdir(exist_ok=True)
        args = [
            a.binary,
            "--net",
            net,
            "--mdp_depth",
            "2",
            "--num_faces",
            "4",
            "--num_dice",
            "1",
            "--subgame_iters",
            str(a.iterations),
            "--num_repeats",
            str(a.repeats),
            "--num_threads",
            str(a.threads),
            "--cfr",
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter()
        with (run_dir / "stdout.log").open("w") as f:
            proc = subprocess.run(
                args,
                check=False,
                cwd=run_dir,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=7200,
            )
        elapsed = time.perf_counter() - start
        end_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
        stdout = (run_dir / "stdout.log").read_text()
        metrics = {}
        for line in stdout.splitlines():
            if line.startswith("XXX "):
                metrics["exploitability"] = json.loads(line[4:])
            if line.startswith("YYY "):
                metrics["full_tree_opponent_ev"] = json.loads(line[4:])
        row = dict(
            variant=variant,
            command=args,
            started_utc=started,
            elapsed_seconds=elapsed,
            cpu_seconds=(
                end_cpu.ru_utime
                + end_cpu.ru_stime
                - start_cpu.ru_utime
                - start_cpu.ru_stime
            ),
            exit_code=proc.returncode,
            **metrics,
        )
        results["runs"].append(row)
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if proc.returncode or not metrics:
            raise RuntimeError(
                f"{variant} evaluation failed; inspect {run_dir}/stdout.log"
            )
    results["status"] = "complete"
    result_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
