"""Measure one realistic fresh rollout/update per contender without saving candidates."""

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scripts.overnight import atomic_json, digest
from scripts.train_exact_policy import (
    CONTENDERS,
    RECIPE,
    optimizer_for,
    rollout,
    update,
    update_rnad,
)
from src.exact_policy import ExactPolicy, load_exact_policy
from src.history_policy import load_policy
from src.rnad import RNaDConfig, RNaDState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rnad-games-per-minibatch", type=int, default=32)
    parser.add_argument("--contenders", nargs="+", choices=list(CONTENDERS), default=list(CONTENDERS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    initial, _ = load_exact_policy(args.initial, "cpu")
    recipe = dict(RECIPE, games_per_iteration=args.games, batch_size=args.batch_size,
                  rnad_games_per_minibatch=args.rnad_games_per_minibatch)
    cuda = torch.device(args.device).type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    report = {"purpose": "setup throughput; no strength claim or saved candidate",
              "common_checkpoint_sha256": digest(args.initial), "recipe": recipe,
              "device": torch.cuda.get_device_name() if cuda else args.device,
              "results": {}}
    started = time.monotonic()
    for contender in args.contenders:
        torch.manual_seed(RECIPE["seed"])
        variant, algorithm = CONTENDERS[contender]
        model = ExactPolicy.from_history_policy(
            initial.backbone, variant=variant,
            value_mode="unbounded" if algorithm == "rnad" else "bounded",
        ).to(args.device)
        model.load_state_dict(initial.state_dict(), strict=False)
        optimizer = optimizer_for(model, recipe, algorithm)
        if algorithm == "rnad":
            state = RNaDState(model, RNaDConfig(eta=recipe["rnad_eta"], target_avg=recipe["rnad_target_avg"],
                                              reference_interval=recipe["rnad_reference_interval"]))
            opponents = ()
        else:
            state = None
            reference, _ = load_policy(args.reference, args.device)
            opponents = (reference, copy.deepcopy(model).eval())
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        data, rollout_metrics = rollout(model, args.device, args.games, 3900000001,
                                         opponents=opponents, full_trajectory=algorithm == "rnad")
        sync()
        rollout_wall = time.monotonic() - tick
        tick = time.monotonic()
        update_metrics = (update_rnad(model, optimizer, data, state, args.device, recipe)
                          if state is not None else update(model, optimizer, data, args.device, recipe))
        sync()
        update_wall = time.monotonic() - tick
        row = {"rollout": rollout_metrics, "update": update_metrics,
               "rollout_wall_seconds": rollout_wall, "update_wall_seconds": update_wall,
               "complete_iteration_seconds": rollout_wall + update_wall,
               "positions_per_iteration_second": rollout_metrics["positions"] / (rollout_wall + update_wall),
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None}
        report["results"][contender] = row
        report["total_setup_seconds"] = time.monotonic() - started
        atomic_json(args.output, report)
        print(json.dumps({"contender": contender, **row}), flush=True)
        del data, model, optimizer, state, opponents
        if algorithm != "rnad":
            del reference
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
