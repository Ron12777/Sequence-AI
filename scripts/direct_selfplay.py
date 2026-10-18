"""Search-free on-policy self-play for experimental public-history policies.

Both seats sample legal board positions from the frozen rollout policy. PPO uses
terminal returns from each actor's perspective; no hidden deal is sampled by the
agent and no privileged critic input is supplied. This is not an equilibrium
solver. Run bounded rounds and validate independently before promotion.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.fast_game import FastSequenceGame
from src.history_policy import (
    observe,
    apply_public_move,
    batch_inputs,
    load_policy,
    save_policy,
)
from scripts.architecture_experiments import evaluate_policy, inputs_from_data
from src.model import load_model
from scripts.overnight import atomic_json


def legal_position_mask(game):
    mask = np.zeros(100, dtype=bool)
    for move in game.get_legal_moves():
        mask[move.row * 10 + move.col] = True
    if not mask.any():
        raise ValueError("No legal positions")
    return mask


def masked_distribution(logits, mask):
    if not mask.any(dim=1).all():
        raise ValueError("Every sampled state needs a legal action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def ppo_objective(log_probability, old_log_probability, advantage, clip=0.2):
    ratio = (log_probability - old_log_probability).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()


def rollout(model, device, count, seed, opponents=None):
    model.eval()
    games = []
    for i in range(count):
        random.seed(seed + i)
        games.append(FastSequenceGame())
    events = [[] for _ in games]
    histories = [[] for _ in games]
    active = list(range(count))
    start = time.monotonic()
    opponent_moves = 0
    for opponent in opponents or []:
        opponent.eval()
    while active:
        groups = {}
        for i in active:
            # Half the deals are ordinary self-play. The other half face frozen
            # reference/historical policies, balanced across both learner seats.
            kind = (i // 2) % 4
            opponent_index = (
                kind % len(opponents)
                if opponents and kind < 2 and games[i].current_player != 1 + i % 2
                else -1
            )
            groups.setdefault(opponent_index, []).append(i)
        for opponent_index, ids in groups.items():
            actor = model if opponent_index == -1 else opponents[opponent_index]
            observations = [observe(games[i], events[i]) for i in ids]
            masks = np.stack([legal_position_mask(games[i]) for i in ids])
            with torch.inference_mode():
                logpi, values = actor(*batch_inputs(observations, device))
                distribution = masked_distribution(
                    logpi, torch.from_numpy(masks).to(device)
                )
                actions = (
                    distribution.sample()
                    if opponent_index == -1
                    else distribution.logits.argmax(1)
                )
                log_probs = distribution.log_prob(actions).cpu().numpy()
                actions = actions.cpu().numpy()
                values = values.cpu().numpy()
            for j, i in enumerate(ids):
                g = games[i]
                position = int(actions[j])
                player = g.current_player
                move = max(
                    (m for m in g.get_legal_moves() if m.row * 10 + m.col == position),
                    key=lambda m: not m.card.is_jack(),
                )
                if opponent_index == -1:
                    histories[i].append(
                        (
                            observations[j],
                            masks[j],
                            position,
                            float(log_probs[j]),
                            float(values[j]),
                            player,
                        )
                    )
                else:
                    opponent_moves += 1
                apply_public_move(g, events[i], move)
        active = [
            i
            for i in active
            if not games[i].game_over
            and len(events[i]) < 120
            and games[i].get_legal_moves()
        ]
    rows = [
        (record, g.get_result(record[-1]))
        for g, history in zip(games, histories)
        for record in history
    ]
    data = {
        "board": torch.from_numpy(np.stack([r[0][0][0] for r in rows])),
        "context": torch.from_numpy(np.stack([r[0][0][1] for r in rows])),
        "history": torch.from_numpy(
            np.stack([r[0][0][2] for r in rows]).astype(np.uint8)
        ),
        "mask": torch.from_numpy(np.stack([r[0][1] for r in rows])),
        "action": torch.tensor([r[0][2] for r in rows]),
        "old_log_probability": torch.tensor([r[0][3] for r in rows]),
        "old_value": torch.tensor([r[0][4] for r in rows]),
        "return": torch.tensor([r[1] for r in rows], dtype=torch.float32),
    }
    return data, {
        "games": count,
        "positions": len(rows),
        "decisive_games": sum(g.winner is not None for g in games),
        "rollout_seconds": time.monotonic() - start,
        "opponent_moves": opponent_moves,
    }


def update(
    model,
    optimizer,
    data,
    device,
    epochs=2,
    batch_size=256,
    entropy_coefficient=0.01,
    clip=0.2,
    target_kl=0.03,
):
    # Keep BatchNorm statistics fixed to match the rollout policy. cuDNN GRU
    # backward requires training mode, so freeze only normalization modules.
    # Both history encoders use zero dropout.
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    advantage = data["return"] - data["old_value"]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    losses = []
    kls = []
    value_losses = []
    entropies = []
    for _ in range(epochs):
        ids = torch.randperm(len(advantage))
        for batch in ids.split(batch_size):
            logpi, value = model(*inputs_from_data(data, batch, device))
            distribution = masked_distribution(logpi, data["mask"][batch].to(device))
            log_probability = distribution.log_prob(data["action"][batch].to(device))
            old = data["old_log_probability"][batch].to(device)
            policy_loss = ppo_objective(
                log_probability, old, advantage[batch].to(device), clip=clip
            )
            value_loss = torch.nn.functional.mse_loss(
                value, data["return"][batch].to(device)
            )
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite self-play loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
            delta = (log_probability - old).detach()
            kls.append(float((delta.exp() - 1 - delta).mean()))
        if np.mean(kls[-max(1, len(ids) // batch_size) :]) > target_kl:
            break
    return {
        "loss": float(np.mean(losses)),
        "approximate_kl": float(np.mean(kls)),
        "updates": len(losses),
        "value_loss": float(np.mean(value_losses)),
        "policy_entropy": float(np.mean(entropies)),
    }


def seed_schedule(seed, iterations, games_per_iteration, evaluation_games):
    """Reserve evaluation deals after every training seed, including long runs."""
    stride = max(10000, games_per_iteration)
    evaluation_start = seed + iterations * stride
    evaluation_stride = max(1000000, evaluation_games)
    return stride, [evaluation_start + i * evaluation_stride for i in range(3)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", required=True)
    p.add_argument("--champion", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--games-per-iteration", type=int, default=128)
    p.add_argument("--evaluation-games", type=int, default=512)
    p.add_argument("--seed", type=int, default=991713)
    a = p.parse_args()
    if (
        min(a.iterations, a.games_per_iteration) <= 0
        or a.evaluation_games < 4
        or a.evaluation_games % 4
    ):
        p.error("Invalid run sizes")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Self-play campaign requires CUDA")
    device = torch.device("cuda")
    model, metadata = load_policy(a.init_from, device)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    stride, evaluation_seeds = seed_schedule(
        a.seed, a.iterations, a.games_per_iteration, a.evaluation_games
    )
    metrics = []
    start = time.monotonic()
    for iteration in range(a.iterations):
        data, row = rollout(
            model, device, a.games_per_iteration, a.seed + iteration * stride
        )
        row.update(update(model, optimizer, data, device))
        row.update(iteration=iteration + 1, seconds=time.monotonic() - start)
        metrics.append(row)
        print(json.dumps(row), flush=True)
        save_policy(
            model,
            output / "latest.pt",
            {
                "training": "Search-free PPO self-play",
                "source_metadata": metadata,
                "iteration": iteration + 1,
                "seed": a.seed,
            },
        )
        atomic_json(output / "metrics.json", metrics)
    teacher, _, _ = load_model(a.champion, device="cuda", compile_model=False)
    results = {
        "config": vars(a),
        "architecture": model.architecture,
        "evaluations": [],
        "production_changed": False,
        "training_seed_stride": stride,
        "evaluation_seeds": evaluation_seeds,
    }
    for opponent, sims, count, seed in [
        ("champion", 0, a.evaluation_games, evaluation_seeds[0]),
        ("greedy", 0, a.evaluation_games, evaluation_seeds[1]),
        ("champion", 128, a.evaluation_games // 2, evaluation_seeds[2]),
    ]:
        result = evaluate_policy(
            model.eval(), teacher, device, count, seed, opponent, sims
        )
        results["evaluations"].append(result)
        atomic_json(output / "results.json", results)
        print(
            json.dumps({k: v for k, v in result.items() if k != "outcomes"}), flush=True
        )


if __name__ == "__main__":
    main()
