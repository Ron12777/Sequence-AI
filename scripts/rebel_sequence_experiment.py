"""Bounded ReBeL-style learning on the restricted Sequence public-belief game.

Search generates value targets and sampled-iteration policies generate the next
public belief. The learned leaf is refreshed each CFR iteration. This is not a
full Sequence player and does not inherit the paper's equilibrium guarantees.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn

from scripts.rebel_resolve import truncate
from src.public_belief_cfr import LeafValues, PublicBeliefCFR
from src.rebel_sequence_game import SequenceBeliefGame


class BeliefValueNet(nn.Module):
    def __init__(self, inputs, types, width=128):
        super().__init__()
        self.types = types
        self.layers = nn.Sequential(
            nn.Linear(inputs, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2 * types),
            nn.Tanh(),
        )
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)

    def forward(self, x, ranges):
        values = self.layers(x).reshape(-1, 2, self.types)
        residual = (values * ranges).sum((1, 2)) / 2
        # Reconcile the two predicted perspectives while preserving [-1,1].
        return (values - residual[:, None, None]) / (1 + residual.abs()[:, None, None])


class NeuralLeaf:
    def __init__(self, game, model):
        self.game, self.model = game, model
        self.calls = 0

    def __call__(self, node, belief):
        self.calls += 1
        x = torch.from_numpy(self.game.features(node, belief))[None]
        ranges = torch.tensor(np.stack(belief.ranges)[None], dtype=torch.float32)
        with torch.inference_mode():
            v = self.model(x, ranges)[0].numpy().astype(np.float64)
        return LeafValues(v[0], v[1])


def sample_episode(game, model, rng, iterations, depth):
    """True private types affect action execution only, never solver inputs."""
    deal = rng.choice(game.prior.size, p=game.prior.ravel())
    hands = np.unravel_index(deal, game.prior.shape)
    full_node = game.root
    reaches = tuple(np.ones(len(game.types)) for _ in range(2))
    leaf = NeuralLeaf(game, model)
    examples, search_seconds, entropies = [], 0.0, []
    while full_node.actor is not None:
        short = truncate(full_node, depth)
        solver = PublicBeliefCFR(short, game.prior, leaf_value=leaf)
        sample_index = int(rng.integers(iterations))
        selected = None
        cf_sum = np.zeros((2, len(game.types)))
        start = time.monotonic()
        for i, result in enumerate(solver.iterations(iterations, root_reach=reaches)):
            cf_sum += np.stack(result.evaluation.counterfactual_values)
            if i == sample_index:
                selected = result
        search_seconds += time.monotonic() - start
        belief = solver.belief(reaches)
        mass = np.stack(belief.compatible_mass)
        targets = np.divide(
            cf_sum / iterations, mass, out=np.zeros_like(cf_sum), where=mass > 0
        )
        used = [
            np.bincount(s, minlength=len(game.cards)) for s in game.spent[full_node.key]
        ]
        valid = np.stack([np.all(game.counts >= sc, axis=1) for sc in used])
        weights = (mass > 0) * valid
        examples.append(
            (
                game.features(full_node, belief),
                np.stack(belief.ranges).astype(np.float32),
                targets.astype(np.float32),
                weights.astype(np.float32),
            )
        )
        node = short
        while node.actor is not None:
            actor = node.actor
            policy = selected.strategies[node.key]
            probabilities = policy[hands[actor]]
            assert abs(float(probabilities.sum()) - 1) < 1e-8
            entropies.append(
                float(-(probabilities * np.log(probabilities + 1e-30)).sum())
            )
            action_index = int(rng.choice(len(node.children), p=probabilities))
            action = list(node.children)[action_index]
            updated = list(reaches)
            updated[actor] = reaches[actor] * policy[:, action_index]
            reaches = tuple(updated)
            node = node.children[action]
        full_node = game.nodes[node.key]
    return examples, {
        "search_seconds": search_seconds,
        "neural_leaf_calls": leaf.calls,
        "visited_beliefs": len(examples),
        "policy_entropy": float(np.mean(entropies)),
        "terminal_result": float(full_node.payoff),
    }


def update(model, optimizer, replay, rng, batches=8, batch_size=64):
    logs = []
    for _ in range(batches):
        ids = rng.integers(len(replay), size=min(batch_size, len(replay)))
        x, ranges, target, weight = [
            torch.tensor(np.stack([replay[i][j] for i in ids])) for j in range(4)
        ]
        predicted = model(x, ranges)
        squared = (predicted - target).square()
        loss = (squared * weight).sum() / weight.sum().clamp_min(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        decisive = (target.abs() > 0.05) * weight
        logs.append(
            {
                "mse": float(loss.detach()),
                "nonzero_target_mse": float(
                    (squared.detach() * decisive).sum() / decisive.sum().clamp_min(1)
                ),
                "nonzero_target_fraction": float(
                    decisive.sum() / weight.sum().clamp_min(1)
                ),
                "zero_sum_residual": float(
                    (predicted.detach() * ranges).sum((1, 2)).abs().mean()
                ),
            }
        )
    return {k: float(np.mean([r[k] for r in logs])) for k in logs[0]}


def save(path, model, optimizer, rng, replay, metadata):
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "replay": replay,
            "metadata": metadata,
        },
        temp,
    )
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seconds", type=float, default=1800)
    p.add_argument("--episodes", type=int, default=100000)
    p.add_argument("--iterations", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261017)
    a = p.parse_args()
    if a.seconds <= 0 or a.iterations < 2 or a.depth < 1:
        p.error("Positive budget and depth, at least two CFR iterations required")
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    game = SequenceBeliefGame()
    solver = PublicBeliefCFR(game.root, game.prior)
    inputs = len(game.features(game.root, solver.belief()))
    model = BeliefValueNet(inputs, len(game.types))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    replay = []
    recipe = {
        "iterations": a.iterations,
        "depth": a.depth,
        "seed": a.seed,
        "width": 128,
        "lr": 3e-4,
        "replay_capacity": 4096,
        "updates_every_episodes": 4,
        "batches_per_update": 8,
        "inputs": inputs,
        "types": len(game.types),
    }
    latest = output / "latest.pt"
    completed, prior_seconds = 0, 0.0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state["metadata"]["recipe"] != recipe:
            raise ValueError("Resume requires identical recipe")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        replay = state["replay"]
        completed = state["metadata"]["episodes"]
        prior_seconds = state["metadata"]["training_seconds"]
    metadata = {
        "recipe": recipe,
        "game": game.summary(),
        "episodes": completed,
        "training_seconds": prior_seconds,
        "pid": os.getpid(),
        "device": "cpu",
        "status": "running",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    for episode in range(completed, completed + a.episodes):
        if time.monotonic() - start >= a.seconds:
            break
        examples, row = sample_episode(game, model.eval(), rng, a.iterations, a.depth)
        replay.extend(examples)
        replay = replay[-recipe["replay_capacity"] :]
        if (episode + 1) % recipe["updates_every_episodes"] == 0:
            row.update(update(model.train(), optimizer, replay, rng))
        metadata.update(
            episodes=episode + 1,
            training_seconds=prior_seconds + time.monotonic() - start,
            replay_beliefs=len(replay),
        )
        row.update(
            episode=episode + 1,
            training_seconds=metadata["training_seconds"],
            replay_beliefs=len(replay),
        )
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if (episode + 1) % 20 == 0:
            save(latest, model, optimizer, rng, replay, metadata)
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
            print(json.dumps(row), flush=True)
    metadata["status"] = "complete"
    save(latest, model, optimizer, rng, replay, metadata)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
