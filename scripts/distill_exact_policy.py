"""One shared, measured canonical/exact-action warm start before the comparison."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scripts.overnight import atomic_json, digest
from scripts.train_exact_policy import (
    native_environment,
    observations,
    optimizer_for,
    tensors,
)
from src.exact_policy import ExactPolicy, save_exact_policy
from src.history_policy import load_policy


def teacher_targets(spatial, actions):
    """Transfer the reference's exact-card tie semantics as training targets."""
    mask = actions >= 0
    positions = actions.clamp_min(0) % 100
    normal = (actions.clamp_min(0) // 100) % 13 != 10
    index = torch.arange(actions.shape[1], device=actions.device)[None, :]
    priority = (normal.long() * (actions.shape[1] + 1) - index).masked_fill(~mask, -100000)
    best = torch.full((len(actions), 100), -100000, dtype=torch.long, device=actions.device)
    best.scatter_reduce_(1, positions, priority, reduce="amax", include_self=True)
    preferred = mask & (priority == best.gather(1, positions))
    probability = spatial.masked_fill(best == -100000, -1e9).softmax(1)
    exact = probability.gather(1, positions) * preferred
    return exact, (exact.argmax(1))


def collect(teacher, device, minimum_positions=65536, seed=3100000001, games=512):
    teacher.eval()
    teacher.requires_grad_(False)
    chunks, targets, values, heldout = [], [], [], []
    collected, batch = 0, 0
    start = time.monotonic()
    while collected < minimum_positions:
        env = native_environment(games, seed + batch * games)
        active = env.active_ids()
        while active:
            original = observations(env, active, canonical=False)
            canonical = observations(env, active, canonical=True)
            with torch.inference_mode():
                board, context, history, actions, _ = tensors(original, device)
                spatial, value = teacher(board, context, history)
                target, chosen = teacher_targets(spatial, actions)
                exact = actions.gather(1, chosen[:, None]).flatten().cpu().tolist()
                targets.append(target.cpu().numpy())
                values.append(value.flatten().cpu().numpy())
            chunks.append(canonical)
            heldout.extend([(batch * games + i) % 5 == 0 for i in active])
            collected += len(active)
            env.step(active, exact)
            active = env.active_ids()
        batch += 1
    width = max(chunk["actions"].shape[1] for chunk in chunks)
    data = {key: torch.from_numpy(np.concatenate([chunk[key] for chunk in chunks]))
            for key in ("board", "context", "history")}
    data["actions"] = torch.from_numpy(np.concatenate([
        np.pad(c["actions"], ((0, 0), (0, width - c["actions"].shape[1])), constant_values=-1) for c in chunks
    ]))
    data["target_policy"] = torch.from_numpy(np.concatenate([
        np.pad(p, ((0, 0), (0, width - p.shape[1]))) for p in targets
    ]))
    data["target_value"] = torch.from_numpy(np.concatenate(values))
    data["heldout"] = torch.tensor(heldout, dtype=torch.bool)
    return data, {"positions": collected, "games": batch * games,
                  "collection_seconds": time.monotonic() - start,
                  "heldout_split": "Every fifth complete deal, never individual states from a training deal",
                  "teacher_action_semantics": "Spatial soft probabilities assigned to existing teacher's preferred legal card per square"}


def measure(model, data, device, batch_size=1024):
    model.eval()
    totals = np.zeros(4)
    count = 0
    with torch.inference_mode():
        for ids in torch.where(data["heldout"])[0].split(batch_size):
            board, context, history, actions, mask = tensors(data, device, ids)
            logits, value = model(board, context, history, actions, mask)
            log_policy = logits.masked_fill(~mask, -1e9).log_softmax(1)
            target = data["target_policy"][ids].to(device)
            kl = (target * (target.clamp_min(1e-30).log() - log_policy)).sum(1)
            chosen = logits.argmax(1)
            teacher = target.argmax(1)
            totals += [float(kl.sum()), float((value.flatten() - data["target_value"][ids].to(device)).square().sum()),
                       float((chosen == teacher).sum()),
                       float(((actions.gather(1, chosen[:, None]) % 100) == (actions.gather(1, teacher[:, None]) % 100)).sum())]
            count += len(ids)
    if not count:
        raise ValueError("Distillation requires held-out deals")
    return dict(zip(("policy_kl", "value_mse", "exact_agreement", "position_agreement"), (totals / count).tolist()))


def distill(teacher_path, output, device="cuda", positions=65536, epochs=4, batch_size=1024, seed=3100000001, collection_games=512):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    teacher, _ = load_policy(teacher_path, device)
    model = ExactPolicy.from_history_policy(teacher, variant="cnn").to(device)
    before_teacher = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
    started = time.monotonic()
    data, report = collect(teacher, device, positions, seed, collection_games)
    report.update(teacher_sha256=digest(teacher_path), seed=seed, scope="Shared distillation, excluded from per-contender PPO/RNaD training budgets")
    report["before"] = measure(model, data, device, batch_size)
    optimizer = optimizer_for(model, {"trunk_lr": 3e-5, "new_lr": 1e-4})
    best = copy.deepcopy(model.state_dict())
    best_kl = report["before"]["policy_kl"]
    curve = []
    train_ids = torch.where(~data["heldout"])[0]
    for epoch in range(epochs):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        losses = []
        for ids in train_ids[torch.randperm(len(train_ids))].split(batch_size):
            board, context, history, actions, mask = tensors(data, device, ids)
            logits, value = model(board, context, history, actions, mask)
            target = data["target_policy"][ids].to(device)
            log_policy = logits.masked_fill(~mask, -1e9).log_softmax(1)
            policy_loss = -(target * log_policy).sum(1).mean()
            value_loss = (value.flatten() - data["target_value"][ids].to(device)).square().mean()
            loss = policy_loss + 0.25 * value_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite common warm-start distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = measure(model, data, device, batch_size)
        curve.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), **metrics})
        print(json.dumps(curve[-1]), flush=True)
        if metrics["policy_kl"] < best_kl:
            best_kl, best = metrics["policy_kl"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    model.eval()
    report.update(curve=curve, after=measure(model, data, device, batch_size), elapsed_seconds=time.monotonic() - started)
    if any(not torch.equal(v.detach().cpu(), before_teacher[k]) for k, v in teacher.state_dict().items()):
        raise RuntimeError("Distillation changed frozen teacher parameters or buffers")
    save_exact_policy(model, output, {"shared_distillation": report})
    report["checkpoint_sha256"] = digest(output)
    atomic_json(output.with_suffix(".json"), report)
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positions", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(1)
    distill(args.teacher, args.output, args.device, args.positions, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
