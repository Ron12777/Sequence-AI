"""Frozen pre-staging PPO update for execution-equivalence regression tests."""

import math
import time

import numpy as np
import torch

from scripts.train_exact_policy import RECIPE, tensors


def update(model, optimizer, data, device, recipe=RECIPE, entropy_coefficient=None):
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    if entropy_coefficient is None:
        entropy_coefficient = recipe["entropy_initial"]
    advantage = data["return"] - data["old_value"]
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    rows, stopped, epochs_completed = [], False, 0
    started = time.monotonic()
    for _ in range(recipe["epochs"]):
        for ids in torch.randperm(len(advantage)).split(recipe["batch_size"]):
            board, context, history, actions, mask = tensors(data, device, ids)
            logits, values = model(board, context, history, actions, mask)
            distribution = torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))
            log_prob = distribution.log_prob(data["action"][ids].to(device))
            delta = log_prob - data["old_log_probability"][ids].to(device)
            ratio = delta.exp()
            kl = (ratio - 1 - delta).mean()
            if float(kl.detach()) > recipe["target_kl"]:
                stopped = True
                break
            adv = advantage[ids].to(device)
            policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - recipe["ppo_clip"], 1 + recipe["ppo_clip"]) * adv).mean()
            value_loss = (values.flatten() - data["return"][ids].to(device)).square().mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + recipe["value_coefficient"] * value_loss - entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite exact policy PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                probabilities = distribution.probs
                positions = actions.clamp_min(0) % 100
                positional = torch.zeros((len(ids), 100), device=device).scatter_add_(1, positions, probabilities * mask)
                position_entropy = -(positional * positional.clamp_min(1e-30).log()).sum(1).mean()
            rows.append({
                "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()), "approximate_kl": float(kl.detach()),
                "policy_entropy": float(entropy.detach()), "position_entropy": float(position_entropy),
                "card_given_position_entropy": float(entropy.detach() - position_entropy),
                "clip_fraction": float(((ratio - 1).abs() > recipe["ppo_clip"]).float().mean().detach()),
                "gradient_norm": float(norm),
            })
        epochs_completed += 1
        if stopped:
            break
    model.eval()
    if not rows:
        raise RuntimeError("PPO policy differed from frozen rollout before its first update")
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    variance = float(data["return"].var(unbiased=False))
    metrics.update(updates=len(rows), epochs_started=epochs_completed, kl_early_stop=stopped,
                   update_seconds=time.monotonic() - started,
                   samples_per_fresh_position=sum(min(recipe["batch_size"], len(advantage) - (i % math.ceil(len(advantage) / recipe["batch_size"])) * recipe["batch_size"]) for i in range(len(rows))) / len(advantage),
                   rollout_value_explained_variance=1 - float((data["return"] - data["old_value"]).var(unbiased=False)) / variance if variance else None)
    return metrics

