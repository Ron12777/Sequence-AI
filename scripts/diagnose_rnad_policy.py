"""CPU-only frozen-policy and NeuRD gating diagnostics; never updates weights.

The state distribution is fresh full-game R-NaD self-play. Both checkpoints are
measured on exactly those observations. Force diagnostics use the saved EMA and
regularization references and fresh on-policy data; they do not reproduce the
staleness of later minibatches within the original training rollout.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.evaluate_exact_policy import digest, freeze, verify_frozen
from scripts.overnight import atomic_json
from scripts.train_exact_policy import rollout, tensors
from src.exact_policy import load_exact_policy
from src.rnad import RNaDState, legal_policy, v_trace

BETAS = (2.0, 4.0, 8.0)


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values), "mean": float(values.mean()), "min": float(values.min()),
        "p10": float(np.quantile(values, 0.1)), "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)), "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)), "max": float(values.max()),
    }


def centered_logits(logits, legal):
    mean = torch.where(legal, logits, 0).sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(legal, logits - mean, 0)


@torch.no_grad()
def predictions(model, data, batch_size=256):
    logits, values = [], []
    for start in range(0, len(data["actions"]), batch_size):
        ids = torch.arange(start, min(start + batch_size, len(data["actions"])))
        logit, value = model(*tensors(data, "cpu", ids))
        logits.append(logit)
        values.append(value.flatten())
    return torch.cat(logits), torch.cat(values)


@torch.no_grad()
def policy_metrics(logits, values, data):
    actions = data["actions"].long()
    legal = actions >= 0
    policy, log_policy = legal_policy(logits, legal)
    centered = centered_logits(logits, legal)
    top = logits.masked_fill(~legal, -torch.inf).argmax(-1, keepdim=True)
    top_center = centered.gather(-1, top).flatten()
    position_policy = torch.zeros(len(logits), 100)
    position_policy.scatter_add_(1, actions.clamp_min(0) % 100, policy)
    entropy = -(policy * log_policy).sum(-1)
    position_entropy = -(position_policy * position_policy.clamp_min(1e-30).log()).sum(-1)
    result = {
        "legal_actions": describe(legal.sum(-1)), "centered_legal_logits": describe(centered[legal]),
        "top_centered_logit": describe(top_center),
        "top_action_probability": describe(policy.gather(-1, top)),
        "exact_action_entropy": describe(entropy), "position_entropy": describe(position_entropy),
        "card_given_position_entropy": describe(entropy - position_entropy),
        "value": describe(values),
        "unregularized_terminal_outcome_mse": float((values - data["return"]).square().mean()),
        "thresholds": {},
    }
    for beta in BETAS:
        high, low = (centered >= beta) & legal, (centered <= -beta) & legal
        result["thresholds"][str(beta)] = {
            "fraction_top_actions_at_or_above": float((top_center >= beta).float().mean()),
            "mean_policy_mass_at_or_above": float((policy * high).sum(-1).mean()),
            "fraction_legal_logits_at_or_above": float(high.sum() / legal.sum()),
            "fraction_legal_logits_at_or_below_negative": float(low.sum() / legal.sum()),
        }
    return result, policy, log_policy


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if float(denominator) else None


@torch.no_grad()
def force_metrics(logits, policy, q, legal, clip):
    """Exact force gates, plus an explicitly limited logit-space direction proxy."""
    centered = centered_logits(logits, legal)
    advantage = q - (policy * q).sum(-1, keepdim=True)
    clipped = advantage.clamp(-clip, clip)
    positive = clipped.clamp_min(0) * legal
    negative = (-clipped).clamp_min(0) * legal
    top = logits.masked_fill(~legal, -torch.inf).argmax(-1, keepdim=True)
    top_positive = positive.gather(-1, top)
    result = {"advantage_abs": describe(advantage[legal].abs()),
              "fraction_legal_advantages_clipped": float((advantage[legal].abs() > clip).float().mean()),
              "thresholds": {}}
    for beta in BETAS:
        blocked_positive = positive * (centered >= beta)
        blocked_negative = negative * (centered <= -beta)
        force = positive * (centered < beta) - negative * (centered > -beta)
        # If logits themselves were trainable parameters updated by SGD, this
        # is d log pi(top)/d step. It is NOT the shared-network Adam update.
        log_top_proxy = force.gather(-1, top).flatten() - (policy * force).sum(-1)
        result["thresholds"][str(beta)] = {
            "mean_positive_force_before_gate_per_state": float(positive.sum(-1).mean()),
            "mean_negative_force_abs_before_gate_per_state": float(negative.sum(-1).mean()),
            "mean_blocked_positive_force_per_state": float(blocked_positive.sum(-1).mean()),
            "mean_blocked_negative_force_abs_per_state": float(blocked_negative.sum(-1).mean()),
            "mean_signed_blocked_force_per_state": float((blocked_positive - blocked_negative).sum(-1).mean()),
            "positive_force_abs_blocked_fraction": safe_fraction(blocked_positive.sum(), positive.sum()),
            "negative_force_abs_blocked_fraction": safe_fraction(blocked_negative.sum(), negative.sum()),
            "top_positive_force_abs_blocked_fraction": safe_fraction(blocked_positive.gather(-1, top).sum(), top_positive.sum()),
            "fraction_top_positive_events_blocked": safe_fraction(((top_positive > 0) & (centered.gather(-1, top) >= beta)).sum(), (top_positive > 0).sum()),
            "mean_log_top_probability_sgd_proxy": float(log_top_proxy.mean()),
            "fraction_negative_log_top_probability_sgd_proxy": float((log_top_proxy < 0).float().mean()),
        }
    return result


@torch.no_grad()
def reconstructed_forces(model, data, logits, values, payload):
    continuation = payload.get("training_state", {}).get("rnad")
    if continuation is None:
        return {"status": "unavailable", "reason": "Checkpoint lacks saved R-NaD EMA/reference state"}
    state = RNaDState(model)
    state.load_state_dict(continuation)
    _, target_values = predictions(state.target, data)
    reference_logits, _ = predictions(state.reference, data)
    previous_logits, _ = predictions(state.previous_reference, data)
    legal = data["actions"] >= 0
    policy, log_pi = legal_policy(logits, legal)
    _, log_reference = legal_policy(reference_logits, legal)
    _, log_previous = legal_policy(previous_logits, legal)
    reg = log_pi - state.alpha * log_reference - (1 - state.alpha) * log_previous
    rows = []
    for game in data["game_index"].unique():
        ids = torch.where(data["game_index"] == game)[0]
        ids = ids[data["ply"][ids].argsort()]
        if not torch.equal(data["ply"][ids], torch.arange(len(ids))):
            raise ValueError("Diagnostic requires complete game trajectories")
        rows.append(ids)
    length, batch = max(map(len, rows)), len(rows)
    lookup = torch.full((length, batch), -1, dtype=torch.long)
    for column, ids in enumerate(rows):
        lookup[:len(ids), column] = ids
    valid = lookup >= 0

    def padded(value):
        result = torch.zeros((*lookup.shape, *value.shape[1:]), dtype=value.dtype)
        result[valid] = value[lookup[valid]]
        return result

    v_targets = torch.zeros_like(padded(values))
    q_targets = torch.zeros_like(padded(logits))
    for player in (0, 1):
        v, _, q = v_trace(
            padded(target_values), valid, padded(data["player"]), padded(data["behavior_policy"]),
            padded(policy), padded(reg), padded(data["action"]), padded(data["rewards"])[..., player],
            player, state.config,
        )
        v_targets += v
        q_targets += q
    selected = lookup[valid]
    target = v_targets[valid]
    predicted = values[selected]
    pi_selected = policy.gather(-1, data["action"][:, None]).flatten()
    mu_selected = data["behavior_policy"].gather(-1, data["action"][:, None]).flatten()
    config = asdict(state.config)
    config["rho"] = "infinity" if not np.isfinite(config["rho"]) else config["rho"]
    return {
        "status": "complete", "config": config, "learner_steps": state.learner_steps,
        "reference_alpha": state.alpha,
        "scope": "Fresh frozen on-policy complete games; no optimizer; does not reproduce stale minibatches",
        "value_target": describe(target), "value_target_variance": float(target.var(unbiased=False)),
        "critic_target_mse_per_state": float((predicted - target).square().mean()),
        "critic_target_explained_variance": 1 - float((predicted - target).var(unbiased=False) / target.var(unbiased=False)) if target.var(unbiased=False) > 0 else None,
        "selected_behavior_probability": describe(mu_selected),
        "selected_current_over_behavior": describe(pi_selected / mu_selected),
        "current_to_regularization_references_weighted_kl": float((policy * reg).sum(-1).mean()),
        "force_gating": force_metrics(logits[selected], policy[selected], q_targets[valid], legal[selected], state.config.nerd_clip),
        "proxy_limitation": "Logit-space SGD direction only, not a prediction of the shared network's Adam step; no strength claim",
    }


def diagnose(common_checkpoint, rnad_checkpoint, output, games=64, seed=7300000000001):
    if games < 2:
        raise ValueError("At least two complete diagnostic games are required")
    torch.set_num_threads(1)
    common, _ = load_exact_policy(common_checkpoint, "cpu")
    current, metadata = load_exact_policy(rnad_checkpoint, "cpu")
    snapshots = [freeze(model) for model in (common, current)]
    hashes = {"common": digest(common_checkpoint), "rnad": digest(rnad_checkpoint)}
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    data, rollout_metrics = rollout(current, "cpu", games, seed, full_trajectory=True)
    current_logits, current_values = predictions(current, data)
    common_logits, common_values = predictions(common, data)
    current_metrics, current_pi, current_logpi = policy_metrics(current_logits, current_values, data)
    common_metrics, common_pi, common_logpi = policy_metrics(common_logits, common_values, data)
    payload = torch.load(rnad_checkpoint, map_location="cpu", weights_only=True)
    forces = reconstructed_forces(current, data, current_logits, current_values, payload)
    result = {
        "purpose": "Frozen CPU diagnostic, not evaluation or recipe change", "seed": seed,
        "games": games, "positions": len(data["actions"]), "rollout": rollout_metrics,
        "state_distribution": "Fresh R-NaD checkpoint self-play; both policies measured on identical visited observations",
        "checkpoint_sha256": hashes, "rnad_metadata": metadata,
        "common_policy": common_metrics, "rnad_policy": current_metrics,
        "warm_start_drift": {
            "mean_kl_common_to_rnad": float((common_pi * (common_logpi - current_logpi)).sum(-1).mean()),
            "mean_kl_rnad_to_common": float((current_pi * (current_logpi - common_logpi)).sum(-1).mean()),
            "top_exact_action_agreement": float((common_logits.argmax(-1) == current_logits.argmax(-1)).float().mean()),
        },
        "neurd": forces, "parameters_and_buffers_unchanged": True,
        "optimizer_constructed": False, "optimizer_steps": 0, "device": "cpu",
        "source_sha256": {name: digest(Path(__file__).resolve().parents[1] / name) for name in (
            "scripts/diagnose_rnad_policy.py", "src/rnad.py", "src/exact_policy.py", "scripts/train_exact_policy.py"
        )},
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }
    for model, snapshot in zip((common, current), snapshots):
        verify_frozen(model, snapshot)
    if hashes != {"common": digest(common_checkpoint), "rnad": digest(rnad_checkpoint)}:
        raise RuntimeError("Checkpoint changed during diagnostics")
    result["sampled_actions_sha256"] = hashlib.sha256(
        data["actions"].gather(-1, data["action"][:, None]).numpy().tobytes()
    ).hexdigest()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", required=True, type=Path)
    parser.add_argument("--rnad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7300000000001)
    args = parser.parse_args()
    report = diagnose(args.common, args.rnad, args.output, args.games, args.seed)
    print(json.dumps({"output": str(args.output), "positions": report["positions"], "force_status": report["neurd"]["status"]}))
