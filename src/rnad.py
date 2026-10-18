# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch R-NaD update for complete two-player, zero-sum trajectories.

Adapted from OpenSpiel's R-NaD implementation at commit 0af322f, including
the corrected legal-action logit centering from PR #1157:
https://github.com/google-deepmind/open_spiel/blob/0af322f/open_spiel/python/algorithms/rnad/rnad.py
Paper: https://arxiv.org/abs/2206.15378

This retains player-aware V-trace, reward regularization against rotating
reference policies, EMA value targets, and thresholded NeuRD forces. It is not
a PPO objective. Differences: PyTorch, a single repeating reference interval,
no optional policy discretization/fine tuning, and a pretrained shared network.
Inputs must contain complete episodes, each in its own batch column. Rewards
are received AFTER each action; padding follows the final action. Truncated
episodes must not silently be treated as terminal by this API.
"""

import copy
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RNaDConfig:
    eta: float = 0.2
    c: float = 1.0
    rho: float = float("inf")
    lambda_: float = 1.0
    nerd_threshold: float = 2.0
    nerd_clip: float = 10_000.0
    target_avg: float = 0.001
    reference_interval: int = 20_000

    def __post_init__(self):
        if self.reference_interval < 2:
            raise ValueError("reference_interval must be at least two updates")
        if not 0 < self.target_avg <= 1:
            raise ValueError("target_avg must be in (0,1]")
        if self.eta < 0 or self.c < 0 or self.rho <= 0:
            raise ValueError("Invalid regularization or importance sampling limits")


def legal_policy(logits, legal):
    """Return policy and finite log policy; illegal/padded entries are zero."""
    legal = legal.bool()
    masked = logits.masked_fill(~legal, -torch.inf)
    # Padding can have no actions. A dummy row avoids undefined softmax(-inf).
    masked = torch.where(legal.any(-1, keepdim=True), masked, 0.0)
    log_policy = masked.log_softmax(-1)
    return log_policy.exp().masked_fill(~legal, 0), log_policy.masked_fill(~legal, 0)


@torch.no_grad()
def v_trace(
    values, valid, player_ids, behavior_policy, policy, log_policy_ratio,
    actions, rewards, player, config=None,
):
    """Official alternating-player V-trace recurrence, with detached outputs.

    Scalars are [T,B], distributions [T,B,A], and rewards here are [T,B]
    from the requested player's perspective. Value estimates are from the
    actor's perspective and only read as this player's estimates on own turns.
    Returns own-turn value targets, own-turn mask, and per-action Q estimates.
    """
    config = config or RNaDConfig()
    if values.ndim == 3:
        values = values.squeeze(-1)
    valid = valid.bool()
    indices = actions.long().clamp_min(0).unsqueeze(-1)
    mu = behavior_policy.gather(-1, indices).squeeze(-1)
    pi = policy.gather(-1, indices).squeeze(-1)
    mu = torch.where(valid, mu, 1.0)
    pi = torch.where(valid, pi, 1.0)
    if torch.any(valid & (mu <= 0)):
        raise ValueError("Every sampled valid action needs positive behavior probability")
    ratio, inv_mu = pi / mu, 1 / mu
    onehot = torch.zeros_like(policy).scatter_(-1, indices, 1)
    sign = torch.where(player_ids == player, 1.0, -1.0) * valid
    entropy_reward = -config.eta * (policy * log_policy_ratio).sum(-1) * sign
    regularization = -config.eta * log_policy_ratio * sign.unsqueeze(-1)
    value_targets, q_targets = torch.zeros_like(values), torch.zeros_like(policy)
    zero = torch.zeros_like(values[0])
    carry_reward = zero.clone()
    carry_uncorrected = zero.clone()
    next_value = zero.clone()
    next_target = zero.clone()
    carry_importance = torch.ones_like(zero)
    for t in range(len(values) - 1, -1, -1):
        live, ours = valid[t], player_ids[t] == player
        uncorrected = rewards[t] + carry_uncorrected + entropy_reward[t]
        discounted = rewards[t] + carry_reward
        combined_is = ratio[t] * carry_importance
        target = (
            values[t]
            + combined_is.clamp(max=config.rho) * (uncorrected + next_value - values[t])
            + config.lambda_ * combined_is.clamp(max=config.c) * (next_target - next_value)
        )
        q = (
            values[t].unsqueeze(-1) + regularization[t]
            + onehot[t] * inv_mu[t].unsqueeze(-1)
            * (discounted + carry_importance * next_target - values[t]).unsqueeze(-1)
        )
        active = live & ours
        value_targets[t] = torch.where(active, target, 0)
        q_targets[t] = torch.where(active.unsqueeze(-1), q, 0)
        carry_reward = torch.where(live & ~ours, entropy_reward[t] + ratio[t] * discounted, 0)
        carry_uncorrected = torch.where(live & ~ours, uncorrected, 0)
        next_value = torch.where(live, torch.where(ours, values[t], next_value), 0)
        next_target = torch.where(live, torch.where(ours, target, next_target), 0)
        carry_importance = torch.where(live & ~ours, combined_is, 1)
    return value_targets, valid & (player_ids == player), q_targets


def nerd_loss(logits, policy, q_targets, legal, mask, threshold=2.0, clip=10_000.0):
    """Thresholded NeuRD logit force, centered on LEGAL logits only.

    Keeping the mean differentiable projects the force onto the legal logit
    subspace. Illegal logits never contribute, including their arbitrary pad
    constants. The detached force has no softmax-Jacobian damping.
    """
    legal = legal.bool()
    valid_logits = torch.where(legal, logits, 0)
    mean = valid_logits.sum(-1, keepdim=True) / legal.sum(-1, keepdim=True).clamp_min(1)
    centered = torch.where(legal, logits - mean, 0)
    advantage = (q_targets - (policy * q_targets).sum(-1, keepdim=True)).detach()
    advantage = advantage.clamp(-clip, clip)
    force = (
        (centered.detach() > -threshold) * advantage.clamp(max=0)
        + (centered.detach() < threshold) * advantage.clamp(min=0)
    ).detach()
    per_state = -(centered * force * legal).sum(-1)
    return (per_state * mask).sum() / mask.sum().clamp_min(1)


def rnad_loss(
    logits, values, target_values, behavior_policy, reference_log_policy,
    previous_reference_log_policy, actions, legal, valid, player_ids, rewards,
    alpha, config=None,
):
    """Loss and diagnostic tensors; input shapes follow module documentation."""
    config = config or RNaDConfig()
    if values.ndim == 3:
        values = values.squeeze(-1)
    policy, log_policy = legal_policy(logits, legal)
    with torch.no_grad():
        log_ratio = log_policy - (
            alpha * reference_log_policy + (1 - alpha) * previous_reference_log_policy
        )
        log_ratio = log_ratio.masked_fill(~legal.bool(), 0)
    critic_loss, policy_loss = logits.new_zeros(()), logits.new_zeros(())
    targets = []
    for player in range(2):
        target, mask, q = v_trace(
            target_values, valid, player_ids, behavior_policy, policy, log_ratio,
            actions, rewards[..., player], player, config,
        )
        critic_loss = critic_loss + ((values - target).square() * mask).sum() / mask.sum().clamp_min(1)
        policy_loss = policy_loss + nerd_loss(
            logits, policy, q, legal, mask, config.nerd_threshold, config.nerd_clip,
        )
        targets.append(target)
    denominator = valid.sum().clamp_min(1)
    metrics = {
        "critic_loss": critic_loss.detach(), "nerd_loss": policy_loss.detach(),
        "entropy": (-(policy * log_policy).sum(-1) * valid).sum().detach() / denominator,
        "regularizer_kl": ((policy * log_ratio).sum(-1) * valid).sum().detach() / denominator,
        "value_target_abs": ((targets[0] + targets[1]).abs() * valid).sum() / denominator,
        "reference_alpha": logits.new_tensor(alpha),
    }
    return critic_loss + policy_loss, metrics


class RNaDState:
    """Frozen EMA target and two rotating regularizers, fully resumable."""

    def __init__(self, model, config=None):
        self.config = config or RNaDConfig()
        self.learner_steps = 0
        self.target = copy.deepcopy(model).eval().requires_grad_(False)
        self.reference = copy.deepcopy(model).eval().requires_grad_(False)
        self.previous_reference = copy.deepcopy(model).eval().requires_grad_(False)

    @property
    def alpha(self):
        interval = self.config.reference_interval
        return min(2 * (self.learner_steps % interval) / interval, 1.0)

    @torch.no_grad()
    def after_update(self, model):
        """Call exactly once AFTER the optimizer update, including on resume."""
        target = self.target.state_dict()
        for name, value in model.state_dict().items():
            if target[name].is_floating_point():
                target[name].lerp_(value, self.config.target_avg)
            else:
                target[name].copy_(value)
        rotate = (self.learner_steps + 1) % self.config.reference_interval == 0
        if rotate:
            self.previous_reference.load_state_dict(self.reference.state_dict())
            self.reference.load_state_dict(target)
        self.learner_steps += 1
        return rotate

    def state_dict(self):
        return {
            "config": asdict(self.config), "learner_steps": self.learner_steps,
            "target": self.target.state_dict(), "reference": self.reference.state_dict(),
            "previous_reference": self.previous_reference.state_dict(),
        }

    def load_state_dict(self, state):
        self.config = RNaDConfig(**state["config"])
        self.learner_steps = state["learner_steps"]
        for name in ("target", "reference", "previous_reference"):
            getattr(self, name).load_state_dict(state[name])
