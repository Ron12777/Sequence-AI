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
"""Isolated execution candidate: scripted R-NaD scan, no production integration.

The public wrapper retains the original checks and preprocessing. Only the
backward recurrence moves from Python dispatch to TorchScript. No compiler,
precision, learner recipe, or global execution setting changes are made.
"""

import torch

from src.rnad import RNaDConfig


@torch.jit.script
def _scan(
    values: torch.Tensor,
    valid: torch.Tensor,
    player_ids: torch.Tensor,
    ratio: torch.Tensor,
    inv_mu: torch.Tensor,
    onehot: torch.Tensor,
    entropy_reward: torch.Tensor,
    regularization: torch.Tensor,
    rewards: torch.Tensor,
    player: int,
    c: float,
    rho: float,
    lambda_: float,
):
    value_targets, q_targets = torch.zeros_like(values), torch.zeros_like(onehot)
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
            + combined_is.clamp(max=rho) * (uncorrected + next_value - values[t])
            + lambda_ * combined_is.clamp(max=c) * (next_target - next_value)
        )
        q = (
            values[t].unsqueeze(-1)
            + regularization[t]
            + onehot[t]
            * inv_mu[t].unsqueeze(-1)
            * (discounted + carry_importance * next_target - values[t]).unsqueeze(-1)
        )
        active = live & ours
        value_targets[t] = torch.where(active, target, 0)
        q_targets[t] = torch.where(active.unsqueeze(-1), q, 0)
        carry_reward = torch.where(
            live & ~ours, entropy_reward[t] + ratio[t] * discounted, 0
        )
        carry_uncorrected = torch.where(live & ~ours, uncorrected, 0)
        next_value = torch.where(live, torch.where(ours, values[t], next_value), 0)
        next_target = torch.where(live, torch.where(ours, target, next_target), 0)
        carry_importance = torch.where(live & ~ours, combined_is, 1)
    return value_targets, valid & (player_ids == player), q_targets


@torch.no_grad()
def v_trace(
    values,
    valid,
    player_ids,
    behavior_policy,
    policy,
    log_policy_ratio,
    actions,
    rewards,
    player,
    config=None,
):
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
        raise ValueError(
            "Every sampled valid action needs positive behavior probability"
        )
    ratio, inv_mu = pi / mu, 1 / mu
    onehot = torch.zeros_like(policy).scatter_(-1, indices, 1)
    sign = torch.where(player_ids == player, 1.0, -1.0) * valid
    entropy_reward = -config.eta * (policy * log_policy_ratio).sum(-1) * sign
    regularization = -config.eta * log_policy_ratio * sign.unsqueeze(-1)
    return _scan(
        values,
        valid,
        player_ids,
        ratio,
        inv_mu,
        onehot,
        entropy_reward,
        regularization,
        rewards,
        player,
        float(config.c),
        float(config.rho),
        float(config.lambda_),
    )
