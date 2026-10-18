"""Generate golden R-NaD targets by executing unmodified official functions.

Run in an isolated environment containing JAX, NumPy and Chex:
  python rnad_reference_validation.py /path/to/official/rnad.py output.npz
The checked source is OpenSpiel commit 0af322f, including PR #1157.
AST extraction avoids importing unrelated OpenSpiel game/Haiku dependencies.
"""

import ast
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util as tree

jax.config.update("jax_enable_x64", True)
source = Path(sys.argv[1]).read_text()
expected_sha = "c22c161705a0c9edc7ead61c3942174ce9f9de390c595d4148afbd6749c0f16b"
if hashlib.sha256(source.encode()).hexdigest() != expected_sha:
    raise ValueError("Source does not match the pinned official implementation")
names = {
    "_policy_ratio", "_where", "_has_played", "v_trace", "get_loss_v",
    "apply_force_with_threshold", "renormalize", "get_loss_nerd",
}
parsed = ast.parse(source)
parsed.body = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
if len(parsed.body) != len(names):
    raise ValueError("Official source is missing a required function")
namespace = {
    "chex": chex, "jax": jax, "jnp": jnp, "lax": lax, "tree": tree,
    "Any": Any, "Sequence": Sequence, "Tuple": tuple,
}
exec(compile(parsed, sys.argv[1], "exec"), namespace)  # noqa: S102 - SHA-pinned upstream functions
v_trace = namespace["v_trace"]
get_loss_v = namespace["get_loss_v"]
get_loss_nerd = namespace["get_loss_nerd"]

# JAX >=0.8 renamed a_min/a_max to min/max. This shim changes API spelling
# only; the original official function AST is not edited.
clip_impl = jnp.clip


def clip_compat(a, a_min=None, a_max=None, **kwargs):
    return clip_impl(a, min=a_min, max=a_max, **kwargs)


jnp.clip = clip_compat
rng = np.random.default_rng(20261017)
t_count, batch, a_count = 9, 4, 7
valid = np.arange(t_count)[:, None] < np.array([9, 6, 8, 4])[None]
player_ids = rng.integers(0, 2, valid.shape)
legal = rng.random((t_count, batch, a_count)) > 0.3
legal[..., 0] = True
# Invalid padded rows deliberately retain arbitrary valid action lists, as in
# the official environment. Valid masks must isolate all boundary data.
logits = rng.normal(size=legal.shape) * 4
logits[~legal] = 1000


def distribution(logit):
    log_pi = jax.nn.log_softmax(jnp.where(legal, logit, -jnp.inf), axis=-1)
    return jnp.where(legal, jnp.exp(log_pi), 0), jnp.where(legal, log_pi, 0)


pi, log_pi = distribution(jnp.asarray(logits))
behavior, _ = distribution(jnp.asarray(rng.normal(size=legal.shape)))
_, reference = distribution(jnp.asarray(rng.normal(size=legal.shape)))
_, previous = distribution(jnp.asarray(rng.normal(size=legal.shape)))
alpha = 0.35
reg = log_pi - (alpha * reference + (1 - alpha) * previous)
actions = np.zeros(valid.shape, np.int64)
for t in range(t_count):
    for b in range(batch):
        actions[t, b] = rng.choice(a_count, p=np.asarray(behavior[t, b]))
actions_oh = jax.nn.one_hot(actions, a_count)
target_values = rng.normal(size=valid.shape)
values = rng.normal(size=valid.shape)
reward0 = rng.normal(size=valid.shape) * 0.05
for b, length in enumerate(valid.sum(axis=0)):
    reward0[length - 1, b] += rng.choice([-1, 1])
rewards = np.stack((reward0, -reward0), axis=-1)
eta, c, rho, lambda_ = 0.2, 0.8, 2.3, 0.9
target_v, masks, target_q = [], [], []
for player in (0, 1):
    sign = np.where(player_ids == player, 1, -1) * valid
    v, mask, q = v_trace(
        jnp.asarray(target_values[..., None]), jnp.asarray(valid), jnp.asarray(player_ids),
        behavior, pi, reg, jnp.asarray(sign[..., None]), actions_oh,
        jnp.asarray(rewards[..., player]), player,
        eta=eta, lambda_=lambda_, c=c, rho=rho,
    )
    target_v.append(v)
    masks.append(mask)
    target_q.append(q)


def official_nerd(x):
    return get_loss_nerd(
        [x, x], [pi, pi], target_q, jnp.asarray(valid), jnp.asarray(player_ids),
        jnp.asarray(legal), [jnp.ones((*valid.shape, 1))] * 2, clip=1.7, threshold=2,
    )


nerd_value, nerd_grad = jax.value_and_grad(official_nerd)(jnp.asarray(logits))
critic_value, critic_grad = jax.value_and_grad(
    lambda x: get_loss_v([x, x], target_v, masks)
)(jnp.asarray(values[..., None]))
np.savez_compressed(
    sys.argv[2], logits=logits, values=values, target_values=target_values,
    valid=valid, player_ids=player_ids, legal=legal, behavior=behavior,
    policy=pi, reference=reference, previous_reference=previous,
    regularizer=reg, actions=actions, rewards=rewards,
    config=np.array([eta, c, rho, lambda_, alpha]),
    target_v=np.stack(target_v), target_q=np.stack(target_q), masks=np.stack(masks),
    nerd_value=nerd_value, nerd_grad=nerd_grad,
    critic_value=critic_value, critic_grad=critic_grad,
)
print({
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "fixture_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "jax": jax.__version__, "nerd_loss": float(nerd_value), "critic_loss": float(critic_value),
})
