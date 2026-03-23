from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def init_policy_params(
    key: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden1: int = 128,
    hidden2: int = 128,
) -> dict[str, jnp.ndarray]:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    scale = 0.08
    return {
        "w1": scale * jax.random.normal(k1, (obs_dim, hidden1)),
        "b1": jnp.zeros((hidden1,), dtype=jnp.float32),
        "w2": scale * jax.random.normal(k2, (hidden1, hidden2)),
        "b2": jnp.zeros((hidden2,), dtype=jnp.float32),
        "w_mu": scale * jax.random.normal(k3, (hidden2, action_dim)),
        "b_mu": jnp.zeros((action_dim,), dtype=jnp.float32),
        "w_v": scale * jax.random.normal(k4, (hidden2, 1)),
        "b_v": jnp.zeros((1,), dtype=jnp.float32),
        "log_std": -0.35 + 0.02 * jax.random.normal(k5, (action_dim,)),
        "value_gain": jnp.array([1.0], dtype=jnp.float32)
        + 0.0 * jax.random.normal(k6, (1,), dtype=jnp.float32),
    }


def _forward_hidden(params: dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    h1 = jnp.tanh(obs @ params["w1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["w2"] + params["b2"])
    return h2


def policy_forward(params: dict[str, jnp.ndarray], obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    h = _forward_hidden(params, obs)
    mu = h @ params["w_mu"] + params["b_mu"]
    value = (h @ params["w_v"] + params["b_v"])[0] * params["value_gain"][0]
    std = jnp.exp(params["log_std"])
    return mu, value, std


def gaussian_log_prob(action: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    var = jnp.square(std)
    return -0.5 * (((action - mean) ** 2) / (var + 1e-8) + 2.0 * jnp.log(std + 1e-8) + jnp.log(2.0 * jnp.pi))


def sample_action(
    params: dict[str, jnp.ndarray],
    obs: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean, value, std = policy_forward(params, obs)
    noise = jax.random.normal(key, mean.shape)
    action = jnp.clip(mean + noise * std, -1.0, 1.0)
    logp = jnp.sum(gaussian_log_prob(action, mean, std))
    return action, logp, value, mean


def deterministic_action(params: dict[str, jnp.ndarray], obs: jnp.ndarray) -> jnp.ndarray:
    mean, _, _ = policy_forward(params, obs)
    return jnp.clip(mean, -1.0, 1.0)


def loss_components(
    params: dict[str, jnp.ndarray],
    obs_batch: jnp.ndarray,
    act_batch: jnp.ndarray,
    old_logp_batch: jnp.ndarray,
    adv_batch: jnp.ndarray,
    return_batch: jnp.ndarray,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    means, values, stds = jax.vmap(policy_forward, in_axes=(None, 0))(params, obs_batch)
    logp = jnp.sum(gaussian_log_prob(act_batch, means, stds), axis=-1)

    ratio = jnp.exp(logp - old_logp_batch)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -jnp.mean(jnp.minimum(ratio * adv_batch, clipped * adv_batch))

    value_loss = jnp.mean(jnp.square(return_batch - values))
    entropy = jnp.mean(jnp.sum(0.5 + 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(stds + 1e-8), axis=-1))
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy

    aux = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": 0.5 * jnp.mean(jnp.square(old_logp_batch - logp)),
        "clip_fraction": jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)),
    }
    return total, aux


def to_numpy_tree(tree: Any) -> Any:
    return jax.tree.map(lambda x: jax.device_get(x), tree)
