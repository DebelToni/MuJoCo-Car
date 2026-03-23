from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy import deterministic_action, init_policy_params, loss_components, sample_action, to_numpy_tree
from src.tank_env import EnvConfig, TankNavEnv


def _save_artifacts(
    output_dir: Path,
    params: dict[str, jnp.ndarray],
    history: list[dict[str, float]],
    eval_rows: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "policy.pkl").open("wb") as f:
        pickle.dump(to_numpy_tree(params), f)

    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(eval_rows, f, indent=2)

    if history:
        iters = np.arange(1, len(history) + 1)
        reward = np.asarray([h["reward_mean"] for h in history], dtype=np.float32)
        success = np.asarray([h["success_rate"] for h in history], dtype=np.float32)
        collision = np.asarray([h["collision_rate"] for h in history], dtype=np.float32)
        value_loss = np.asarray([h["value_loss"] for h in history], dtype=np.float32)

        plt.style.use("seaborn-v0_8-darkgrid")
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

        axes[0, 0].plot(iters, reward, color="#2a9d8f")
        axes[0, 0].set_title("Parallel Rollout Reward")

        axes[0, 1].plot(iters, success, color="#264653", label="success")
        axes[0, 1].plot(iters, collision, color="#e76f51", label="collision")
        axes[0, 1].set_title("Parallel Rollout Success/Collision")
        axes[0, 1].legend(loc="best", fontsize=8)

        axes[1, 0].plot(iters, value_loss, color="#9c6644")
        axes[1, 0].set_title("Value Loss")

        if eval_rows:
            ex = np.arange(1, len(eval_rows) + 1)
            acc = np.asarray([r["accuracy"] for r in eval_rows], dtype=np.float32)
            axes[1, 1].plot(ex, acc, marker="o", color="#1d3557")
            axes[1, 1].set_ylim(0.0, 1.02)
            axes[1, 1].set_title("Eval Accuracy")
        else:
            axes[1, 1].axis("off")

        fig.savefig(output_dir / "training_curves.png", dpi=170)
        plt.close(fig)


def gae_and_returns(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    lam: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    t_steps, n_envs = rewards.shape
    adv = jnp.zeros((t_steps, n_envs), dtype=jnp.float32)
    gae = jnp.zeros((n_envs,), dtype=jnp.float32)
    next_value = jnp.zeros((n_envs,), dtype=jnp.float32)

    for t in range(t_steps - 1, -1, -1):
        non_terminal = 1.0 - dones[t].astype(jnp.float32)
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv = adv.at[t].set(gae)
        next_value = values[t]

    returns = adv + values
    return adv, returns


def evaluate_policy(
    env: TankNavEnv,
    params: dict[str, jnp.ndarray],
    seed: int,
    episodes: int,
    max_steps: int,
) -> dict[str, float]:
    success: list[float] = []
    collision: list[float] = []
    final_dist: list[float] = []
    total_reward: list[float] = []

    for ep in range(episodes):
        key = jax.random.PRNGKey(seed + ep)
        state, obs = env.reset(key)
        rew = 0.0
        s = 0.0
        c = 0.0
        d = float(np.asarray(state["prev_dist"]))

        for _ in range(max_steps):
            action = deterministic_action(params, obs)
            state, obs, reward, done, metrics = env.step(state, action)
            rew += float(np.asarray(reward))
            s = max(s, float(np.asarray(metrics["success"])))
            c = max(c, float(np.asarray(metrics["collision"])))
            d = float(np.asarray(metrics["dist_target"]))
            if bool(np.asarray(done)):
                break

        success.append(s)
        collision.append(c)
        final_dist.append(d)
        total_reward.append(rew)

    return {
        "eval_success_rate": float(np.mean(success)),
        "eval_collision_rate": float(np.mean(collision)),
        "eval_distance_mean": float(np.mean(final_dist)),
        "eval_reward_mean": float(np.mean(total_reward)),
        "accuracy": float(np.mean(success)),
    }


def collect_parallel_batch(
    env: TankNavEnv,
    params: dict[str, jnp.ndarray],
    rng: jax.Array,
    num_envs: int,
    horizon: int,
    gamma: float,
    gae_lam: float,
):
    vm_reset = jax.jit(jax.vmap(env.reset))
    vm_step = jax.jit(jax.vmap(env.step))
    vm_sample = jax.jit(jax.vmap(sample_action, in_axes=(None, 0, 0)))

    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, num_envs)
    states, obs = vm_reset(reset_keys)

    done_mask = jnp.zeros((num_envs,), dtype=jnp.bool_)

    obs_hist = []
    act_hist = []
    logp_hist = []
    val_hist = []
    rew_hist = []
    done_hist = []
    active_hist = []
    succ_hist = []
    coll_hist = []
    dist_hist = []

    for _ in range(horizon):
        rng, act_key = jax.random.split(rng)
        act_keys = jax.random.split(act_key, num_envs)

        active = jnp.logical_not(done_mask)
        actions, logp, values, _ = vm_sample(params, obs, act_keys)
        next_states, next_obs, reward, done, metrics = vm_step(states, actions)

        reward = jnp.where(active, reward, 0.0)
        done_eff = jnp.where(active, done, jnp.ones_like(done, dtype=jnp.bool_))

        obs_hist.append(obs)
        act_hist.append(actions)
        logp_hist.append(logp)
        val_hist.append(values)
        rew_hist.append(reward)
        done_hist.append(done_eff)
        active_hist.append(active)
        succ_hist.append(metrics["success"])
        coll_hist.append(metrics["collision"])
        dist_hist.append(metrics["dist_target"])

        done_mask = done_eff
        states = next_states
        obs = next_obs

    obs_t = jnp.stack(obs_hist, axis=0)
    act_t = jnp.stack(act_hist, axis=0)
    logp_t = jnp.stack(logp_hist, axis=0)
    val_t = jnp.stack(val_hist, axis=0)
    rew_t = jnp.stack(rew_hist, axis=0)
    done_t = jnp.stack(done_hist, axis=0)
    active_t = jnp.stack(active_hist, axis=0)

    adv_t, ret_t = gae_and_returns(rew_t, val_t, done_t, gamma=gamma, lam=gae_lam)

    mask_b = active_t.reshape((-1,))
    obs_b = obs_t.reshape((-1, obs_t.shape[-1]))[mask_b]
    act_b = act_t.reshape((-1, act_t.shape[-1]))[mask_b]
    logp_b = logp_t.reshape((-1,))[mask_b]
    adv_b = adv_t.reshape((-1,))[mask_b]
    ret_b = ret_t.reshape((-1,))[mask_b]

    adv_b = (adv_b - jnp.mean(adv_b)) / (jnp.std(adv_b) + 1e-6)

    batch = {
        "obs": obs_b,
        "act": act_b,
        "logp": logp_b,
        "adv": adv_b,
        "ret": ret_b,
    }

    active_np = np.asarray(active_t)
    succ_np = np.asarray(jnp.stack(succ_hist, axis=0))
    coll_np = np.asarray(jnp.stack(coll_hist, axis=0))
    dist_np = np.asarray(jnp.stack(dist_hist, axis=0))

    succ = float(np.mean(np.max(np.where(active_np, succ_np, 0.0), axis=0)))
    coll = float(np.mean(np.max(np.where(active_np, coll_np, 0.0), axis=0)))

    active_counts = np.maximum(active_np.sum(axis=0) - 1, 0).astype(np.int32)
    dist = float(np.mean(dist_np[active_counts, np.arange(num_envs)]))
    reward_mean = float(np.mean(np.sum(np.asarray(rew_t), axis=0)))

    metrics = {
        "reward_mean": reward_mean,
        "success_rate": succ,
        "collision_rate": coll,
        "distance_mean": dist,
        "samples": int(obs_b.shape[0]),
    }
    return batch, metrics, rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel MJX PPO training for GPU")
    parser.add_argument("--xml", type=Path, default=Path("mjcf/tank_car_nav.xml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=90)
    parser.add_argument("--update-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.7)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/parallel_gpu_run"))
    args = parser.parse_args()

    env = TankNavEnv(EnvConfig(xml_path=args.xml, max_steps=args.max_steps))

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = init_policy_params(init_key, env.obs_size, env.action_size)

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(p: dict[str, jnp.ndarray], s: optax.OptState, batch: dict[str, jnp.ndarray]):
        (loss, aux), grads = jax.value_and_grad(loss_components, has_aux=True)(
            p,
            batch["obs"],
            batch["act"],
            batch["logp"],
            batch["adv"],
            batch["ret"],
            args.clip_eps,
            args.value_coef,
            args.entropy_coef,
        )
        updates, s2 = optimizer.update(grads, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, loss, aux

    history: list[dict[str, float]] = []
    eval_rows: list[dict[str, float]] = []
    start = time.monotonic()

    horizon = min(args.horizon, args.max_steps)

    for it in range(1, args.iterations + 1):
        batch, rollout_metrics, rng = collect_parallel_batch(
            env=env,
            params=params,
            rng=rng,
            num_envs=args.num_envs,
            horizon=horizon,
            gamma=args.gamma,
            gae_lam=args.gae_lam,
        )

        aux = {
            "policy_loss": jnp.array(0.0),
            "value_loss": jnp.array(0.0),
            "entropy": jnp.array(0.0),
            "approx_kl": jnp.array(0.0),
            "clip_fraction": jnp.array(0.0),
        }
        for _ in range(args.update_epochs):
            params, opt_state, _, aux = update_step(params, opt_state, batch)

        row = {
            "iter": it,
            **rollout_metrics,
            "policy_loss": float(np.asarray(aux["policy_loss"])),
            "value_loss": float(np.asarray(aux["value_loss"])),
            "entropy": float(np.asarray(aux["entropy"])),
            "approx_kl": float(np.asarray(aux["approx_kl"])),
            "clip_fraction": float(np.asarray(aux["clip_fraction"])),
            "elapsed_min": float((time.monotonic() - start) / 60.0),
        }
        history.append(row)
        print(json.dumps(row))

        if it % args.eval_every == 0 or it == args.iterations:
            e = evaluate_policy(env, params, seed=args.seed + 100_000 * it, episodes=args.eval_episodes, max_steps=args.max_steps)
            eval_row = {"iter": it, **e}
            eval_rows.append(eval_row)
            print(json.dumps({"eval": eval_row}))

        _save_artifacts(args.output_dir, params, history, eval_rows)

    final = {
        "finished": True,
        "iterations": args.iterations,
        "best_accuracy": max((r["accuracy"] for r in eval_rows), default=0.0),
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(final))


if __name__ == "__main__":
    main()
