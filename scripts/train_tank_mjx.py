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

from src.policy import (
    deterministic_action,
    init_policy_params,
    loss_components,
    policy_forward,
    sample_action,
    to_numpy_tree,
)
from src.tank_env import EnvConfig, TankNavEnv


def _load_json_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(rows, list):
        return rows
    return []


def _save_artifacts(
    output_dir: Path,
    params: dict[str, jnp.ndarray],
    history: list[dict[str, float]],
    segment_rows: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = output_dir / "policy.pkl"
    with policy_path.open("wb") as f:
        pickle.dump(to_numpy_tree(params), f)

    latest_policy_path = output_dir / "policy_latest.pkl"
    with latest_policy_path.open("wb") as f:
        pickle.dump(to_numpy_tree(params), f)

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    segments_path = output_dir / "episode_summaries.json"
    segments_path.write_text(json.dumps(segment_rows, indent=2), encoding="utf-8")

    chart_path = output_dir / "training_curves.png"
    plot_training(history, segment_rows, chart_path)


def gae_and_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


def collect_batch(
    env: TankNavEnv,
    params: dict[str, jnp.ndarray],
    rng: jax.Array,
    episodes: int,
    max_steps: int,
    gamma: float,
    gae_lam: float,
):
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    logp_all: list[np.ndarray] = []
    ret_all: list[np.ndarray] = []
    adv_all: list[np.ndarray] = []

    ep_rewards: list[float] = []
    ep_lens: list[int] = []
    ep_success: list[float] = []
    ep_collision: list[float] = []
    ep_final_dist: list[float] = []

    for ep in range(episodes):
        rng, reset_key, act_key = jax.random.split(rng, 3)
        state, obs = env.reset(reset_key)

        traj_obs: list[np.ndarray] = []
        traj_act: list[np.ndarray] = []
        traj_logp: list[float] = []
        traj_val: list[float] = []
        traj_reward: list[float] = []
        traj_done: list[float] = []

        success = 0.0
        collision = 0.0
        final_dist = float(np.asarray(state["prev_dist"]))

        for _ in range(max_steps):
            act_key, sample_key = jax.random.split(act_key)
            action, logp, value, _ = sample_action(params, obs, sample_key)
            state, obs_next, reward, done, metrics = env.step(state, action)

            traj_obs.append(np.asarray(obs, dtype=np.float32))
            traj_act.append(np.asarray(action, dtype=np.float32))
            traj_logp.append(float(logp))
            traj_val.append(float(value))
            traj_reward.append(float(reward))
            done_f = float(np.asarray(done))
            traj_done.append(done_f)

            success = max(success, float(np.asarray(metrics["success"])))
            collision = max(collision, float(np.asarray(metrics["collision"])))
            final_dist = float(np.asarray(metrics["dist_target"]))

            obs = obs_next
            if done_f > 0.5:
                break

        rewards = np.asarray(traj_reward, dtype=np.float32)
        values = np.asarray(traj_val, dtype=np.float32)
        dones = np.asarray(traj_done, dtype=np.float32)
        adv, ret = gae_and_returns(rewards, values, dones, gamma=gamma, lam=gae_lam)

        obs_all.append(np.asarray(traj_obs, dtype=np.float32))
        act_all.append(np.asarray(traj_act, dtype=np.float32))
        logp_all.append(np.asarray(traj_logp, dtype=np.float32))
        adv_all.append(np.asarray(adv, dtype=np.float32))
        ret_all.append(np.asarray(ret, dtype=np.float32))

        ep_rewards.append(float(np.sum(rewards)))
        ep_lens.append(len(rewards))
        ep_success.append(success)
        ep_collision.append(collision)
        ep_final_dist.append(final_dist)

    obs_batch = jnp.asarray(np.concatenate(obs_all, axis=0))
    act_batch = jnp.asarray(np.concatenate(act_all, axis=0))
    logp_batch = jnp.asarray(np.concatenate(logp_all, axis=0))
    adv_batch = jnp.asarray(np.concatenate(adv_all, axis=0))
    ret_batch = jnp.asarray(np.concatenate(ret_all, axis=0))

    adv_batch = (adv_batch - jnp.mean(adv_batch)) / (jnp.std(adv_batch) + 1e-6)

    batch = {
        "obs": obs_batch,
        "act": act_batch,
        "logp": logp_batch,
        "adv": adv_batch,
        "ret": ret_batch,
    }

    metrics = {
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "len_mean": float(np.mean(ep_lens)),
        "success_rate": float(np.mean(ep_success)),
        "collision_rate": float(np.mean(ep_collision)),
        "distance_mean": float(np.mean(ep_final_dist)),
    }
    return batch, metrics, rng


def oracle_action_from_state(env: TankNavEnv, state: dict[str, jnp.ndarray]) -> np.ndarray:
    data = state["data"]
    car = np.asarray([data.qpos[env.car_x_qpos], data.qpos[env.car_y_qpos]], dtype=np.float32)
    yaw = float(np.asarray(data.qpos[env.car_yaw_qpos]))
    heading = yaw + 0.5 * np.pi

    target = np.asarray(state["target_xy"], dtype=np.float32)
    obstacles = np.asarray(state["obstacles_xy"], dtype=np.float32)
    obstacle_radii = np.asarray(env.obstacle_radii, dtype=np.float32)

    to_goal = target - car
    goal_dist = np.linalg.norm(to_goal) + 1e-6
    force = to_goal / goal_dist

    for center, radius in zip(obstacles, obstacle_radii, strict=False):
        rel = car - center
        center_dist = np.linalg.norm(rel) + 1e-6
        clearance = center_dist - (float(radius) + float(env.config.car_radius))
        if clearance < 2.2:
            strength = 1.35 / (clearance + 0.45)
            force += strength * (rel / center_dist)

    desired_heading = float(np.arctan2(force[1], force[0]))
    err = (desired_heading - heading + np.pi) % (2.0 * np.pi) - np.pi

    speed = float(np.clip(1.0 * np.exp(-0.22 * abs(err)), 0.22, 1.0))
    yaw_rate = float(np.clip(2.7 * err, -2.6, 2.6))

    sep = float(env.config.track_separation)
    max_speed = float(env.config.max_track_speed)
    v_l = speed - 0.5 * yaw_rate * sep
    v_r = speed + 0.5 * yaw_rate * sep

    action = np.asarray([v_l / max_speed, v_r / max_speed], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)


def collect_bc_dataset(
    env: TankNavEnv,
    seed: int,
    episodes: int,
    max_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []

    for ep in range(episodes):
        state, obs = env.reset(jax.random.PRNGKey(seed + ep))
        for _ in range(max_steps):
            action_np = oracle_action_from_state(env, state)
            action = jnp.asarray(action_np)
            obs_all.append(np.asarray(obs, dtype=np.float32))
            act_all.append(action_np.astype(np.float32))
            state, obs, _, done, _ = env.step(state, action)
            if bool(np.asarray(done)):
                break

    return jnp.asarray(np.asarray(obs_all, dtype=np.float32)), jnp.asarray(np.asarray(act_all, dtype=np.float32))


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


def plot_training(history: list[dict[str, float]], segment_rows: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    iters = np.arange(1, len(history) + 1)
    reward = np.asarray([h["reward_mean"] for h in history], dtype=np.float32)
    success = np.asarray([h["success_rate"] for h in history], dtype=np.float32)
    collision = np.asarray([h["collision_rate"] for h in history], dtype=np.float32)
    value_loss = np.asarray([h["value_loss"] for h in history], dtype=np.float32)

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(iters, reward, color="#2a9d8f", marker="o", markersize=3)
    axes[0, 0].set_title("Rollout Reward")
    axes[0, 0].set_xlabel("Iteration")

    axes[0, 1].plot(iters, success, color="#264653", label="success")
    axes[0, 1].plot(iters, collision, color="#e76f51", label="collision")
    axes[0, 1].set_title("Rollout Success / Collision")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].legend(loc="best", fontsize=8)

    axes[1, 0].plot(iters, value_loss, color="#9c6644")
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_xlabel("Iteration")

    if segment_rows:
        seg_x = np.arange(1, len(segment_rows) + 1)
        seg_acc = np.asarray([r["accuracy"] for r in segment_rows], dtype=np.float32)
        axes[1, 1].plot(seg_x, seg_acc, color="#1d3557", marker="o")
        axes[1, 1].axhline(0.95, color="#d62828", linestyle="--", linewidth=1)
        axes[1, 1].set_ylim(0.0, 1.02)
        axes[1, 1].set_title("Episode Accuracy")
        axes[1, 1].set_xlabel("Training episode")
        axes[1, 1].set_ylabel("Eval success rate")
    else:
        axes[1, 1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tank-car navigation PPO with MuJoCo MJX")
    parser.add_argument("--xml", type=Path, default=Path("mjcf/tank_car_nav.xml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-minutes", type=float, default=30.0)
    parser.add_argument("--max-training-episodes", type=int, default=6)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--rollout-episodes", type=int, default=18)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--update-epochs", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lam", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.7)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--bc-episodes", type=int, default=80)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--resume-policy", type=Path, default=None)
    parser.add_argument("--checkpoint-every-iters", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/main_run"))
    args = parser.parse_args()

    env = TankNavEnv(EnvConfig(xml_path=args.xml, max_steps=args.max_steps))

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    params = init_policy_params(init_key, env.obs_size, env.action_size)

    if args.resume_policy is not None:
        with args.resume_policy.open("rb") as f:
            loaded_params = pickle.load(f)
        params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float32), loaded_params)

        if params["w1"].shape[0] != env.obs_size:
            raise ValueError(
                f"Resume policy obs dim mismatch: policy expects {params['w1'].shape[0]}, env provides {env.obs_size}."
            )
        if params["w_mu"].shape[1] != env.action_size:
            raise ValueError(
                f"Resume policy action dim mismatch: policy expects {params['w_mu'].shape[1]}, env provides {env.action_size}."
            )

    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(
        p: dict[str, jnp.ndarray],
        s: optax.OptState,
        batch: dict[str, jnp.ndarray],
    ):
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

    @jax.jit
    def bc_update_step(
        p: dict[str, jnp.ndarray],
        s: optax.OptState,
        obs_batch: jnp.ndarray,
        act_batch: jnp.ndarray,
    ):
        def bc_loss_fn(params: dict[str, jnp.ndarray]):
            means, _, _ = jax.vmap(policy_forward, in_axes=(None, 0))(params, obs_batch)
            return jnp.mean(jnp.square(means - act_batch))

        loss, grads = jax.value_and_grad(bc_loss_fn)(p)
        updates, s2 = optimizer.update(grads, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, loss

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "train_metrics.json"
    segments_path = args.output_dir / "episode_summaries.json"

    history: list[dict[str, float]] = _load_json_rows(metrics_path)
    segment_rows: list[dict[str, float]] = _load_json_rows(segments_path)

    if args.resume_policy is not None:
        print(json.dumps({"resume": True, "resume_policy": str(args.resume_policy)}))

    if args.resume_policy is None and args.bc_episodes > 0 and args.bc_epochs > 0:
        bc_obs, bc_act = collect_bc_dataset(
            env=env,
            seed=args.seed + 100_000,
            episodes=args.bc_episodes,
            max_steps=args.max_steps,
        )
        for epoch in range(1, args.bc_epochs + 1):
            params, opt_state, bc_loss = bc_update_step(params, opt_state, bc_obs, bc_act)
            if epoch == 1 or epoch % 10 == 0 or epoch == args.bc_epochs:
                print(json.dumps({"bc_epoch": epoch, "bc_loss": float(np.asarray(bc_loss))}))

    solved = False
    global_iter = max((int(r.get("global_iter", 0)) for r in history), default=0)
    start_episode = len(segment_rows) + 1

    if start_episode > args.max_training_episodes:
        done_row = {
            "finished": True,
            "solved": False,
            "target_accuracy": args.target_accuracy,
            "best_accuracy": max((r["accuracy"] for r in segment_rows), default=0.0),
            "episodes_run": len(segment_rows),
            "note": "No remaining episodes to run; increase --max-training-episodes to continue.",
        }
        print(json.dumps(done_row))
        return

    for train_ep in range(start_episode, args.max_training_episodes + 1):
        segment_start = time.monotonic()
        max_seconds = args.episode_minutes * 60.0
        local_iter = 0

        while time.monotonic() - segment_start < max_seconds:
            batch, rollout_metrics, rng = collect_batch(
                env=env,
                params=params,
                rng=rng,
                episodes=args.rollout_episodes,
                max_steps=args.max_steps,
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

            global_iter += 1
            local_iter += 1
            row = {
                "global_iter": global_iter,
                "training_episode": train_ep,
                "episode_iter": local_iter,
                **rollout_metrics,
                "policy_loss": float(np.asarray(aux["policy_loss"])),
                "value_loss": float(np.asarray(aux["value_loss"])),
                "entropy": float(np.asarray(aux["entropy"])),
                "approx_kl": float(np.asarray(aux["approx_kl"])),
                "clip_fraction": float(np.asarray(aux["clip_fraction"])),
                "samples": int(np.asarray(batch["obs"]).shape[0]),
                "elapsed_min": float((time.monotonic() - segment_start) / 60.0),
            }
            history.append(row)
            print(json.dumps(row))

            if args.checkpoint_every_iters > 0 and global_iter % args.checkpoint_every_iters == 0:
                _save_artifacts(args.output_dir, params, history, segment_rows)

        eval_seed = args.seed + 10_000 * train_ep
        eval_metrics = evaluate_policy(
            env=env,
            params=params,
            seed=eval_seed,
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
        )
        seg_row = {
            "training_episode": train_ep,
            "minutes_budget": args.episode_minutes,
            "iterations": local_iter,
            **eval_metrics,
        }
        segment_rows.append(seg_row)
        print(json.dumps({"segment_eval": seg_row}))

        _save_artifacts(args.output_dir, params, history, segment_rows)

        if eval_metrics["accuracy"] >= args.target_accuracy:
            solved = True
            break

    done_row = {
        "finished": True,
        "solved": solved,
        "target_accuracy": args.target_accuracy,
        "best_accuracy": max((r["accuracy"] for r in segment_rows), default=0.0),
        "episodes_run": len(segment_rows),
    }
    print(json.dumps(done_row))


if __name__ == "__main__":
    main()
