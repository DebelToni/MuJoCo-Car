from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy import deterministic_action
from src.tank_env import EnvConfig, TankNavEnv


def load_policy(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def plot_rollout(sensor_hist: np.ndarray, dist_hist: np.ndarray, reward_hist: np.ndarray, out_path: Path) -> None:
    t = np.arange(sensor_hist.shape[0])
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)

    axes[0].plot(t, sensor_hist[:, 0], label="sensor pi/5")
    axes[0].plot(t, sensor_hist[:, 1], label="sensor 2pi/5")
    axes[0].plot(t, sensor_hist[:, 2], label="sensor 3pi/5")
    axes[0].plot(t, sensor_hist[:, 3], label="sensor 4pi/5")
    axes[0].set_title("Normalized Front Sensor Distances")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Distance / range")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(t, dist_hist, color="#2a9d8f", label="target distance (m)")
    ax2 = axes[1].twinx()
    ax2.plot(t, reward_hist, color="#e76f51", alpha=0.7, label="reward")

    axes[1].set_title("Task Progress")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Meters")
    ax2.set_ylabel("Reward")

    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render deterministic tank rollout")
    parser.add_argument("--xml", type=Path, default=Path("mjcf/tank_car_nav.xml"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/main_run/policy.pkl"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--camera", type=str, default="topdown")
    parser.add_argument("--substep-frames", action="store_true")
    parser.add_argument("--gif", type=Path, default=Path("outputs/main_run/rollout.gif"))
    parser.add_argument("--chart", type=Path, default=Path("outputs/main_run/rollout_dashboard.png"))
    args = parser.parse_args()

    env = TankNavEnv(EnvConfig(xml_path=args.xml, max_steps=args.steps))
    policy = load_policy(args.policy)

    key = jax.random.PRNGKey(args.seed)
    state, obs = env.reset(key)

    host_data = mujoco.MjData(env.host_model)
    renderer = mujoco.Renderer(env.host_model, width=640, height=480)

    frames: list[np.ndarray] = []
    sensors: list[np.ndarray] = []
    dists: list[float] = []
    rewards: list[float] = []
    sensor_idx_start = (env.stack_size - 1) * 4
    sensor_idx_end = sensor_idx_start + 4

    for _ in range(args.steps):
        action = deterministic_action(policy, obs)
        if args.substep_frames:
            state, obs, reward, done, metrics, trajectory = env.step_with_trajectory(state, action)
            for sub in trajectory:
                host_data.qpos[:] = sub["qpos"]
                host_data.qvel[:] = sub["qvel"]
                host_data.ctrl[:] = sub["ctrl"]
                mujoco.mj_forward(env.host_model, host_data)

                renderer.update_scene(host_data, camera=args.camera)
                frames.append(renderer.render())
        else:
            state, obs, reward, done, metrics = env.step(state, action)
            host_data.qpos[:] = np.asarray(state["data"].qpos)
            host_data.qvel[:] = np.asarray(state["data"].qvel)
            host_data.ctrl[:] = np.asarray(state["data"].ctrl)
            mujoco.mj_forward(env.host_model, host_data)
            renderer.update_scene(host_data, camera=args.camera)
            frames.append(renderer.render())

        if obs.shape[0] >= sensor_idx_end:
            sensors.append(np.asarray(obs[sensor_idx_start:sensor_idx_end], dtype=np.float32))
        else:
            sensors.append(np.full((4,), np.nan, dtype=np.float32))
        dists.append(float(np.asarray(metrics["dist_target"])))
        rewards.append(float(np.asarray(reward)))

        if bool(np.asarray(done)):
            break

    args.gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.gif, frames, fps=30)
    print(f"Saved rollout GIF: {args.gif}")

    plot_rollout(np.asarray(sensors), np.asarray(dists), np.asarray(rewards), args.chart)
    print(f"Saved rollout chart: {args.chart}")


if __name__ == "__main__":
    main()
