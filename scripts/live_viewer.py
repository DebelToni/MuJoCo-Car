from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.policy import deterministic_action
from src.tank_env import EnvConfig, TankNavEnv


def load_policy(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live viewer for trained tank policy")
    parser.add_argument("--xml", type=Path, default=Path("mjcf/tank_car_nav.xml"))
    parser.add_argument("--policy", type=Path, default=Path("outputs/main_run/policy.pkl"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--camera", type=str, default="topdown")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    env = TankNavEnv(EnvConfig(xml_path=args.xml, max_steps=min(900, args.steps)))
    policy = load_policy(args.policy)

    episode_idx = 0

    def _reset(ep: int):
        return env.reset(jax.random.PRNGKey(args.seed + ep))

    state, obs = _reset(episode_idx)
    data = mujoco.MjData(env.host_model)
    dt = env.host_model.opt.timestep * env.config.frame_skip * env.config.action_hold_steps

    with mujoco.viewer.launch_passive(env.host_model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(env.host_model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)

        for _ in range(args.steps):
            t0 = time.time()

            action = deterministic_action(policy, obs)
            state, obs, _, done, _ = env.step(state, action)

            data.qpos[:] = np.asarray(state["data"].qpos)
            data.qvel[:] = np.asarray(state["data"].qvel)
            data.ctrl[:] = np.asarray(state["data"].ctrl)
            mujoco.mj_forward(env.host_model, data)
            viewer.sync()

            if bool(np.asarray(done)):
                if not args.loop:
                    break
                episode_idx += 1
                state, obs = _reset(episode_idx)

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
