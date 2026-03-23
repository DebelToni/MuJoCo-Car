from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


def _name_to_id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise ValueError(f"Missing '{name}' in model for object type {obj_type}.")
    return int(idx)


@dataclass
class EnvConfig:
    xml_path: Path
    frame_skip: int = 4
    action_hold_steps: int = 6
    max_steps: int = 420
    sensor_range: float = 3.0

    room_half_size: float = 1.0
    wall_thickness: float = 0.05
    interior_wall_length: float = 0.50
    interior_wall_width: float = 0.05

    car_spawn_margin: float = 0.22
    target_spawn_margin: float = 0.22
    min_spawn_separation: float = 0.35
    target_success_radius: float = 0.12

    car_radius: float = 0.10
    max_track_speed: float = 1.25
    track_separation: float = 0.036


class TankNavEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.host_model = mujoco.MjModel.from_xml_path(str(config.xml_path))
        self.model = mjx.put_model(self.host_model)

        self.nu = int(self.host_model.nu)
        self.nq = int(self.host_model.nq)
        self.nv = int(self.host_model.nv)

        self.ctrl_min = jnp.asarray(self.host_model.actuator_ctrlrange[:, 0], dtype=jnp.float32)
        self.ctrl_max = jnp.asarray(self.host_model.actuator_ctrlrange[:, 1], dtype=jnp.float32)

        self.car_x_qpos, self.car_x_qvel = self._joint_indices("x_joint")
        self.car_y_qpos, self.car_y_qvel = self._joint_indices("y_joint")
        self.car_yaw_qpos, self.car_yaw_qvel = self._joint_indices("yaw_joint")

        self.target_x_qpos, _ = self._joint_indices("target_x")
        self.target_y_qpos, _ = self._joint_indices("target_y")

        self.home_qpos = jnp.asarray(self.host_model.qpos0, dtype=jnp.float32)

        self.sensor_angles = jnp.asarray(
            [jnp.pi / 5.0, 2.0 * jnp.pi / 5.0, 3.0 * jnp.pi / 5.0, 4.0 * jnp.pi / 5.0],
            dtype=jnp.float32,
        )
        self.sensor_front_offset = jnp.asarray([0.0, 0.052], dtype=jnp.float32)
        self.stack_size = 3

        self.obs_size = 4 * self.stack_size + 3
        self.action_size = 2

        self.kp_lin = jnp.array(58.0, dtype=jnp.float32)
        self.kp_yaw = jnp.array(9.5, dtype=jnp.float32)

        self.wall_centers, self.wall_half_extents, self.wall_yaws = self._build_wall_boxes()

        self.reset = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    def _joint_indices(self, joint_name: str) -> tuple[int, int]:
        jid = _name_to_id(self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.host_model.jnt_qposadr[jid]), int(self.host_model.jnt_dofadr[jid])

    def _build_wall_boxes(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        room = self.config.room_half_size
        thick_half = 0.5 * self.config.wall_thickness

        perimeter_centers = jnp.asarray(
            [
                [0.0, room + thick_half],
                [0.0, -(room + thick_half)],
                [room + thick_half, 0.0],
                [-(room + thick_half), 0.0],
            ],
            dtype=jnp.float32,
        )
        perimeter_half_extents = jnp.asarray(
            [
                [room + thick_half, thick_half],
                [room + thick_half, thick_half],
                [thick_half, room + thick_half],
                [thick_half, room + thick_half],
            ],
            dtype=jnp.float32,
        )
        perimeter_yaws = jnp.zeros((4,), dtype=jnp.float32)

        inner_half_len = 0.5 * self.config.interior_wall_length
        inner_half_wid = 0.5 * self.config.interior_wall_width
        inner_centers = jnp.asarray(
            [
                [0.35, 0.20],
                [-0.30, -0.25],
            ],
            dtype=jnp.float32,
        )
        inner_half_extents = jnp.asarray(
            [
                [inner_half_len, inner_half_wid],
                [inner_half_len, inner_half_wid],
            ],
            dtype=jnp.float32,
        )
        inner_yaws = jnp.asarray([0.0, 0.5 * jnp.pi], dtype=jnp.float32)

        centers = jnp.concatenate([perimeter_centers, inner_centers], axis=0)
        half_extents = jnp.concatenate([perimeter_half_extents, inner_half_extents], axis=0)
        yaws = jnp.concatenate([perimeter_yaws, inner_yaws], axis=0)
        return centers, half_extents, yaws

    @staticmethod
    def _rotate(v: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        return jnp.asarray([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=jnp.float32)

    @staticmethod
    def _to_local(point: jnp.ndarray, center: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        rel = point - center
        return jnp.asarray([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]], dtype=jnp.float32)

    @staticmethod
    def _ray_box_distance(
        origin: jnp.ndarray,
        direction: jnp.ndarray,
        center: jnp.ndarray,
        half_extents: jnp.ndarray,
        yaw: jnp.ndarray,
        max_range: jnp.ndarray,
    ) -> jnp.ndarray:
        o = TankNavEnv._to_local(origin, center, yaw)
        d_world_point = origin + direction
        d = TankNavEnv._to_local(d_world_point, center, yaw) - o

        eps = 1e-6
        big = 1e4

        parallel_x = jnp.abs(d[0]) < eps
        parallel_y = jnp.abs(d[1]) < eps

        tx1 = (-half_extents[0] - o[0]) / jnp.where(parallel_x, 1.0, d[0])
        tx2 = (half_extents[0] - o[0]) / jnp.where(parallel_x, 1.0, d[0])
        ty1 = (-half_extents[1] - o[1]) / jnp.where(parallel_y, 1.0, d[1])
        ty2 = (half_extents[1] - o[1]) / jnp.where(parallel_y, 1.0, d[1])

        tx_near = jnp.minimum(tx1, tx2)
        tx_far = jnp.maximum(tx1, tx2)
        ty_near = jnp.minimum(ty1, ty2)
        ty_far = jnp.maximum(ty1, ty2)

        tx_near = jnp.where(parallel_x, -big, tx_near)
        tx_far = jnp.where(parallel_x, big, tx_far)
        ty_near = jnp.where(parallel_y, -big, ty_near)
        ty_far = jnp.where(parallel_y, big, ty_far)

        valid_parallel = jnp.logical_and(
            jnp.logical_or(jnp.logical_not(parallel_x), jnp.abs(o[0]) <= half_extents[0]),
            jnp.logical_or(jnp.logical_not(parallel_y), jnp.abs(o[1]) <= half_extents[1]),
        )

        t_enter = jnp.maximum(tx_near, ty_near)
        t_exit = jnp.minimum(tx_far, ty_far)
        hit = jnp.logical_and(valid_parallel, t_exit >= jnp.maximum(t_enter, 0.0))

        dist = jnp.maximum(t_enter, 0.0)
        return jnp.where(hit, jnp.minimum(dist, max_range), max_range)

    @staticmethod
    def _point_box_signed_distance(
        point: jnp.ndarray,
        center: jnp.ndarray,
        half_extents: jnp.ndarray,
        yaw: jnp.ndarray,
    ) -> jnp.ndarray:
        p_local = TankNavEnv._to_local(point, center, yaw)
        q = jnp.abs(p_local) - half_extents
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0))
        inside = jnp.minimum(jnp.maximum(q[0], q[1]), 0.0)
        return outside + inside

    def _sensor_distances(self, car_xy: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
        sensor_origin = car_xy + self._rotate(self.sensor_front_offset, yaw)
        angles_world = self.sensor_angles + yaw
        dirs = jnp.stack([jnp.cos(angles_world), jnp.sin(angles_world)], axis=1)
        max_range = jnp.asarray(self.config.sensor_range, dtype=jnp.float32)

        def cast_one(d: jnp.ndarray) -> jnp.ndarray:
            dists = jax.vmap(
                lambda c, h, a: self._ray_box_distance(sensor_origin, d, c, h, a, max_range)
            )(self.wall_centers, self.wall_half_extents, self.wall_yaws)
            return jnp.min(dists)

        return jax.vmap(cast_one)(dirs)

    def _raw_sensor_observation(self, data: Any) -> jnp.ndarray:
        car_xy = jnp.asarray([data.qpos[self.car_x_qpos], data.qpos[self.car_y_qpos]], dtype=jnp.float32)
        yaw = data.qpos[self.car_yaw_qpos]
        dists = self._sensor_distances(car_xy, yaw)
        return jnp.clip(dists / self.config.sensor_range, 0.0, 1.0)

    def _goal_features(self, car_xy: jnp.ndarray, yaw: jnp.ndarray, target_xy: jnp.ndarray) -> jnp.ndarray:
        heading = yaw + 0.5 * jnp.pi
        heading_vec = jnp.asarray([jnp.cos(heading), jnp.sin(heading)], dtype=jnp.float32)
        to_goal = target_xy - car_xy
        dist = jnp.linalg.norm(to_goal)
        goal_dir = to_goal / (dist + 1e-6)

        cos_err = jnp.dot(heading_vec, goal_dir)
        sin_err = heading_vec[0] * goal_dir[1] - heading_vec[1] * goal_dir[0]
        dist_norm = jnp.clip(dist / (2.0 * self.config.room_half_size), 0.0, 1.0)
        return jnp.asarray([dist_norm, cos_err, sin_err], dtype=jnp.float32)

    def _compose_observation(
        self,
        sensor_hist: jnp.ndarray,
        car_xy: jnp.ndarray,
        yaw: jnp.ndarray,
        target_xy: jnp.ndarray,
    ) -> jnp.ndarray:
        sensor_flat = jnp.reshape(sensor_hist, (-1,))
        goal_features = self._goal_features(car_xy, yaw, target_xy)
        return jnp.concatenate([sensor_flat, goal_features], axis=0)

    def _sample_xy(self, key: jax.Array, margin: float) -> jnp.ndarray:
        lim = self.config.room_half_size - margin
        return jax.random.uniform(key, (2,), minval=-lim, maxval=lim, dtype=jnp.float32)

    def _reset(self, key: jax.Array) -> tuple[dict[str, Any], jnp.ndarray]:
        key, car_key, target_key, yaw_key, sep_key = jax.random.split(key, 5)

        qpos = jnp.array(self.home_qpos)
        qvel = jnp.zeros((self.nv,), dtype=jnp.float32)

        car_xy = self._sample_xy(car_key, self.config.car_spawn_margin)
        target_xy = self._sample_xy(target_key, self.config.target_spawn_margin)
        yaw = jax.random.uniform(yaw_key, (), minval=-jnp.pi, maxval=jnp.pi)

        vec = target_xy - car_xy
        dist = jnp.linalg.norm(vec)
        sep_angle = jax.random.uniform(sep_key, (), minval=-jnp.pi, maxval=jnp.pi)
        fallback_dir = jnp.asarray([jnp.cos(sep_angle), jnp.sin(sep_angle)], dtype=jnp.float32)
        dir_vec = jnp.where(dist > 1e-3, vec / (dist + 1e-6), fallback_dir)
        adjusted_dist = jnp.maximum(dist, self.config.min_spawn_separation)
        target_xy = car_xy + adjusted_dist * dir_vec
        lim_t = self.config.room_half_size - self.config.target_spawn_margin
        target_xy = jnp.clip(target_xy, -lim_t, lim_t)

        qpos = qpos.at[self.car_x_qpos].set(car_xy[0])
        qpos = qpos.at[self.car_y_qpos].set(car_xy[1])
        qpos = qpos.at[self.car_yaw_qpos].set(yaw)
        qpos = qpos.at[self.target_x_qpos].set(target_xy[0])
        qpos = qpos.at[self.target_y_qpos].set(target_xy[1])

        ctrl = jnp.zeros((self.nu,), dtype=jnp.float32)
        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.model, data)

        dist0 = jnp.linalg.norm(target_xy - car_xy)
        raw_obs = self._raw_sensor_observation(data)
        sensor_hist = jnp.repeat(raw_obs[None, :], self.stack_size, axis=0)

        state = {
            "data": data,
            "step": jnp.array(0, dtype=jnp.int32),
            "prev_dist": dist0,
            "target_xy": target_xy,
            "success": jnp.array(False),
            "collision": jnp.array(False),
            "last_action": jnp.zeros((2,), dtype=jnp.float32),
            "sensor_hist": sensor_hist,
            "rng": key,
        }
        obs = self._compose_observation(sensor_hist, car_xy, yaw, target_xy)
        return state, obs

    def _step(self, state: dict[str, Any], action: jnp.ndarray):
        action = jnp.clip(action, -1.0, 1.0)

        data = state["data"]
        yaw = data.qpos[self.car_yaw_qpos]
        heading_cmd = yaw + 0.5 * jnp.pi

        left = action[0]
        right = action[1]
        v_l = self.config.max_track_speed * left
        v_r = self.config.max_track_speed * right
        v = 0.5 * (v_l + v_r)
        omega = (v_r - v_l) / (self.config.track_separation + 1e-6)

        vx_des = v * jnp.cos(heading_cmd)
        vy_des = v * jnp.sin(heading_cmd)
        wz_des = omega

        qvel = data.qvel
        fx = self.kp_lin * (vx_des - qvel[self.car_x_qvel])
        fy = self.kp_lin * (vy_des - qvel[self.car_y_qvel])
        tz = self.kp_yaw * (wz_des - qvel[self.car_yaw_qvel])

        ctrl = jnp.asarray([fx, fy, tz, 4.0 * left, 4.0 * right], dtype=jnp.float32)
        ctrl = jnp.clip(ctrl, self.ctrl_min, self.ctrl_max)
        data = data.replace(ctrl=ctrl)

        def _substep(_, d):
            return mjx.step(self.model, d)

        total_substeps = self.config.frame_skip * self.config.action_hold_steps
        data = jax.lax.fori_loop(0, total_substeps, _substep, data)

        car_xy = jnp.asarray([data.qpos[self.car_x_qpos], data.qpos[self.car_y_qpos]], dtype=jnp.float32)
        yaw = data.qpos[self.car_yaw_qpos]
        target_xy = state["target_xy"]

        sensor_dists = self._sensor_distances(car_xy, yaw)
        raw_obs = jnp.clip(sensor_dists / self.config.sensor_range, 0.0, 1.0)
        sensor_hist = jnp.concatenate([state["sensor_hist"][1:], raw_obs[None, :]], axis=0)
        obs = self._compose_observation(sensor_hist, car_xy, yaw, target_xy)

        dist_target = jnp.linalg.norm(target_xy - car_xy)
        progress = state["prev_dist"] - dist_target

        clearances = jax.vmap(
            lambda c, h, a: self._point_box_signed_distance(car_xy, c, h, a) - self.config.car_radius
        )(self.wall_centers, self.wall_half_extents, self.wall_yaws)
        min_clearance = jnp.min(clearances)
        collision = min_clearance < 0.0

        heading = yaw + 0.5 * jnp.pi
        heading_vec = jnp.asarray([jnp.cos(heading), jnp.sin(heading)], dtype=jnp.float32)
        goal_vec = (target_xy - car_xy) / (dist_target + 1e-6)
        heading_align = jnp.dot(heading_vec, goal_vec)

        sensor_min = jnp.min(sensor_dists)
        obstacle_penalty = jnp.clip((1.0 - sensor_min / 0.45), 0.0, 1.0)

        action_cost = 0.008 * jnp.sum(jnp.square(action - state["last_action"]))
        success = dist_target < self.config.target_success_radius
        out_of_bounds = jnp.any(jnp.abs(car_xy) > (self.config.room_half_size + 0.10))

        reward = (
            24.0 * progress
            + 0.60 * heading_align
            - 0.35 * obstacle_penalty
            - action_cost
            - 0.01
            + jnp.where(success, 30.0, 0.0)
            - jnp.where(collision, 10.0, 0.0)
            - jnp.where(out_of_bounds, 3.0, 0.0)
        )

        step = state["step"] + 1
        done = jnp.logical_or(step >= self.config.max_steps, jnp.logical_or(success, jnp.logical_or(collision, out_of_bounds)))

        new_state = {
            "data": data,
            "step": step,
            "prev_dist": dist_target,
            "target_xy": target_xy,
            "success": jnp.logical_or(state["success"], success),
            "collision": jnp.logical_or(state["collision"], collision),
            "last_action": action,
            "sensor_hist": sensor_hist,
            "rng": state["rng"],
        }

        metrics = {
            "dist_target": dist_target,
            "progress": progress,
            "sensor_min": sensor_min,
            "heading_align": heading_align,
            "min_clearance": min_clearance,
            "success": success.astype(jnp.float32),
            "collision": collision.astype(jnp.float32),
        }
        return new_state, obs, reward, done, metrics
