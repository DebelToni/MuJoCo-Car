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
    max_steps: int = 520
    sensor_range: float = 12.0
    target_min_dist: float = 5.0
    target_max_dist: float = 10.0
    spawn_angle_min: float = 0.45 * jnp.pi
    spawn_angle_max: float = 0.55 * jnp.pi
    obstacle_min_dist: float = 1.8
    obstacle_max_dist: float = 9.5
    world_limit: float = 14.0
    target_half_size: float = 0.025
    target_success_radius: float = 0.12
    obstacle_half_sizes: tuple[tuple[float, float], ...] = (
        (0.05, 0.05),
        (0.05, 0.05),
        (0.05, 0.05),
        (0.05, 0.05),
        (0.50, 0.05),
        (0.50, 0.05),
        (0.50, 0.05),
        (0.50, 0.05),
    )
    obstacle_yaws: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5 * jnp.pi,
        0.0,
        0.5 * jnp.pi,
    )
    car_radius: float = 0.17
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

        if len(config.obstacle_half_sizes) != 8:
            raise ValueError("This scene expects exactly 8 obstacle footprints.")
        if len(config.obstacle_yaws) != 8:
            raise ValueError("This scene expects exactly 8 obstacle yaws.")
        self.obstacle_half_extents = jnp.asarray(config.obstacle_half_sizes, dtype=jnp.float32)
        self.obstacle_yaws = jnp.asarray(config.obstacle_yaws, dtype=jnp.float32)
        self.obstacle_footprint_radii = jnp.linalg.norm(self.obstacle_half_extents, axis=1)

        obstacle_x_idx = []
        obstacle_y_idx = []
        for i in range(8):
            ox, _ = self._joint_indices(f"obstacle_{i}_x")
            oy, _ = self._joint_indices(f"obstacle_{i}_y")
            obstacle_x_idx.append(ox)
            obstacle_y_idx.append(oy)
        self.obstacle_x_qpos_idx = jnp.asarray(obstacle_x_idx, dtype=jnp.int32)
        self.obstacle_y_qpos_idx = jnp.asarray(obstacle_y_idx, dtype=jnp.int32)

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

        self.reset = jax.jit(self._reset)
        self.step = jax.jit(self._step)

    def _joint_indices(self, joint_name: str) -> tuple[int, int]:
        jid = _name_to_id(self.host_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.host_model.jnt_qposadr[jid]), int(self.host_model.jnt_dofadr[jid])

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

    def _sensor_distances(self, car_xy: jnp.ndarray, yaw: jnp.ndarray, obstacles_xy: jnp.ndarray):
        sensor_origin = car_xy + self._rotate(self.sensor_front_offset, yaw)

        angles_world = self.sensor_angles + yaw
        dirs = jnp.stack([jnp.cos(angles_world), jnp.sin(angles_world)], axis=1)
        max_range = jnp.asarray(self.config.sensor_range, dtype=jnp.float32)

        def cast_one(d: jnp.ndarray) -> jnp.ndarray:
            dists = jax.vmap(
                lambda c, h, a: self._ray_box_distance(sensor_origin, d, c, h, a, max_range)
            )(obstacles_xy, self.obstacle_half_extents, self.obstacle_yaws)
            return jnp.min(dists)

        return jax.vmap(cast_one)(dirs)

    def _raw_sensor_observation(self, data: Any, obstacles_xy: jnp.ndarray) -> jnp.ndarray:
        car_xy = jnp.asarray([data.qpos[self.car_x_qpos], data.qpos[self.car_y_qpos]], dtype=jnp.float32)
        yaw = data.qpos[self.car_yaw_qpos]
        dists = self._sensor_distances(car_xy, yaw, obstacles_xy)
        return jnp.clip(dists / self.config.sensor_range, 0.0, 1.0)

    def _goal_features(self, car_xy: jnp.ndarray, yaw: jnp.ndarray, target_xy: jnp.ndarray) -> jnp.ndarray:
        heading = yaw + 0.5 * jnp.pi
        heading_vec = jnp.asarray([jnp.cos(heading), jnp.sin(heading)], dtype=jnp.float32)
        to_goal = target_xy - car_xy
        dist = jnp.linalg.norm(to_goal)
        goal_dir = to_goal / (dist + 1e-6)

        cos_err = jnp.dot(heading_vec, goal_dir)
        sin_err = heading_vec[0] * goal_dir[1] - heading_vec[1] * goal_dir[0]
        dist_norm = jnp.clip(dist / self.config.target_max_dist, 0.0, 1.0)
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

    def _spawn_target(self, key: jax.Array) -> tuple[jax.Array, jnp.ndarray]:
        key, k_r, k_a = jax.random.split(key, 3)
        r = jax.random.uniform(
            k_r,
            (),
            minval=jnp.asarray(self.config.target_min_dist, dtype=jnp.float32),
            maxval=jnp.asarray(self.config.target_max_dist, dtype=jnp.float32),
        )
        a = jax.random.uniform(
            k_a,
            (),
            minval=jnp.asarray(self.config.spawn_angle_min, dtype=jnp.float32),
            maxval=jnp.asarray(self.config.spawn_angle_max, dtype=jnp.float32),
        )
        target = jnp.asarray([r * jnp.cos(a), r * jnp.sin(a)], dtype=jnp.float32)
        return key, target

    def _spawn_obstacles(self, key: jax.Array, target_xy: jnp.ndarray) -> tuple[jax.Array, jnp.ndarray]:
        key, k_r, k_a = jax.random.split(key, 3)
        n = self.obstacle_half_extents.shape[0]

        radii = jax.random.uniform(
            k_r,
            (n,),
            minval=jnp.asarray(self.config.obstacle_min_dist, dtype=jnp.float32),
            maxval=jnp.asarray(self.config.obstacle_max_dist, dtype=jnp.float32),
        )
        angles = jax.random.uniform(
            k_a,
            (n,),
            minval=jnp.asarray(-jnp.pi, dtype=jnp.float32),
            maxval=jnp.asarray(jnp.pi, dtype=jnp.float32),
        )

        pts = jnp.stack([radii * jnp.cos(angles), radii * jnp.sin(angles)], axis=1)

        dist_from_car = jnp.linalg.norm(pts, axis=1)
        min_from_car = 1.2 + self.obstacle_footprint_radii
        scale_car = jnp.maximum(1.0, min_from_car / (dist_from_car + 1e-6))
        pts = pts * scale_car[:, None]

        vec_t = pts - target_xy[None, :]
        dist_t = jnp.linalg.norm(vec_t, axis=1)
        min_from_target = 1.0 + self.obstacle_footprint_radii + self.config.target_half_size
        scale_t = jnp.maximum(1.0, min_from_target / (dist_t + 1e-6))
        pts = target_xy[None, :] + vec_t * scale_t[:, None]

        target_dist = jnp.linalg.norm(target_xy) + 1e-6
        fwd = target_xy / target_dist
        side = jnp.asarray([-fwd[1], fwd[0]], dtype=jnp.float32)

        proj = jnp.sum(pts * fwd[None, :], axis=1)
        lat = jnp.sum(pts * side[None, :], axis=1)
        in_segment = jnp.logical_and(proj > 0.6, proj < target_dist - 0.8)
        corridor_half = 0.9 + self.obstacle_footprint_radii
        corridor_hit = jnp.logical_and(in_segment, jnp.abs(lat) < corridor_half)
        lat_target = jnp.sign(lat + 1e-3) * corridor_half
        pts = pts + jnp.where(corridor_hit, lat_target - lat, 0.0)[:, None] * side[None, :]

        pts = jnp.clip(pts, -self.config.world_limit + 0.5, self.config.world_limit - 0.5)
        return key, pts

    def _reset(self, key: jax.Array) -> tuple[dict[str, Any], jnp.ndarray]:
        key, target_key, obs_key = jax.random.split(key, 3)

        qpos = jnp.array(self.home_qpos)
        qvel = jnp.zeros((self.nv,), dtype=jnp.float32)

        qpos = qpos.at[self.car_x_qpos].set(0.0)
        qpos = qpos.at[self.car_y_qpos].set(0.0)
        qpos = qpos.at[self.car_yaw_qpos].set(0.0)

        _, target_xy = self._spawn_target(target_key)
        _, obstacles_xy = self._spawn_obstacles(obs_key, target_xy)

        qpos = qpos.at[self.target_x_qpos].set(target_xy[0])
        qpos = qpos.at[self.target_y_qpos].set(target_xy[1])

        qpos = qpos.at[self.obstacle_x_qpos_idx].set(obstacles_xy[:, 0])
        qpos = qpos.at[self.obstacle_y_qpos_idx].set(obstacles_xy[:, 1])

        ctrl = jnp.zeros((self.nu,), dtype=jnp.float32)
        data = mjx.make_data(self.model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.model, data)

        dist0 = jnp.linalg.norm(target_xy)
        raw_obs = self._raw_sensor_observation(data, obstacles_xy)
        sensor_hist = jnp.repeat(raw_obs[None, :], self.stack_size, axis=0)

        state = {
            "data": data,
            "step": jnp.array(0, dtype=jnp.int32),
            "prev_dist": dist0,
            "target_xy": target_xy,
            "obstacles_xy": obstacles_xy,
            "success": jnp.array(False),
            "collision": jnp.array(False),
            "last_action": jnp.zeros((2,), dtype=jnp.float32),
            "sensor_hist": sensor_hist,
            "rng": key,
        }
        car_xy = jnp.asarray([data.qpos[self.car_x_qpos], data.qpos[self.car_y_qpos]], dtype=jnp.float32)
        yaw = data.qpos[self.car_yaw_qpos]
        obs = self._compose_observation(sensor_hist, car_xy, yaw, target_xy)
        return state, obs

    def _step(self, state: dict[str, Any], action: jnp.ndarray):
        action = jnp.clip(action, -1.0, 1.0)

        data = state["data"]
        yaw = data.qpos[self.car_yaw_qpos]
        heading = yaw + 0.5 * jnp.pi

        left = action[0]
        right = action[1]
        v_l = self.config.max_track_speed * left
        v_r = self.config.max_track_speed * right
        v = 0.5 * (v_l + v_r)
        omega = (v_r - v_l) / (self.config.track_separation + 1e-6)

        vx_des = v * jnp.cos(heading)
        vy_des = v * jnp.sin(heading)
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

        data = jax.lax.fori_loop(0, self.config.frame_skip, _substep, data)

        car_xy = jnp.asarray([data.qpos[self.car_x_qpos], data.qpos[self.car_y_qpos]], dtype=jnp.float32)
        yaw = data.qpos[self.car_yaw_qpos]
        target_xy = state["target_xy"]
        obstacles_xy = state["obstacles_xy"]

        sensor_dists = self._sensor_distances(car_xy, yaw, obstacles_xy)
        raw_obs = jnp.clip(sensor_dists / self.config.sensor_range, 0.0, 1.0)
        sensor_hist = jnp.concatenate([state["sensor_hist"][1:], raw_obs[None, :]], axis=0)
        obs = self._compose_observation(sensor_hist, car_xy, yaw, target_xy)

        dist_target = jnp.linalg.norm(target_xy - car_xy)
        progress = state["prev_dist"] - dist_target

        clearances = jax.vmap(
            lambda c, h, a: self._point_box_signed_distance(car_xy, c, h, a) - self.config.car_radius
        )(obstacles_xy, self.obstacle_half_extents, self.obstacle_yaws)
        min_clearance = jnp.min(clearances)
        collision = min_clearance < 0.0

        heading_vec = jnp.asarray([jnp.cos(heading), jnp.sin(heading)], dtype=jnp.float32)
        goal_vec = (target_xy - car_xy) / (dist_target + 1e-6)
        heading_align = jnp.dot(heading_vec, goal_vec)

        sensor_min = jnp.min(sensor_dists)
        obstacle_penalty = jnp.clip((1.0 - sensor_min / 1.2), 0.0, 1.0)

        action_cost = 0.008 * jnp.sum(jnp.square(action - state["last_action"]))
        success = dist_target < self.config.target_success_radius
        out_of_bounds = jnp.any(jnp.abs(car_xy) > self.config.world_limit)

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
            "obstacles_xy": obstacles_xy,
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
