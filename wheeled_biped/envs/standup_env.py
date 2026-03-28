"""
Stand-Up & Height Transition Environment - Stage 2 Training.

Task: Robot học chuyển đổi chiều cao (đứng lên/ngồi xuống) và phục hồi từ ngã.

Mỗi episode:
  - 70% starts: đứng ở random height ∈ [0.38, 0.72m] → chuyển sang height_command mới
  - 30% starts: bắt đầu từ tư thế ngã → phục hồi về height_command

Stage 2 trong curriculum. Obs = 40 dims (39 base + height_command 1).
Khớp kích thước với balance_env → warm-start từ balance checkpoint.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from wheeled_biped.envs.base_env import EnvState, WheeledBipedEnv
from wheeled_biped.rewards.reward_functions import (
    compute_total_reward,
    penalty_action_rate,
    penalty_body_angular_velocity,
    penalty_joint_torque,
    penalty_joint_velocity,
    penalty_position_drift,
    reward_alive,
    reward_body_level,
    reward_heading,
    reward_height,
    reward_leg_symmetry,
    reward_natural_pose,
    reward_upright,
)
from wheeled_biped.utils.math_utils import quat_to_euler


class StandUpEnv(WheeledBipedEnv):
    """Environment cho task đứng lên/ngồi xuống và phục hồi từ ngã.

    Stage 2: Warm-start từ balance checkpoint (obs=40, cùng kích thước).

    Mỗi episode:
      - Start: 70% đứng ở random height, 30% ngã
      - Target: random height_command ∈ [0.38, 0.72m]
      - Robot phải chuyển đến height_command và giữ ổn định

    Obs (40 dims) = base 39 + height_command_norm 1 → khớp balance_env.
    Reward = balance-like + upright bonus (tín hiệu recovery từ ngã).
    Termination: chỉ khi torso_height < 0.05m (không check tilt).
    """

    MIN_HEIGHT_CMD = 0.38  # ngồi thấp nhất
    MAX_HEIGHT_CMD = 0.72  # đứng thẳng nhất

    # (hip_pitch, knee, approx_torso_z) cho 7 tư thế đứng
    _STANDING_POSES = [
        (0.00, 0.00, 0.720),  # thẳng nhất
        (0.15, 0.25, 0.715),
        (0.30, 0.50, 0.710),  # keyframe
        (0.60, 1.00, 0.600),
        (0.90, 1.50, 0.520),
        (1.20, 2.00, 0.450),
        (1.50, 2.50, 0.380),  # ngồi thấp nhất
    ]

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "body_level": reward_cfg.get("body_level", 1.5),
            "height": reward_cfg.get("height", 2.0),
            "upright": reward_cfg.get("upright", 1.0),
            "natural_pose": reward_cfg.get("natural_pose", 1.5),
            "symmetry": reward_cfg.get("symmetry", 0.3),
            "alive": reward_cfg.get("alive", 0.5),
            "heading": reward_cfg.get("heading", 0.5),
            "position_drift": reward_cfg.get("position_drift", 0.3),
            "orientation": reward_cfg.get("orientation", 0.5),
            "yaw_rate": reward_cfg.get("yaw_rate", 0.3),
            "joint_torque": reward_cfg.get("joint_torque", -0.0005),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.0002),
            "action_rate": reward_cfg.get("action_rate", -0.005),
        }

        task_cfg = self.config.get("task", {})
        # Tỉ lệ bắt đầu từ tư thế ngã (0.0–1.0); default 30%
        self._fallen_ratio = float(task_cfg.get("fallen_ratio", 0.3))

        # Pre-compute init poses (JIT-compatible indexing)
        self._standing_qpos = self._precompute_standing_qpos()
        self._fallen_qpos = self._precompute_fallen_qpos()
        self._base_mjx_data = self._create_base_mjx_data()

    def _compute_obs_size(self) -> int:
        """Obs = base 39 + height_command 1 = 40. Khớp balance_env."""
        return super()._compute_obs_size() + 1

    def _precompute_standing_qpos(self) -> jnp.ndarray:
        """7 tư thế đứng từ thẳng cao (~0.72m) đến ngồi thấp (~0.38m).

        Set cả z (torso height) và joint angles để tránh floating artifact.
        Pattern khớp fallen_qpos: q[7:] = [0, 0, hp, kn, 0, 0, 0, hp, kn, 0]
        """
        import numpy as np

        nq = self.mj_model.nq
        poses = []
        for hp, kn, z in self._STANDING_POSES:
            q = np.zeros(nq)
            q[2] = z  # torso z (phù hợp với góc khớp)
            q[3] = 1.0  # quat w=1 → thẳng đứng [w, x, y, z] = [1, 0, 0, 0]
            q[7:] = [0, 0, hp, kn, 0, 0, 0, hp, kn, 0]  # 10 joints
            poses.append(q.copy())
        return jnp.array(poses)

    def _precompute_fallen_qpos(self) -> jnp.ndarray:
        """7 tư thế ngã cho recovery training (kế thừa từ design cũ)."""
        import numpy as np

        nq = self.mj_model.nq
        poses = []

        def _q(xyz, wxyz, joints):
            q = np.zeros(nq)
            q[:3] = xyz
            q[3:7] = wxyz
            q[7:] = joints
            return q.copy()

        # 0: supine — nằm ngửa
        poses.append(
            _q(
                [0, 0, 0.15],
                [0.7071, -0.7071, 0, 0],
                [0, 0, -0.5, -1.0, 0, 0, 0, -0.5, -1.0, 0],
            )
        )
        # 1: prone — nằm sấp
        poses.append(
            _q(
                [0, 0, 0.15],
                [0.7071, 0.7071, 0, 0],
                [0, 0, -0.8, -0.5, 0, 0, 0, -0.8, -0.5, 0],
            )
        )
        # 2: nghiêng trái
        poses.append(
            _q(
                [0, 0, 0.15],
                [0.7071, 0, 0.7071, 0],
                [0.3, 0, -0.3, -0.5, 0, -0.3, 0, -0.3, -0.5, 0],
            )
        )
        # 3: nghiêng phải
        poses.append(
            _q(
                [0, 0, 0.15],
                [0.7071, 0, -0.7071, 0],
                [-0.3, 0, -0.3, -0.5, 0, 0.3, 0, -0.3, -0.5, 0],
            )
        )
        # 4: pitch nặng (~70°)
        angle = 1.2
        poses.append(
            _q(
                [0, 0, 0.20],
                [np.cos(angle / 2), np.sin(angle / 2), 0, 0],
                [0, 0, -0.3, -1.0, 0, 0, 0, -0.3, -1.0, 0],
            )
        )
        # 5: roll nặng (~70°)
        poses.append(
            _q(
                [0, 0, 0.20],
                [np.cos(angle / 2), 0, np.sin(angle / 2), 0],
                [0, 0, -0.3, -1.0, 0, 0, 0, -0.3, -1.0, 0],
            )
        )
        # 6: combined tilt
        sa = np.sin(angle / 2) * 0.7071
        poses.append(
            _q(
                [0, 0, 0.20],
                [np.cos(angle / 2), sa, sa, 0],
                [0, 0, -0.3, -0.5, 0, 0, 0, -0.3, -0.5, 0],
            )
        )

        return jnp.array(poses)

    def _create_base_mjx_data(self) -> mjx.Data:
        """Tạo base MJX data (keyframe) để clone trong reset."""
        mj_data = mujoco.MjData(self.mj_model)
        if self.mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
        mujoco.mj_forward(self.mj_model, mj_data)
        return mjx.put_data(self.mj_model, mj_data)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset: mixed start (70% đứng random height, 30% ngã) + random height_command."""
        rng, type_key, stand_key, fall_key, noise_key, height_key = jax.random.split(rng, 6)

        n_standing = self._standing_qpos.shape[0]  # 7
        n_fallen = self._fallen_qpos.shape[0]  # 7

        is_fallen_start = jax.random.uniform(type_key) < self._fallen_ratio
        standing_idx = jax.random.randint(stand_key, (), 0, n_standing)
        fallen_idx = jax.random.randint(fall_key, (), 0, n_fallen)

        standing_qpos = self._standing_qpos[standing_idx]
        fallen_qpos = self._fallen_qpos[fallen_idx]
        selected_qpos = jnp.where(is_fallen_start, fallen_qpos, standing_qpos)

        mjx_data = self._base_mjx_data.replace(qpos=selected_qpos)

        # Nhiễu nhỏ vào joint angles để đa dạng hóa start state
        joint_noise = jax.random.uniform(
            noise_key, shape=(self.NUM_JOINTS,), minval=-0.05, maxval=0.05
        )
        mjx_data = mjx_data.replace(qpos=mjx_data.qpos.at[7:].add(joint_noise))

        # Random height_command: full range ngay từ đầu (không cần curriculum)
        height_command = jax.random.uniform(
            height_key, shape=(), minval=self.MIN_HEIGHT_CMD, maxval=self.MAX_HEIGHT_CMD
        )

        prev_action = jnp.zeros(self.num_actions)
        base_obs = self._extract_obs(mjx_data, prev_action)
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        obs = jnp.concatenate([base_obs, jnp.array([height_norm])])

        anchor_xy = mjx_data.qpos[:2]
        initial_yaw = quat_to_euler(mjx_data.qpos[3:7])[2]

        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            prev_action=prev_action,
            info={
                "is_fallen": jnp.bool_(False),
                "time_limit": jnp.bool_(False),
                "height_command": height_command,
                "anchor_xy": anchor_xy,
                "initial_yaw": initial_yaw,
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step: physics + reward + termination (không có push disturbance)."""
        action = jnp.clip(action, -1.0, 1.0)

        ctrl_range = self.mjx_model.actuator_ctrlrange
        scaled_action = ctrl_range[:, 0] + (action + 1.0) * 0.5 * (
            ctrl_range[:, 1] - ctrl_range[:, 0]
        )
        mjx_data = state.mjx_data.replace(ctrl=scaled_action)

        def physics_step(data, _):
            data = mjx.step(self.mjx_model, data)
            return data, None

        mjx_data, _ = jax.lax.scan(physics_step, mjx_data, None, length=self._n_substeps)

        base_obs = self._extract_obs(mjx_data, action)
        height_command = state.info["height_command"]
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        obs = jnp.concatenate([base_obs, jnp.array([height_norm])])

        reward = self._compute_reward(mjx_data, action, state)
        is_fallen = self._check_termination(mjx_data)
        step_count = state.step_count + 1
        time_limit = step_count >= self._episode_length
        done = is_fallen | time_limit

        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            step_count=step_count,
            prev_action=action,
            info={
                "is_fallen": is_fallen,
                "time_limit": time_limit,
                "height_command": height_command,
                "anchor_xy": state.info["anchor_xy"],
                "initial_yaw": state.info["initial_yaw"],
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _check_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Chỉ terminate khi torso < 0.05m. Không check tilt (robot có thể bắt đầu ngã)."""
        return mjx_data.qpos[2] < self._min_height

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Reward = balance-like + upright bonus cho recovery từ ngã."""
        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_pos = mjx_data.qpos[7:17]
        joint_vel = mjx_data.qvel[6:]
        ang_vel = mjx_data.qvel[3:6]
        torques = mjx_data.ctrl

        is_fallen = self._check_termination(mjx_data)
        height_command = prev_state.info["height_command"]
        current_xy = mjx_data.qpos[:2]
        anchor_xy = prev_state.info["anchor_xy"]
        initial_yaw = prev_state.info["initial_yaw"]

        components = {
            # Thân nằm ngang (phạt roll + pitch)
            "body_level": reward_body_level(torso_quat, sigma_roll=0.15, sigma_pitch=0.15),
            # Đạt chiều cao mục tiêu (σ=0.25: gradient trong ±0.5m)
            "height": reward_height(torso_height, height_command, sigma=0.25),
            # Đứng thẳng — tín hiệu mạnh khi recover từ ngã
            # (1.0 khi thẳng đứng, 0.0 khi nằm ngang)
            "upright": reward_upright(torso_quat),
            # Tư thế khớp tự nhiên theo chiều cao mục tiêu
            "natural_pose": reward_natural_pose(joint_pos, height_command, sigma=0.8),
            # 2 chân đối xứng
            "symmetry": reward_leg_symmetry(joint_pos, sigma=0.5),
            # Bonus sống sót
            "alive": reward_alive(~is_fallen),
            # Giữ hướng ban đầu (nới lỏng hơn balance)
            "heading": reward_heading(torso_quat, initial_yaw, sigma=0.5),
            # Ổn định góc
            "orientation": jnp.clip(
                1.0 - jnp.sqrt(penalty_body_angular_velocity(ang_vel)) * 0.15, 0.0, 1.0
            ),
            # Chống xoay yaw
            "yaw_rate": jnp.clip(1.0 - jnp.abs(ang_vel[2]) * 0.5, 0.0, 1.0),
            # Drift nới lỏng (sigma=1.0) — cho phép pivot khi recovery
            "position_drift": penalty_position_drift(current_xy, anchor_xy, sigma=1.0),
            # Penalties nhẹ hơn balance — robot cần torque lớn khi chuyển chiều cao
            "joint_torque": penalty_joint_torque(torques),
            "joint_velocity": penalty_joint_velocity(joint_vel),
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
        }

        return compute_total_reward(components, self._reward_weights)
