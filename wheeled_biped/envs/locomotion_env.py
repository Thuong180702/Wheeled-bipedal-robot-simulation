"""
Locomotion Environment - Di chuyển bằng bánh xe.

Task: Bám theo vận tốc mong muốn (linear + angular) sử dụng bánh xe,
đồng thời giữ thăng bằng.

Stage 2 trong curriculum learning.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from wheeled_biped.envs.base_env import EnvState, WheeledBipedEnv
from wheeled_biped.rewards.reward_functions import (
    compute_total_reward,
    penalty_action_rate,
    penalty_joint_torque,
    penalty_joint_velocity,
    reward_alive,
    reward_height,
    reward_tracking_velocity,
    reward_upright,
)


class LocomotionEnv(WheeledBipedEnv):
    """Environment cho task di chuyển bằng bánh xe.

    Mỗi episode, robot nhận lệnh vận tốc (vel_x, ang_vel_z) và phải
    di chuyển theo lệnh sử dụng bánh xe, giữ thăng bằng.

    Observation mở rộng thêm: command velocity (2 thêm).
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Lệnh vận tốc
        cmd_cfg = self.config.get("command", {})
        self._vel_x_range = cmd_cfg.get("lin_vel_x_range", [-1.0, 2.0])
        self._ang_vel_z_range = cmd_cfg.get("ang_vel_z_range", [-1.5, 1.5])
        self._resample_interval = cmd_cfg.get("resample_interval", 500)

        # Reward weights
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "tracking_velocity": reward_cfg.get("tracking_velocity", 1.5),
            "upright": reward_cfg.get("upright", 0.5),
            "height": reward_cfg.get("height", 0.3),
            "joint_torque": reward_cfg.get("joint_torque", -0.0001),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.0001),
            "action_rate": reward_cfg.get("action_rate", -0.001),
            "alive": reward_cfg.get("alive", 0.1),
        }

        # Obs thêm 2 chiều cho command
        self.obs_size += 2

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        """Lấy mẫu lệnh vận tốc ngẫu nhiên.

        Returns:
            [cmd_vel_x, cmd_ang_vel_z]
        """
        rng, k1, k2 = jax.random.split(rng, 3)
        cmd_vel_x = jax.random.uniform(
            k1, minval=self._vel_x_range[0], maxval=self._vel_x_range[1]
        )
        cmd_ang_vel_z = jax.random.uniform(
            k2, minval=self._ang_vel_z_range[0], maxval=self._ang_vel_z_range[1]
        )
        return jnp.array([cmd_vel_x, cmd_ang_vel_z])

    def _extract_obs(
        self,
        mjx_data: mjx.Data,
        prev_action: jnp.ndarray,
        command: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Trích xuất obs + command velocity."""
        base_obs = super()._extract_obs(mjx_data, prev_action)
        if command is None:
            command = jnp.zeros(2)
        return jnp.concatenate([base_obs, command])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset kèm lấy mẫu command mới."""
        rng, cmd_key = jax.random.split(rng)
        state = super().reset(rng)

        # Lấy lệnh vận tốc
        command = self._sample_command(cmd_key)

        # Cập nhật obs bao gồm command
        obs = self._extract_obs(state.mjx_data, state.prev_action, command)

        return state._replace(
            obs=obs,
            info={"command": command, "is_fallen": jnp.bool_(False)},
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step kèm command tracking."""
        # Lấy command từ info
        command = state.info.get("command", jnp.zeros(2))

        # Gọi base step
        new_state = super().step(state, action)

        # Cập nhật obs với command
        obs = self._extract_obs(new_state.mjx_data, action, command)

        return new_state._replace(
            obs=obs,
            info={**new_state.info, "command": command},
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho locomotion."""

        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_vel = mjx_data.qvel[6:]
        torques = mjx_data.ctrl

        # Vận tốc thực tế
        base_vel_x = mjx_data.qvel[0]
        base_vel_y = mjx_data.qvel[1]
        base_ang_vel_z = mjx_data.qvel[5]

        # Command
        command = prev_state.info.get("command", jnp.zeros(2))
        cmd_vel_x = command[0]
        cmd_ang_vel_z = command[1]

        # Kiểm tra sống sót
        is_fallen = self._check_termination(mjx_data)
        is_alive = ~is_fallen

        components = {
            "tracking_velocity": reward_tracking_velocity(
                base_vel_x,
                base_vel_y,
                base_ang_vel_z,
                cmd_vel_x,
                jnp.float32(0.0),
                cmd_ang_vel_z,
            ),
            "upright": reward_upright(torso_quat),
            "height": reward_height(torso_height, 0.65),
            "joint_torque": penalty_joint_torque(torques),
            "joint_velocity": penalty_joint_velocity(joint_vel),
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
            "alive": reward_alive(is_alive),
        }

        return compute_total_reward(components, self._reward_weights)
