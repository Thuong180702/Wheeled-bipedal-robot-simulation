"""
Rough Terrain Environment - Đi trên địa hình gồ ghề.

Task: Di chuyển ổn định trên heightfield terrain với độ gồ ghề
tăng dần (curriculum trong curriculum).

Stage 5 (cuối) trong curriculum learning.
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
    penalty_body_angular_velocity,
    reward_height,
    reward_tracking_velocity,
    reward_upright,
)
from wheeled_biped.utils.math_utils import quat_conjugate, quat_rotate


class TerrainEnv(WheeledBipedEnv):
    """Environment cho task đi trên địa hình gồ ghề.

    Sử dụng heightfield terrain, độ khó tăng dần.
    Robot dùng kết hợp bánh xe và chân để duy trì ổn định.

    Observation thêm:
      - Command velocity (2)
      - Heightfield sample quanh chân (tùy chọn)
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Cấu hình terrain
        terrain_cfg = self.config.get("terrain", {})
        self._terrain_type = terrain_cfg.get("type", "heightfield")
        self._difficulty_levels = terrain_cfg.get(
            "difficulty_levels",
            [
                {"name": "mild", "max_height": 0.03, "frequency": 0.5},
            ],
        )
        self._current_difficulty = 0

        # Command
        cmd_cfg = self.config.get("command", {})
        self._vel_x_range = cmd_cfg.get("lin_vel_x_range", [0.2, 1.5])
        self._ang_vel_z_range = cmd_cfg.get("ang_vel_z_range", [-1.0, 1.0])

        # Reward weights
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "tracking_velocity": reward_cfg.get("tracking_velocity", 1.5),
            "upright": reward_cfg.get("upright", 1.0),
            "height": reward_cfg.get("height", 0.5),
            "foot_contact": reward_cfg.get("foot_contact", 0.5),
            "body_stability": reward_cfg.get("body_stability", 0.5),
            "joint_torque": reward_cfg.get("joint_torque", -0.0002),
            "action_rate": reward_cfg.get("action_rate", -0.003),
            "alive": reward_cfg.get("alive", 0.2),
        }

        # Obs thêm: command(2)
        self.obs_size += 2

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        """Lấy mẫu lệnh vận tốc."""
        rng, k1, k2 = jax.random.split(rng, 3)
        cmd_vel_x = jax.random.uniform(k1, minval=self._vel_x_range[0], maxval=self._vel_x_range[1])
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
        """Trích xuất obs + command."""
        base_obs = super()._extract_obs(mjx_data, prev_action)
        if command is None:
            command = jnp.zeros(2)
        return jnp.concatenate([base_obs, command])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset kèm command mới."""
        rng, cmd_key = jax.random.split(rng)
        state = super().reset(rng)
        command = self._sample_command(cmd_key)
        obs = self._extract_obs(state.mjx_data, state.prev_action, command)
        return state._replace(
            obs=obs,
            info={
                "command": command,
                "is_fallen": jnp.bool_(False),
                "time_limit": jnp.bool_(False),
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step kèm command tracking."""
        command = state.info.get("command", jnp.zeros(2))
        new_state = super().step(state, action)
        obs = self._extract_obs(new_state.mjx_data, action, command)
        return new_state._replace(
            obs=obs,
            info={**new_state.info, "command": command},
        )

    def increase_difficulty(self) -> bool:
        """Tăng độ khó terrain.

        Returns:
            True nếu còn mức khó hơn, False nếu đã max.
        """
        if self._current_difficulty < len(self._difficulty_levels) - 1:
            self._current_difficulty += 1
            return True
        return False

    def get_current_difficulty(self) -> dict:
        """Trả về cấu hình độ khó hiện tại."""
        return self._difficulty_levels[self._current_difficulty]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho rough terrain."""

        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        torques = mjx_data.ctrl

        # Body-frame velocity (-Y = forward, X = lateral)
        quat_inv = quat_conjugate(torso_quat)
        body_vel = quat_rotate(quat_inv, mjx_data.qvel[:3])
        body_ang_vel = quat_rotate(quat_inv, mjx_data.qvel[3:6])
        body_vel_forward = -body_vel[1]  # robot faces -Y
        body_vel_lateral = body_vel[0]
        body_ang_vel_z = body_ang_vel[2]

        # Command
        command = prev_state.info.get("command", jnp.zeros(2))

        is_fallen = self._check_termination(mjx_data)
        is_alive = ~is_fallen

        components = {
            "tracking_velocity": reward_tracking_velocity(
                body_vel_forward,
                body_vel_lateral,
                body_ang_vel_z,
                command[0],
                jnp.float32(0.0),
                command[1],
            ),
            "upright": reward_upright(torso_quat),
            "height": reward_height(torso_height, 0.55),
            "foot_contact": jnp.float32(1.0),  # placeholder
            "body_stability": jnp.clip(
                1.0 - penalty_body_angular_velocity(body_ang_vel) * 0.01, 0.0, 1.0
            ),
            "joint_torque": jnp.sum(jnp.square(torques)),
            "action_rate": jnp.sum(jnp.square(action - prev_state.prev_action)),
            "alive": jnp.where(is_alive, 1.0, 0.0),
        }

        return compute_total_reward(components, self._reward_weights)
