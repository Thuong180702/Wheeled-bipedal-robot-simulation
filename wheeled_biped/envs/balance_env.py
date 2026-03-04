"""
Balance Environment - Huấn luyện robot đứng vững.

Task: Giữ thân robot thẳng đứng, duy trì chiều cao ổn định,
chống chịu nhiễu loạn bên ngoài (đẩy ngẫu nhiên).

Đây là stage đầu tiên trong curriculum learning.
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
    penalty_body_angular_velocity,
    penalty_joint_torque,
    penalty_joint_velocity,
    reward_alive,
    reward_height,
    reward_upright,
)
from wheeled_biped.utils.math_utils import quat_to_euler


class BalanceEnv(WheeledBipedEnv):
    """Environment cho task đứng vững.

    Reward:
      - Thưởng đứng thẳng (upright)
      - Thưởng duy trì chiều cao
      - Thưởng ổn định hướng
      - Phạt dùng mô-men lớn
      - Phạt action rate
      - Bonus sống sót mỗi step
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Lấy trọng số reward
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "upright": reward_cfg.get("upright", 1.0),
            "height": reward_cfg.get("height", 0.5),
            "joint_torque": reward_cfg.get("joint_torque", -0.0001),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.0001),
            "action_rate": reward_cfg.get("action_rate", -0.001),
            "orientation": reward_cfg.get("orientation", 0.3),
            "alive": reward_cfg.get("alive", 0.2),
        }

        # Chiều cao mục tiêu
        self._target_height = 0.65

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho task balance."""

        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_vel = mjx_data.qvel[6:]  # vận tốc khớp
        ang_vel = mjx_data.qvel[3:6]  # vận tốc góc thân
        torques = mjx_data.ctrl  # mô-men điều khiển

        # Kiểm tra còn sống
        is_fallen = self._check_termination(mjx_data)
        is_alive = ~is_fallen

        # Tính các thành phần reward
        components = {
            "upright": reward_upright(torso_quat),
            "height": reward_height(torso_height, self._target_height),
            "joint_torque": penalty_joint_torque(torques),
            "joint_velocity": penalty_joint_velocity(joint_vel),
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
            "orientation": 1.0 - penalty_body_angular_velocity(ang_vel) * 0.01,
            "alive": reward_alive(is_alive),
        }

        return compute_total_reward(components, self._reward_weights)
