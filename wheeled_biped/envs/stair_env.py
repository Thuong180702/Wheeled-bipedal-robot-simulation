"""
Stair Climbing Environment - Leo cầu thang.

Task: Robot phải leo lên/xuống cầu thang sử dụng kết hợp
chân và bánh xe, duy trì thăng bằng.

Stage 4 trong curriculum learning.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from wheeled_biped.envs.base_env import EnvState, WheeledBipedEnv
from wheeled_biped.rewards.reward_functions import (
    compute_total_reward,
    penalty_action_rate,
    penalty_joint_torque,
    reward_alive,
    reward_foot_contact,
    reward_stair_progress,
    reward_upright,
)
from wheeled_biped.sim.terrain_generator import generate_stair_terrain


class StairEnv(WheeledBipedEnv):
    """Environment cho task leo cầu thang.

    Tạo terrain cầu thang bằng cách inject geom vào model.
    Reward dựa trên:
      - Tiến bộ theo hướng x (tiến)
      - Tiến bộ theo hướng z (leo cao)
      - Giữ thăng bằng
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        # Lưu stair config trước khi gọi super()
        self._stair_config = config.get("stairs", {}) if config else {}
        super().__init__(config=config, **kwargs)

        # Thay thế model bằng bản có cầu thang
        stair_model = self._build_stair_model()
        self.mj_model = stair_model
        self.mjx_model = mjx.put_model(stair_model)

        # Cập nhật body ID
        self._torso_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Tính vị trí đích (đỉnh cầu thang, phía -Y = forward)
        num_steps = self._stair_config.get("num_steps", 8)
        step_height = self._stair_config.get("step_height", 0.15)
        step_depth = self._stair_config.get("step_depth", 0.30)
        platform_length = 1.0
        self._goal_y = -(platform_length + num_steps * step_depth)
        self._goal_z = num_steps * step_height

        # Reward weights
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "forward_progress": reward_cfg.get("forward_progress", 2.0),
            "height_progress": reward_cfg.get("height_progress", 1.5),
            "upright": reward_cfg.get("upright", 0.8),
            "foot_contact": reward_cfg.get("foot_contact", 0.5),
            "joint_torque": reward_cfg.get("joint_torque", -0.0002),
            "action_rate": reward_cfg.get("action_rate", -0.002),
            "alive": reward_cfg.get("alive", 0.2),
            "stair_clearance": reward_cfg.get("stair_clearance", 0.5),
        }

        # Obs thêm: vị trí đích (2) + tiến bộ (1)
        self.obs_size += 3

    def _build_stair_model(self) -> mujoco.MjModel:
        """Tạo MuJoCo model bao gồm cầu thang.

        Đọc model robot gốc, thêm geom cầu thang vào XML.
        Đổi wheel type cylinder→sphere để tương thích MJX (cylinder-box
        collision không được hỗ trợ trong MJX, nhưng sphere-box thì có).
        """
        from wheeled_biped.utils.config import get_model_path
        import re

        # Đọc XML gốc
        model_path = get_model_path()
        with open(model_path, "r", encoding="utf-8") as f:
            base_xml = f.read()

        # Tạo stair geoms
        stair_geoms = generate_stair_terrain(
            num_steps=self._stair_config.get("num_steps", 8),
            step_height=self._stair_config.get("step_height", 0.15),
            step_depth=self._stair_config.get("step_depth", 0.30),
            step_width=1.0,
        )

        # Inject vào worldbody (trước tag đóng </worldbody>)
        stair_xml = f"\n    <!-- Cầu thang -->\n    {stair_geoms}\n"
        modified_xml = base_xml.replace("</worldbody>", stair_xml + "  </worldbody>")

        # MJX workaround: đổi wheel default class từ cylinder → sphere
        # (MJX không hỗ trợ cylinder-box collision)
        modified_xml = modified_xml.replace(
            '<default class="wheel">',
            '<default class="wheel_stair">',
        )
        # Thay type cylinder → sphere trong default wheel
        modified_xml = re.sub(
            r'(<default class="wheel_stair">\s*<geom\s+)type="cylinder"',
            r'\1type="sphere"',
            modified_xml,
        )
        # Cập nhật class reference trong geom instances
        modified_xml = modified_xml.replace(
            'class="wheel"',
            'class="wheel_stair"',
        )
        # Đổi size wheel geom: "0.06 0.02" → "0.06" (sphere chỉ cần radius)
        modified_xml = re.sub(
            r'(name="[lr]_wheel_geom"\s+class="wheel_stair"\s+)size="0\.06 0\.02"',
            r'\1size="0.06"',
            modified_xml,
        )

        return mujoco.MjModel.from_xml_string(modified_xml)

    def _extract_obs(
        self,
        mjx_data: mjx.Data,
        prev_action: jnp.ndarray,
        goal: jnp.ndarray | None = None,
        progress: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Trích xuất obs + goal position + progress."""
        base_obs = super()._extract_obs(mjx_data, prev_action)
        if goal is None:
            goal = jnp.zeros(2)
        if progress is None:
            progress = jnp.zeros(1)
        return jnp.concatenate([base_obs, goal, progress])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset kèm goal mới."""
        state = super().reset(rng)
        goal = jnp.array([self._goal_y, self._goal_z])
        progress = jnp.zeros(1)
        obs = self._extract_obs(state.mjx_data, state.prev_action, goal, progress)
        return state._replace(
            obs=obs,
            info={
                "goal": goal,
                "progress": progress,
                "is_fallen": jnp.bool_(False),
                "time_limit": jnp.bool_(False),
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step kèm progress tracking."""
        goal = state.info.get("goal", jnp.array([self._goal_y, self._goal_z]))
        new_state = super().step(state, action)
        # Tính tiến bộ: tỷ lệ khoảng cách y đến goal (-Y = forward)
        torso_y = new_state.mjx_data.qpos[1]
        progress = jnp.clip(torso_y / self._goal_y, 0.0, 1.0).reshape(1)
        obs = self._extract_obs(new_state.mjx_data, action, goal, progress)
        return new_state._replace(
            obs=obs,
            info={**new_state.info, "goal": goal, "progress": progress},
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho stair climbing."""

        torso_quat = mjx_data.qpos[3:7]
        torso_pos = mjx_data.qpos[:3]
        torques = mjx_data.ctrl
        prev_torso_pos = prev_state.mjx_data.qpos[:3]

        is_fallen = self._check_termination(mjx_data)
        is_alive = ~is_fallen

        components = {
            "forward_progress": reward_stair_progress(
                torso_pos,
                prev_torso_pos,
                forward_weight=2.0,
                height_weight=1.5,
            ),
            "height_progress": jnp.clip(torso_pos[2] - prev_torso_pos[2], 0.0, None),
            "upright": reward_upright(torso_quat),
            "foot_contact": jnp.float32(1.0),  # placeholder
            "joint_torque": jnp.sum(jnp.square(torques)),
            "action_rate": jnp.sum(jnp.square(action - prev_state.prev_action)),
            "alive": jnp.where(is_alive, 1.0, 0.0),
            "stair_clearance": jnp.float32(0.0),  # sẽ tính bằng contact detection
        }

        return compute_total_reward(components, self._reward_weights)
