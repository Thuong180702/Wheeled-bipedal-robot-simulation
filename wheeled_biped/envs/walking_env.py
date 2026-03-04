"""
Walking Environment - Đi bộ bằng chân.

Task: Robot bước chân đi bộ (không dùng bánh), dáng đi đối xứng,
bám theo vận tốc mong muốn.

Stage 3 trong curriculum learning.
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
    reward_foot_clearance,
    reward_foot_contact,
    reward_gait_symmetry,
    reward_height,
    reward_tracking_velocity,
    reward_upright,
)


class WalkingEnv(WheeledBipedEnv):
    """Environment cho task đi bộ.

    Mở rộng observation:
      - Command velocity: 2
      - Pha dáng đi (gait phase): 2 (sin, cos)
    Tổng thêm: 4
    """

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Lệnh vận tốc
        cmd_cfg = self.config.get("command", {})
        self._vel_x_range = cmd_cfg.get("lin_vel_x_range", [0.1, 1.0])
        self._ang_vel_z_range = cmd_cfg.get("ang_vel_z_range", [-0.5, 0.5])

        # Tham số dáng đi
        gait_cfg = self.config.get("gait", {})
        self._step_frequency = gait_cfg.get("step_frequency", 1.5)  # Hz
        self._swing_height = gait_cfg.get("swing_height", 0.05)
        self._stance_ratio = gait_cfg.get("stance_ratio", 0.6)

        # Reward weights
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "tracking_velocity": reward_cfg.get("tracking_velocity", 1.5),
            "foot_clearance": reward_cfg.get("foot_clearance", 0.5),
            "gait_symmetry": reward_cfg.get("gait_symmetry", 0.3),
            "upright": reward_cfg.get("upright", 0.5),
            "height": reward_cfg.get("height", 0.3),
            "joint_torque": reward_cfg.get("joint_torque", -0.0002),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.0001),
            "action_rate": reward_cfg.get("action_rate", -0.002),
            "foot_contact": reward_cfg.get("foot_contact", 0.3),
            "alive": reward_cfg.get("alive", 0.1),
        }

        # Obs thêm: command(2) + gait_phase(2)
        self.obs_size += 4

    def _sample_command(self, rng: jax.Array) -> jnp.ndarray:
        """Lấy mẫu lệnh vận tốc."""
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
        gait_phase: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Trích xuất obs + command + gait phase."""
        base_obs = super()._extract_obs(mjx_data, prev_action)
        if command is None:
            command = jnp.zeros(2)
        if gait_phase is None:
            gait_phase = jnp.zeros(2)
        return jnp.concatenate([base_obs, command, gait_phase])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset kèm command mới."""
        rng, cmd_key = jax.random.split(rng)
        state = super().reset(rng)
        command = self._sample_command(cmd_key)
        gait_phase = self._compute_gait_phase(jnp.int32(0))
        obs = self._extract_obs(state.mjx_data, state.prev_action, command, gait_phase)
        return state._replace(
            obs=obs,
            info={
                "command": command,
                "gait_phase": gait_phase,
                "is_fallen": jnp.bool_(False),
                "time_limit": jnp.bool_(False),
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step kèm gait phase update."""
        command = state.info.get("command", jnp.zeros(2))
        new_state = super().step(state, action)
        gait_phase = self._compute_gait_phase(new_state.step_count)
        obs = self._extract_obs(new_state.mjx_data, action, command, gait_phase)
        return new_state._replace(
            obs=obs,
            info={**new_state.info, "command": command, "gait_phase": gait_phase},
        )

    def _compute_gait_phase(self, step_count: jnp.ndarray) -> jnp.ndarray:
        """Tính pha dáng đi dựa trên bước đếm.

        Dùng clock signal (sin/cos) để đồng bộ nhịp bước.

        Args:
            step_count: bước đếm hiện tại.

        Returns:
            [sin(phase), cos(phase)] — tín hiệu clock.
        """
        time_s = step_count * self.CONTROL_DT
        phase = 2 * jnp.pi * self._step_frequency * time_s
        return jnp.array([jnp.sin(phase), jnp.cos(phase)])

    def _get_desired_contacts(self, step_count: jnp.ndarray) -> tuple:
        """Xác định chân nào nên chạm đất dựa trên pha.

        Quy ước: pha [0, π) → chân trái stance, phải swing
                 pha [π, 2π) → ngược lại

        Returns:
            (left_should_contact, right_should_contact)
        """
        time_s = step_count * self.CONTROL_DT
        phase = (2 * jnp.pi * self._step_frequency * time_s) % (2 * jnp.pi)

        # Stance phase: 0 → stance_ratio * 2π
        stance_end = self._stance_ratio * 2 * jnp.pi

        left_stance = phase < stance_end
        right_stance = ((phase + jnp.pi) % (2 * jnp.pi)) < stance_end

        return left_stance, right_stance

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho walking."""

        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_vel = mjx_data.qvel[6:]
        torques = mjx_data.ctrl

        # Vận tốc
        base_vel_x = mjx_data.qvel[0]
        base_vel_y = mjx_data.qvel[1]
        base_ang_vel_z = mjx_data.qvel[5]

        # Command
        command = prev_state.info.get("command", jnp.zeros(2))
        cmd_vel_x = command[0]
        cmd_ang_vel_z = command[1]

        # Vị trí khớp chân trái vs phải
        # [l_hip_roll, l_hip_pitch, l_knee, l_ankle, l_wheel]
        left_joint_pos = mjx_data.qpos[7:12]
        right_joint_pos = mjx_data.qpos[12:17]

        # Pha dáng đi
        left_desired, right_desired = self._get_desired_contacts(prev_state.step_count)

        # Vị trí z của wheel (ước lượng bằng wheel contact site)
        # Dùng xpos của wheel body
        l_wheel_height = mjx_data.xpos[self._get_body_id("l_wheel_link"), 2]
        r_wheel_height = mjx_data.xpos[self._get_body_id("r_wheel_link"), 2]

        # Pha swing: chân không chạm đất
        l_is_swing = ~left_desired
        r_is_swing = ~right_desired

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
            "foot_clearance": 0.5
            * (
                reward_foot_clearance(l_wheel_height, self._swing_height, l_is_swing)
                + reward_foot_clearance(r_wheel_height, self._swing_height, r_is_swing)
            ),
            "gait_symmetry": reward_gait_symmetry(
                left_joint_pos[:4], right_joint_pos[:4]
            ),
            "upright": reward_upright(torso_quat),
            "height": reward_height(torso_height, 0.65),
            "joint_torque": penalty_joint_torque(torques),
            "joint_velocity": penalty_joint_velocity(joint_vel),
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
            "foot_contact": reward_foot_contact(
                left_desired, right_desired, left_desired, right_desired
            ),
            "alive": reward_alive(is_alive),
        }

        return compute_total_reward(components, self._reward_weights)

    def _get_body_id(self, body_name: str) -> int:
        """Lấy ID body từ tên (cache kết quả)."""
        import mujoco

        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
