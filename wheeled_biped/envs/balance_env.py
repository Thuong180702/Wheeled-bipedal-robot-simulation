"""
Balance Environment - Huấn luyện robot đứng vững ở nhiều chiều cao.

Task: Giữ thân robot nằm ngang, duy trì chiều cao theo lệnh (height_command),
2 chân đối xứng, đứng yên ổn định, chống chịu nhiễu loạn.

Observation: 40 dims = base 39 + height_command 1
Height command ngẫu nhiên mỗi episode trong khoảng [min_height_cmd, max_height_cmd].

Đây là stage đầu tiên trong curriculum learning.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx

from wheeled_biped.envs.base_env import EnvState, WheeledBipedEnv
from wheeled_biped.sim.domain_randomization import (
    apply_external_force,
    clear_external_force,
)
from wheeled_biped.rewards.reward_functions import (
    compute_total_reward,
    penalty_action_rate,
    penalty_body_angular_velocity,
    penalty_joint_torque,
    penalty_joint_velocity,
    penalty_position_drift,
    penalty_wheel_velocity,
    reward_alive,
    reward_body_level,
    reward_heading,
    reward_height,
    reward_leg_symmetry,
    reward_legs_forward,
    reward_legs_vertical,
    reward_natural_pose,
    reward_no_motion,
)
from wheeled_biped.utils.math_utils import quat_conjugate, quat_rotate, quat_to_euler


class BalanceEnv(WheeledBipedEnv):
    """Environment cho task đứng vững ở nhiều chiều cao.

    Mỗi episode random một height_command ∈ [0.35, 0.71].
    Robot phải giữ chiều cao theo lệnh, thân ngang, 2 chân đối xứng, đứng yên.

    Observation (40 dims):
      - base obs (39): gravity_body, lin_vel, ang_vel, joint_pos, joint_vel, prev_action
      - height_command (1): chiều cao mục tiêu (normalized về [0, 1])
    """

    # Khoảng chiều cao lệnh (m) — đo từ kinematics thực tế
    # Thẳng nhất (hp=0, kn=0): ~0.73m, Gập tối đa (hp=1.5, kn=2.5): ~0.36m
    MIN_HEIGHT_CMD = 0.38  # ngồi thấp nhất (an toàn trên min_height=0.3)
    MAX_HEIGHT_CMD = 0.72  # đứng thẳng nhất

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Lấy trọng số reward
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "body_level": reward_cfg.get("body_level", 1.5),
            "height": reward_cfg.get("height", 1.5),
            "legs_forward": reward_cfg.get("legs_forward", 0.3),
            "legs_vertical": reward_cfg.get("legs_vertical", 0.3),
            "joint_torque": reward_cfg.get("joint_torque", -0.0001),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.00005),
            "action_rate": reward_cfg.get("action_rate", -0.005),
            "orientation": reward_cfg.get("orientation", 0.0),
            "alive": reward_cfg.get("alive", 0.5),
            "no_motion": reward_cfg.get("no_motion", 0.0),
            "symmetry": reward_cfg.get("symmetry", 0.3),
            "wheel_velocity": reward_cfg.get("wheel_velocity", 0.0),
            "position_drift": reward_cfg.get("position_drift", 0.2),
            "heading": reward_cfg.get("heading", 0.3),
            "natural_pose": reward_cfg.get("natural_pose", 1.5),
        }

        # Push disturbance config
        dr_cfg = self.config.get("domain_randomization", {})
        self._push_interval = int(dr_cfg.get("push_interval", 200))
        self._push_magnitude = float(dr_cfg.get("push_magnitude", 30.0))
        self._push_duration = int(dr_cfg.get("push_duration", 5))  # số steps giữ lực
        self._push_enabled = bool(dr_cfg.get("enabled", True))
        # Lấy torso body_id từ mj_model
        import mujoco

        self._torso_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

    def _compute_obs_size(self) -> int:
        """Observation = base 39 + height_command 1 = 40."""
        return super()._compute_obs_size() + 1

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset environment với random height_command."""
        rng, sub_key = jax.random.split(rng)
        mjx_data = self._get_initial_mjx_data(sub_key)

        # Nhiễu nhỏ vào vị trí khớp
        rng, noise_key = jax.random.split(rng)
        joint_noise = jax.random.uniform(
            noise_key, shape=(self.NUM_JOINTS,), minval=-0.05, maxval=0.05
        )
        new_qpos = mjx_data.qpos.at[7:].add(joint_noise)
        mjx_data = mjx_data.replace(qpos=new_qpos)

        # Random height command cho episode này
        rng, height_key = jax.random.split(rng)
        height_command = jax.random.uniform(
            height_key, shape=(), minval=self.MIN_HEIGHT_CMD, maxval=self.MAX_HEIGHT_CMD
        )

        prev_action = jnp.zeros(self.num_actions)
        base_obs = self._extract_obs(mjx_data, prev_action)

        # Normalize height_command về [0, 1]
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        obs = jnp.concatenate([base_obs, jnp.array([height_norm])])

        # Lưu vị trí XY neo và yaw ban đầu để tính reward
        anchor_xy = mjx_data.qpos[:2]
        initial_yaw = quat_to_euler(mjx_data.qpos[3:7])[2]

        # Random key riêng cho push disturbance
        rng, push_key = jax.random.split(rng)

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
                "push_rng": push_key,
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step với height_command trong observation."""
        action = jnp.clip(action, -1.0, 1.0)

        # Scale action theo ctrlrange
        ctrl_range = self.mjx_model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]
        scaled_action = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

        mjx_data = state.mjx_data.replace(ctrl=scaled_action)

        # Push disturbance: áp dụng lực ngẫu nhiên mỗi push_interval steps
        # Lực được giữ trong push_duration steps rồi xóa
        push_rng = state.info["push_rng"]
        step_count = state.step_count
        is_push_step = (step_count % self._push_interval) == 0
        is_push_active = (step_count % self._push_interval) < self._push_duration

        push_rng, new_push_rng = jax.random.split(push_rng)
        mjx_data_pushed, _ = apply_external_force(
            mjx_data,
            push_rng,
            body_id=self._torso_id,
            magnitude=self._push_magnitude,
        )
        # Chỉ áp dụng khi enabled và đang trong push_duration window
        apply_push = self._push_enabled & is_push_active
        mjx_data = jax.lax.cond(
            apply_push,
            lambda: mjx_data_pushed,
            lambda: clear_external_force(mjx_data),
        )

        def physics_step(data, _):
            data = mjx.step(self.mjx_model, data)
            return data, None

        mjx_data, _ = jax.lax.scan(
            physics_step, mjx_data, None, length=self._n_substeps
        )

        # Base obs (39 dims)
        base_obs = self._extract_obs(mjx_data, action)

        # Append height_command (normalized)
        height_command = state.info["height_command"]
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        obs = jnp.concatenate([base_obs, jnp.array([height_norm])])

        # Reward
        reward = self._compute_reward(mjx_data, action, state)

        # Termination
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
                "height_command": height_command,  # giữ nguyên trong episode
                "anchor_xy": state.info["anchor_xy"],  # giữ nguyên
                "initial_yaw": state.info["initial_yaw"],  # giữ nguyên
                "push_rng": new_push_rng,  # cập nhật rng cho push tiếp theo
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho task balance multi-height."""

        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_pos = mjx_data.qpos[7:17]
        joint_vel = mjx_data.qvel[6:]
        ang_vel = mjx_data.qvel[3:6]
        torques = mjx_data.ctrl

        # Vận tốc tuyến tính body frame
        quat_inv = quat_conjugate(torso_quat)
        base_lin_vel = quat_rotate(quat_inv, mjx_data.qvel[:3])

        # Kiểm tra còn sống
        is_fallen = self._check_termination(mjx_data)
        is_alive = ~is_fallen

        # Height command từ state
        height_command = prev_state.info["height_command"]

        # Vị trí XY hiện tại và neo
        current_xy = mjx_data.qpos[:2]
        anchor_xy = prev_state.info["anchor_xy"]
        initial_yaw = prev_state.info["initial_yaw"]

        components = {
            # Thân nằm ngang song song mặt đất (phạt roll + pitch riêng)
            "body_level": reward_body_level(
                torso_quat, sigma_roll=0.15, sigma_pitch=0.15
            ),
            # Giữ chiều cao theo lệnh
            "height": reward_height(
                torso_height, height_command, sigma=0.25
            ),  # σ=0.25 → gradient đủ mạnh cả khi h_cmd xa keyframe
            # Chân hướng thẳng về phía trước (hip_yaw ≈ 0)
            "legs_forward": reward_legs_forward(joint_pos, sigma=0.15),
            # Chân vuông góc mặt đất (hip_roll ≈ 0)
            "legs_vertical": reward_legs_vertical(joint_pos, sigma=0.15),
            # 2 chân đối xứng
            "symmetry": reward_leg_symmetry(joint_pos, sigma=0.5),
            # Đứng yên, không di chuyển
            "no_motion": reward_no_motion(base_lin_vel, sigma=0.2),
            # Ổn định hướng (phạt rung lắc góc)
            "orientation": jnp.clip(
                1.0 - penalty_body_angular_velocity(ang_vel) * 0.01, 0.0, 1.0
            ),
            # Bonus sống sót
            "alive": reward_alive(is_alive),
            # Phạt mô-men lớn (tiết kiệm năng lượng)
            "joint_torque": penalty_joint_torque(torques),
            # Phạt vận tốc khớp lớn
            "joint_velocity": penalty_joint_velocity(joint_vel),
            # Phạt thay đổi action đột ngột
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
            # Phạt bánh xe quay (giữ robot đứng yên tại chỗ)
            "wheel_velocity": penalty_wheel_velocity(joint_vel),
            # Giữ vị trí neo — cho phép drift nhẹ khi cân bằng
            "position_drift": penalty_position_drift(current_xy, anchor_xy, sigma=0.5),
            # Giữ hướng ban đầu — phạt xoay yaw khỏi hướng reset
            "heading": reward_heading(torso_quat, initial_yaw),
            # Tư thế khớp tự nhiên — hp/kn phù hợp với chiều cao mục tiêu
            "natural_pose": reward_natural_pose(joint_pos, height_command, sigma=0.8),
        }

        return compute_total_reward(components, self._reward_weights)
