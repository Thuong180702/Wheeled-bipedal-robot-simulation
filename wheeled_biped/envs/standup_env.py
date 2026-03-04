"""
Stand-Up Environment - Huấn luyện robot đứng dậy từ trạng thái nằm/ngã.

Task: Robot bắt đầu ở tư thế nằm ngửa, nằm sấp, hoặc ngẫu nhiên trên mặt đất.
Mục tiêu: Đứng dậy ổn định về tư thế đứng thẳng (target height ~0.65m).

Đây là task bổ sung quan trọng — khi robot bị ngã trong thực tế,
nó cần khả năng tự đứng dậy.
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
    reward_alive,
    reward_height_progress,
    reward_stand_up_phase,
    reward_upright,
)
from wheeled_biped.utils.math_utils import quat_to_euler


class StandUpEnv(WheeledBipedEnv):
    """Environment cho task đứng dậy.

    Robot khởi tạo ở tư thế ngã (nằm ngửa, nằm sấp, hoặc nghiêng),
    và phải tự đứng dậy về tư thế thẳng đứng.

    Observation mở rộng thêm 2 chiều:
      - target_height (1): chiều cao mục tiêu
      - height_error (1): sai lệch chiều cao hiện tại so với mục tiêu
    Total obs: 39 + 2 = 41

    Reward:
      - Thưởng tiến bộ chiều cao (height progress)
      - Thưởng đứng thẳng (upright) — bonus khi gần thẳng đứng
      - Thưởng stand-up phase (kết hợp height + upright)
      - Phạt mô-men lớn
      - Phạt action rate
      - Phạt vận tốc góc (rung lắc)
      - Bonus sống sót
    """

    # Các tư thế khởi tạo ngã
    FALL_MODES = ["supine", "prone", "left_side", "right_side", "random_tilt"]

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Cấu hình task
        task_cfg = self.config.get("task", {})
        self._target_height = task_cfg.get("target_height", 0.65)

        # Termination nới lỏng hơn — cho phép nằm trên đất lâu hơn
        term_cfg = self.config.get("termination", {})
        self._max_tilt = term_cfg.get("max_tilt_rad", 3.14)  # gần như không giới hạn
        self._min_height = term_cfg.get("min_height", 0.05)  # rất thấp
        self._episode_length = task_cfg.get("episode_length", 1500)  # dài hơn

        # Reward weights
        reward_cfg = self.config.get("rewards", {})
        self._reward_weights = {
            "height_progress": reward_cfg.get("height_progress", 2.0),
            "stand_up_phase": reward_cfg.get("stand_up_phase", 1.5),
            "upright": reward_cfg.get("upright", 1.0),
            "height_target": reward_cfg.get("height_target", 0.8),
            "joint_torque": reward_cfg.get("joint_torque", -0.00005),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.00005),
            "action_rate": reward_cfg.get("action_rate", -0.0005),
            "angular_velocity": reward_cfg.get("angular_velocity", -0.001),
            "alive": reward_cfg.get("alive", 0.1),
        }

        # Obs thêm 2 chiều: target_height + height_error
        self.obs_size += 2

    def _compute_obs_size(self) -> int:
        """Base obs = 39, sẽ cộng thêm 2 trong __init__."""
        return super()._compute_obs_size()

    def _get_fallen_mjx_data(self, rng: jax.Array) -> mjx.Data:
        """Tạo MJX data ở tư thế ngã ngẫu nhiên.

        Có 5 chế độ: nằm ngửa, nằm sấp, nghiêng trái/phải, nghiêng ngẫu nhiên.
        """
        mj_data = mujoco.MjData(self.mj_model)

        # Lấy tư thế đứng mặc định trước
        if self.mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
        else:
            mj_data.qpos[:3] = [0, 0, 0.68]
            mj_data.qpos[3:7] = [1, 0, 0, 0]
            mj_data.qpos[7:] = [0, 0.4, 0.8, -0.4, 0, 0, 0.4, 0.8, -0.4, 0]

        # Thả robot nằm trên mặt đất — chiều cao thấp
        # Quaternion [w, x, y, z] theo MuJoCo convention
        # Nằm ngửa = xoay 90° quanh trục X (pitch = -π/2)
        # Nằm sấp = xoay 90° quanh trục X (pitch = +π/2)

        # Chuyển RNG sang numpy seed (vì MuJoCo CPU)
        rng_int = int(jax.random.randint(rng, (), 0, len(self.FALL_MODES)))

        # Chọn ngẫu nhiên trong 5 chế độ (cần deterministic với rng)
        # Dùng modulo đơn giản
        mode_idx = rng_int % len(self.FALL_MODES)

        if mode_idx == 0:  # supine — nằm ngửa
            mj_data.qpos[:3] = [0, 0, 0.15]
            # Quaternion cho pitch = -π/2 (nằm ngửa, bụng hướng lên)
            mj_data.qpos[3:7] = [0.7071, -0.7071, 0, 0]
            # Các khớp gập lại
            mj_data.qpos[7:] = [0, -0.5, 1.5, -0.3, 0, 0, -0.5, 1.5, -0.3, 0]
        elif mode_idx == 1:  # prone — nằm sấp
            mj_data.qpos[:3] = [0, 0, 0.15]
            # pitch = +π/2 (nằm sấp, mặt hướng xuống)
            mj_data.qpos[3:7] = [0.7071, 0.7071, 0, 0]
            mj_data.qpos[7:] = [0, 0.8, 0.5, -0.2, 0, 0, 0.8, 0.5, -0.2, 0]
        elif mode_idx == 2:  # left side — nghiêng trái
            mj_data.qpos[:3] = [0, 0, 0.15]
            # roll = +π/2 (nghiêng sang trái)
            mj_data.qpos[3:7] = [0.7071, 0, 0.7071, 0]
            mj_data.qpos[7:] = [0.3, 0.4, 0.8, -0.4, 0, -0.3, 0.4, 0.8, -0.4, 0]
        elif mode_idx == 3:  # right side — nghiêng phải
            mj_data.qpos[:3] = [0, 0, 0.15]
            # roll = -π/2 (nghiêng sang phải)
            mj_data.qpos[3:7] = [0.7071, 0, -0.7071, 0]
            mj_data.qpos[7:] = [-0.3, 0.4, 0.8, -0.4, 0, 0.3, 0.4, 0.8, -0.4, 0]
        else:  # random tilt — nghiêng ngẫu nhiên
            mj_data.qpos[:3] = [0, 0, 0.20]
            # Nghiêng ngẫu nhiên ~60-90 độ
            angle = 1.0 + (rng_int % 100) / 100.0 * 0.57  # 1.0 ~ 1.57 rad
            axis_choice = rng_int % 3
            if axis_choice == 0:
                mj_data.qpos[3:7] = [
                    float(jnp.cos(angle / 2)),
                    float(jnp.sin(angle / 2)),
                    0, 0,
                ]
            elif axis_choice == 1:
                mj_data.qpos[3:7] = [
                    float(jnp.cos(angle / 2)),
                    0,
                    float(jnp.sin(angle / 2)),
                    0,
                ]
            else:
                sa = float(jnp.sin(angle / 2)) * 0.7071
                mj_data.qpos[3:7] = [float(jnp.cos(angle / 2)), sa, sa, 0]
            mj_data.qpos[7:] = [0, 0.3, 1.0, -0.3, 0, 0, 0.3, 1.0, -0.3, 0]

        # Forward kinematics để cập nhật contact
        mujoco.mj_forward(self.mj_model, mj_data)

        return mjx.put_data(self.mj_model, mj_data)

    def _extract_obs(
        self,
        mjx_data: mjx.Data,
        prev_action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Trích xuất obs + height info cho stand-up task."""
        base_obs = super()._extract_obs(mjx_data, prev_action)
        torso_height = mjx_data.qpos[2]
        height_error = self._target_height - torso_height
        extra = jnp.array([self._target_height, height_error])
        return jnp.concatenate([base_obs, extra])

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset — robot bắt đầu ở tư thế ngã."""
        rng, fall_key, noise_key = jax.random.split(rng, 3)

        # Tạo tư thế ngã ngẫu nhiên
        mjx_data = self._get_fallen_mjx_data(fall_key)

        # Thêm nhiễu nhỏ vào khớp
        joint_noise = jax.random.uniform(
            noise_key, shape=(self.NUM_JOINTS,), minval=-0.03, maxval=0.03
        )
        new_qpos = mjx_data.qpos.at[7:].add(joint_noise)
        mjx_data = mjx_data.replace(qpos=new_qpos)

        prev_action = jnp.zeros(self.num_actions)
        obs = self._extract_obs(mjx_data, prev_action)

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
                "prev_height": mjx_data.qpos[2],  # Để tính height progress
            },
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step — mục tiêu đứng dậy."""
        # Lưu chiều cao trước khi step
        prev_height = state.mjx_data.qpos[2]

        # Gọi base step
        new_state = super().step(state, action)

        # Cập nhật obs (đã override _extract_obs)
        obs = self._extract_obs(new_state.mjx_data, action)

        # Cập nhật info với prev_height
        new_info = {**new_state.info, "prev_height": prev_height}

        return new_state._replace(obs=obs, info=new_info)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _check_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Termination nới lỏng — chỉ kết thúc nếu chiều cao cực thấp.

        Không dùng tilt check vì robot ban đầu đã nằm.
        """
        torso_height = mjx_data.qpos[2]
        # Chỉ terminate khi quá thấp (ví dụ rơi khỏi map)
        return torso_height < self._min_height

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward cho task đứng dậy."""
        torso_quat = mjx_data.qpos[3:7]
        torso_height = mjx_data.qpos[2]
        joint_vel = mjx_data.qvel[6:]
        ang_vel = mjx_data.qvel[3:6]
        torques = mjx_data.ctrl
        prev_height = prev_state.info.get("prev_height", torso_height)

        components = {
            # Thưởng nâng cao (delta height)
            "height_progress": reward_height_progress(
                torso_height, prev_height, scale=5.0
            ),
            # Thưởng phase tổng hợp (height × upright)
            "stand_up_phase": reward_stand_up_phase(
                torso_height, torso_quat, self._target_height
            ),
            # Thưởng đứng thẳng
            "upright": reward_upright(torso_quat),
            # Bonus khi đạt chiều cao gần mục tiêu
            "height_target": jnp.where(
                torso_height > self._target_height * 0.85,
                1.0,
                torso_height / self._target_height,
            ),
            # Phạt
            "joint_torque": penalty_joint_torque(torques),
            "joint_velocity": penalty_joint_velocity(joint_vel),
            "action_rate": penalty_action_rate(action, prev_state.prev_action),
            "angular_velocity": penalty_body_angular_velocity(ang_vel),
            # Bonus sống sót
            "alive": jnp.float32(1.0),
        }

        return compute_total_reward(components, self._reward_weights)
