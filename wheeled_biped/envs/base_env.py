"""
Base Environment cho Wheeled Bipedal Robot trên MuJoCo MJX.

Cung cấp:
  - Khởi tạo MuJoCo/MJX model
  - State dataclass tương thích JAX
  - reset() / step() interface
  - Trích xuất observation từ mjx_data
  - Kiểm tra termination

Tất cả method đều tương thích jax.jit và jax.vmap.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.utils.math_utils import get_gravity_in_body_frame, quat_to_euler


class EnvState(NamedTuple):
    """Trạng thái environment - immutable, JAX-compatible."""

    mjx_data: mjx.Data  # Dữ liệu vật lý MJX
    obs: jnp.ndarray  # Observation vector
    reward: jnp.ndarray  # Reward bước hiện tại
    done: jnp.ndarray  # Episode kết thúc?
    step_count: jnp.ndarray  # Bước đếm trong episode
    prev_action: jnp.ndarray  # Action bước trước (cho action rate penalty)
    info: dict[str, Any]  # Thông tin phụ


class WheeledBipedEnv:
    """Base environment cho robot hai chân có bánh xe.

    Sử dụng MuJoCo MJX để chạy physics trên GPU thông qua JAX.

    Attributes:
        mj_model: MuJoCo model (CPU).
        mjx_model: MJX model (GPU).
        config: dict cấu hình.
        num_actions: số lượng actuator (10).
        obs_size: kích thước observation vector.
    """

    # --- Thông số cố định ---
    NUM_JOINTS: int = 10  # 5 loại × 2 bên
    PHYSICS_DT: float = 0.002  # 500Hz
    CONTROL_DT: float = 0.02  # 50Hz → 10 substeps

    # Tên khớp theo thứ tự
    JOINT_NAMES: list[str] = [
        "l_hip_roll",
        "l_hip_pitch",
        "l_knee",
        "l_ankle",
        "l_wheel",
        "r_hip_roll",
        "r_hip_pitch",
        "r_knee",
        "r_ankle",
        "r_wheel",
    ]

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_path: str | Path | None = None,
    ):
        """Khởi tạo environment.

        Args:
            config: dict cấu hình task.
            model_path: đường dẫn file MJCF (mặc định: model trong assets/).
        """
        self.config = config or {}

        # Tải MuJoCo model
        model_path = model_path or get_model_path()
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))

        # Cấu hình physics
        substeps = int(self.CONTROL_DT / self.PHYSICS_DT)
        self._n_substeps = substeps

        # Tạo MJX model (GPU)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Lấy thông tin kích thước
        self.num_actions = self.mj_model.nu  # 10 actuators
        self._num_qpos = self.mj_model.nq  # 7 (freejoint) + 10 (joints)
        self._num_qvel = self.mj_model.nv  # 6 (freejoint) + 10 (joints)

        # Lấy ID các body/joint quan trọng
        self._torso_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

        # Tính kích thước observation
        self.obs_size = self._compute_obs_size()

        # Cấu hình termination
        term_cfg = self.config.get("termination", {})
        self._max_tilt = term_cfg.get("max_tilt_rad", 0.8)
        self._min_height = term_cfg.get("min_height", 0.3)

        # Episode length
        task_cfg = self.config.get("task", {})
        self._episode_length = task_cfg.get("episode_length", 1000)

    def _compute_obs_size(self) -> int:
        """Tính kích thước observation vector.

        Observation bao gồm:
          - Trọng lực trong body frame: 3
          - Vận tốc thân (linear): 3
          - Vận tốc thân (angular): 3
          - Vị trí khớp: 10
          - Vận tốc khớp: 10
          - Action bước trước: 10
          Total: 39
        """
        return 3 + 3 + 3 + self.NUM_JOINTS + self.NUM_JOINTS + self.NUM_JOINTS

    def _get_initial_mjx_data(self, rng: jax.Array) -> mjx.Data:
        """Tạo dữ liệu ban đầu cho MJX (tư thế đứng).

        Args:
            rng: JAX random key.

        Returns:
            mjx.Data ở tư thế khởi tạo.
        """
        mj_data = mujoco.MjData(self.mj_model)

        # Đặt tư thế đứng từ keyframe nếu có
        if self.mj_model.nkey > 0:
            key_id = 0
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, key_id)
        else:
            # Tư thế mặc định
            mj_data.qpos[:3] = [0, 0, 0.68]  # vị trí thân
            mj_data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (đứng thẳng)
            # Các khớp: [l_hip_roll, l_hip_pitch, l_knee, l_ankle, l_wheel,
            #            r_hip_roll, r_hip_pitch, r_knee, r_ankle, r_wheel]
            mj_data.qpos[7:] = [0, 0.4, 0.8, -0.4, 0, 0, 0.4, 0.8, -0.4, 0]

        # Forward kinematics
        mujoco.mj_forward(self.mj_model, mj_data)

        # Chuyển sang MJX
        return mjx.put_data(self.mj_model, mj_data)

    def _extract_obs(self, mjx_data: mjx.Data, prev_action: jnp.ndarray) -> jnp.ndarray:
        """Trích xuất observation từ MJX data.

        Args:
            mjx_data: dữ liệu MJX.
            prev_action: action bước trước.

        Returns:
            Observation vector.
        """
        # Quaternion thân (qpos[3:7])
        torso_quat = mjx_data.qpos[3:7]

        # Trọng lực trong body frame (3,) - cảm biến quan trọng nhất
        gravity_body = get_gravity_in_body_frame(torso_quat)

        # Vận tốc thân trong world frame
        base_lin_vel = mjx_data.qvel[:3]  # (3,) linear velocity
        base_ang_vel = mjx_data.qvel[3:6]  # (3,) angular velocity

        # Vị trí + vận tốc các khớp (trừ freejoint)
        joint_pos = mjx_data.qpos[7:]  # (10,)
        joint_vel = mjx_data.qvel[6:]  # (10,)

        # Ghép lại
        obs = jnp.concatenate(
            [
                gravity_body,  # 3
                base_lin_vel,  # 3
                base_ang_vel,  # 3
                joint_pos,  # 10
                joint_vel,  # 10
                prev_action,  # 10
            ]
        )

        return obs

    def _check_termination(self, mjx_data: mjx.Data) -> jnp.ndarray:
        """Kiểm tra điều kiện kết thúc episode.

        Robot bị coi là ngã khi:
          - Nghiêng quá nhiều
          - Chiều cao quá thấp

        Args:
            mjx_data: dữ liệu MJX.

        Returns:
            Boolean: True nếu cần kết thúc.
        """
        # Chiều cao thân
        torso_height = mjx_data.qpos[2]

        # Góc nghiêng từ quaternion
        torso_quat = mjx_data.qpos[3:7]
        euler = quat_to_euler(torso_quat)
        tilt = jnp.sqrt(euler[0] ** 2 + euler[1] ** 2)  # roll² + pitch²

        is_fallen = (torso_height < self._min_height) | (tilt > self._max_tilt)
        return is_fallen

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset environment về trạng thái ban đầu.

        Args:
            rng: JAX random key.

        Returns:
            EnvState ban đầu.
        """
        rng, sub_key = jax.random.split(rng)
        mjx_data = self._get_initial_mjx_data(sub_key)

        # Thêm nhiễu nhỏ vào vị trí khớp (tránh overfitting)
        rng, noise_key = jax.random.split(rng)
        joint_noise = jax.random.uniform(
            noise_key, shape=(self.NUM_JOINTS,), minval=-0.05, maxval=0.05
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
            info={"is_fallen": jnp.bool_(False), "time_limit": jnp.bool_(False)},
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Thực hiện một step điều khiển.

        Mỗi step chạy n_substeps bước physics (50Hz → 10 × 500Hz).

        Args:
            state: trạng thái hiện tại.
            action: action từ policy (10,).

        Returns:
            EnvState mới.
        """
        # Clip action theo giới hạn actuator
        action = jnp.clip(action, -1.0, 1.0)

        # Scale action theo ctrlrange
        ctrl_range = self.mjx_model.actuator_ctrlrange
        ctrl_min = ctrl_range[:, 0]
        ctrl_max = ctrl_range[:, 1]
        scaled_action = ctrl_min + (action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

        # Đặt control vào MJX data
        mjx_data = state.mjx_data.replace(ctrl=scaled_action)

        # Chạy n bước physics
        def physics_step(data, _):
            data = mjx.step(self.mjx_model, data)
            return data, None

        mjx_data, _ = jax.lax.scan(
            physics_step, mjx_data, None, length=self._n_substeps
        )

        # Trích xuất observation
        obs = self._extract_obs(mjx_data, action)

        # Tính reward (sẽ được override bởi subclass)
        reward = self._compute_reward(mjx_data, action, state)

        # Kiểm tra termination
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
            info={"is_fallen": is_fallen, "time_limit": time_limit},
        )

    def _compute_reward(
        self,
        mjx_data: mjx.Data,
        action: jnp.ndarray,
        prev_state: EnvState,
    ) -> jnp.ndarray:
        """Tính reward (sẽ được override bởi các task env cụ thể).

        Returns:
            Scalar reward.
        """
        return jnp.float32(0.0)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_if_done(self, state: EnvState, rng: jax.Array) -> EnvState:
        """Auto-reset nếu episode kết thúc (dùng trong training loop).

        Args:
            state: trạng thái hiện tại.
            rng: random key.

        Returns:
            State mới (reset nếu done=True, giữ nguyên nếu chưa).
        """
        new_state = self.reset(rng)
        return jax.tree.map(
            lambda new, old: jnp.where(state.done, new, old),
            new_state,
            state,
        )

    # --- Vectorized interface (cho training song song) ---

    def v_reset(self, rng: jax.Array, num_envs: int) -> EnvState:
        """Reset num_envs môi trường song song.

        Args:
            rng: random key.
            num_envs: số lượng env.

        Returns:
            Batched EnvState.
        """
        keys = jax.random.split(rng, num_envs)
        return jax.vmap(self.reset)(keys)

    def v_step(self, states: EnvState, actions: jnp.ndarray) -> EnvState:
        """Step num_envs môi trường song song.

        Args:
            states: batched EnvState.
            actions: (num_envs, num_actions).

        Returns:
            Batched EnvState mới.
        """
        return jax.vmap(self.step)(states, actions)

    def v_reset_if_done(self, states: EnvState, rng: jax.Array) -> EnvState:
        """Auto-reset các env đã done.

        Args:
            states: batched EnvState.
            rng: random key.

        Returns:
            Batched EnvState (đã reset nếu cần).
        """
        num_envs = states.done.shape[0]
        keys = jax.random.split(rng, num_envs)
        return jax.vmap(self.reset_if_done)(states, keys)
