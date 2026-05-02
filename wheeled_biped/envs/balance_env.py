"""
Balance Environment - Huấn luyện robot đứng vững ở nhiều chiều cao.

Task: Giữ thân robot nằm ngang, duy trì chiều cao theo lệnh (height_command),
2 chân đối xứng, đứng yên ổn định, chống chịu nhiễu loạn.

Observation: 42 dims = base (39 or 36) + height_command (1) + current_height (1) + yaw_error (1)
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
from wheeled_biped.sim.domain_randomization import randomize_mjx_model
from wheeled_biped.sim.low_level_control import pid_control
from wheeled_biped.sim.push_disturbance import apply_push_disturbance
from wheeled_biped.utils.math_utils import (
    quat_conjugate,
    quat_rotate,
    quat_to_euler,
    wrap_angle,
)


class BalanceEnv(WheeledBipedEnv):
    """Environment cho task đứng vững ở nhiều chiều cao.

    Mỗi episode random một height_command ∈ [0.40, 0.70].
    Với curriculum: khởi đầu [0.69, 0.70],
    mở rộng dần → [0.40, 0.70] khi reward đạt ngưỡng (trainer quản lý).
    Robot phải giữ chiều cao theo lệnh, thân ngang, 2 chân đối xứng, đứng yên.

    Observation (42 dims when lin_vel_mode != "disabled", 39 dims when "disabled"):
        - base obs (39 or 36): gravity_body, [lin_vel,] ang_vel, joint_pos, joint_vel, prev_action
        - height_command (1): chiều cao mục tiêu (normalized về [0, 1] theo [MIN=0.40, MAX=0.70])
        - current_height (1): chiều cao torso thực tế (normalized về [0, 1] theo [MIN=0.40, MAX=0.70])
        - yaw_error (1): góc lệch yaw so với hướng mục tiêu (radians, wrapped to [-π, π])
    """

    # Khoảng chiều cao lệnh (m) — đo từ kinematics thực tế
    MIN_HEIGHT_CMD = 0.40
    MAX_HEIGHT_CMD = 0.70

    def __init__(self, config: dict[str, Any] | None = None, **kwargs):
        super().__init__(config=config, **kwargs)

        # Lấy trọng số reward
        reward_cfg = self.config.get("rewards", {})
        reward_params_cfg = self.config.get("reward_params", {})
        # Defaults mirror configs/training/balance.yaml so partial configs behave
        # consistently with the production config.  Any key present in the config
        # overrides the default via .get(); keys absent in the config use these values.
        self._reward_weights = {
            "body_level": reward_cfg.get("body_level", 1.5),
            "height": reward_cfg.get("height", 2.5),
            "legs_forward": reward_cfg.get("legs_forward", 0.5),
            "legs_vertical": reward_cfg.get("legs_vertical", 0.5),
            "joint_torque": reward_cfg.get("joint_torque", -0.0008),
            "joint_velocity": reward_cfg.get("joint_velocity", -0.001),
            "action_rate": reward_cfg.get("action_rate", -0.08),
            "orientation": reward_cfg.get("orientation", 1.0),
            "alive": reward_cfg.get("alive", 0.3),
            "no_motion": reward_cfg.get("no_motion", 1.0),
            "symmetry": reward_cfg.get("symmetry", 1.0),
            "wheel_velocity": reward_cfg.get("wheel_velocity", -0.006),
            "position_drift": reward_cfg.get("position_drift", 2.5),
            "heading": reward_cfg.get("heading", 0.5),
            "natural_pose": reward_cfg.get("natural_pose", 0.4),
            "yaw_rate": reward_cfg.get("yaw_rate", 0.5),
        }
        self._position_drift_sigma = float(reward_params_cfg.get("position_drift_sigma", 0.5))

        # Height curriculum config
        # Trainer sẽ quản lý curriculum_min_height dựa trên reward threshold
        # Env chỉ lưu initial_min_height để dùng làm giá trị mặc định ban đầu
        task_cfg = self.config.get("task", {})
        self._initial_min_height = float(
            task_cfg.get("initial_min_height", self.MIN_HEIGHT_CMD)
        )

        # Push disturbance + per-episode model DR config
        dr_cfg = self.config.get("domain_randomization", {})
        self._push_interval = int(dr_cfg.get("push_interval", 200))
        self._push_magnitude = float(dr_cfg.get("push_magnitude", 20.0))
        self._push_duration = int(dr_cfg.get("push_duration", 5))  # số steps giữ lực
        self._push_enabled = bool(dr_cfg.get("enabled", True))
        # Full DR config stored for randomize_mjx_model() (mass/friction/damping)
        self._dr_enabled = self._push_enabled  # reuse 'enabled' flag for all DR
        self._dr_config = dr_cfg

        # Low-level PID config: policy -> target, PID -> actuator ctrl
        pid_cfg = self.config.get("low_level_pid", {})
        self._pid_enabled = bool(pid_cfg.get("enabled", False))
        self._pid_smoothing_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
        self._pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
        self._wheel_vel_limit = float(pid_cfg.get("wheel_vel_limit", 20.0))
        # Action delay: simulates communication latency between policy computer and
        # motor drivers (CAN bus / EtherCAT typically 1–3 ms, i.e. 1 control step).
        # 0 = no delay (default, original behaviour).
        # N = hold the smoothed normalized target for N additional control steps
        #     before it is delivered to the PID controller.
        self._action_delay_steps = int(pid_cfg.get("action_delay_steps", 0))

        # Joint range theo thứ tự action/joint của env (qpos[7:])
        joint_mins = []
        joint_maxs = []
        for joint_name in self.JOINT_NAMES:
            jid = self.mj_model.joint(joint_name).id
            jrange = self.mj_model.jnt_range[jid]
            joint_mins.append(float(jrange[0]))
            joint_maxs.append(float(jrange[1]))
        self._joint_mins = jnp.array(joint_mins, dtype=jnp.float32)
        self._joint_maxs = jnp.array(joint_maxs, dtype=jnp.float32)

        # Wheel joint dùng velocity target thay vì position target
        wheel_indices = [i for i, n in enumerate(self.JOINT_NAMES) if "wheel" in n]
        wheel_mask = [
            1.0 if i in wheel_indices else 0.0 for i in range(self.num_actions)
        ]
        self._wheel_mask = jnp.array(wheel_mask, dtype=jnp.float32)

        # PID action bias: shifts policy action space so action=0 targets the
        # keyframe joint positions instead of the joint-range midpoints.
        #
        # Problem without this: action=0 drives joints to range midpoints:
        #   hip_pitch → 0.65 rad (midpoint of [-0.5, 1.8])
        #   knee      → 1.10 rad (midpoint of [-0.5, 2.7])
        # But the keyframe (0.71 m standing height) requires:
        #   hip_pitch → 0.30 rad, knee → 0.50 rad
        # This 0.35/0.60 rad mismatch causes the robot to deep-squat and lose
        # balance immediately at episode start (every episode), making training fail.
        #
        # With bias: action=0 → pos_target = keyframe → robot holds standing pose.
        # bias_i = 2*(kf_i - min_i) / (max_i - min_i) - 1  (wheel joints → 0)
        # hip_pitch: 2*(0.3 - (-0.5)) / (1.8 - (-0.5)) - 1 = -0.304
        # knee:      2*(0.5 - (-0.5)) / (2.7 - (-0.5)) - 1 = -0.375
        _kf = jnp.array(
            [0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.3, 0.5, 0.0], dtype=jnp.float32
        )
        # Wheel joints are unlimited (range=[0,0] in model) → avoid div-by-zero NaN.
        # Replace range with 1.0 for wheels; the result is zeroed by (1-wheel_mask) anyway.
        _jrange = jnp.where(
            self._wheel_mask > 0.5,
            1.0,
            self._joint_maxs - self._joint_mins,
        )
        self._pid_action_bias = (
            2.0 * (_kf - self._joint_mins) / _jrange - 1.0
        ) * (1.0 - self._wheel_mask)

        # PID gains (vector theo 10 joints), có fallback an toàn
        default_kp = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
        default_ki = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
        # Wheel kd (indices 4, 9) is 0.0: wheels use PI velocity control (no derivative).
        # pid_control() also masks wheel kd to 0 internally, so any non-zero value here
        # has no effect; using 0.0 avoids misleading readers of the fallback defaults.
        default_kd = [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0]
        kp_cfg = pid_cfg.get("kp", default_kp)
        ki_cfg = pid_cfg.get("ki", default_ki)
        kd_cfg = pid_cfg.get("kd", default_kd)
        if not isinstance(kp_cfg, list) or len(kp_cfg) != self.num_actions:
            kp_cfg = default_kp
        if not isinstance(ki_cfg, list) or len(ki_cfg) != self.num_actions:
            ki_cfg = default_ki
        if not isinstance(kd_cfg, list) or len(kd_cfg) != self.num_actions:
            kd_cfg = default_kd
        self._pid_kp = jnp.array(kp_cfg, dtype=jnp.float32)
        self._pid_ki = jnp.array(ki_cfg, dtype=jnp.float32)
        self._pid_kd = jnp.array(kd_cfg, dtype=jnp.float32)

        # Cache actuator ctrl range cho clip output PID
        ctrl_range = self.mjx_model.actuator_ctrlrange
        self._ctrl_min = ctrl_range[:, 0]
        self._ctrl_max = ctrl_range[:, 1]

        # Lấy torso body_id từ mj_model
        import mujoco

        self._torso_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

    def _pid_low_level_ctrl(
        self,
        mjx_data: mjx.Data,
        normalized_target: jnp.ndarray,
        pid_integral: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Delegate to the reusable pid_control helper in sim.low_level_control."""
        return pid_control(
            mjx_data,
            normalized_target,
            pid_integral,
            kp=self._pid_kp,
            ki=self._pid_ki,
            kd=self._pid_kd,
            joint_mins=self._joint_mins,
            joint_maxs=self._joint_maxs,
            wheel_mask=self._wheel_mask,
            wheel_vel_limit=self._wheel_vel_limit,
            i_limit=self._pid_i_limit,
            ctrl_min=self._ctrl_min,
            ctrl_max=self._ctrl_max,
            control_dt=self.CONTROL_DT,
        )

    def _compute_obs_size(self) -> int:
        """Observation = base (39 or 36) + height_command (1) + current_height (1) + yaw_error (1).

        Default (clean lin_vel mode): 39 + 1 + 1 + 1 = 42
        Disabled lin_vel mode:        36 + 1 + 1 + 1 = 39
        """
        return super()._compute_obs_size() + 3

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.Array) -> EnvState:
        """Reset environment với random height_command."""
        rng, sub_key = jax.random.split(rng)
        mjx_data = self._get_initial_mjx_data(sub_key)

        # Per-episode model DR (mass/friction/damping) — widened ranges in balance.yaml
        if self._dr_enabled:
            rng, dr_key = jax.random.split(rng)
            dr_mjx_model, _ = randomize_mjx_model(
                self.mjx_model, dr_key, self._dr_config
            )
        else:
            dr_mjx_model = self.mjx_model

        # Nhiễu vị trí khớp tại reset (widened ±0.05 → ±0.10 rad for DR)
        rng, joint_noise_key = jax.random.split(rng)
        joint_noise = jax.random.uniform(
            joint_noise_key, shape=(self.NUM_JOINTS,), minval=-0.10, maxval=0.10
        )
        new_qpos = mjx_data.qpos.at[7:].add(joint_noise)
        mjx_data = mjx_data.replace(qpos=new_qpos)

        # Random height command cho episode này
        # Ban đầu dùng initial_min_height (gần keyframe), curriculum sẽ mở rộng sau
        rng, height_key = jax.random.split(rng)
        height_command = jax.random.uniform(
            height_key,
            shape=(),
            minval=self._initial_min_height,
            maxval=self.MAX_HEIGHT_CMD,
        )

        # Obs noise key — separate from push_rng for independent noise stream
        rng, obs_noise_key = jax.random.split(rng)

        prev_action = jnp.zeros(self.num_actions)
        base_obs = self._extract_obs(mjx_data, prev_action, obs_noise_key)

        # Normalize height_command về [0, 1]
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        # current_height_norm: actual torso height normalized to [0, 1]
        current_height_norm = (mjx_data.qpos[2] - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        # yaw_error = 0 at episode start (robot begins at initial heading).
        # This extra dim makes heading control observable to the MLP policy:
        # gravity_body is yaw-invariant, so without yaw_error the policy cannot
        # sense accumulated heading drift.
        obs = jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, 0.0])])

        # Lưu vị trí XY neo và yaw ban đầu để tính reward
        anchor_xy = mjx_data.qpos[:2]
        initial_yaw = quat_to_euler(mjx_data.qpos[3:7])[2]

        # Random key riêng cho push disturbance
        rng, push_key = jax.random.split(rng)

        info = {
            "is_fallen": jnp.bool_(False),
            "time_limit": jnp.bool_(False),
            "height_command": height_command,
            "anchor_xy": anchor_xy,
            "initial_yaw": initial_yaw,
            "push_rng": push_key,
            "noise_rng": obs_noise_key,
            "dr_mjx_model": dr_mjx_model,
            "lifetime_steps": jnp.int32(0),
            "curriculum_min_height": jnp.float32(self._initial_min_height),
            "pid_integral": jnp.zeros(self.num_actions, dtype=jnp.float32),
        }
        # Action delay buffer: stores the last N smoothed targets waiting to be
        # applied.  Slot [0] = oldest (applied next step), slot [-1] = newest.
        # Initialized to zeros so early steps see zero command (safe initial state).
        if self._action_delay_steps > 0:
            info["action_delay_buffer"] = jnp.zeros(
                (self._action_delay_steps, self.num_actions), dtype=jnp.float32
            )

        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            prev_action=prev_action,
            info=info,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """Step với height_command trong observation."""
        action = jnp.clip(action, -1.0, 1.0)

        # Step 1 — Smooth the raw policy output (reduces chattering).
        # smooth_action is the policy's intended target this step.
        if self._pid_enabled and self._pid_smoothing_alpha > 0.0:
            smooth_action = (
                self._pid_smoothing_alpha * state.prev_action
                + (1.0 - self._pid_smoothing_alpha) * action
            )
        else:
            smooth_action = action

        # Step 2 — Action delay: hold smooth_action for _action_delay_steps
        # steps before it reaches the PID controller.  This models communication
        # latency between the policy computer and motor drivers (~20 ms / step).
        # When delay=0 the buffer is absent and control_action = smooth_action.
        if self._action_delay_steps > 0:
            # Oldest entry in the buffer is the delayed target to apply now.
            delayed_action = state.info["action_delay_buffer"][0]
            # Shift left, enqueue the current smooth_action at the end.
            new_delay_buffer = jnp.concatenate(
                [state.info["action_delay_buffer"][1:], smooth_action[None, :]],
                axis=0,
            )
            control_action = delayed_action
        else:
            control_action = smooth_action

        # Step 3 — Direct torque mode (cũ) hoặc PID low-level mode
        if self._pid_enabled:
            # Apply keyframe bias so policy action=0 → PID targets keyframe pose.
            biased_action = jnp.clip(control_action + self._pid_action_bias, -1.0, 1.0)
            scaled_action, pid_integral = self._pid_low_level_ctrl(
                state.mjx_data,
                biased_action,
                state.info["pid_integral"],
            )
        else:
            scaled_action = self._ctrl_min + (control_action + 1.0) * 0.5 * (
                self._ctrl_max - self._ctrl_min
            )
            pid_integral = state.info["pid_integral"]

        mjx_data = state.mjx_data.replace(ctrl=scaled_action)

        # Push disturbance: áp dụng lực ngẫu nhiên mỗi push_interval steps
        push_rng = state.info["push_rng"]
        mjx_data, new_push_rng = apply_push_disturbance(
            mjx_data,
            push_rng,
            body_id=self._torso_id,
            step_count=state.step_count,
            push_interval=self._push_interval,
            push_duration=self._push_duration,
            push_magnitude=self._push_magnitude,
            push_enabled=self._push_enabled,
        )

        # Use per-episode DR model for physics (randomized mass/friction/damping)
        dr_mjx_model = state.info["dr_mjx_model"]

        def physics_step(data, _):
            data = mjx.step(dr_mjx_model, data)
            return data, None

        mjx_data, _ = jax.lax.scan(
            physics_step, mjx_data, None, length=self._n_substeps
        )

        # Advance obs noise RNG and extract noisy base obs (39 dims)
        noise_key, new_noise_rng = jax.random.split(state.info["noise_rng"])
        base_obs = self._extract_obs(mjx_data, control_action, noise_key)

        # Append height_command (normalized), current_height (normalized), and yaw_error.
        height_command = state.info["height_command"]
        height_norm = (height_command - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        current_height_norm = (mjx_data.qpos[2] - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        # yaw_error: wrap_angle produces a value in [-π, π].
        # The obs normalizer (Welford running mean/std) will standardize this
        # during training.  Directly observable yaw drift closes the heading loop
        # that ang_vel alone cannot close in a memoryless MLP policy.
        current_yaw = quat_to_euler(mjx_data.qpos[3:7])[2]
        yaw_error = wrap_angle(current_yaw - state.info["initial_yaw"])
        obs = jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, yaw_error])])

        # Reward
        reward = self._compute_reward(mjx_data, control_action, state)

        # Termination
        is_fallen = self._check_termination(mjx_data)
        step_count = state.step_count + 1
        time_limit = step_count >= self._episode_length
        done = is_fallen | time_limit

        # prev_action stored in state = smooth_action (the policy's intended target).
        # This is used for: (a) the action-smoothing recurrence next step, and
        # (b) the prev_action channel in the observation.  Using smooth_action
        # (not delayed_action) keeps the observation semantics consistent with
        # "what did the policy last request", regardless of pipeline delay.
        new_info = {
            "is_fallen": is_fallen,
            "time_limit": time_limit,
            "height_command": height_command,
            "anchor_xy": state.info["anchor_xy"],
            "initial_yaw": state.info["initial_yaw"],
            "push_rng": new_push_rng,
            "noise_rng": new_noise_rng,
            "dr_mjx_model": state.info["dr_mjx_model"],
            "lifetime_steps": state.info["lifetime_steps"] + 1,
            "curriculum_min_height": state.info["curriculum_min_height"],
            "pid_integral": pid_integral,
        }
        if self._action_delay_steps > 0:
            new_info["action_delay_buffer"] = new_delay_buffer

        return EnvState(
            mjx_data=mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            step_count=step_count,
            prev_action=smooth_action,  # policy's intended target (pre-delay)
            info=new_info,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset_if_done(self, state: EnvState, rng) -> EnvState:
        """Override base: carry lifetime_steps + curriculum_min_height.

        curriculum_min_height được trainer quản lý (dựa trên reward threshold).
        Env chỉ carry giá trị này qua các episode và dùng nó để sample height.
        """
        new_state = self.reset(rng)

        # Carry từ old state (KHÔNG reset)
        carried_lifetime = state.info["lifetime_steps"]
        curriculum_min = state.info["curriculum_min_height"]

        # Re-sample height trong curriculum range hiện tại
        rng_h, _ = jax.random.split(rng)
        curriculum_height = jax.random.uniform(
            rng_h, shape=(), minval=curriculum_min, maxval=self.MAX_HEIGHT_CMD
        )
        # Normalize vẫn dùng FULL range để obs nhất quán
        height_norm = (curriculum_height - self.MIN_HEIGHT_CMD) / (
            self.MAX_HEIGHT_CMD - self.MIN_HEIGHT_CMD
        )
        # obs[-3] = height_norm, obs[-2] = current_height_norm (set by reset()), obs[-1] = yaw_error
        new_obs = new_state.obs.at[-3].set(height_norm)
        new_info = {
            **new_state.info,
            "height_command": curriculum_height,
            "curriculum_min_height": curriculum_min,
            "lifetime_steps": carried_lifetime,
        }
        new_state = new_state._replace(info=new_info, obs=new_obs)

        return jax.tree.map(
            lambda new, old: jnp.where(state.done, new, old),
            new_state,
            state,
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
            "symmetry": reward_leg_symmetry(joint_pos, sigma=0.3),
            # Đứng yên, không di chuyển — công thức tuyến tính để gradient còn tác dụng
            # ngay cả khi robot đang drift nhanh (exp_kernel sigma=0.2 → 0 tại >0.5 m/s)
            "no_motion": reward_no_motion(base_lin_vel, sigma=0.2),
            # Ổn định tổng thể (phạt rung lắc góc tất cả trục)
            # Linear: gradient tuyến tính trong [0, 6.67 rad/s]; hiệu quả hơn exp-saturation
            "orientation": jnp.clip(
                1.0 - jnp.sqrt(penalty_body_angular_velocity(ang_vel)) * 0.15, 0.0, 1.0
            ),
            # Phạt riêng tốc độ xoay yaw (chống spinning)
            "yaw_rate": jnp.clip(1.0 - jnp.abs(ang_vel[2]) * 0.5, 0.0, 1.0),
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
            # Giữ vị trí neo.  balance.yaml tightens sigma to 0.25m for
            # stationary standing; robust/recovery configs can keep it wider.
            "position_drift": penalty_position_drift(
                current_xy, anchor_xy, sigma=self._position_drift_sigma
            ),
            # Giữ hướng ban đầu — sigma=0.5 rad: gradient tới ~80° (thay vì 12° với sigma=0.1)
            "heading": reward_heading(torso_quat, initial_yaw, sigma=0.5),
            # Tư thế khớp tự nhiên — hp/kn phù hợp với chiều cao mục tiêu
            "natural_pose": reward_natural_pose(joint_pos, height_command, sigma=0.5),
        }

        return compute_total_reward(components, self._reward_weights)
