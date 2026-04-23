"""
Unified Multi-Skill Controller
================================

Nạp nhiều checkpoint (balance, locomotion, walking, stair, terrain)
và tự động chuyển đổi giữa các skill dựa trên:
  - IMU orientation (ngã / đứng)
  - Velocity command (đứng yên / di chuyển)
  - Terrain cảm biến (phẳng / bậc thang / gồ ghề)
  - Trạng thái chân (chạm đất / lơ lửng)

Ưu tiên:  Balance → Locomotion → Walking → Stair → Terrain
(nếu không có checkpoint nào thì bỏ qua skill đó)

Changes from original
---------------------
Fix 1 – Blend counter only resets when the *target* skill changes (not on
         every step where desired != active). Transitions now complete
         naturally after ``_blend_steps`` consecutive steps.
Fix 2 – Explicit per-skill obs adapters replace the silent generic
         zero-pad / truncation fallback. A ``ValueError`` is raised for
         genuine schema mismatches, keeping the ``unknown_pad`` path as an
         explicit escape hatch that logs a warning.
Fix 3 – Dwell-time hysteresis: a raw skill detection must be stable for
         ``dwell_threshold`` (default 3) consecutive calls before the
         controller acts on it, preventing single-frame spikes from
         triggering transitions.
Fix 4 – Per-skill ``_prev_actions`` buffers replace the single shared
         ``_prev_action``. Each skill's observation is always built from
         its own last action, even during blending.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.training.networks import create_actor_critic
from wheeled_biped.training.ppo import normalize_obs
from wheeled_biped.utils.math_utils import (
    get_gravity_in_body_frame,
    quat_conjugate,
    quat_rotate,
    quat_to_euler,
    wrap_angle,
)


class Skill(Enum):
    """Các kỹ năng có thể chuyển đổi."""

    BALANCE = auto()
    LOCOMOTION = auto()
    WALKING = auto()
    STAIR = auto()
    TERRAIN = auto()
    STAND_UP = auto()


# Thứ tự ưu tiên: skill nào chuyên biệt hơn sẽ ưu tiên khi điều kiện phù hợp
_SKILL_PRIORITY = [
    Skill.TERRAIN,
    Skill.STAIR,
    Skill.WALKING,
    Skill.LOCOMOTION,
    Skill.STAND_UP,
    Skill.BALANCE,
]

# ---------------------------------------------------------------------------
# Observation adapter registry
# ---------------------------------------------------------------------------
# Each adapter name describes exactly how to build the observation vector
# for a skill at inference time, matching what the training env produced.
#
# Base sizes:
#   _BASE_OBS_FULL    = 39  (lin_vel included: "clean" or "noisy" mode)
#   _BASE_OBS_NOVELIN = 36  (lin_vel disabled: "disabled" mode)
#
# Adapters:
#   "exact"                 → obs_size == 39; no extras
#   "height_cmd"            → obs_size == 40; append height_cmd (1 dim)
#   "velocity_cmd"          → obs_size == 41, needs_command; append [vel_x, ang_vel_z]
#   "height_cmd_yaw"        → obs_size == 42, !needs_command; append height_cmd + current_height + yaw_error
#                             (BalanceEnv with clean/noisy lin_vel + current_height + yaw obs added)
#   "novelin_height_cmd_yaw"→ obs_size == 39; 36-dim base (no lin_vel) + height_cmd + current_height + yaw_error
#                             (BalanceEnv with lin_vel_mode="disabled")
#   "unknown_pad"           → any other size; zero-pad with warning
#
_VALID_ADAPTERS = frozenset(
    {
        "exact",
        "height_cmd",
        "velocity_cmd",
        "height_cmd_yaw",
        "novelin_height_cmd_yaw",
        "unknown_pad",
    }
)

# gravity(3) + lin_vel(3) + ang_vel(3) + qpos(10) + qvel(10) + prev_action(10)
_BASE_OBS_FULL = 39
_BASE_OBS_NOVELIN = 36  # same layout without lin_vel(3)
# Legacy alias kept for code that references _BASE_OBS_SIZE by name
_BASE_OBS_SIZE = _BASE_OBS_FULL


def _infer_adapter(obs_size: int, needs_command: bool) -> str:
    """Choose an obs adapter from obs_size and needs_command flag."""
    if obs_size == _BASE_OBS_FULL:  # 39
        return "exact"
    if obs_size == _BASE_OBS_FULL + 1 and not needs_command:  # 40
        return "height_cmd"
    if obs_size == _BASE_OBS_FULL + 2 and needs_command:  # 41
        return "velocity_cmd"
    if obs_size == _BASE_OBS_FULL + 3 and not needs_command:  # 42
        # BalanceEnv trained with clean/noisy lin_vel + current_height + yaw_error
        return "height_cmd_yaw"
    if obs_size == _BASE_OBS_NOVELIN + 3 and not needs_command:  # 39
        # BalanceEnv trained with lin_vel_mode="disabled" + current_height + yaw_error
        return "novelin_height_cmd_yaw"
    return "unknown_pad"


@dataclass
class SkillPolicy:
    """Chứa thông tin policy cho một skill."""

    network: Any
    params: Any
    obs_rms: Any
    obs_size: int
    config: dict[str, Any]
    needs_command: bool = False  # Locomotion/Walking cần velocity command
    # Fix 2: explicit adapter chosen at load time
    obs_adapter: str = "exact"  # one of _VALID_ADAPTERS


@dataclass
class ControlCommand:
    """Lệnh điều khiển từ bên ngoài (bàn phím, joystick, planner)."""

    vel_x: float = 0.0  # m/s, tiến(+)/lùi(-)
    ang_vel_z: float = 0.0  # rad/s, xoay trái(+)/phải(-)
    height_target: float = 0.71  # m, chiều cao mong muốn (0.38–0.72)
    mode: Skill | None = None  # Ép chọn skill (None = tự động)


class UnifiedController:
    """Bộ điều khiển thống nhất nhiều skills."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        mj_model: mujoco.MjModel,
        *,
        stage_map: dict[str, str] | None = None,
        dwell_threshold: int = 3,
    ):
        """
        Args:
            checkpoint_dir: Root outputs directory.  Default layout (new convention):
                            ``outputs/<stage>/rl/seed<N>/checkpoints/final/``.
                            Pass ``stage_map`` to override the per-skill subfolder
                            mapping for non-default seeds or custom layouts.
            mj_model: MuJoCo model (để lấy actuator_ctrlrange, nq, nv, ...)
            stage_map: Custom mapping {skill_name: checkpoint_subfolder} relative
                       to checkpoint_dir.  Defaults to seed42 paths under the new
                       ``outputs/<stage>/rl/seed42/checkpoints/final`` convention.
            dwell_threshold: Number of consecutive calls _detect_skill_raw must
                             return the same skill before the controller acts on
                             the switch request (hysteresis, Fix 3).
        """
        self.mj_model = mj_model
        self.ckpt_dir = Path(checkpoint_dir)
        self.rng = jax.random.PRNGKey(42)

        # Default mapping: skill name → subfolder relative to checkpoint_dir.
        # Uses new output layout: outputs/<stage>/rl/seed42/checkpoints/final
        default_map = {
            "balance": "balance/rl/seed42/checkpoints/final",
            "locomotion": "wheeled_locomotion/rl/seed42/checkpoints/final",
            "walking": "walking/rl/seed42/checkpoints/final",
            "stair": "stair_climbing/rl/seed42/checkpoints/final",
            "terrain": "rough_terrain/rl/seed42/checkpoints/final",
            "stand_up": "stand_up/rl/seed42/checkpoints/final",
        }
        smap = stage_map or default_map

        self.skills: dict[Skill, SkillPolicy] = {}

        _skill_enum = {
            "balance": Skill.BALANCE,
            "locomotion": Skill.LOCOMOTION,
            "walking": Skill.WALKING,
            "stair": Skill.STAIR,
            "terrain": Skill.TERRAIN,
            "stand_up": Skill.STAND_UP,
        }

        for name, subfolder in smap.items():
            skill_enum = _skill_enum.get(name)
            if skill_enum is None:
                continue
            ckpt_path = self.ckpt_dir / subfolder / "checkpoint.pkl"
            if not ckpt_path.exists():
                # Cũng thử trực tiếp tên thư mục
                ckpt_path2 = self.ckpt_dir / subfolder.split("/")[0] / "final" / "checkpoint.pkl"
                if ckpt_path2.exists():
                    ckpt_path = ckpt_path2
                else:
                    print(f"  [skip] {name}: không tìm thấy {ckpt_path}")
                    continue

            try:
                sp = self._load_skill(ckpt_path, needs_command=(name in ("locomotion", "walking")))
                self.skills[skill_enum] = sp
                print(f"  [ok]   {name}: obs_size={sp.obs_size} adapter={sp.obs_adapter}")
            except Exception as e:
                print(f"  [fail] {name}: {e}")

        if not self.skills:
            raise RuntimeError("Không tải được skill nào! Kiểm tra checkpoint_dir.")

        # Skill hiện tại
        self._active_skill = self._pick_default_skill()

        # Fix 4: per-skill prev_action buffers
        self._prev_actions: dict[Skill, jnp.ndarray] = {
            sk: jnp.zeros(mj_model.nu) for sk in self.skills
        }

        self._prev_ctrl = np.zeros(mj_model.nu)
        self._transition_alpha = 0.0  # 0=old, 1=new (smooth blending)
        self._transition_target: Skill | None = None
        self._blend_steps = 10
        self._blend_counter = 0

        # Fix 3: dwell-time hysteresis state
        self._dwell_threshold = max(1, int(dwell_threshold))
        self._dwell_counts: dict[Skill, int] = {}  # raw-detection vote counter

        # Yaw tracking for balance skill: records the robot's heading when
        # balance mode first activates in a session.  Reset on every non-balance
        # → balance transition so the policy sees yaw_error=0 at entry.
        self._balance_initial_yaw: float | None = None

        print(f"\n  Active skills: {[s.name for s in self.skills]}")
        print(f"  Default skill: {self._active_skill.name}")
        print(f"  Dwell threshold: {self._dwell_threshold} steps")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_skill(self, ckpt_path: Path, needs_command: bool) -> SkillPolicy:
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)

        params = jax.device_put(ckpt["params"])
        obs_rms = jax.device_put(ckpt["obs_rms"])
        config = ckpt["config"]
        obs_size = int(obs_rms.mean.shape[0])

        # Fix 2: choose adapter at load time
        adapter = _infer_adapter(obs_size, needs_command)
        if adapter == "unknown_pad":
            warnings.warn(
                f"Checkpoint at {ckpt_path} has obs_size={obs_size} which does not match "
                f"any known adapter (base={_BASE_OBS_SIZE}, height_cmd={_BASE_OBS_SIZE + 1}, "
                f"velocity_cmd={_BASE_OBS_SIZE + 2}). Using 'unknown_pad' — obs semantics "
                f"may be wrong. Consider adding an explicit adapter.",
                stacklevel=3,
            )

        self.rng, rng_init = jax.random.split(self.rng)
        network, _ = create_actor_critic(
            obs_size=obs_size,
            action_size=self.mj_model.nu,
            config=config,
            rng=rng_init,
        )

        return SkillPolicy(
            network=network,
            params=params,
            obs_rms=obs_rms,
            obs_size=obs_size,
            config=config,
            needs_command=needs_command,
            obs_adapter=adapter,
        )

    # ------------------------------------------------------------------
    # Observation extraction (CPU side, giống visualize.py policy)
    # ------------------------------------------------------------------
    def _get_balance_yaw_error(self, mj_data: mujoco.MjData) -> float:
        """Return yaw_error (rad) for the balance skill, tracking initial heading.

        The first call (or after a non-balance → balance transition) records the
        current yaw as the reference heading.  Subsequent calls return the
        accumulated heading drift in [-π, π].
        """
        torso_quat = jnp.array(mj_data.qpos[3:7])
        current_yaw = float(quat_to_euler(torso_quat)[2])
        if self._balance_initial_yaw is None:
            self._balance_initial_yaw = current_yaw
            return 0.0
        return float(wrap_angle(jnp.array(current_yaw - self._balance_initial_yaw)))

    def _build_obs(self, mj_data: mujoco.MjData, skill: Skill, cmd: ControlCommand) -> jnp.ndarray:
        """Xây obs vector phù hợp với obs_size của skill.

        Fix 2: uses the explicit ``obs_adapter`` stored in SkillPolicy instead
        of the old silent generic pad/cut fallback.
        """
        sp = self.skills[skill]
        # Fix 4: use per-skill prev_action buffer
        prev_action = self._prev_actions[skill]

        torso_quat = jnp.array(mj_data.qpos[3:7])
        gravity_body = get_gravity_in_body_frame(torso_quat)

        quat_inv = quat_conjugate(torso_quat)
        world_lin_vel = jnp.array(mj_data.qvel[:3])
        world_ang_vel = jnp.array(mj_data.qvel[3:6])
        body_lin_vel = quat_rotate(quat_inv, world_lin_vel)
        body_ang_vel = quat_rotate(quat_inv, world_ang_vel)

        joint_pos = jnp.array(mj_data.qpos[7:17])  # 10
        joint_vel = jnp.array(mj_data.qvel[6:16])  # 10

        # Full 39-dim base (lin_vel included)
        base_obs_full = jnp.concatenate(
            [
                gravity_body,  # 3
                body_lin_vel,  # 3
                body_ang_vel,  # 3
                joint_pos,  # 10
                joint_vel,  # 10
                prev_action,  # 10
            ]
        )  # 39

        # 36-dim base (lin_vel excluded — matches "disabled" training mode)
        base_obs_novelin = jnp.concatenate(
            [
                gravity_body,  # 3
                body_ang_vel,  # 3
                joint_pos,  # 10
                joint_vel,  # 10
                prev_action,  # 10
            ]
        )  # 36

        adapter = sp.obs_adapter

        if adapter == "exact":
            if base_obs_full.shape[0] != sp.obs_size:
                raise ValueError(
                    f"Skill {skill.name}: adapter='exact' but computed obs has "
                    f"{base_obs_full.shape[0]} dims, expected {sp.obs_size}."
                )
            return base_obs_full

        if adapter == "height_cmd":
            # Normalization matches BalanceEnv: (h - 0.40) / (0.70 - 0.40).
            # Note: previous inference code used (h-0.38)/0.34 — that was a
            # pre-existing mismatch with training; fixed here.
            height_norm = float(np.clip((cmd.height_target - 0.40) / (0.70 - 0.40), 0.0, 1.0))
            obs = jnp.concatenate([base_obs_full, jnp.array([height_norm])])
            if obs.shape[0] != sp.obs_size:
                raise ValueError(
                    f"Skill {skill.name}: adapter='height_cmd' produced {obs.shape[0]} "
                    f"dims, expected {sp.obs_size}."
                )
            return obs

        if adapter == "velocity_cmd":
            command_vec = jnp.array([cmd.vel_x, cmd.ang_vel_z])
            obs = jnp.concatenate([base_obs_full, command_vec])
            if obs.shape[0] != sp.obs_size:
                raise ValueError(
                    f"Skill {skill.name}: adapter='velocity_cmd' produced {obs.shape[0]} "
                    f"dims, expected {sp.obs_size}."
                )
            return obs

        if adapter == "height_cmd_yaw":
            # BalanceEnv with clean/noisy lin_vel + current_height + yaw_error (42 dims).
            height_norm = float(np.clip((cmd.height_target - 0.40) / (0.70 - 0.40), 0.0, 1.0))
            current_h_norm = float(np.clip((float(mj_data.qpos[2]) - 0.40) / (0.70 - 0.40), 0.0, 1.0))
            yaw_error = self._get_balance_yaw_error(mj_data)
            obs = jnp.concatenate([base_obs_full, jnp.array([height_norm, current_h_norm, yaw_error])])
            if obs.shape[0] != sp.obs_size:
                raise ValueError(
                    f"Skill {skill.name}: adapter='height_cmd_yaw' produced {obs.shape[0]} "
                    f"dims, expected {sp.obs_size}."
                )
            return obs

        if adapter == "novelin_height_cmd_yaw":
            # BalanceEnv with lin_vel_mode="disabled" + current_height + yaw_error (39 dims).
            height_norm = float(np.clip((cmd.height_target - 0.40) / (0.70 - 0.40), 0.0, 1.0))
            current_h_norm = float(np.clip((float(mj_data.qpos[2]) - 0.40) / (0.70 - 0.40), 0.0, 1.0))
            yaw_error = self._get_balance_yaw_error(mj_data)
            obs = jnp.concatenate([base_obs_novelin, jnp.array([height_norm, current_h_norm, yaw_error])])
            if obs.shape[0] != sp.obs_size:
                raise ValueError(
                    f"Skill {skill.name}: adapter='novelin_height_cmd_yaw' produced "
                    f"{obs.shape[0]} dims, expected {sp.obs_size}."
                )
            return obs

        # unknown_pad — explicit escape hatch with warning already issued at load
        current_len = base_obs_full.shape[0]
        if current_len < sp.obs_size:
            obs = jnp.concatenate([base_obs_full, jnp.zeros(sp.obs_size - current_len)])
        elif current_len > sp.obs_size:
            obs = base_obs_full[: sp.obs_size]
        else:
            obs = base_obs_full
        return obs

    # ------------------------------------------------------------------
    # Skill selection logic
    # ------------------------------------------------------------------
    def _pick_default_skill(self) -> Skill:
        """Chọn skill mặc định (ưu tiên balance nếu có, rồi locomotion, ...)."""
        for s in [
            Skill.BALANCE,
            Skill.LOCOMOTION,
            Skill.WALKING,
            Skill.STAIR,
            Skill.TERRAIN,
        ]:
            if s in self.skills:
                return s
        return list(self.skills.keys())[0]

    def _detect_skill_raw(self, mj_data: mujoco.MjData, cmd: ControlCommand) -> Skill:
        """Raw skill detection — single-step heuristic (no hysteresis).

        Internal method; callers should use ``_detect_skill`` which adds
        dwell-time filtering (Fix 3).
        """
        # Nếu user ép mode
        if cmd.mode is not None and cmd.mode in self.skills:
            return cmd.mode

        torso_height = float(mj_data.qpos[2])
        torso_quat = jnp.array(mj_data.qpos[3:7])
        # Physically meaningful tilt: angle between body-frame z-axis and world up.
        #
        # Old formula: arccos(2*qw^2 - 1) = total rotation angle from identity.
        # This INCLUDES yaw: a robot facing sideways (90° yaw) triggers tilt≈1.57 rad,
        # exceeding the 0.5 rad threshold and falsely activating BALANCE mode.
        #
        # New formula: project gravity into body frame; the z-component tells us how
        # much the body aligns with world-up.  Pure yaw leaves gz = -1 (tilt=0).
        # Roll/pitch of θ degrees shifts gz toward 0, giving tilt = arccos(-gz) = θ.
        # get_gravity_in_body_frame is already imported and used in _build_obs.
        g_body = np.array(get_gravity_in_body_frame(torso_quat))  # [gx, gy, gz]
        tilt = float(np.arccos(np.clip(-g_body[2], -1.0, 1.0)))

        v_cmd = abs(cmd.vel_x) + abs(cmd.ang_vel_z)

        # 1. Robot đang ngã hoặc nghiêng nhiều → Balance
        if tilt > 0.5 or torso_height < 0.4:
            if Skill.BALANCE in self.skills:
                return Skill.BALANCE

        # 2. Phát hiện bề mặt không bằng phẳng qua chênh lệch chiều cao chân
        left_foot_z = self._foot_height(mj_data, "l_wheel_link")
        right_foot_z = self._foot_height(mj_data, "r_wheel_link")
        foot_diff = (
            abs(left_foot_z - right_foot_z)
            if left_foot_z is not None and right_foot_z is not None
            else 0
        )

        # 3. Nếu chênh lệch chân > ngưỡng → có thể đang leo bậc
        if foot_diff > 0.03:
            if Skill.STAIR in self.skills:
                return Skill.STAIR
            if Skill.TERRAIN in self.skills:
                return Skill.TERRAIN

        # 4. Đang di chuyển nhanh → locomotion hoặc walking
        if v_cmd > 0.3:
            if v_cmd > 0.8 and Skill.LOCOMOTION in self.skills:
                return Skill.LOCOMOTION
            if Skill.WALKING in self.skills:
                return Skill.WALKING
            if Skill.LOCOMOTION in self.skills:
                return Skill.LOCOMOTION

        # 5. Đứng yên → balance
        if Skill.BALANCE in self.skills:
            return Skill.BALANCE

        return self._active_skill  # Giữ nguyên

    def _detect_skill(self, mj_data: mujoco.MjData, cmd: ControlCommand) -> Skill:
        """Hysteresis-filtered skill detection (Fix 3).

        A raw skill detection must be returned for ``_dwell_threshold``
        consecutive calls before the controller acts on it.  This prevents
        single-frame sensor spikes (e.g. a brief foot height difference)
        from triggering unwanted transitions.

        Forced-mode commands (``cmd.mode is not None``) bypass hysteresis.
        """
        # Forced mode bypasses dwell filter — user intent is immediate
        if cmd.mode is not None and cmd.mode in self.skills:
            self._dwell_counts.clear()
            return cmd.mode

        raw = self._detect_skill_raw(mj_data, cmd)

        # Decay counts for any skill that is no longer the raw winner
        stale = [sk for sk in list(self._dwell_counts) if sk != raw]
        for sk in stale:
            del self._dwell_counts[sk]

        self._dwell_counts[raw] = self._dwell_counts.get(raw, 0) + 1

        if self._dwell_counts[raw] >= self._dwell_threshold:
            return raw

        # Not yet stable enough — keep current skill
        return self._active_skill

    def _foot_height(self, mj_data: mujoco.MjData, body_name: str) -> float | None:
        """Trả về chiều cao (z) của body wheel."""
        try:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            return float(mj_data.xpos[body_id, 2])
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Main action computation
    # ------------------------------------------------------------------
    def get_action(
        self,
        mj_data: mujoco.MjData,
        cmd: ControlCommand | None = None,
    ) -> np.ndarray:
        """Tính control signal cho robot.

        Args:
            mj_data: MuJoCo data hiện tại.
            cmd: Lệnh điều khiển (mặc định = đứng yên).

        Returns:
            ctrl array (nu,) đã clip theo actuator_ctrlrange.
        """
        if cmd is None:
            cmd = ControlCommand()

        # Phát hiện skill phù hợp (with dwell-time filter)
        desired_skill = self._detect_skill(mj_data, cmd)

        # Fix 1: only reset blend counter when the *target* changes
        if desired_skill != self._active_skill:
            if desired_skill != self._transition_target:
                # Truly new target → start a fresh blend
                self._transition_target = desired_skill
                self._blend_counter = 0
                self._transition_alpha = 0.0
                # Reset yaw reference when transitioning INTO balance so the
                # policy sees yaw_error=0 at entry rather than accumulated drift
                # from a prior non-balance episode.
                if desired_skill == Skill.BALANCE:
                    self._balance_initial_yaw = None
            # else: same target already in progress → keep incrementing
        else:
            # desired == active → cancel any in-progress transition
            if self._transition_target is not None:
                self._transition_target = None
                self._blend_counter = 0
                self._transition_alpha = 0.0

        # Tính action từ skill hiện tại
        current_ctrl = self._compute_skill_ctrl(mj_data, self._active_skill, cmd)

        # Blending nếu đang chuyển skill
        if self._transition_target is not None:
            target_ctrl = self._compute_skill_ctrl(mj_data, self._transition_target, cmd)
            self._blend_counter += 1
            self._transition_alpha = min(1.0, self._blend_counter / self._blend_steps)
            blended = (
                1 - self._transition_alpha
            ) * current_ctrl + self._transition_alpha * target_ctrl

            if self._transition_alpha >= 1.0:
                self._active_skill = self._transition_target
                self._transition_target = None

            ctrl = blended
        else:
            ctrl = current_ctrl

        # Clip
        ctrl = np.clip(
            ctrl,
            self.mj_model.actuator_ctrlrange[:, 0],
            self.mj_model.actuator_ctrlrange[:, 1],
        )

        self._prev_ctrl = ctrl
        return ctrl

    def _compute_skill_ctrl(
        self,
        mj_data: mujoco.MjData,
        skill: Skill,
        cmd: ControlCommand,
    ) -> np.ndarray:
        """Chạy policy network và trả về ctrl array."""
        sp = self.skills[skill]
        obs = self._build_obs(mj_data, skill, cmd)
        obs_norm = normalize_obs(obs, sp.obs_rms)
        dist, _ = sp.network.apply(sp.params, obs_norm)
        action = jnp.clip(dist.loc, -1.0, 1.0)

        # Fix 4: update only this skill's prev_action buffer
        self._prev_actions[skill] = action

        # Scale action → ctrl range
        ctrl_range = self.mj_model.actuator_ctrlrange
        ctrl = ctrl_range[:, 0] + (action + 1) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        return np.array(ctrl)

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------
    @property
    def active_skill(self) -> Skill:
        return self._active_skill

    @property
    def available_skills(self) -> list[Skill]:
        return list(self.skills.keys())

    def force_skill(self, skill: Skill) -> None:
        """Ép chuyển sang skill (bỏ qua auto-detect và hysteresis)."""
        if skill in self.skills:
            self._active_skill = skill
            self._transition_target = None
            self._dwell_counts.clear()
