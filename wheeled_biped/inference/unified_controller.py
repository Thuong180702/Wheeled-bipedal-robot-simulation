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
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

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


@dataclass
class SkillPolicy:
    """Chứa thông tin policy cho một skill."""

    network: Any
    params: Any
    obs_rms: Any
    obs_size: int
    config: Dict[str, Any]
    needs_command: bool = False  # Locomotion/Walking cần velocity command


@dataclass
class ControlCommand:
    """Lệnh điều khiển từ bên ngoài (bàn phím, joystick, planner)."""

    vel_x: float = 0.0  # m/s, tiến(+)/lùi(-)
    ang_vel_z: float = 0.0  # rad/s, xoay trái(+)/phải(-)
    height_target: float = 0.71  # m, chiều cao mong muốn (0.38–0.72)
    mode: Optional[Skill] = None  # Ép chọn skill (None = tự động)


class UnifiedController:
    """Bộ điều khiển thống nhất nhiều skills."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        mj_model: mujoco.MjModel,
        *,
        stage_map: Dict[str, str] | None = None,
    ):
        """
        Args:
            checkpoint_dir: Thư mục chứa các sub-folder checkpoint
                            (balance/, wheeled_locomotion/, walking/, ...)
            mj_model: MuJoCo model (để lấy actuator_ctrlrange, nq, nv, ...)
            stage_map: Mapping tùy chỉnh  {skill_name: checkpoint_subfolder}
        """
        self.mj_model = mj_model
        self.ckpt_dir = Path(checkpoint_dir)
        self.rng = jax.random.PRNGKey(42)

        # Mapping mặc định: tên stage → thư mục con
        default_map = {
            "balance": "balance/final",
            "locomotion": "wheeled_locomotion/final",
            "walking": "walking/final",
            "stair": "stair_climbing/final",
            "terrain": "rough_terrain/final",
            "stand_up": "stand_up/final",
        }
        smap = stage_map or default_map

        self.skills: Dict[Skill, SkillPolicy] = {}

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
                ckpt_path2 = (
                    self.ckpt_dir / subfolder.split("/")[0] / "final" / "checkpoint.pkl"
                )
                if ckpt_path2.exists():
                    ckpt_path = ckpt_path2
                else:
                    print(f"  [skip] {name}: không tìm thấy {ckpt_path}")
                    continue

            try:
                sp = self._load_skill(
                    ckpt_path, needs_command=(name in ("locomotion", "walking"))
                )
                self.skills[skill_enum] = sp
                print(f"  [ok]   {name}: obs_size={sp.obs_size}")
            except Exception as e:
                print(f"  [fail] {name}: {e}")

        if not self.skills:
            raise RuntimeError("Không tải được skill nào! Kiểm tra checkpoint_dir.")

        # Skill hiện tại
        self._active_skill = self._pick_default_skill()
        self._prev_ctrl = np.zeros(mj_model.nu)
        self._prev_action = jnp.zeros(mj_model.nu)  # normalized [-1,1]
        self._transition_alpha = 0.0  # 0=old, 1=new (smooth blending)
        self._transition_target: Optional[Skill] = None
        self._blend_steps = 10
        self._blend_counter = 0

        print(f"\n  Active skills: {[s.name for s in self.skills]}")
        print(f"  Default skill: {self._active_skill.name}")

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
        )

    # ------------------------------------------------------------------
    # Observation extraction (CPU side, giống visualize.py policy)
    # ------------------------------------------------------------------
    def _build_obs(
        self, mj_data: mujoco.MjData, skill: Skill, cmd: ControlCommand
    ) -> jnp.ndarray:
        """Xây obs vector phù hợp với obs_size của skill."""
        torso_quat = jnp.array(mj_data.qpos[3:7])
        gravity_body = get_gravity_in_body_frame(torso_quat)

        # Body-frame velocity (giống training - dùng quat_conjugate + quat_rotate)
        quat_inv = quat_conjugate(torso_quat)
        world_lin_vel = jnp.array(mj_data.qvel[:3])
        world_ang_vel = jnp.array(mj_data.qvel[3:6])
        body_lin_vel = quat_rotate(quat_inv, world_lin_vel)
        body_ang_vel = quat_rotate(quat_inv, world_ang_vel)

        base_obs = jnp.concatenate(
            [
                gravity_body,  # 3
                body_lin_vel,  # 3  body-frame linear vel
                body_ang_vel,  # 3  body-frame angular vel
                jnp.array(mj_data.qpos[7:17]),  # 10 joint pos
                jnp.array(mj_data.qvel[6:16]),  # 10 joint vel
                self._prev_action,  # 10 prev action (normalized [-1,1])
            ]
        )  # total 39

        sp = self.skills[skill]
        if sp.obs_size == 39:
            return base_obs

        # Balance skill: obs_size=40 → thêm height_command
        if skill == Skill.BALANCE and sp.obs_size == 40:
            # height_cmd normalized [0,1]: (target - 0.38) / (0.72 - 0.38)
            # cmd.height_target mặc định ~0.71m → norm ≈ 0.97
            height_norm = np.clip((cmd.height_target - 0.38) / (0.72 - 0.38), 0.0, 1.0)
            return jnp.concatenate([base_obs, jnp.array([height_norm])])

        # Thêm command nếu cần (locomotion / walking: +2)
        if sp.needs_command:
            command_vec = jnp.array([cmd.vel_x, cmd.ang_vel_z])
            obs = jnp.concatenate([base_obs, command_vec])
        else:
            obs = base_obs

        # Pad hoặc cắt nếu chênh size (vd walking thêm gait phase)
        current_len = obs.shape[0]
        if current_len < sp.obs_size:
            obs = jnp.concatenate([obs, jnp.zeros(sp.obs_size - current_len)])
        elif current_len > sp.obs_size:
            obs = obs[: sp.obs_size]

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

    def _detect_skill(self, mj_data: mujoco.MjData, cmd: ControlCommand) -> Skill:
        """Tự động phát hiện skill phù hợp dựa trên trạng thái hiện tại."""

        # Nếu user ép mode
        if cmd.mode is not None and cmd.mode in self.skills:
            return cmd.mode

        torso_height = float(mj_data.qpos[2])
        torso_quat = mj_data.qpos[3:7]
        # Tilt angle: arccos(2*qw^2 - 1) — xấp xỉ
        tilt = float(np.arccos(np.clip(2 * torso_quat[0] ** 2 - 1, -1, 1)))

        # Vận tốc mong muốn
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
            # Nếu vận tốc cao → locomotion (wheel-based)
            if v_cmd > 0.8 and Skill.LOCOMOTION in self.skills:
                return Skill.LOCOMOTION
            # Vận tốc trung bình → walking
            if Skill.WALKING in self.skills:
                return Skill.WALKING
            if Skill.LOCOMOTION in self.skills:
                return Skill.LOCOMOTION

        # 5. Đứng yên → balance
        if Skill.BALANCE in self.skills:
            return Skill.BALANCE

        return self._active_skill  # Giữ nguyên

    def _foot_height(self, mj_data: mujoco.MjData, body_name: str) -> Optional[float]:
        """Trả về chiều cao (z) của body wheel."""
        try:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
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

        # Phát hiện skill phù hợp
        desired_skill = self._detect_skill(mj_data, cmd)

        # Smooth transition nếu đổi skill
        if desired_skill != self._active_skill:
            self._transition_target = desired_skill
            self._blend_counter = 0
            self._transition_alpha = 0.0

        # Tính action từ skill hiện tại
        current_ctrl = self._compute_skill_ctrl(mj_data, self._active_skill, cmd)

        # Blending nếu đang chuyển skill
        if self._transition_target is not None:
            target_ctrl = self._compute_skill_ctrl(
                mj_data, self._transition_target, cmd
            )
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

        # Lưu prev_action (normalized [-1,1]) cho obs step tiếp theo
        self._prev_action = action

        # Scale action → ctrl range
        ctrl_range = self.mj_model.actuator_ctrlrange
        ctrl = ctrl_range[:, 0] + (action + 1) * 0.5 * (
            ctrl_range[:, 1] - ctrl_range[:, 0]
        )
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
        """Ép chuyển sang skill (bỏ qua auto-detect)."""
        if skill in self.skills:
            self._active_skill = skill
            self._transition_target = None
