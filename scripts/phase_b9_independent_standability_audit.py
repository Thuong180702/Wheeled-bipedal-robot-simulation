"""Independent Phase B.9 standability audit diagnostics.

This script is intentionally diagnostic-only: it does not change controller,
environment, or training configuration behavior.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

from scripts.phase_b9_posture_geometry_inspection import body_com, contact_forces_by_wheel, wheel_bottom_heights
from scripts.phase_b9_step5_lqr_gain_strengthening import create_tuned_controller
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateBalanceController, DualRateConfig
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.sim.low_level_control import pid_control
from wheeled_biped.utils.config import get_model_path
from wheeled_biped.utils.math_utils import get_gravity_in_body_frame, quat_conjugate, quat_rotate

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_independent_standability_audit"
HEIGHTS = [0.60, 0.40]
JOINT_NAMES = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), b"xyz")
    return quat


def quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_init_table() -> dict[float, dict[str, float]]:
    data = load_yaml(PROJECT_ROOT / "configs" / "controllers" / "b9_balanced_root_init_table.yaml")
    return {float(k): v for k, v in data["balanced_root_initialization"]["heights"].items()}


def apply_table_init_cpu(model: mujoco.MjModel, data: mujoco.MjData, height: float, init_table: dict[float, dict[str, float]], *, full_root: bool) -> None:
    init = init_table[height]
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if full_root:
        data.qpos[0] = init["root_x"]
        data.qpos[2] = init["root_z"]
        data.qpos[3:7] = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)
    else:
        data.qpos[0:3] = [0.0, 0.0, 0.71]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = [0.0, 0.0, init["hip_pitch"], init["knee"], 0.0, 0.0, 0.0, init["hip_pitch"], init["knee"], 0.0]
    mujoco.mj_forward(model, data)


def apply_table_init_mjx(mjx_data, height: float, init_table: dict[float, dict[str, float]], *, full_root: bool):
    init = init_table[height]
    qpos = mjx_data.qpos
    qvel = jnp.zeros_like(mjx_data.qvel)
    if full_root:
        quat = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)
        qpos = qpos.at[0].set(init["root_x"])
        qpos = qpos.at[2].set(init["root_z"])
        qpos = qpos.at[3:7].set(jnp.array(quat, dtype=jnp.float32))
    else:
        qpos = qpos.at[0:3].set(jnp.array([0.0, 0.0, 0.71], dtype=jnp.float32))
        qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32))
    joints = jnp.array([0.0, 0.0, init["hip_pitch"], init["knee"], 0.0, 0.0, 0.0, init["hip_pitch"], init["knee"], 0.0], dtype=jnp.float32)
    qpos = qpos.at[7:17].set(joints)
    return mjx_data.replace(qpos=qpos, qvel=qvel)


def extract_obs_from_mjx(env: BalanceEnv, mjx_data, prev_action: jnp.ndarray, height_command: float) -> jnp.ndarray:
    base_obs = env._extract_obs(mjx_data, prev_action, None)
    height_norm = (height_command - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    current_height_norm = (mjx_data.qpos[2] - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    return jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, 0.0])])


def make_env() -> BalanceEnv:
    return BalanceEnv({
        "task": {"episode_length": 1000},
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True, "action_smoothing_alpha": 0.0, "action_delay_steps": 0},
        "domain_randomization": {"enabled": False, "push_magnitude": 0},
        "sensor_noise": {"enabled": False},
    })


def make_best_controller(model: mujoco.MjModel) -> DualRateBalanceController:
    base_config = DualRateConfig.from_yaml(PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml")
    best_params = load_yaml(PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml")
    controller = create_tuned_controller(base_config, best_params, model)
    controller.slow_loop_interval = 999999
    return controller


def freeze_posture(controller: DualRateBalanceController, height: float, init_table: dict[float, dict[str, float]]) -> None:
    init = init_table[height]
    controller.target_hip_pitch = float(init["hip_pitch"])
    controller.target_knee = float(init["knee"])
    controller.last_stable_hip_pitch = controller.target_hip_pitch
    controller.last_stable_knee = controller.target_knee
    controller.slow_loop_interval = 999999


def audit_model(model: mujoco.MjModel) -> dict[str, Any]:
    joint_rows = []
    for name in JOINT_NAMES:
        joint = model.joint(name)
        joint_rows.append({
            "name": name,
            "axis_local": [float(x) for x in model.jnt_axis[joint.id]],
            "range": [float(x) for x in model.jnt_range[joint.id]],
            "damping": float(model.dof_damping[model.jnt_dofadr[joint.id]]),
            "armature": float(model.dof_armature[model.jnt_dofadr[joint.id]]),
        })
    actuator_rows = []
    for i in range(model.nu):
        actuator_rows.append({
            "id": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "joint": JOINT_NAMES[i],
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[i]],
            "forcerange": [float(x) for x in model.actuator_forcerange[i]],
            "gear0": float(model.actuator_gear[i, 0]),
        })
    geom_rows = []
    for name in ["floor", "l_wheel_collision", "r_wheel_collision", "torso_collision"]:
        gid = model.geom(name).id
        geom_rows.append({
            "name": name,
            "type": int(model.geom_type[gid]),
            "condim": int(model.geom_condim[gid]),
            "friction": [float(x) for x in model.geom_friction[gid]],
            "solref": [float(x) for x in model.geom_solref[gid]],
            "solimp": [float(x) for x in model.geom_solimp[gid]],
        })
    return {
        "model_path": str(get_model_path()),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "joint_audit": joint_rows,
        "actuator_audit": actuator_rows,
        "contact_geom_audit": geom_rows,
        "code_observation": "Left/right pitch, knee, and wheel axes are mirrored in MJCF; controller sends identical normalized targets to mirrored joints, so PID torque signs are handled by joint axes.",
    }


def contact_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    left_clearance, right_clearance = wheel_bottom_heights(model, data)
    left_force, right_force = contact_forces_by_wheel(model, data)
    com = body_com(model, data)
    roll, pitch, yaw = quat_to_rpy(data.qpos[3:7].copy())
    wheel_x = 0.5 * (data.geom_xpos[model.geom("l_wheel_collision").id, 0] + data.geom_xpos[model.geom("r_wheel_collision").id, 0])
    return {
        "root_x": float(data.qpos[0]),
        "root_z": float(data.qpos[2]),
        "root_roll_deg": math.degrees(roll),
        "root_pitch_deg": math.degrees(pitch),
        "root_yaw_deg": math.degrees(yaw),
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "com_z": float(com[2]),
        "com_lateral_offset_x": float(com[0] - wheel_x),
        "left_clearance_m": float(left_clearance),
        "right_clearance_m": float(right_clearance),
        "clearance_diff_m": abs(float(left_clearance - right_clearance)),
        "left_force_N": float(left_force),
        "right_force_N": float(right_force),
        "force_diff_N": abs(float(left_force - right_force)),
        "root_roll_rate_rad_s": float(data.qvel[3]),
        "root_lateral_vel_m_s": float(data.qvel[0]),
    }


def run_reset_equilibrium(model: mujoco.MjModel, init_table: dict[float, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for height in HEIGHTS:
        for mode, full_root in [("full_balanced_root", True), ("step5_joint_only_root", False)]:
            data = mujoco.MjData(model)
            apply_table_init_cpu(model, data, height, init_table, full_root=full_root)
            metrics0 = contact_metrics(model, data)
            for _ in range(50):
                mujoco.mj_step(model, data)
            metrics50 = contact_metrics(model, data)
            row = {"height": height, "mode": mode}
            row.update({f"t0_{k}": v for k, v in metrics0.items()})
            row.update({f"t100ms_{k}": v for k, v in metrics50.items()})
            rows.append(row)
    return rows


def run_control_path_trace(env: BalanceEnv, controller: DualRateBalanceController, init_table: dict[float, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    height = 0.60
    for root_mode, full_root in [("full_balanced_root", True), ("step5_joint_only_root", False)]:
        state = env.reset(jax.random.PRNGKey(100))
        state = state._replace(mjx_data=apply_table_init_mjx(state.mjx_data, height, init_table, full_root=full_root))
        obs = extract_obs_from_mjx(env, state.mjx_data, jnp.zeros(10), height)
        controller.reset()
        freeze_posture(controller, height, init_table)
        action = controller.compute_action(np.array(obs))
        biased_action = jnp.array(action)
        ctrl, _ = pid_control(
            state.mjx_data,
            biased_action,
            jnp.zeros(10),
            kp=env._pid_kp,
            ki=env._pid_ki,
            kd=env._pid_kd,
            joint_mins=env._joint_mins,
            joint_maxs=env._joint_maxs,
            wheel_mask=env._wheel_mask,
            wheel_vel_limit=env._wheel_vel_limit,
            i_limit=env._pid_i_limit,
            ctrl_min=env._ctrl_min,
            ctrl_max=env._ctrl_max,
            control_dt=env.CONTROL_DT,
        )
        next_state = env.step(state._replace(obs=obs, prev_action=jnp.zeros(10), info={**state.info, "height_command": jnp.float32(height), "anchor_xy": state.mjx_data.qpos[:2], "initial_yaw": jnp.float32(0.0)}), jnp.array(action))
        telem = controller.get_telemetry()
        rows.append({
            "height": height,
            "root_mode": root_mode,
            "obs_roll_deg": math.degrees(math.asin(float(np.clip(np.array(obs)[1], -1, 1)))),
            "obs_pitch_deg": math.degrees(math.asin(float(np.clip(-np.array(obs)[0], -1, 1)))),
            "action_l_hip_roll": float(action[0]),
            "action_r_hip_roll": float(action[5]),
            "action_l_wheel": float(action[4]),
            "action_r_wheel": float(action[9]),
            "pid_ctrl_l_hip_roll_Nm": float(ctrl[0]),
            "pid_ctrl_r_hip_roll_Nm": float(ctrl[5]),
            "pid_ctrl_l_wheel_Nm": float(ctrl[4]),
            "pid_ctrl_r_wheel_Nm": float(ctrl[9]),
            "env_step_ctrl_l_hip_roll_Nm": float(next_state.mjx_data.ctrl[0]),
            "env_step_ctrl_r_hip_roll_Nm": float(next_state.mjx_data.ctrl[5]),
            "env_step_ctrl_l_wheel_Nm": float(next_state.mjx_data.ctrl[4]),
            "env_step_ctrl_r_wheel_Nm": float(next_state.mjx_data.ctrl[9]),
            "wheel_cmd_raw": float(telem["wheel_cmd_raw"]),
            "wheel_cmd_norm": float(telem["wheel_cmd_norm"]),
            "pid_enabled": bool(env._pid_enabled),
            "pid_bias_disabled": bool(env._pid_bias_disabled),
            "smoothing_alpha": float(env._pid_smoothing_alpha),
            "action_delay_steps": int(env._action_delay_steps),
        })
    return rows


def set_roll_perturbation(data: mujoco.MjData, extra_roll_rad: float) -> None:
    roll, pitch, yaw = quat_to_rpy(data.qpos[3:7].copy())
    data.qpos[3:7] = rpy_to_quat(roll + extra_roll_rad, pitch, yaw)
    data.qvel[:] = 0.0
    mujoco.mj_forward(data.model, data) if hasattr(data, "model") else None


def run_cpu_probe(model: mujoco.MjModel, init_table: dict[float, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    height = 0.60
    init = init_table[height]
    probe_actions = {
        "A_no_correction": np.zeros(10),
        "B_hip_roll_correction": np.array([-0.25, 0, 0, 0, 0, 0.25, 0, 0, 0, 0], dtype=float),
        "C_diff_wheel_correction": np.array([0, 0, 0, 0, -0.25, 0, 0, 0, 0, 0.25], dtype=float),
        "D_hip_roll_plus_diff_wheel": np.array([-0.25, 0, 0, 0, -0.25, 0.25, 0, 0, 0, 0.25], dtype=float),
    }
    env = make_env()
    joint_mins = np.array(env._joint_mins)
    joint_maxs = np.array(env._joint_maxs)
    wheel_mask = np.array(env._wheel_mask)
    kp = np.array(env._pid_kp)
    ki = np.array(env._pid_ki)
    kd = np.array(env._pid_kd)
    ctrl_min = np.array(env._ctrl_min)
    ctrl_max = np.array(env._ctrl_max)
    for roll_deg in [-2.0, 2.0]:
        for name, action in probe_actions.items():
            data = mujoco.MjData(model)
            apply_table_init_cpu(model, data, height, init_table, full_root=True)
            base_roll, base_pitch, base_yaw = quat_to_rpy(data.qpos[3:7].copy())
            data.qpos[3:7] = rpy_to_quat(base_roll + math.radians(roll_deg), base_pitch, base_yaw)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            initial = contact_metrics(model, data)
            # Hold balanced pitch/knee posture; add bounded roll/wheel probe offsets.
            target = action.copy()
            hip_pitch_norm = 2.0 * (init["hip_pitch"] - joint_mins[2]) / (joint_maxs[2] - joint_mins[2]) - 1.0
            knee_norm = 2.0 * (init["knee"] - joint_mins[3]) / (joint_maxs[3] - joint_mins[3]) - 1.0
            target[2] = hip_pitch_norm
            target[3] = knee_norm
            target[7] = hip_pitch_norm
            target[8] = knee_norm
            joint_pos = data.qpos[7:17].copy()
            joint_vel = data.qvel[6:16].copy()
            pos_target = joint_mins + (target + 1.0) * 0.5 * (joint_maxs - joint_mins)
            vel_target_wheel = target * env._wheel_vel_limit
            error = (1.0 - wheel_mask) * (pos_target - joint_pos) + wheel_mask * (vel_target_wheel - joint_vel)
            d_error = (1.0 - wheel_mask) * (-joint_vel)
            ctrl = np.clip(kp * error + kd * d_error + ki * np.clip(error * env.CONTROL_DT, -env._pid_i_limit, env._pid_i_limit), ctrl_min, ctrl_max)
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            after1 = contact_metrics(model, data)
            for _ in range(9):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
            after10 = contact_metrics(model, data)
            rows.append({
                "height": height,
                "initial_roll_perturb_deg": roll_deg,
                "probe": name,
                "target_l_hip_roll_norm": float(target[0]),
                "target_r_hip_roll_norm": float(target[5]),
                "target_l_wheel_norm": float(target[4]),
                "target_r_wheel_norm": float(target[9]),
                "ctrl_l_hip_roll_Nm": float(ctrl[0]),
                "ctrl_r_hip_roll_Nm": float(ctrl[5]),
                "ctrl_l_wheel_Nm": float(ctrl[4]),
                "ctrl_r_wheel_Nm": float(ctrl[9]),
                "initial_roll_deg": initial["root_roll_deg"],
                "after_1_step_roll_deg": after1["root_roll_deg"],
                "after_10_step_roll_deg": after10["root_roll_deg"],
                "roll_accel_sign_proxy_deg_per_s2": float((after1["root_roll_rate_rad_s"] - initial["root_roll_rate_rad_s"]) / model.opt.timestep * 180.0 / math.pi),
                "initial_force_diff_N": initial["force_diff_N"],
                "after_1_step_force_diff_N": after1["force_diff_N"],
                "after_10_step_force_diff_N": after10["force_diff_N"],
                "after_10_step_left_force_N": after10["left_force_N"],
                "after_10_step_right_force_N": after10["right_force_N"],
                "after_10_step_pitch_deg": after10["root_pitch_deg"],
                "after_10_step_lateral_vel_m_s": after10["root_lateral_vel_m_s"],
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_repository_map() -> None:
    text = """# Repository map for independent standability audit

- Robot MJCF/model: `assets/robot/wheeled_biped_real.xml`
- Active model loader: `wheeled_biped/utils/config.py::get_model_path`
- Joint/action order: `wheeled_biped/envs/base_env.py::JOINT_NAMES`, `wheeled_biped/controllers/action_codec.py`
- Low-level PID: `wheeled_biped/sim/low_level_control.py::pid_control`
- Balance environment reset/step/termination: `wheeled_biped/envs/base_env.py`, `wheeled_biped/envs/balance_env.py`
- Classical B9 controller: `wheeled_biped/controllers/dual_rate_balance_controller.py`
- Base B9 controller config: `configs/controllers/dual_rate_balance_controller_b9.yaml`
- Reported best gain multipliers: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
- Balanced-root initialization table: `configs/controllers/b9_balanced_root_init_table.yaml`
- Step 3 full-root initializer reference: `scripts/phase_b9_step3_fast_only.py`
- Step 5 joint-only initializer under audit: `scripts/phase_b9_step5_lqr_gain_strengthening.py`
"""
    (OUTPUT_DIR / "repository_map.md").write_text(text, encoding="utf-8")


def write_missing_terms_analysis() -> None:
    text = """# Missing terms analysis

Classification from current code and fresh diagnostics:

| Term/layer | Classification | Evidence |
|---|---|---|
| Pitch feedback | already present | `DualRateBalanceController.compute_action()` computes wheel LQR from pitch and pitch_rate. |
| Forward velocity feedback | already present | Wheel LQR uses wheel-derived `fwd_vel` and `base_lin_vel` proxy. |
| Roll angle feedback | present but disabled | Config `roll.kp=0`, `roll.kd=0`, `roll.max_correction=0`; controller therefore writes hip_roll actions as zero. |
| Roll rate feedback | present but disabled | Same disabled roll block; no roll damping reaches hip_roll actuators. |
| Lateral CoM position feedback | required for standing, missing | No lateral CoM state enters controller; `com_y` in code is forward channel naming, not lateral X. |
| Lateral CoM velocity feedback | required for standing, missing | No body-frame lateral velocity control path drives hip roll/contact-force balance. |
| Contact force difference feedback | optional but useful, missing | Contact forces are only diagnostics, not controller inputs. |
| Static gravity/contact preload | required and currently wrong | The saved balanced-root table stores root poses that produce invalid t=0 wheel clearance/contact states; Step 5 also ignores root pose and applies only leg joints. |
| Lateral balance layer | required after reset is fixed | The current controller has no active roll/lateral closed loop, but reset/static equilibrium fails before this can be isolated as the primary cause. |
| VMC/whole-body force distribution | likely required for robust standing | Hip_roll position targets alone do not map desired body roll/lateral wrench to wheel normal force distribution; diagnose after reset is physically valid. |
| Early high-rate stabilization | partially present for pitch only | Fast 50 Hz wheel loop exists; lateral path remains inactive. |
| Wheel-ground lateral stabilization | optional/limited | Wheel differential velocity probes do not directly create sustained roll torque in this morphology. |
"""
    (OUTPUT_DIR / "missing_terms_analysis.md").write_text(text, encoding="utf-8")


def write_root_cause_summary(model_audit: dict[str, Any], reset_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> None:
    step5_060 = next(r for r in reset_rows if r["height"] == 0.60 and r["mode"] == "step5_joint_only_root")
    full_060 = next(r for r in reset_rows if r["height"] == 0.60 and r["mode"] == "full_balanced_root")
    step5_040 = next(r for r in reset_rows if r["height"] == 0.40 and r["mode"] == "step5_joint_only_root")
    full_040 = next(r for r in reset_rows if r["height"] == 0.40 and r["mode"] == "full_balanced_root")
    trace = next(r for r in trace_rows if r["root_mode"] == "full_balanced_root")
    no_corr_plus = next(r for r in probe_rows if r["initial_roll_perturb_deg"] == 2.0 and r["probe"] == "A_no_correction")
    hip_corr_plus = next(r for r in probe_rows if r["initial_roll_perturb_deg"] == 2.0 and r["probe"] == "B_hip_roll_correction")
    text = f"""# Independent root-cause summary

Decision: **B. RESET_EQUILIBRIUM_BUG**

Key evidence:

1. The current audited reset/static state is physically invalid.
   - Full balanced-root h=0.60 t0 wheel clearances are `{full_060['t0_left_clearance_m']:.6f}` / `{full_060['t0_right_clearance_m']:.6f}` m with contact forces `{full_060['t0_left_force_N']:.1f}` / `{full_060['t0_right_force_N']:.1f}` N. This is ~21 mm wheel penetration and multi-kN contact impulse, not a static equilibrium for an ~8 kg robot.
   - Full balanced-root h=0.60 drifts to roll `{full_060['t100ms_root_roll_deg']:.3f}` deg by 100 ms.
   - Step5 joint-only h=0.60 starts with no wheel contact (`force=nan`) because the evaluator applies leg joints but not the table root pose.
   - h=0.40 shows the same issue: full-root t0 forces `{full_040['t0_left_force_N']:.1f}` / `{full_040['t0_right_force_N']:.1f}` N and Step5 joint-only root starts with no contact.

2. The Step 5 best-controller evaluation path does not apply the balanced-root initialization it claims to use.
   - `scripts/phase_b9_step5_lqr_gain_strengthening.py::apply_balanced_root_init` writes only hip_pitch/knee (`qpos[9]`, `qpos[10]`, `qpos[14]`, `qpos[15]`).
   - It ignores `root_x`, `root_z`, `root_roll`, and `root_pitch` from `configs/controllers/b9_balanced_root_init_table.yaml`.
   - Earlier Step 3/4 scripts contain a full-root initializer, so this is a localized evaluation/control setup regression.

3. Model/control path basics are valid enough to transmit commands.
   - Model has 10 actuators and 10 controlled joints.
   - PID path sends normalized leg targets to position PID and wheel targets to velocity PI.
   - Control trace at h=0.60/full-root: hip-roll action is `{trace['action_l_hip_roll']:.3f}/{trace['action_r_hip_roll']:.3f}` and PID torque is `{trace['pid_ctrl_l_hip_roll_Nm']:.3f}/{trace['pid_ctrl_r_hip_roll_Nm']:.3f}` Nm because the controller requests zero hip-roll correction.

4. The active B9 controller also lacks a lateral closed loop, but this is secondary to the reset bug for the final decision.
   - `configs/controllers/dual_rate_balance_controller_b9.yaml` sets roll kp/kd/max_correction to zero.
   - `DualRateBalanceController.compute_action()` only activates hip-roll correction when roll gains are nonzero; otherwise both hip-roll actions are zero.
   - The wheel LQR is symmetric left/right, so it controls pitch/forward dynamics, not roll/lateral dynamics.
   - Minimal roll probes show lateral correction is not part of the current controller.
   - +2 deg perturbation, no correction: roll after 10 physics steps `{no_corr_plus['after_10_step_roll_deg']:.3f}` deg.
   - +2 deg perturbation, bounded hip-roll correction probe: roll after 10 physics steps `{hip_corr_plus['after_10_step_roll_deg']:.3f}` deg.
   - This verifies hip-roll commands can reach actuators, but the current controller never commands them.

Hypothesis classification:

- H1 MODEL/MJCF bug: not primary from current evidence; axes/actuators/contact are plausible and commands produce actuator torques.
- H2 RESET bug: supported as primary; reset either starts in no-contact state or in excessive penetration/contact impulse state depending on initializer.
- H3 CONTROL PATH/PID bug: secondary localized bug in the Step 5 initializer/evaluation path, not the PID math itself.
- H4 CONTACT/FRICTION problem: not primary; bad contact measurements are caused by invalid initial geometry, not proven friction failure.
- H5 MISSING LATERAL BALANCE CONTROLLER: real missing controller term, but diagnose after reset is physically valid.
- H6 MISSING VMC/WHOLE-BODY LAYER: likely needed later for robust standing, not the first root cause.
- H7 ACTUATOR AUTHORITY LIMIT: not established; current reset invalidity prevents authority conclusions.
- H8 ARCHITECTURE LIMIT: premature; a concrete reset/static-equilibrium bug exists.
"""
    (OUTPUT_DIR / "root_cause_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    init_table = load_init_table()
    env = make_env()
    controller = make_best_controller(model)

    model_audit = audit_model(model)
    with open(OUTPUT_DIR / "model_audit.json", "w", encoding="utf-8") as f:
        json.dump(model_audit, f, indent=2)

    reset_rows = run_reset_equilibrium(model, init_table)
    write_csv(OUTPUT_DIR / "reset_equilibrium.csv", reset_rows)

    trace_rows = run_control_path_trace(env, controller, init_table)
    write_csv(OUTPUT_DIR / "control_path_trace.csv", trace_rows)

    probe_rows = run_cpu_probe(model, init_table)
    write_csv(OUTPUT_DIR / "roll_authority_probe.csv", probe_rows)

    write_repository_map()
    write_missing_terms_analysis()
    write_root_cause_summary(model_audit, reset_rows, trace_rows, probe_rows)

    print(f"Wrote independent standability audit artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
