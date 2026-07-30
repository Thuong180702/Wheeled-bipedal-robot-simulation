"""
Record envelope follower + gate activation states during push-recovery.
Produces time-series data for paper Figure: EMA, g_prox, g_env, g_boost, kp_eff.

Usage:
  JAX_ENABLE_X64=True mjpython scripts/record_gate_activation.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import mujoco
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
MODEL_PATH = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
DT = 0.01; SUBSTEPS = 5
TOTAL_S = 20.0; PUSH_START_S = 3.0
PUSH_DUR_STEPS = 7; PUSH_FORCE_N = 90.0

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([
    nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])
JOINT_NAMES = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
               "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]

_S_ANCHOR_ACT_EMA = 844
DEG = 180.0 / np.pi


def smoothstep01(x):
    """Hermite smoothstep: 3t^2 - 2t^3, t clipped to [0,1]."""
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_ctx(model):
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
    est = CentroidalStateEstimator(cfg, mj_model=model)
    return {"centroidal_estimator": est, "initial_yaw_z": 0.0,
            "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
            "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}


def compute_gate_states(sag_pos_err, ema_val, pitch_rad, pitch_rate_rad_s):
    """Recompute gate states from raw signals using JAX-equivalent formulas."""
    # Proximity gate: g_prox = 1 - smoothstep(|Δx|, 0.05, 0.15)
    g_prox = 1.0 - smoothstep01((abs(sag_pos_err) - 0.05) / (0.15 - 0.05))

    # Quiet-stance gate: g_env = 1 - smoothstep(EMA, 0.25, 0.50)
    g_env = 1.0 - smoothstep01((ema_val - 0.25) / (0.50 - 0.25))

    # Stability gate: g_stab = smoothstep((HI - |pitch|)/(HI-LO)) * smoothstep((HI - |rate|)/(HI-LO))
    # HI=0.21 rad (12°), LO=0.035 rad (2°) for pitch; HI=0.262, LO=0.035 for rate
    g_stab = (smoothstep01((0.21 - abs(pitch_rad)) / (0.21 - 0.035))
              * smoothstep01((0.262 - abs(pitch_rate_rad_s)) / (0.262 - 0.035)))

    # Boost proximity gate: wider band (0.08-0.18 m) than anchor prox
    g_prox_boost = 1.0 - smoothstep01((abs(sag_pos_err) - 0.08) / (0.18 - 0.08))

    # Effective boost gate
    g_boost = g_prox_boost * g_env * g_stab

    # k_p softness gate: max(EMA-based, displacement-based)
    g_kp_soft = max(
        smoothstep01((ema_val - 0.30) / (0.45 - 0.30)),
        smoothstep01((abs(sag_pos_err) - 0.10) / (0.25 - 0.10)))
    kp_eff = 50.0 - 15.0 * g_kp_soft

    return {
        "g_prox": float(g_prox),
        "g_env": float(g_env),
        "g_stab": float(g_stab),
        "g_boost": float(g_boost),
        "g_kp_soft": float(g_kp_soft),
        "kp_eff": float(kp_eff),
    }


def run():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Init controller
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    data.qpos[7:17] = POSTURE
    data.qpos[2] = ROOT_Z
    mujoco.mj_forward(model, data)
    ctx = build_ctx(model)

    total_steps = int(TOTAL_S / DT)
    push_start_step = int(PUSH_START_S / DT)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    records = []
    for step in range(total_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])

        # Apply push force during push window
        pushing = push_start_step <= step < push_start_step + PUSH_DUR_STEPS
        data.xfrc_applied[torso_id, :3] = 0.0
        if pushing:
            data.xfrc_applied[torso_id, 0] = PUSH_FORCE_N  # forward push (+x)

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        # Extract raw signals
        jax_state_flat = np.array(v3["jax_state"])
        ema_val = float(jax_state_flat[_S_ANCHOR_ACT_EMA])

        # Sagittal velocity (body-frame x velocity)
        quat = data.qpos[3:7].copy()
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        # Body-frame velocity
        cvel = data.cvel[1]  # torso body velocity [vx, vy, vz, wx, wy, wz]
        sag_vel = float(cvel[0])  # body-frame sagittal velocity

        # Sagittal position error (from torso x position vs home)
        # Home position is latched at t=0; drift from initial x
        sag_pos = float(data.qpos[0])  # world-frame torso x
        if step == 0:
            home_x = sag_pos
        sag_pos_err = sag_pos - home_x

        # Pitch and pitch rate
        pitch = float(np.arcsin(np.clip(-2.0*(quat[1]*quat[3] - quat[0]*quat[2]), -1.0, 1.0)))
        pitch_rate = float(cvel[4])  # body-frame pitch rate (wy)

        t = (step + 1) * DT
        rec = {
            "t": t,
            "sag_vel": float(sag_vel),
            "abs_sag_vel": float(abs(sag_vel)),
            "ema": ema_val,
            "sag_pos_err": float(sag_pos_err),
            "pitch_rad": pitch,
            "pitch_rate_rad_s": pitch_rate,
            "pushing": pushing,
            "push_force_N": PUSH_FORCE_N if pushing else 0.0,
        }
        rec.update(compute_gate_states(sag_pos_err, ema_val, pitch, pitch_rate))
        records.append(rec)

    out_path = OUT_DIR / "gate_activation_timeseries.json"
    json.dump({
        "metadata": {
            "profile": PROFILE,
            "push_force_N": PUSH_FORCE_N,
            "push_start_s": PUSH_START_S,
            "push_dur_steps": PUSH_DUR_STEPS,
            "total_s": TOTAL_S,
            "dt_s": DT,
            "control_hz": int(1/DT),
        },
        "records": records,
    }, out_path.open("w"), indent=2, default=str)
    print(f"Saved {len(records)} steps → {out_path}")


if __name__ == "__main__":
    run()
