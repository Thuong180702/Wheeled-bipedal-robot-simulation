#!/usr/bin/env python3
"""Measure idle CoM RMS for L2.5 (L2 + proximity gate only).

Applies the L2.5 config patches and param overrides from replicate_ablation_n10.py,
settles for 5s, then measures CoM XY RMS over 10s of quiet standing.

Usage:
  .venv/bin/python scripts/measure_l25_idle.py
"""
import json, os, sys, shutil, time
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
CTRL_FILE = ROOT / "wheeled_biped/controllers/k2_jax_controller.py"
BACKUP = str(CTRL_FILE) + ".l25_backup"

# ── L2.5 config (matches replicate_ablation_n10.py) ──
PATCHES = [
    # Disable envelope gate (anchor_quiet = always 1.0)
    ('_anchor_quiet = 1.0 - _jax_smoothstep01((_act_ema - 0.25) / (0.50 - 0.25))',
     '_anchor_quiet = 1.0  # L2.5: no envelope gate (prox gate only)'),
]
PARAM_OVERRIDES = {32: 1.0, 54: 6.0, 91: 0.0, 94: 50.0}

MODEL_PATH = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
DT = 0.01; SUBSTEPS = 5
SETTLE_S = 5.0; MEASURE_S = 10.0
N_REPS = 10

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])


def apply_patches():
    shutil.copy2(str(CTRL_FILE), BACKUP)
    with open(CTRL_FILE) as f:
        content = f.read()
    for old, new in PATCHES:
        if old not in content:
            raise ValueError(f"Patch not found: {old[:80]}...")
        content = content.replace(old, new)
    with open(CTRL_FILE, "w") as f:
        f.write(content)
    # Clear pyc cache
    pkg = ROOT / "wheeled_biped"
    for root, dirs, files in os.walk(str(pkg)):
        if "__pycache__" in dirs:
            for f in os.listdir(os.path.join(root, "__pycache__")):
                if any(k in f for k in ["k2_jax_controller", "offline_three_arm",
                                         "sagittal_velocity", "centroidal"]):
                    os.remove(os.path.join(root, "__pycache__", f))


def restore():
    if os.path.exists(BACKUP):
        shutil.copy2(BACKUP, str(CTRL_FILE))
        os.remove(BACKUP)


def measure_one_rep(seed):
    """Run one rep: init, settle, measure idle RMS. Returns CoM XY RMS in mm."""
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        compute_v3_torque_for_state, init_v3_controller)
    from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator, CentroidalStateEstimatorConfig)

    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
    est = CentroidalStateEstimator(cfg, mj_model=model)

    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    ctx = {"centroidal_estimator": est, "initial_yaw_z": 0.0,
           "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
           "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}

    # Init with perturbations
    perturbed = POSTURE + rng.normal(0.0, 0.005, size=10)
    JOINT_NAMES = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
                   "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]
    for j, jname in enumerate(JOINT_NAMES):
        jid = model.joint(jname).id
        lo_j, hi_j = model.jnt_range[jid]
        perturbed[j] = float(np.clip(perturbed[j], lo_j, hi_j))
    data.qpos[7:17] = perturbed
    data.qpos[2] = ROOT_Z + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)

    v3 = dict(init_v3_controller(profile_name="K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR", model=model))
    v3["jax_state"] = pack_state_k2()
    params = v3["jax_params"]
    for idx, val in PARAM_OVERRIDES.items():
        params = params.at[int(idx)].set(float(val))
    v3["jax_params"] = params

    # Settle
    for _ in range(int(SETTLE_S / DT)):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    # Measure idle RMS
    com_x = []; com_y = []
    for _ in range(int(MEASURE_S / DT)):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        com = data.subtree_com[0].copy()
        com_x.append(float(com[0]))
        com_y.append(float(com[1]))

    com_x = np.array(com_x); com_y = np.array(com_y)
    com_xy_rms = float(np.sqrt(np.mean(com_x**2 + com_y**2)) * 1000)  # mm
    return com_xy_rms


def main():
    apply_patches()
    try:
        rms_vals = []
        for rep in range(N_REPS):
            seed = 20260730 * 100 + rep
            rms = measure_one_rep(seed)
            rms_vals.append(rms)
            print(f"  Rep {rep+1}/{N_REPS}: CoM XY RMS = {rms:.4f} mm")
        rms_mean = float(np.mean(rms_vals))
        rms_std = float(np.std(rms_vals, ddof=1))
        print(f"\n  L2.5 Idle: {rms_mean:.4f} ± {rms_std:.4f} mm (N={N_REPS})")
        out = {"config": "L2.5", "idle_rms_mm_mean": rms_mean,
               "idle_rms_mm_std": rms_std, "n_reps": N_REPS,
               "values": rms_vals}
        out_path = ROOT / "outputs" / "paper_statistics" / "l25_idle.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved → {out_path}")
    finally:
        restore()


if __name__ == "__main__":
    main()
