#!/usr/bin/env python3
"""Evaluate ACC (V3_ANCHOR) at 50 Hz — disentangle architecture vs control rate.

Key changes from nominal 100 Hz operation:
  - control_dt: 0.01 → 0.02
  - fs_hz (notch): 100 → 50 (preserves 2.5 Hz physical centre)
  - SUBSTEPS: 5 → 10 (keeps 500 Hz physics)
  - All per-step rate limits and EMA alphas in the profile are UNCHANGED
    (naive dt change — tests whether architecture survives slower updates
    without per-parameter recomputation)

Outputs:
  - idle RMS (CoM std after settle)
  - push F_min (binary-search threshold)
  - ringdown time series (pitch/roll/yaw/x/y after 90 N push)
"""
import json, os, sys, time, argparse
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "acc_50hz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.signal_filters import biquad_notch_coefficients
from wheeled_biped.teleop_shaper import HeightPosture

PROFILE_NAME = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
CTRL_DT_50HZ = 0.02
FS_HZ_50HZ = 50.0
SUBSTEPS_50HZ = 10  # 500 Hz physics = 0.02 / 10
MODEL_PATH = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])

PITCH_LIMIT = 0.8; HEIGHT_LIMIT = 0.30
PUSH_DUR_S = 0.07; SETTLE_S = 3.0; POST_PUSH_S = 10.0


def get_pitch_roll_yaw(data):
    q = data.qpos[3:7]
    w, x, y, z = q[0], q[1], q[2], q[3]
    pitch = float(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    roll = float(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    yaw = float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return pitch, roll, yaw


def settle(model, data, v3, ctx, steps):
    """Settle robot for `steps` controller steps."""
    for _ in range(steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS_50HZ):
            mujoco.mj_step(model, data)


def run_single_push(model, data, v3, ctx, force_N, angle_deg):
    """Return True if robot survives force_N push at angle_deg."""
    settle_steps = int(SETTLE_S / CTRL_DT_50HZ)
    push_steps = int(PUSH_DUR_S / CTRL_DT_50HZ)
    post_steps = int(POST_PUSH_S / CTRL_DT_50HZ)
    total = push_steps + post_steps

    angle_rad = np.deg2rad(angle_deg)
    fx = force_N * np.cos(angle_rad)
    fy = force_N * np.sin(angle_rad)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    for step in range(total):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])

        data.xfrc_applied[torso_id, :3] = 0.0
        if step < push_steps:
            data.xfrc_applied[torso_id, 0] = fx
            data.xfrc_applied[torso_id, 1] = fy

        for _ in range(SUBSTEPS_50HZ):
            mujoco.mj_step(model, data)

        q = data.qpos[3:7]
        pitch = np.arcsin(-2*(q[1]*q[3] - q[0]*q[2]))
        if abs(pitch) > PITCH_LIMIT or data.subtree_com[0][2] < HEIGHT_LIMIT:
            return False
    return True


def bisect_fmin(model, data, v3, ctx, angle_deg, lo=10.0, hi=160.0, n_iter=8):
    """Binary search for minimum force that causes fall."""
    survived_lo = run_single_push(model, data, v3, ctx, lo, angle_deg)
    if not survived_lo:
        return lo  # even lo fails
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        model_copy = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data_copy = mujoco.MjData(model_copy)
        data_copy.qpos[7:17] = POSTURE.copy()
        data_copy.qpos[2] = ROOT_Z
        mujoco.mj_forward(model_copy, data_copy)
        v3_fresh = init_controller(model_copy)
        ctx_fresh = build_ctx(model_copy, data_copy, v3_fresh)
        settle(model_copy, data_copy, v3_fresh, ctx_fresh, int(SETTLE_S / CTRL_DT_50HZ))
        if run_single_push(model_copy, data_copy, v3_fresh, ctx_fresh, mid, angle_deg):
            lo = mid
        else:
            hi = mid
    return lo


def init_controller(model):
    """Init V3_ANCHOR with 50 Hz params override.

    Uses the EXACT V3_ANCHOR profile (all gains, gates, drift, homing, anchor)
    and patches only time-dependent parameters:
      - control_dt: 0.01 → 0.02
      - notch fs_hz: 100 → 50 (recomputes biquad coefficients for 2.5 Hz physical)
    All other params (rate limits, gains, torque caps, etc.) are unchanged.
    """
    v3 = dict(init_v3_controller(profile_name=PROFILE_NAME, model=model))

    # Patch control_dt in JAX params array
    old_params = v3["jax_params"]
    new_params = old_params.at[28].set(CTRL_DT_50HZ)  # _IDX_CONTROL_DT = 28

    # Patch notch filter: recompute coefficients for fs=50 Hz (keep fc=2.5 Hz, Q=2.0)
    b0, b1, b2, a1, a2 = biquad_notch_coefficients(FS_HZ_50HZ, 2.5, 2.0)
    new_params = new_params.at[0].set(float(b0))  # _IDX_NOTCH_B0
    new_params = new_params.at[1].set(float(b1))  # _IDX_NOTCH_B1
    new_params = new_params.at[2].set(float(b2))  # _IDX_NOTCH_B2
    new_params = new_params.at[3].set(float(a1))  # _IDX_NOTCH_A1
    new_params = new_params.at[4].set(float(a2))  # _IDX_NOTCH_A2
    new_params = new_params.at[5].set(FS_HZ_50HZ) # _IDX_NOTCH_FS
    # Anchor integral leak: maintain physical τ ≈ 2.0s
    # 100 Hz: leak=0.005 → (1-0.005)^100 = 0.606 per s
    # 50 Hz:  leak=0.010 → (1-0.010)^50  = 0.605 per s
    new_params = new_params.at[90].set(0.010)  # _IDX_ANCHOR_INTEGRAL_LEAK

    v3["jax_params"] = new_params
    v3["jax_state"] = pack_state_k2()
    v3["control_dt"] = CTRL_DT_50HZ
    return v3


def build_ctx(model, data, v3):
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator, CentroidalStateEstimatorConfig)
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
    est = CentroidalStateEstimator(cfg, mj_model=model)
    return {"centroidal_estimator": est, "initial_yaw_z": 0.0,
            "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
            "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}


def main():
    p = argparse.ArgumentParser(description="ACC @ 50 Hz evaluation")
    p.add_argument("--idle", action="store_true", help="Measure idle RMS precision")
    p.add_argument("--push", action="store_true", help="Measure push F_min (4 directions)")
    p.add_argument("--ringdown", action="store_true", help="Record post-push ringdown")
    p.add_argument("--all", action="store_true", help="Run all evaluations")
    p.add_argument("--push-n", type=float, default=90.0, help="Push force for ringdown")
    args = p.parse_args()

    if not (args.idle or args.push or args.ringdown or args.all):
        p.print_help()
        return

    run_idle = args.idle or args.all
    run_push = args.push or args.all
    run_ringdown = args.ringdown or args.all

    print(f"=== ACC @ 50 Hz Evaluation ({PROFILE_NAME}) ===")
    print(f"    control_dt = {CTRL_DT_50HZ}s, fs_hz = {FS_HZ_50HZ}, substeps = {SUBSTEPS_50HZ}")

    # ── Init ──
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    v3 = init_controller(model)
    ctx = build_ctx(model, data := mujoco.MjData(model), v3)
    data.qpos[7:17] = POSTURE.copy()
    data.qpos[2] = ROOT_Z
    mujoco.mj_forward(model, data)

    settle_steps = int(SETTLE_S / CTRL_DT_50HZ)
    settle(model, data, v3, ctx, settle_steps)

    home_x, home_y = float(data.qpos[0]), float(data.qpos[1])
    home_z = float(data.qpos[2])

    # ═══════════════════════════════════════════════════════════════
    # IDLE RMS
    # ═══════════════════════════════════════════════════════════════
    if run_idle:
        print("\n--- Idle RMS ---")
        idle_s = 10.0
        idle_steps = int(idle_s / CTRL_DT_50HZ)
        com_x = np.zeros(idle_steps)
        com_y = np.zeros(idle_steps)
        pitch_vals = np.zeros(idle_steps)

        for step in range(idle_steps):
            r = compute_v3_torque_for_state(
                data, model, v3["jax_step_fn"], v3["jax_state"],
                v3["jax_params"], ctx, teleop=None)
            v3["jax_state"] = r["next_jax_state"]
            data.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS_50HZ):
                mujoco.mj_step(model, data)
            com_x[step] = float(data.qpos[0] - home_x)
            com_y[step] = float(data.qpos[1] - home_y)
            pitch_vals[step], _, _ = get_pitch_roll_yaw(data)

        idle_result = {
            "dt_s": CTRL_DT_50HZ,
            "idle_duration_s": idle_s,
            "com_x_rms_mm": float(np.std(com_x) * 1000),
            "com_y_rms_mm": float(np.std(com_y) * 1000),
            "com_xy_rms_mm": float(np.sqrt(np.var(com_x) + np.var(com_y)) * 1000),
            "pitch_rms_deg": float(np.degrees(np.std(pitch_vals))),
            "com_z_mean_m": float(np.mean([float(data.subtree_com[0][2]) for _ in [0]] + [home_z])),
        }
        print(f"  COM XY RMS: {idle_result['com_xy_rms_mm']:.3f} mm")
        print(f"  COM X RMS:  {idle_result['com_x_rms_mm']:.3f} mm")
        print(f"  COM Y RMS:  {idle_result['com_y_rms_mm']:.3f} mm")
        print(f"  Pitch RMS:  {idle_result['pitch_rms_deg']:.4f} deg")
        json.dump(idle_result, open(OUT_DIR / "idle_50hz.json", "w"), indent=2)
        print(f"  → {OUT_DIR / 'idle_50hz.json'}")

    # ═══════════════════════════════════════════════════════════════
    # PUSH F_MIN (4 directions)
    # ═══════════════════════════════════════════════════════════════
    if run_push:
        print("\n--- Push F_min (4 directions) ---")
        angles = [0, 90, 180, -90]  # fwd, left, bwd, right
        fmin_by_angle = {}
        for ang in angles:
            model_fresh = mujoco.MjModel.from_xml_path(MODEL_PATH)
            data_fresh = mujoco.MjData(model_fresh)
            data_fresh.qpos[7:17] = POSTURE.copy()
            data_fresh.qpos[2] = ROOT_Z
            mujoco.mj_forward(model_fresh, data_fresh)
            v3_fresh = init_controller(model_fresh)
            ctx_fresh = build_ctx(model_fresh, data_fresh, v3_fresh)
            settle(model_fresh, data_fresh, v3_fresh, ctx_fresh, settle_steps)
            fmin = bisect_fmin(model_fresh, data_fresh, v3_fresh, ctx_fresh, ang)
            fmin_by_angle[str(ang)] = float(fmin)
            print(f"  {ang:4d}° → F_min = {fmin:6.1f} N")

        push_result = {
            "dt_s": CTRL_DT_50HZ,
            "fmin_by_angle_N": fmin_by_angle,
            "fmin_overall_N": float(min(fmin_by_angle.values())),
        }
        json.dump(push_result, open(OUT_DIR / "push_fmin_50hz.json", "w"), indent=2)
        print(f"  Overall F_min = {push_result['fmin_overall_N']:.1f} N")
        print(f"  → {OUT_DIR / 'push_fmin_50hz.json'}")

    # ═══════════════════════════════════════════════════════════════
    # RINGDOWN
    # ═══════════════════════════════════════════════════════════════
    if run_ringdown:
        push_n = args.push_n
        print(f"\n--- Ringdown ({push_n} N forward push) ---")
        ringdown_s = 20.0
        push_start_s = 3.0
        total_steps = int(ringdown_s / CTRL_DT_50HZ)
        push_s = int(PUSH_DUR_S / CTRL_DT_50HZ)
        push_start = int(push_start_s / CTRL_DT_50HZ)
        push_end = push_start + push_s

        t_log = np.zeros(total_steps)
        pitch_log = np.zeros(total_steps)
        roll_log = np.zeros(total_steps)
        yaw_log = np.zeros(total_steps)
        com_x_log = np.zeros(total_steps)
        com_y_log = np.zeros(total_steps)
        com_z_log = np.zeros(total_steps)

        for step in range(total_steps):
            data.xfrc_applied[torso_id, :3] = 0.0
            if push_start <= step < push_end:
                data.xfrc_applied[torso_id, 0] = push_n

            r = compute_v3_torque_for_state(
                data, model, v3["jax_step_fn"], v3["jax_state"],
                v3["jax_params"], ctx, teleop=None)
            v3["jax_state"] = r["next_jax_state"]
            data.ctrl[:] = np.array(r["tau_v3"])
            for _ in range(SUBSTEPS_50HZ):
                mujoco.mj_step(model, data)

            t_log[step] = step * CTRL_DT_50HZ
            pitch_log[step], roll_log[step], yaw_log[step] = get_pitch_roll_yaw(data)
            com_x_log[step] = float(data.qpos[0] - home_x)
            com_y_log[step] = float(data.qpos[1] - home_y)
            com_z_log[step] = float(data.qpos[2])

            if abs(pitch_log[step]) > PITCH_LIMIT or data.qpos[2] < HEIGHT_LIMIT:
                print(f"  FALL at {step*CTRL_DT_50HZ:.1f}s")
                t_log = t_log[:step+1]
                pitch_log = pitch_log[:step+1]
                roll_log = roll_log[:step+1]
                yaw_log = yaw_log[:step+1]
                com_x_log = com_x_log[:step+1]
                com_y_log = com_y_log[:step+1]
                com_z_log = com_z_log[:step+1]
                break

        peak_pitch_deg = float(np.degrees(np.max(np.abs(pitch_log[push_start:]))))
        settle_after = np.abs(pitch_log[push_end:])
        recovery_time_s = None
        for i, p in enumerate(settle_after):
            if p < np.deg2rad(2.0):
                recovery_time_s = float(i * CTRL_DT_50HZ)
                break

        ringdown_result = {
            "metadata": {"profile": PROFILE_NAME, "dt_s": CTRL_DT_50HZ,
                         "push_N": push_n, "push_start_s": push_start_s,
                         "push_dur_s": PUSH_DUR_S},
            "time_s": t_log.tolist(),
            "pitch_rad": pitch_log.tolist(),
            "roll_rad": roll_log.tolist(),
            "yaw_rad": yaw_log.tolist(),
            "com_x_m": com_x_log.tolist(),
            "com_y_m": com_y_log.tolist(),
            "com_z_m": com_z_log.tolist(),
            "peak_pitch_deg": peak_pitch_deg,
            "recovery_time_s": recovery_time_s,
        }
        json.dump(ringdown_result, open(OUT_DIR / "ringdown_50hz.json", "w"), indent=2)
        print(f"  Peak pitch: {peak_pitch_deg:.2f} deg")
        print(f"  Recovery <2°: {recovery_time_s:.2f} s" if recovery_time_s else "  Recovery: N/A (fell)")
        print(f"  → {OUT_DIR / 'ringdown_50hz.json'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
