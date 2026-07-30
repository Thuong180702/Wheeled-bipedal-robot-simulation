"""
Collect ACC metrics for LQR comparison table with N=10 independent trials.
Matches the metrics reported for LQR baselines in Table VI (tab:lqr_baselines):
  - Survival time (s), Fall rate (%)
  - Pitch RMS (deg), Roll RMS (deg)
  - Height RMS (mm) — CoM Z oscillation around mean
  - Torque RMS (Nm) — across all 10 joints

Based on run_r2_between_trial_stats.py's working controller initialization.
"""
import json, sys, time
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
DT = 0.01
SUBSTEPS = 5
DURATION_S = 25.0
SETTLE_S = 5.0
N_TRIALS = 10

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
h0 = float(nom["target_com_z_m"])
POSTURE = np.array([
    nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])


def quat_to_rpy(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    roll = float(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    pitch = float(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    return roll, pitch


def build_controller_context(model, posture, h0_ref):
    """Build controller context — inlined from run_r2_between_trial_stats.py.
    Avoids promote_v3_vs_assist which has JAX import-order issues."""
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass, torso_inertia=torso_inertia)
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=model)
    return {
        "centroidal_estimator": centroidal_estimator,
        "initial_yaw_z": 0.0,
        "l_wheel_id": l_wheel_id,
        "r_wheel_id": r_wheel_id,
        "eq_joint": posture,
        "height_ref": h0_ref,
        "prev_com_pos": None,
    }


def run_one_trial(trial_id, model, posture, root_z, h0):
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    if not v3.get("initialized"):
        return {"trial": trial_id, "fell": True, "error": v3.get("error", "init failed"),
                "survival_s": 0, "pitch_rms_deg": float("nan"),
                "roll_rms_deg": float("nan"), "height_rms_mm": float("nan"),
                "torque_rms_Nm": float("nan")}

    v3["jax_state"] = pack_state_k2()

    # Randomized initial conditions (matching run_r2_between_trial_stats.py)
    rng = np.random.default_rng(20260727 + trial_id)
    perturbed = posture + rng.normal(0.0, 0.005, size=10)
    joint_names = [
        "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
        "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
    ]
    for j, jname in enumerate(joint_names):
        jid = model.joint(jname).id
        lo, hi = model.jnt_range[jid]
        perturbed[j] = float(np.clip(perturbed[j], lo, hi))

    data = mujoco.MjData(model)
    data.qpos[7:17] = perturbed
    data.qpos[2] = root_z + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)

    ctx = build_controller_context(model, posture, h0)

    total_steps = int(DURATION_S / DT)
    settle_steps = int(SETTLE_S / DT)

    pitch_vals = []
    roll_vals = []
    com_z_vals = []
    torque_vals = []
    fell = False
    step = 0

    for step in range(total_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        q = data.qpos[3:7]
        roll, pitch = quat_to_rpy(q)

        if abs(pitch) > 0.8 or data.qpos[2] < 0.30:
            fell = True
            break

        if step >= settle_steps:
            pitch_vals.append(np.degrees(pitch))
            roll_vals.append(np.degrees(roll))
            com_z_vals.append(float(data.subtree_com[0][2]) * 1000)
            torque_vals.append(np.array(data.ctrl[:10]).copy())

    survival_s = (step + 1) * DT

    if len(pitch_vals) > 0:
        p_arr = np.array(pitch_vals)
        r_arr = np.array(roll_vals)
        z_arr = np.array(com_z_vals)
        t_arr = np.array(torque_vals)

        return {
            "trial": trial_id, "fell": fell, "survival_s": survival_s,
            # pitch/roll: std (oscillation around mean), height: std, torque: RMS
            "pitch_rms_deg": float(np.std(p_arr)),
            "roll_rms_deg": float(np.std(r_arr)),
            "height_rms_mm": float(np.std(z_arr)),
            "torque_rms_Nm": float(np.sqrt(np.mean(t_arr**2))),
        }
    else:
        return {
            "trial": trial_id, "fell": True, "survival_s": survival_s,
            "pitch_rms_deg": float("nan"), "roll_rms_deg": float("nan"),
            "height_rms_mm": float("nan"), "torque_rms_Nm": float("nan"),
        }


def main():
    print("=" * 60)
    print(f"ACC LQR Comparison — {N_TRIALS} trials")
    print(f"Profile: {PROFILE}, duration: {DURATION_S}s (settle: {SETTLE_S}s)")
    print("=" * 60)

    # Pre-load model once
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    trials = []
    t0 = time.time()
    for i in range(N_TRIALS):
        r = run_one_trial(i, model, POSTURE, ROOT_Z, h0)
        trials.append(r)
        status = "FELL" if r["fell"] else "OK"
        print(f"  [{i+1:2d}/{N_TRIALS}] {status} | "
              f"pitch={r['pitch_rms_deg']:.3f}° | roll={r['roll_rms_deg']:.3f}° | "
              f"H={r['height_rms_mm']:.2f}mm | τ={r['torque_rms_Nm']:.2f}Nm")

    fell_count = sum(1 for t in trials if t["fell"])
    survivals = [t["survival_s"] for t in trials]
    ok = [t for t in trials if not t["fell"]]

    stats = {
        "test": "acc_lqr_comparison",
        "profile": PROFILE,
        "n_trials": N_TRIALS,
        "duration_s": DURATION_S - SETTLE_S,
        "control_hz": int(1/DT),
        "sim_hz": int(1/(DT/SUBSTEPS)),
        "fell_count": fell_count,
        "fall_rate_pct": 100.0 * fell_count / N_TRIALS,
        "survival_s_mean": float(np.mean(survivals)),
        "survival_s_std": float(np.std(survivals, ddof=1)) if N_TRIALS > 1 else 0.0,
    }

    for metric in ["pitch_rms_deg", "roll_rms_deg", "height_rms_mm", "torque_rms_Nm"]:
        vals = [t[metric] for t in ok if not np.isnan(t[metric])]
        stats[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
        stats[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print("PAPER-READY: ACC row for LQR comparison Table VI")
    print(f"{'='*60}")
    print(f"  N = {N_TRIALS}")
    print(f"  Survival:   {stats['survival_s_mean']:.2f} ± {stats['survival_s_std']:.2f} s")
    print(f"  Fall rate:  {stats['fall_rate_pct']:.1f}%")
    print(f"  Pitch RMS:  {stats['pitch_rms_deg_mean']:.3f} ± {stats['pitch_rms_deg_std']:.3f}°")
    print(f"  Roll RMS:   {stats['roll_rms_deg_mean']:.3f} ± {stats['roll_rms_deg_std']:.3f}°")
    print(f"  Height RMS: {stats['height_rms_mm_mean']:.2f} ± {stats['height_rms_mm_std']:.2f} mm")
    print(f"  Torque RMS: {stats['torque_rms_Nm_mean']:.2f} ± {stats['torque_rms_Nm_std']:.2f} Nm")

    out_path = OUT_DIR / "acc_lqr_comparison.json"
    json.dump(stats, out_path.open("w"), indent=2, default=str)
    print(f"\n  Saved → {out_path}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
