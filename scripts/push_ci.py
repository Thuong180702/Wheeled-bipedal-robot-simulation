"""
Push sweep with repetitions for confidence intervals (R2 fix).
Extends push_sweep_paper.py to run N≥3 independent reps with randomized
initial joint positions (±0.005 rad) per direction per config.

Reports F_min ± std and F_med ± std across repetitions.

Usage:
  JAX_ENABLE_X64=True mjpython scripts/push_ci.py --profile=K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR --reps=3 --quick
  JAX_ENABLE_X64=True mjpython scripts/push_ci.py --profile=V3_HOMING --reps=3 --quick
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)

DT = 0.01; SUBSTEPS = 5
PUSH_DUR = 7; PUSH_START = 300; POST_PUSH_S = 17.0
POST_PUSH_STEPS = int(POST_PUSH_S / DT)
TOTAL_STEPS = PUSH_START + PUSH_DUR + POST_PUSH_STEPS
PITCH_LIMIT = 0.8; HEIGHT_LIMIT = 0.30
FORCE_MIN, FORCE_MAX = 10.0, 160.0
N_BISECT = 8; TOLERANCE = 5.0

MODEL_PATH = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])

JOINT_NAMES = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
               "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]


def build_ctx(model, data, v3):
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
    est = CentroidalStateEstimator(cfg, mj_model=model)
    return {"centroidal_estimator": est, "initial_yaw_z": 0.0,
            "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
            "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}


def run_push(model, data, v3, ctx, force_N, angle_deg):
    """Single push trial. Returns True if survived."""
    angle_rad = np.deg2rad(angle_deg)
    force = np.array([force_N * np.cos(angle_rad), force_N * np.sin(angle_rad), 0.0])
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    for step in range(POST_PUSH_STEPS + PUSH_DUR):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])

        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, :3] = force

        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        quat = data.qpos[3:7].copy()
        pitch = np.arcsin(-2*(quat[1]*quat[3] - quat[0]*quat[2]))
        if abs(pitch) > PITCH_LIMIT or data.subtree_com[0][2] < HEIGHT_LIMIT:
            return False
    return True


def settle(model, data, v3, ctx, steps=300):
    for _ in range(steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)


def bisect_one_direction(model, profile, ctx, angle_deg, seed):
    """Binary search for max survived force at one angle, with randomized init."""
    rng = np.random.default_rng(seed)
    lo, hi = FORCE_MIN, FORCE_MAX
    best = lo

    for i in range(N_BISECT):
        mid = (lo + hi) / 2
        # Fresh physics + controller each bisect iteration (avoids state carryover)
        data = mujoco.MjData(model)
        perturbed = POSTURE + rng.normal(0.0, 0.005, size=10)
        for j, jname in enumerate(JOINT_NAMES):
            jid = model.joint(jname).id
            lo_j, hi_j = model.jnt_range[jid]
            perturbed[j] = float(np.clip(perturbed[j], lo_j, hi_j))
        data.qpos[7:17] = perturbed
        data.qpos[2] = ROOT_Z + rng.normal(0.0, 0.001)
        mujoco.mj_forward(model, data)

        v3 = dict(init_v3_controller(profile_name=profile, model=model))
        if not v3.get("initialized"):
            return best  # fallback — unlikely
        v3["jax_state"] = pack_state_k2()
        ctx_fresh = build_ctx(model, data, v3)
        settle(model, data, v3, ctx_fresh, steps=PUSH_START)

        if run_push(model, data, v3, ctx_fresh, mid, angle_deg):
            best = mid
            lo = mid + TOLERANCE / 2
        else:
            hi = mid - TOLERANCE / 2
    return best


def run_config(model, profile, angles, n_reps, label):
    """Run push sweep with n_reps per direction. Returns per-direction stats."""
    print(f"\n{'='*60}")
    print(f"Config: {label}  ({profile})")
    print(f"Directions: {len(angles)}, Reps: {n_reps}")
    print(f"{'='*60}")

    t0 = time.time()
    all_reps = {ang: [] for ang in angles}

    for rep in range(n_reps):
        seed = 20260728 * 100 + rep
        print(f"  Rep {rep+1}/{n_reps}: ", end="", flush=True)
        for ang in angles:
            th = bisect_one_direction(model, profile, angles, ang, seed + int(ang))
            all_reps[ang].append(th)
            print(f"{th:.0f} ", end="", flush=True)
        elapsed = time.time() - t0
        eta = elapsed / (rep+1) * (n_reps - rep - 1)
        print(f" | {elapsed/60:.1f}m ETA {eta/60:.1f}m")

    # Per-direction stats
    per_dir = {}
    f_min_vals = []  # one F_min per rep
    f_med_vals = []  # one F_med per rep

    for rep in range(n_reps):
        rep_vals = [all_reps[ang][rep] for ang in angles]
        f_min_vals.append(min(rep_vals))
        f_med_vals.append(sorted(rep_vals)[len(rep_vals)//2])

    f_min_mean = np.mean(f_min_vals)
    f_min_std = np.std(f_min_vals, ddof=1) if n_reps > 1 else 0.0
    f_med_mean = np.mean(f_med_vals)
    f_med_std = np.std(f_med_vals, ddof=1) if n_reps > 1 else 0.0

    elapsed = time.time() - t0
    print(f"\n  F_min = {f_min_mean:.1f} ± {f_min_std:.1f} N")
    print(f"  F_med = {f_med_mean:.1f} ± {f_med_std:.1f} N")
    print(f"  Time: {elapsed/60:.1f}m")

    return {
        "label": label, "profile": profile, "n_reps": n_reps,
        "n_directions": len(angles),
        "F_min_mean": float(f_min_mean), "F_min_std": float(f_min_std),
        "F_med_mean": float(f_med_mean), "F_med_std": float(f_med_std),
        "F_min_ci95": float(1.96 * f_min_std / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
        "F_med_ci95": float(1.96 * f_med_std / np.sqrt(n_reps)) if n_reps > 1 else 0.0,
        "f_min_per_rep": f_min_vals,
        "f_med_per_rep": f_med_vals,
        "all_reps": {int(k): v for k, v in all_reps.items()},
        "elapsed_min": elapsed/60,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="8 directions (every 45°)")
    p.add_argument("--label", default=None)
    p.add_argument("--all-ablations", action="store_true",
                   help="Run all ablation configs (takes ~8h)")
    a = p.parse_args()

    angles = list(range(0, 360, 45)) if a.quick else list(range(0, 360, 15))
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    if a.all_ablations:
        # All configs from Table III additive + subtractive
        # Each needs specific parameter overrides — for now run the main
        # profiles that have named definitions
        configs = [
            ("K2_JAX_DEDICATED_DEFAULT_V3", "L0_P-only"),
            ("K2_JAX_DEDICATED_DEFAULT_V3_HOMING", "L1_P-homing"),
            ("K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR", "L3_S0_Full-ACC"),
        ]
        all_results = {}
        for profile, label in configs:
            r = run_config(model, profile, angles, a.reps, label)
            all_results[label] = r
        out_path = OUT_DIR / "push_ci_all_ablations.json"
    else:
        label = a.label or a.profile.split("_")[-1]
        r = run_config(model, a.profile, angles, a.reps, label)
        all_results = {label: r}
        out_path = OUT_DIR / f"push_ci_{label.lower()}.json"

    all_results["metadata"] = {
        "n_reps": a.reps, "n_directions": len(angles),
        "binary_search_tol_N": TOLERANCE,
        "binary_search_iters": N_BISECT,
        "force_range_N": [FORCE_MIN, FORCE_MAX],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(all_results, out_path.open("w"), indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
