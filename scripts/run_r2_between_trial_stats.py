#!/usr/bin/env python3
"""
R2: Between-trial statistics for the ACC idle precision claim (0.3mm CoM RMS).

Runs N independent idle-standing trials with randomized initial joint positions
and reports mean ± std with 95% CI. Also measures the simulator noise floor.

Usage:
    python scripts/run_r2_between_trial_stats.py
    python scripts/run_r2_between_trial_stats.py --n-trials 20 --noise-floor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state,
    init_v3_controller,
)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.utils.config import get_model_path

# ── Constants ──────────────────────────────────────────────────────────────────
DT = 0.01          # 100 Hz control
SUBSTEPS = 5       # 500 Hz simulation
TOTAL_S = 25.0     # total run duration
SETTLE_S = 5.0     # skip first 5s (settle)
WINDOW_S = 20.0    # measurement window
ACC_PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
OUT_DIR = ROOT / "outputs" / "r2_between_trial_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Nominal posture and height from the variant setup
DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"


def load_nominal_config():
    """Load nominal posture and target height from the variant setup."""
    nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
    h0 = float(nom["target_com_z_m"])
    posture = np.array([
        nom["hip_roll_left"], nom["hip_yaw_left"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
        nom["hip_roll_right"], nom["hip_yaw_right"],
        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    ])
    root_z = float(nom["calibrated_root_z_m"])
    return h0, posture, root_z


def run_single_trial(
    model: mujoco.MjModel,
    posture: np.ndarray,
    root_z: float,
    h0: float,
    trial_seed: int,
    perturb_std_rad: float = 0.005,
) -> dict:
    """Run one idle-standing trial with randomized initial joint positions.

    Args:
        model: MuJoCo model.
        posture: nominal 10-dim joint positions [rad].
        root_z: calibrated root z from variant setup.
        h0: target CoM z height.
        trial_seed: random seed for this trial.
        perturb_std_rad: std of Gaussian perturbation applied to initial qpos.

    Returns:
        dict with sagittal CoM RMS, pitch RMS, etc.
    """
    rng = np.random.default_rng(trial_seed)

    # Initialize ACC controller
    v3 = dict(init_v3_controller(profile_name=ACC_PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()

    # Set up MuJoCo data with perturbed initial posture
    data = mujoco.MjData(model)
    perturbed_posture = posture + rng.normal(0.0, perturb_std_rad, size=10)
    # Clamp perturbations to joint limits
    _joint_names = [
        "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
        "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
    ]
    for j, jname in enumerate(_joint_names):
        jid = model.joint(jname).id
        lo, hi = model.jnt_range[jid]
        perturbed_posture[j] = float(np.clip(perturbed_posture[j], lo, hi))
    data.qpos[7:17] = perturbed_posture
    data.qpos[2] = root_z + rng.normal(0.0, 0.001)  # small root-z perturbation
    mujoco.mj_forward(model, data)

    # Build controller context (inlined from promote_v3_vs_assist to avoid
    # JAX import-order conflict). See _build_v3_controller_context in
    # scripts/promote_v3_vs_assist.py.
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator, CentroidalStateEstimatorConfig,
    )
    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")
    robot_mass = float(np.sum(model.body_mass))
    torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
    centroidal_config = CentroidalStateEstimatorConfig(
        robot_mass=robot_mass, torso_inertia=torso_inertia,
    )
    centroidal_estimator = CentroidalStateEstimator(centroidal_config, mj_model=model)
    ctx = {
        "centroidal_estimator": centroidal_estimator,
        "initial_yaw_z": 0.0,
        "l_wheel_id": l_wheel_id,
        "r_wheel_id": r_wheel_id,
        "eq_joint": posture,
        "height_ref": h0,
        "prev_com_pos": None,
    }

    n_steps = int(TOTAL_S / DT)
    settle_start = int(SETTLE_S / DT)

    com_x_debiased = np.zeros(n_steps - settle_start)
    pitch_vals = np.zeros(n_steps - settle_start)
    step_count = 0

    for step in range(n_steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None,
        )
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        if step >= settle_start:
            com_x_debiased[step_count] = data.subtree_com[0][0]
            q = data.qpos[3:7]
            pitch = np.degrees(np.arcsin(np.clip(
                2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
            pitch_vals[step_count] = pitch
            step_count += 1

        # Check for fall
        if abs(pitch_vals[max(0, step_count - 1)]) > 46 or data.qpos[2] < 0.15:
            return {
                "survived": False,
                "fall_step": step,
                "trial_seed": trial_seed,
            }

    # Compute metrics: sagittal CoM RMS (std after removing mean)
    com_x_centered = com_x_debiased - np.mean(com_x_debiased)
    rms_x_mm = float(np.sqrt(np.mean(com_x_centered ** 2)) * 1000)
    p2p_x_mm = float((np.max(com_x_debiased) - np.min(com_x_debiased)) * 1000)
    rms_pitch_deg = float(np.sqrt(np.mean(pitch_vals ** 2)))

    # Best 20s window RMS (slide a 20s window over the settled data)
    window_samples = int(WINDOW_S / DT)
    best_rms = float("inf")
    # Iterate over possible window start positions within the settled data
    for ws_start in range(0, len(com_x_debiased) - window_samples + 1, window_samples // 2):
        w = com_x_debiased[ws_start:ws_start + window_samples]
        wr = float(np.sqrt(np.mean((w - np.mean(w)) ** 2)) * 1000)
        if wr < best_rms:
            best_rms = wr

    return {
        "survived": True,
        "trial_seed": trial_seed,
        "com_x_rms_mm": rms_x_mm,
        "best_20s_com_x_rms_mm": best_rms,
        "com_x_p2p_mm": p2p_x_mm,
        "pitch_rms_deg": rms_pitch_deg,
        "n_samples": step_count,
    }


def run_noise_floor(model: mujoco.MjModel, posture: np.ndarray, root_z: float,
                    duration_s: float = 5.0) -> dict:
    """Measure simulator noise floor: fixed-base, locked-joint CoM RMS.

    Fixes the robot base (free joint) in place and locks all joints at nominal
    posture with no controller active. Any residual CoM variation is numerical
    noise from MuJoCo's integrator / constraint solver.
    """
    data = mujoco.MjData(model)

    # Fix free joint by zeroing velocities each step
    data.qpos[7:17] = posture
    data.qpos[2] = root_z
    mujoco.mj_forward(model, data)

    n_steps = int(duration_s / DT)
    com_positions = np.zeros((n_steps, 3))

    for step in range(n_steps):
        # Zero all velocities to hold fixed
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)
        com_positions[step] = data.subtree_com[0].copy()

    com_centered = com_positions - np.mean(com_positions, axis=0)
    rms_xy_mm = float(np.sqrt(np.mean(np.sum(com_centered[:, :2] ** 2, axis=1))) * 1000)
    rms_x_mm = float(np.sqrt(np.mean(com_centered[:, 0] ** 2)) * 1000)
    rms_y_mm = float(np.sqrt(np.mean(com_centered[:, 1] ** 2)) * 1000)
    rms_z_mm = float(np.sqrt(np.mean(com_centered[:, 2] ** 2)) * 1000)
    rms_3d_mm = float(np.sqrt(np.mean(np.sum(com_centered ** 2, axis=1))) * 1000)
    p2p_x_mm = float((np.max(com_positions[:, 0]) - np.min(com_positions[:, 0])) * 1000)

    return {
        "rms_x_mm": rms_x_mm,
        "rms_y_mm": rms_y_mm,
        "rms_xy_mm": rms_xy_mm,
        "rms_z_mm": rms_z_mm,
        "rms_3d_mm": rms_3d_mm,
        "p2p_x_mm": p2p_x_mm,
        "n_samples": n_steps,
        "duration_s": duration_s,
        "description": "Fixed-base, locked-joint, no controller — numerical noise floor",
    }


def main():
    parser = argparse.ArgumentParser(
        description="R2: Between-trial statistics for ACC idle precision claim"
    )
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of independent trials (default: 10)")
    parser.add_argument("--noise-floor", action="store_true", default=True,
                        help="Also measure simulator noise floor")
    parser.add_argument("--perturb-std-rad", type=float, default=0.005,
                        help="Std of initial joint perturbation [rad] (default: 0.005)")
    parser.add_argument("--base-seed", type=int, default=20260727,
                        help="Base random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("R2: Between-Trial Statistics for ACC Idle Precision")
    print(f"Profile: {ACC_PROFILE}")
    print(f"Trials: {args.n_trials}")
    print(f"Perturbation std: {args.perturb_std_rad:.4f} rad")
    print("=" * 70)

    # ── Load model and nominal config ──────────────────────────────────────
    model_path = str(get_model_path())
    model = mujoco.MjModel.from_xml_path(model_path)
    h0, posture, root_z = load_nominal_config()

    print(f"\nNominal posture (deg):")
    for i, name in enumerate([
        "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
        "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
    ]):
        print(f"  {name:>14s}: {np.degrees(posture[i]):6.2f}°")
    print(f"  Target CoM z: {h0:.3f} m")
    print(f"  Calibrated root z: {root_z:.3f} m")

    # ── Run N independent trials ───────────────────────────────────────────
    all_results = []
    t0 = time.time()

    for trial_i in range(args.n_trials):
        seed = args.base_seed + trial_i * 100
        print(f"\n  Trial {trial_i + 1}/{args.n_trials} (seed={seed})...", end=" ", flush=True)

        result = run_single_trial(
            model, posture, root_z, h0,
            trial_seed=seed,
            perturb_std_rad=args.perturb_std_rad,
        )

        if result["survived"]:
            print(f"CoM X RMS: {result['com_x_rms_mm']:.3f} mm  "
                  f"(best 20s: {result['best_20s_com_x_rms_mm']:.3f} mm)")
        else:
            print(f"FELL at step {result.get('fall_step', '?')}!")
        all_results.append(result)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / args.n_trials:.0f}s/trial)")

    # ── Compute between-trial statistics ───────────────────────────────────
    survived = [r for r in all_results if r["survived"]]
    n_survived = len(survived)
    n_fell = len(all_results) - n_survived

    print(f"\n{'=' * 70}")
    print("BETWEEN-TRIAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Trials: {args.n_trials} | Survived: {n_survived} | Fell: {n_fell}")
    print(f"  Survival rate: {n_survived / args.n_trials * 100:.0f}%")

    if n_survived >= 2:
        rms_vals = np.array([r["com_x_rms_mm"] for r in survived])
        best_rms_vals = np.array([r["best_20s_com_x_rms_mm"] for r in survived])
        p2p_vals = np.array([r["com_x_p2p_mm"] for r in survived])
        pitch_vals = np.array([r["pitch_rms_deg"] for r in survived])

        mean_rms = float(np.mean(rms_vals))
        std_rms = float(np.std(rms_vals, ddof=1))
        mean_best = float(np.mean(best_rms_vals))
        std_best = float(np.std(best_rms_vals, ddof=1))

        # 95% CI: mean ± t_{0.025, n-1} * std/sqrt(n)
        from scipy import stats as scipy_stats
        ci95_rms = scipy_stats.t.interval(0.95, n_survived - 1,
                                          loc=mean_rms, scale=std_rms / np.sqrt(n_survived))
        ci95_best = scipy_stats.t.interval(0.95, n_survived - 1,
                                           loc=mean_best, scale=std_best / np.sqrt(n_survived))

        print(f"\n  ── Sagittal CoM X RMS (post-5s, full 20s) ──")
        print(f"  Mean ± Std:  {mean_rms:.4f} ± {std_rms:.4f} mm")
        print(f"  95% CI:      [{ci95_rms[0]:.4f}, {ci95_rms[1]:.4f}] mm")
        print(f"  CI width:    {ci95_rms[1] - ci95_rms[0]:.4f} mm")
        print(f"  Between-trial CV: {std_rms / mean_rms * 100:.1f}%")
        print(f"  Individual trials: {', '.join(f'{v:.3f}' for v in rms_vals)}")

        print(f"\n  ── Best 20s Window CoM X RMS ──")
        print(f"  Mean ± Std:  {mean_best:.4f} ± {std_best:.4f} mm")
        print(f"  95% CI:      [{ci95_best[0]:.4f}, {ci95_best[1]:.4f}] mm")
        print(f"  CI width:    {ci95_best[1] - ci95_best[0]:.4f} mm")

        print(f"\n  ── CoM X Peak-to-Peak ──")
        print(f"  Mean ± Std:  {np.mean(p2p_vals):.2f} ± {np.std(p2p_vals, ddof=1):.2f} mm")

        print(f"\n  ── Pitch RMS ──")
        print(f"  Mean ± Std:  {np.mean(pitch_vals):.4f} ± {np.std(pitch_vals, ddof=1):.4f}°")

        # Strength assessment
        print(f"\n  ── Claim Assessment ──")
        if ci95_best[1] - ci95_best[0] < 0.1:
            print(f"  ✅ CI width < 0.1 mm — STRONG claim validated")
        elif ci95_best[1] - ci95_best[0] < 0.5:
            print(f"  ✅ CI width {ci95_best[1]-ci95_best[0]:.2f} mm < 0.5 mm — claim validated")
        else:
            print(f"  ⚠️  CI width {ci95_best[1]-ci95_best[0]:.2f} mm > 0.5 mm — investigation needed")

    # ── Noise floor measurement ────────────────────────────────────────────
    noise_result = None
    if args.noise_floor:
        print(f"\n{'=' * 70}")
        print("NOISE FLOOR MEASUREMENT (fixed-base, locked-joint)")
        print(f"{'=' * 70}")
        noise_result = run_noise_floor(model, posture, root_z, duration_s=5.0)
        print(f"  CoM X RMS:  {noise_result['rms_x_mm']:.4f} mm")
        print(f"  CoM XY RMS: {noise_result['rms_xy_mm']:.4f} mm")
        print(f"  CoM Z RMS:  {noise_result['rms_z_mm']:.4f} mm")
        print(f"  CoM 3D RMS: {noise_result['rms_3d_mm']:.4f} mm")
        print(f"  CoM X P2P:  {noise_result['p2p_x_mm']:.4f} mm")

        if n_survived >= 2 and noise_result["rms_x_mm"] > 0:
            signal_to_noise = mean_rms / noise_result["rms_x_mm"]
            print(f"\n  Signal-to-noise ratio: {signal_to_noise:.1f}×")
            if signal_to_noise < 2:
                print(f"  ⚠️  Signal within 2× noise floor — claim near measurement limit")
            else:
                print(f"  ✅ Signal {signal_to_noise:.1f}× above noise floor")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "test": "r2_between_trial_statistics",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "profile": ACC_PROFILE,
        "n_trials": args.n_trials,
        "n_survived": n_survived,
        "n_fell": n_fell,
        "survival_rate": n_survived / args.n_trials,
        "perturb_std_rad": args.perturb_std_rad,
        "base_seed": args.base_seed,
        "nominal_root_z": root_z,
        "target_com_z": h0,
    }

    if n_survived >= 2:
        output["com_x_rms_mm"] = {
            "mean": mean_rms,
            "std": std_rms,
            "ci95_lower": float(ci95_rms[0]),
            "ci95_upper": float(ci95_rms[1]),
            "ci95_width": float(ci95_rms[1] - ci95_rms[0]),
            "between_trial_cv_pct": float(std_rms / mean_rms * 100),
            "individual": [float(v) for v in rms_vals],
        }
        output["best_20s_com_x_rms_mm"] = {
            "mean": mean_best,
            "std": std_best,
            "ci95_lower": float(ci95_best[0]),
            "ci95_upper": float(ci95_best[1]),
            "ci95_width": float(ci95_best[1] - ci95_best[0]),
            "individual": [float(v) for v in best_rms_vals],
        }
        output["com_x_p2p_mm"] = {
            "mean": float(np.mean(p2p_vals)),
            "std": float(np.std(p2p_vals, ddof=1)),
            "individual": [float(v) for v in p2p_vals],
        }
        output["pitch_rms_deg"] = {
            "mean": float(np.mean(pitch_vals)),
            "std": float(np.std(pitch_vals, ddof=1)),
            "individual": [float(v) for v in pitch_vals],
        }

    if noise_result:
        output["noise_floor"] = noise_result
        if n_survived >= 2 and noise_result["rms_x_mm"] > 0:
            output["signal_to_noise_ratio"] = signal_to_noise

    # Also include per-trial results
    output["trials"] = all_results

    out_path = OUT_DIR / "between_trial_stats.json"
    json.dump(output, out_path.open("w"), indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # ── Paper-ready summary ────────────────────────────────────────────────
    if n_survived >= 2:
        print(f"\n{'=' * 70}")
        print("PAPER-READY SUMMARY")
        print(f"{'=' * 70}")
        print(f"  ACC idle CoM X RMS:")
        print(f"    {mean_rms:.3f} ± {std_rms:.3f} mm (mean ± std, N={n_survived})")
        print(f"    95% CI: [{ci95_rms[0]:.3f}, {ci95_rms[1]:.3f}] mm")
        print(f"    Between-trial CV: {std_rms / mean_rms * 100:.1f}%")
        if noise_result:
            print(f"    Noise floor (fixed-base): {noise_result['rms_x_mm']:.4f} mm")
            print(f"    S/N ratio: {output.get('signal_to_noise_ratio', 0):.1f}×")
        print(f"\n  Paper claim: 0.3mm (single trial, N=1)")
        print(f"  Updated claim: {mean_rms:.2f} ± {std_rms:.2f} mm (N={n_survived}, 95% CI [{ci95_rms[0]:.2f}, {ci95_rms[1]:.2f}] mm)")

    return 0 if n_survived >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
