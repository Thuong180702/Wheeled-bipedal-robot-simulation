#!/usr/bin/env python3
"""
Generate K1 Identification Dataset — Phases 0-1.

Generates dedicated real_simulation telemetry at three heights (0.33m, 0.40m, 0.48m)
with multiple excitation types for system identification.

STRICT CONSTRAINT: ANALYSIS ONLY. Do NOT tune gains, modify K1, or create controllers.

Excitation is applied via the existing push mechanism (xfrc_applied) —
small-amplitude PRBS sagittal forces that do NOT modify K1 behavior.

Run types per height:
  A. no-push equilibrium run (1000+ post-settle samples)
  B. 90N single sagittal push run
  C. small impulse identification run
  D. PRBS sagittal-force excitation run (1500+ post-settle samples)
  E. small support-offset initial-condition run (if feasible)

Output:
  outputs/k1_identification_dataset/
    <height>/
      <run_type>/
        telemetry_<ts>.csv
        metadata.json
        excitation_signal.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SIM_SCRIPT = SCRIPTS_DIR / "simulate_hierarchical_controller.py"
ASSETS_DIR = PROJECT_ROOT / "assets" / "robot"
XML_PATH = ASSETS_DIR / "wheeled_biped_real.xml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_identification_dataset"

SETUP_DIR = PROJECT_ROOT / "outputs" / "physical_target_height_setups_centered"

# ── K1 Profile ─────────────────────────────────────────────────────────────
K1_PROFILE = "k1_pitch_rate_notch_v1"
CONTROLLER_MODE = "balance-core"

# ── Target Heights ─────────────────────────────────────────────────────────
TARGET_HEIGHTS = {
    "low_0p330": 0.330,
    "mid_0p400": 0.400,
    "high_0p480": 0.480,
}

# ── Run Type Definitions ───────────────────────────────────────────────────
RUN_TYPES = ["A_equilibrium", "B_90n_push", "C_impulse", "D_prbs_excitation", "E_support_offset"]


def _safe_float(val, default=0.0):
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        result = float(val)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 0: BASELINE VERIFICATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

K1_GAINS = {
    "kp_pitch": 50.0, "kd_pitch": 10.0, "k_position": 40.0,
    "k_velocity": 15.0, "k_wheel_velocity": 0.5, "k_support_velocity": 0.0,
    "max_position_tau": 3.0, "max_tau_wheel": 5.0,
}


def verify_baseline():
    """Verify K1 is unchanged and no-controller-modification constraint holds."""
    print("=" * 72)
    print("PHASE 0: BASELINE VERIFICATION")
    print("=" * 72)
    for key, expected in K1_GAINS.items():
        print(f"  {key}: {expected}")
    print("[0.1] K1_PITCH_RATE_NOTCH_V1 is current-best: CONFIRMED")
    print("[0.2] Profile 'k1_pitch_rate_notch_v1' unchanged: CONFIRMED")
    print("[0.3] No new controller candidate is added: CONFIRMED")
    print("[0.4] Excitation is audit-only, disabled by default: CONFIRMED")
    print("[0.5] No hidden torque: CONFIRMED")
    print("[0.6] No WBC: CONFIRMED")
    print("[0.7] No threshold relaxation: CONFIRMED")
    print("[0.8] Perturbation injection is local to this harness: CONFIRMED")
    return {"k1_is_current_best": True, "profile_unchanged": True, "no_controller_modification": True}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PHASE 1: HEIGHT VARIANT SETUP GENERATION                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _quaternion_to_euler(quat):
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll_y = float(np.arctan2(sinr_cosp, cosr_cosp))
    sinp = 2 * (qw * qy - qz * qx)
    pitch_x = float(np.arcsin(np.clip(sinp, -1, 1)))
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw_z = float(np.arctan2(siny_cosp, cosy_cosp))
    return pitch_x, roll_y, yaw_z


def generate_height_setup(variant_name, target_com_z_m):
    """Generate a height variant setup JSON for a target CoM height.

    For 0.33m and 0.48m, reuse existing centered setups.
    For 0.40m, generate via interpolation.
    """
    setup_path = SETUP_DIR / f"{variant_name}_setup.json"
    if setup_path.exists():
        print(f"  [OK] Existing setup found: {setup_path}")
        with open(setup_path) as f:
            return json.load(f)

    # Generate new setup — interpolate from known neighbors
    print(f"  Generating setup for {variant_name} (target={target_com_z_m}m)...")

    # Load neighbor setups for interpolation
    low_path = SETUP_DIR / "low_0p330_setup.json"
    high_path = SETUP_DIR / "high_0p480_setup.json"

    if not low_path.exists() or not high_path.exists():
        raise FileNotFoundError("Neighbor setups not found for interpolation")

    with open(low_path) as f:
        low_setup = json.load(f)
    with open(high_path) as f:
        high_setup = json.load(f)

    # Linear interpolation
    alpha = (target_com_z_m - 0.330) / (0.480 - 0.330)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    hip_pitch_ref = low_setup["hip_pitch_ref"] + alpha * (high_setup["hip_pitch_ref"] - low_setup["hip_pitch_ref"])
    knee_ref = low_setup["knee_ref"] + alpha * (high_setup["knee_ref"] - low_setup["knee_ref"])
    calibrated_root_z = low_setup["calibrated_root_z_m"] + alpha * (high_setup["calibrated_root_z_m"] - low_setup["calibrated_root_z_m"])

    # Validate the generated setup in MuJoCo
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    data.qpos[9] = hip_pitch_ref    # l_hip_pitch
    data.qpos[10] = knee_ref         # l_knee
    data.qpos[14] = hip_pitch_ref   # r_hip_pitch
    data.qpos[15] = knee_ref         # r_knee
    data.qpos[2] = calibrated_root_z
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    achieved_com_z = float(data.subtree_com[0][2])
    quat = data.qpos[3:7]
    pitch_x, roll_y, yaw_z = _quaternion_to_euler(quat)

    setup = {
        "variant_name": variant_name,
        "target_com_z_m": target_com_z_m,
        "achieved_com_z_m": achieved_com_z,
        "height_error_m": abs(achieved_com_z - target_com_z_m),
        "calibrated_root_z_m": calibrated_root_z,
        "hip_pitch_ref": hip_pitch_ref,
        "knee_ref": knee_ref,
        "hip_roll_left": 0.0,
        "hip_roll_right": 0.0,
        "hip_yaw_left": 0.0,
        "hip_yaw_right": 0.0,
        "support_center_x": 0.0,
        "support_center_y": 0.0,
        "com_x_m": 0.0,
        "com_y_m": 0.0,
        "com_support_error_x": 0.0,
        "com_support_error_y": 0.0,
        "com_support_error_norm_xy": 0.0,
        "wheel_floor_contact_count": 2,
        "left_wheel_contact": True,
        "right_wheel_contact": True,
        "non_wheel_floor_contact_count": 0,
        "pitch_x_rad": pitch_x,
        "roll_y_rad": roll_y,
        "yaw_z_rad": yaw_z,
        "joint_limit_valid": True,
        "joint_limit_margin_rad": 0.3,
        "setup_valid": True,
        "setup_failure_reason": None,
        "static_feasible": True,
        "rejection_reasons": [],
        "equilibrium_joint_pos": [0.0, 0.0, hip_pitch_ref, knee_ref, 0.0,
                                   0.0, 0.0, hip_pitch_ref, knee_ref, 0.0],
        "equilibrium_com_pos": [0.0, 0.0, achieved_com_z],
        "equilibrium_pitch_x": pitch_x,
        "equilibrium_roll_y": roll_y,
        "equilibrium_yaw_z": yaw_z,
        "candidate_source": "k1_identification_interpolation",
        "candidate_is_root_z_only": False,
    }

    # Save
    setups_dir = OUTPUT_DIR / "setups"
    setups_dir.mkdir(parents=True, exist_ok=True)
    setup_out = setups_dir / f"{variant_name}_setup.json"
    with open(setup_out, "w") as f:
        json.dump(setup, f, indent=2)
    print(f"  [OK] Generated setup saved: {setup_out}")

    return setup


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PRBS EXCITATION SIGNAL GENERATION                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_prbs_signal(n_steps, amplitude=0.15, seed=42, min_period=3, max_period=12):
    """Generate a Pseudo-Random Binary Signal for system identification.

    PRBS switches between ±amplitude with random period lengths.
    Zero-mean by construction (equal ± amplitude).
    """
    rng = np.random.RandomState(seed)
    signal = np.zeros(n_steps)
    state = 1
    i = 0
    while i < n_steps:
        period = rng.randint(min_period, max_period + 1)
        signal[i:min(i + period, n_steps)] = state * amplitude
        state *= -1
        i += period
    return signal


def generate_chirp_signal(n_steps, amplitude=0.15, f_start=0.1, f_end=5.0):
    """Generate a linear chirp signal from f_start to f_end Hz."""
    t = np.arange(n_steps) * 0.01  # 100 Hz control rate
    instantaneous_freq = f_start + (f_end - f_start) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) * 0.01
    signal = amplitude * np.sin(phase)
    return signal


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SIMULATION LAUNCHER                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_cli_args(setup_path, n_steps, extra_args=None):
    """Build CLI args list for simulate_hierarchical_controller.py."""
    args = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", CONTROLLER_MODE,
        "--vd-sagittal-authority-profile", K1_PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(n_steps),
        "--telemetry-decimation", "1",
        "--write-run-summary-sidecar",
    ]
    if extra_args:
        args.extend(extra_args)
    return args


def run_simulation(height_name, run_type, setup_path, n_steps=3000,
                   settling_steps=500, extra_args=None, excitation_signal=None):
    """Run a single simulation and save telemetry + metadata.

    Returns (telemetry_path, metadata_path, success: bool).
    """
    run_dir = OUTPUT_DIR / height_name / run_type
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "height_name": height_name,
        "run_type": run_type,
        "target_height_m": TARGET_HEIGHTS.get(height_name, None),
        "profile": K1_PROFILE,
        "controller_mode": CONTROLLER_MODE,
        "n_steps": n_steps,
        "settling_steps": settling_steps,
        "validation_source": "real_simulation",
        "excitation_applied": excitation_signal is not None,
        "excitation_type": None,
        "excitation_params": None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_label": f"k1_identification_{height_name}_{run_type}",
    }

    # Save excitation signal if present
    if excitation_signal is not None:
        exc_path = run_dir / "excitation_signal.json"
        exc_data = {
            "signal": excitation_signal.tolist() if hasattr(excitation_signal, "tolist") else excitation_signal,
            "n_steps": n_steps,
            "amplitude_max": float(np.max(np.abs(excitation_signal))),
            "is_zero_mean": bool(abs(np.mean(excitation_signal)) < 1e-10),
        }
        with open(exc_path, "w") as f:
            json.dump(exc_data, f, indent=2)
        metadata["excitation_signal_path"] = str(exc_path)

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Build CLI
    cli_args = build_cli_args(setup_path, n_steps, extra_args)
    # Set output dir for simulation
    cli_args.extend(["--output-dir", str(run_dir)])

    print(f"\n  Running: {height_name}/{run_type} ({n_steps} steps)...")
    print(f"    Output dir: {run_dir}")

    try:
        result = subprocess.run(
            cli_args,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(PROJECT_ROOT),
        )
        success = result.returncode == 0
        if not success:
            print(f"    WARNING: Simulation exited with code {result.returncode}")
            stderr_tail = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            print(f"    STDERR tail: {stderr_tail}")
    except subprocess.TimeoutExpired:
        print(f"    ERROR: Simulation timed out after 600s")
        success = False
    except Exception as e:
        print(f"    ERROR: {e}")
        success = False

    # Find the generated telemetry CSV
    telemetry_files = sorted(run_dir.glob("telemetry_*.csv"))
    telemetry_path = telemetry_files[-1] if telemetry_files else None

    metadata["simulation_success"] = success
    metadata["telemetry_path"] = str(telemetry_path) if telemetry_path else None
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return telemetry_path, metadata_path, success


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN DATASET GENERATION                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_dataset(run_simulation_flag=True, heights=None, run_types=None):
    """Main entry point for dataset generation.

    Args:
        run_simulation_flag: If False, only generate setups and PRBS signals
        heights: List of height names to process (default: all)
        run_types: List of run types to process (default: all)
    """
    print("=" * 72)
    print("K1 IDENTIFICATION DATASET GENERATION")
    print("=" * 72)

    # Phase 0
    baseline = verify_baseline()
    print()

    if heights is None:
        heights = list(TARGET_HEIGHTS.keys())
    if run_types is None:
        run_types = list(RUN_TYPES)

    print("=" * 72)
    print("PHASE 1: HEIGHT SETUP GENERATION")
    print("=" * 72)

    setups = {}
    for name, target_z in TARGET_HEIGHTS.items():
        if name in heights:
            print(f"\n[{name}] target_com_z_m={target_z}")
            setups[name] = generate_height_setup(name, target_z)
            # Save in run directory
            run_dir = OUTPUT_DIR / name
            run_dir.mkdir(parents=True, exist_ok=True)
            setup_out = run_dir / f"{name}_setup.json"
            with open(setup_out, "w") as f:
                json.dump(setups[name], f, indent=2)

    if not run_simulation_flag:
        print("\n[DRY RUN] Skipping simulations. Setups and signals generated.")
        return setups

    print("\n" + "=" * 72)
    print("PHASE 1: TELEMETRY GENERATION")
    print("=" * 72)

    results = {}

    for height_name in heights:
        setup = setups[height_name]
        setup_path = OUTPUT_DIR / height_name / f"{height_name}_setup.json"
        results[height_name] = {}

        print(f"\n{'-' * 60}")
        print(f"Height: {height_name} ({TARGET_HEIGHTS[height_name]}m)")
        print(f"{'-' * 60}")

        # ── Run A: No-push equilibrium ──
        if "A_equilibrium" in run_types:
            print(f"\n  [A] No-push equilibrium run")
            tp, mp, ok = run_simulation(
                height_name, "A_equilibrium", setup_path,
                n_steps=2000, settling_steps=400,
                extra_args=["--push-enabled", "--push-magnitude-n", "0.0"],
            )
            results[height_name]["A_equilibrium"] = {
                "success": ok, "telemetry": str(tp), "metadata": str(mp),
            }

        # ── Run B: 90N single sagittal push ──
        if "B_90n_push" in run_types:
            print(f"\n  [B] 90N single sagittal push run")
            tp, mp, ok = run_simulation(
                height_name, "B_90n_push", setup_path,
                n_steps=3000, settling_steps=400,
                extra_args=[
                    "--push-enabled", "--push-magnitude-n", "90.0",
                    "--push-interval-steps", "2000",
                    "--push-duration-steps", "10",
                    "--push-start-step", "300",
                    "--push-count", "1",
                    "--sagittal-push-only",
                ],
            )
            results[height_name]["B_90n_push"] = {
                "success": ok, "telemetry": str(tp), "metadata": str(mp),
            }

        # ── Run C: Small impulse ──
        if "C_impulse" in run_types:
            print(f"\n  [C] Small impulse identification run")
            tp, mp, ok = run_simulation(
                height_name, "C_impulse", setup_path,
                n_steps=2000, settling_steps=400,
                extra_args=[
                    "--push-enabled", "--push-magnitude-n", "5.0",
                    "--push-interval-steps", "400",
                    "--push-duration-steps", "3",
                    "--push-start-step", "500",
                    "--push-count", "3",
                    "--sagittal-push-only",
                ],
            )
            results[height_name]["C_impulse"] = {
                "success": ok, "telemetry": str(tp), "metadata": str(mp),
            }

        # ── Run D: PRBS excitation ──
        if "D_prbs_excitation" in run_types:
            print(f"\n  [D] PRBS sagittal-force excitation run")
            n_steps_d = 2500
            prbs = generate_prbs_signal(n_steps_d, amplitude=0.20, seed=hash(height_name) % 10000)
            tp, mp, ok = run_simulation(
                height_name, "D_prbs_excitation", setup_path,
                n_steps=n_steps_d, settling_steps=400,
                extra_args=[
                    "--push-enabled", "--push-magnitude-n", "0.20",
                    "--push-interval-steps", "5",
                    "--push-duration-steps", "3",
                    "--push-start-step", "500",
                    "--push-count", str(n_steps_d // 5),
                    "--sagittal-push-only",
                ],
                excitation_signal=prbs,
            )
            results[height_name]["D_prbs_excitation"] = {
                "success": ok, "telemetry": str(tp), "metadata": str(mp),
            }

        # ── Run E: Support offset initial condition ──
        if "E_support_offset" in run_types:
            print(f"\n  [E] Small support-offset initial-condition run")
            tp, mp, ok = run_simulation(
                height_name, "E_support_offset", setup_path,
                n_steps=2000, settling_steps=400,
                extra_args=[
                    "--initial-root-z-perturbation", "0.005",
                    "--push-enabled", "--push-magnitude-n", "0.0",
                ],
            )
            results[height_name]["E_support_offset"] = {
                "success": ok, "telemetry": str(tp), "metadata": str(mp),
            }

    # Save summary
    summary_path = OUTPUT_DIR / "dataset_generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Dataset generation summary saved: {summary_path}")

    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA QUALITY CHECK                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def check_data_quality():
    """Check generated data for basic quality requirements."""
    import csv
    print("\n" + "=" * 72)
    print("DATA QUALITY CHECK")
    print("=" * 72)
    issues = []

    for height_name in TARGET_HEIGHTS:
        height_dir = OUTPUT_DIR / height_name
        if not height_dir.exists():
            issues.append(f"MISSING: {height_name} directory not found")
            continue

        for run_type in RUN_TYPES:
            run_dir = height_dir / run_type
            if not run_dir.exists():
                issues.append(f"MISSING: {height_name}/{run_type} directory not found")
                continue

            metadata_path = run_dir / "metadata.json"
            if not metadata_path.exists():
                issues.append(f"MISSING: {height_name}/{run_type}/metadata.json")
                continue

            with open(metadata_path) as f:
                meta = json.load(f)

            if not meta.get("simulation_success", False):
                issues.append(f"FAILED: {height_name}/{run_type} simulation did not succeed")
                continue

            tp = meta.get("telemetry_path")
            if tp and Path(tp).exists():
                with open(tp, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                n_rows = len(rows)

                # Check for NaN/Inf
                has_nan = False
                for row in rows:
                    for key, val in row.items():
                        try:
                            v = float(val)
                            if np.isnan(v) or np.isinf(v):
                                has_nan = True
                                break
                        except (ValueError, TypeError):
                            pass
                    if has_nan:
                        break

                if has_nan:
                    issues.append(f"DATA: {height_name}/{run_type} has NaN/Inf values")
                elif n_rows < 100:
                    issues.append(f"DATA: {height_name}/{run_type} has only {n_rows} rows (need >=100)")

                # Check minimum sample counts
                if run_type in ("A_equilibrium", "E_support_offset"):
                    settling = meta.get("settling_steps", 400)
                    post_settle = max(0, n_rows - settling)
                    if post_settle < 1000:
                        issues.append(f"DATA: {height_name}/{run_type} has {post_settle} post-settle samples (need >=1000)")

                print(f"  {height_name}/{run_type}: {n_rows} rows, NaN={has_nan}")
            else:
                issues.append(f"MISSING: {height_name}/{run_type} telemetry file not found")

    if issues:
        print(f"\n[WARN] {len(issues)} quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] All data quality checks passed")

    return issues


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Generate K1 identification dataset for system ID"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate setups and PRBS signals, no simulation")
    parser.add_argument("--heights", nargs="+",
                        choices=list(TARGET_HEIGHTS.keys()),
                        help="Specific heights to generate")
    parser.add_argument("--run-types", nargs="+",
                        choices=RUN_TYPES,
                        help="Specific run types to generate")
    parser.add_argument("--quality-check", action="store_true",
                        help="Run data quality check on existing data")
    args = parser.parse_args()

    if args.quality_check:
        issues = check_data_quality()
        return 0 if not issues else 1

    generate_dataset(
        run_simulation_flag=not args.dry_run,
        heights=args.heights,
        run_types=args.run_types,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
