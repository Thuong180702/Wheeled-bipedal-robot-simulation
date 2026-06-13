#!/usr/bin/env python3
"""
Evaluate Pitch-Safe Joint Sagittal-Yaw Fix Candidates (J2a-J2d)

Progressive evaluation protocol: J2a→J2b→J2c→J2d, stop at first pass.

Candidates designed after audit showing position_authority_induces_pitch_overshoot.
Target: preserve support/hip-yaw improvements while keeping pitch < 0.10 rad.

Stop-at-first-pass: First candidate to pass all phases is selected.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any


OUTPUT_DIR = Path("outputs/pitch_safe_joint_sagittal_yaw_fix")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Acceptance gates (strict, no relaxation)
STEP_E_GATES = {
    "support_position_error_max_abs": 0.15,
    "hip_yaw_abs_max": 0.07,
    "pitch_x_max_abs": 0.10,
    "roll_y_max_abs": 0.05,
    "final_height_error": 0.02,
    "contact_valid_percent": 99.9,
}

STEP_C_GATES = {
    **STEP_E_GATES,
    "height_recovered": True,
}

CANDIDATES = {
    "J2a": {
        "profile": "J2a",
        "description": "Conservative position cap (k_pos=60, max_tau=4.5, k_vel=22)",
        "rationale": "50% authority increase vs baseline, safest pitch profile"
    },
    "J2b": {
        "profile": "J2b",
        "description": "Balanced authority (k_pos=65, max_tau=5.0, k_vel=25)",
        "rationale": "63-67% authority increase, balanced approach"
    },
    "J2c": {
        "profile": "J2c",
        "description": "Velocity damping priority (k_pos=60, max_tau=4.5, k_vel=28)",
        "rationale": "Conservative position + aggressive velocity damping"
    },
    "J2d": {
        "profile": "J2d",
        "description": "Torque cap priority (k_pos=70, max_tau=4.5, k_vel=25)",
        "rationale": "Higher stiffness + capped peak torque"
    },
}


def run_simulation(
    height_setup: str,
    step_type: str,
    num_steps: int,
    profile: str,
    output_name: str,
    dry_run: bool = False,
) -> Tuple[bool, Dict[str, Any], str]:
    """Run a single simulation and return pass/fail, metrics, telemetry path."""

    output_subdir = OUTPUT_DIR / output_name
    output_subdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", height_setup,
        "--steps", str(num_steps),
        "--vd-sagittal-authority-profile", profile,
    ]

    print(f"  Running: {output_name} ({num_steps} steps, profile={profile})...")

    if dry_run:
        print(f"  [DRY-RUN] Command: {' '.join(cmd)}")
        return True, {}, ""

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        print(f"  [ERROR] Simulation failed: {result.stderr}")
        return False, {}, ""

    # Find auto-generated telemetry file
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(sim_output_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not telemetry_files:
        print(f"  [ERROR] No telemetry file found in {sim_output_dir}")
        return False, {}, ""

    # Copy most recent telemetry file to output directory
    latest_telemetry = telemetry_files[0]
    telemetry_path = output_subdir / "telemetry.csv"
    import shutil
    shutil.copy(latest_telemetry, telemetry_path)

    df = pd.read_csv(telemetry_path)

    metrics = compute_metrics(df, step_type)
    passed = check_gates(metrics, step_type)

    return passed, metrics, str(telemetry_path)


def compute_metrics(df: pd.DataFrame, step_type: str) -> Dict[str, Any]:
    """Compute acceptance metrics from telemetry."""

    metrics = {}

    # Support position error
    if "sagittal_position_error_m" in df.columns:
        support_error = df["sagittal_position_error_m"].values
        metrics["support_position_error_max_abs"] = float(abs(support_error).max())

    # Hip-yaw
    if "hip_yaw_abs_max" in df.columns:
        metrics["hip_yaw_abs_max"] = float(df["hip_yaw_abs_max"].max())

    # Pitch
    if "pitch_x" in df.columns:
        metrics["pitch_x_max_abs"] = float(abs(df["pitch_x"].values).max())
    elif "robot_pitch_x" in df.columns:
        metrics["pitch_x_max_abs"] = float(abs(df["robot_pitch_x"].values).max())

    # Roll
    if "roll_y" in df.columns:
        metrics["roll_y_max_abs"] = float(abs(df["roll_y"].values).max())
    elif "robot_roll_y" in df.columns:
        metrics["roll_y_max_abs"] = float(abs(df["robot_roll_y"].values).max())

    # Height error
    if "height_error_m" in df.columns:
        metrics["final_height_error"] = float(abs(df["height_error_m"].values[-1]))

    # Contact validity
    if "contact_force_valid" in df.columns:
        metrics["contact_valid_percent"] = float(100.0 * df["contact_force_valid"].mean())

    # Step C specific
    if step_type == "step_c":
        # Height recovery check
        if "height_error_m" in df.columns:
            final_height_error = abs(df["height_error_m"].values[-1])
            metrics["height_recovered"] = final_height_error <= 0.02

    # WBC/ownership invariants
    if "applied_wbc_contribution_norm" in df.columns:
        metrics["wbc_applied_max"] = float(df["applied_wbc_contribution_norm"].max())
    elif "tau_wbc_norm" in df.columns:
        metrics["wbc_applied_max"] = float(df["tau_wbc_norm"].max())
    else:
        metrics["wbc_applied_max"] = 0.0

    if "hidden_torque_norm" in df.columns:
        metrics["hidden_torque_max"] = float(df["hidden_torque_norm"].max())
    else:
        metrics["hidden_torque_max"] = 0.0

    if "ownership_violation_count" in df.columns:
        metrics["ownership_violations_max"] = int(df["ownership_violation_count"].max())
    else:
        metrics["ownership_violations_max"] = 0

    return metrics


def check_gates(metrics: Dict[str, Any], step_type: str) -> bool:
    """Check if metrics pass acceptance gates."""

    gates = STEP_C_GATES if step_type == "step_c" else STEP_E_GATES

    failures = []

    for gate_name, threshold in gates.items():
        if gate_name not in metrics:
            print(f"    [WARN] Missing metric: {gate_name}")
            failures.append(f"{gate_name}: MISSING")
            continue

        value = metrics[gate_name]

        if isinstance(threshold, bool):
            if value != threshold:
                print(f"    [FAIL] {gate_name}: {value} (expected {threshold})")
                failures.append(f"{gate_name}: {value} != {threshold}")
        else:
            if value > threshold:
                print(f"    [FAIL] {gate_name}: {value:.4f} > {threshold:.4f}")
                failures.append(f"{gate_name}: {value:.4f} > {threshold:.4f}")

    # Invariant checks
    if metrics.get("wbc_applied_max", 0.0) > 1.0:
        print(f"    [FAIL] WBC applied: {metrics['wbc_applied_max']:.2f} > 1.0")
        failures.append(f"WBC applied: {metrics['wbc_applied_max']:.2f}")

    if metrics.get("hidden_torque_max", 0.0) > 1.0:
        print(f"    [FAIL] Hidden torque: {metrics['hidden_torque_max']:.2f} > 1.0")
        failures.append(f"Hidden torque: {metrics['hidden_torque_max']:.2f}")

    if metrics.get("ownership_violations_max", 0) > 0:
        print(f"    [FAIL] Ownership violations: {metrics['ownership_violations_max']} > 0")
        failures.append(f"Ownership violations: {metrics['ownership_violations_max']}")

    return len(failures) == 0


def evaluate_candidate(candidate_name: str, candidate_info: Dict, dry_run: bool = False) -> Dict[str, Any]:
    """Evaluate one candidate through full protocol."""

    print(f"\n{'='*80}")
    print(f"Evaluating {candidate_name}: {candidate_info['description']}")
    print(f"Rationale: {candidate_info['rationale']}")
    print(f"{'='*80}\n")

    profile = candidate_info["profile"]
    results = {
        "candidate": candidate_name,
        "profile": profile,
        "description": candidate_info["description"],
        "rationale": candidate_info["rationale"],
        "phases_completed": {},
        "final_decision": "NOT_EVALUATED",
    }

    # Phase 1: low_0p300 Step E 1000
    phase_name = "phase_1_low_0p300_step_e_1000"
    print(f"[Phase 1] low_0p300 Step E 1000")
    passed, metrics, telem_path = run_simulation(
        "outputs/physical_target_height_setups/low_0p300_setup.json",
        "step_e",
        1000,
        profile,
        f"{candidate_name}_low_0p300_step_e_1000",
        dry_run=dry_run,
    )
    results["phases_completed"][phase_name] = {
        "passed": passed,
        "metrics": metrics,
        "telemetry": telem_path,
    }

    if not passed:
        print(f"  [FAIL] {candidate_name} failed Phase 1")
        results["final_decision"] = "FAILED_PHASE_1_LOW_0P300_STEP_E_1000"
        return results

    print(f"  [PASS] Phase 1")
    print(f"    support_error: {metrics.get('support_position_error_max_abs', 0.0):.4f} m")
    print(f"    hip_yaw: {metrics.get('hip_yaw_abs_max', 0.0):.4f} rad")
    print(f"    pitch: {metrics.get('pitch_x_max_abs', 0.0):.4f} rad")

    # Phase 2: low_0p300 Step E 5000
    phase_name = "phase_2_low_0p300_step_e_5000"
    print(f"\n[Phase 2] low_0p300 Step E 5000")
    passed, metrics, telem_path = run_simulation(
        "outputs/physical_target_height_setups/low_0p300_setup.json",
        "step_e",
        5000,
        profile,
        f"{candidate_name}_low_0p300_step_e_5000",
        dry_run=dry_run,
    )
    results["phases_completed"][phase_name] = {
        "passed": passed,
        "metrics": metrics,
        "telemetry": telem_path,
    }

    if not passed:
        print(f"  [FAIL] {candidate_name} failed Phase 2")
        results["final_decision"] = "FAILED_PHASE_2_LOW_0P300_STEP_E_5000"
        return results

    print(f"  [PASS] Phase 2")

    # Phase 3: high_0p480 Step E 5000 (regression check)
    phase_name = "phase_3_high_0p480_step_e_5000"
    print(f"\n[Phase 3] high_0p480 Step E 5000 (regression check)")
    passed, metrics, telem_path = run_simulation(
        "outputs/physical_target_height_setups/high_0p480_setup.json",
        "step_e",
        5000,
        profile,
        f"{candidate_name}_high_0p480_step_e_5000",
        dry_run=dry_run,
    )
    results["phases_completed"][phase_name] = {
        "passed": passed,
        "metrics": metrics,
        "telemetry": telem_path,
    }

    if not passed:
        print(f"  [FAIL] {candidate_name} failed Phase 3 (high height regression)")
        results["final_decision"] = "FAILED_PHASE_3_HIGH_0P480_REGRESSION"
        return results

    print(f"  [PASS] Phase 3 (no regression at high height)")

    # If we reach here, candidate passed first 3 phases
    print(f"\n[SUCCESS] {candidate_name} passed Phases 1-3!")
    results["final_decision"] = "BOUNDARY_RANGE_PASS"

    return results


def main():
    import sys

    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("\n" + "="*80)
        print("DRY-RUN: Verifying candidate commands")
        print("="*80 + "\n")

        for candidate_name in ["J2a", "J2b", "J2c", "J2d"]:
            candidate_info = CANDIDATES[candidate_name]
            print(f"\n[{candidate_name}] {candidate_info['description']}")
            run_simulation(
                "outputs/physical_target_height_setups/low_0p300_setup.json",
                "step_e",
                1000,
                candidate_info["profile"],
                f"{candidate_name}_low_0p300_step_e_1000",
                dry_run=True,
            )

        print("\n" + "="*80)
        print("DRY-RUN complete. Review commands above.")
        print("="*80)
        return 0

    print("\n" + "="*80)
    print("Pitch-Safe Joint Sagittal-Yaw Fix Evaluation")
    print("="*80 + "\n")

    print("Evaluation protocol: J2a->J2b->J2c->J2d, stop at first pass\n")
    print("Target: All three gates pass at low_0p300")
    print("  - support_position_error <= 0.15 m")
    print("  - hip_yaw_abs_max <= 0.07 rad")
    print("  - pitch_x_max_abs <= 0.10 rad")
    print()

    all_results = []
    selected_candidate = None

    for candidate_name in ["J2a", "J2b", "J2c", "J2d"]:
        candidate_info = CANDIDATES[candidate_name]

        results = evaluate_candidate(candidate_name, candidate_info, dry_run=False)
        all_results.append(results)

        # Check if this candidate passed
        if results["final_decision"] == "BOUNDARY_RANGE_PASS":
            selected_candidate = candidate_name
            print(f"\n{'='*80}")
            print(f"SELECTED CANDIDATE: {candidate_name}")
            print(f"{'='*80}\n")
            break

    # Save summary
    summary = {
        "evaluation_date": "2026-06-05",
        "selected_candidate": selected_candidate,
        "all_results": all_results,
        "audit_classification": "position_authority_induces_pitch_overshoot",
        "design_strategy": "reduce position authority while maintaining velocity damping",
    }

    summary_path = OUTPUT_DIR / "pitch_safe_candidate_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Summary saved: {summary_path}")

    if selected_candidate is None:
        print("\n[RESULT] No candidate passed all phases.")
        print("CLASSIFICATION: PITCH_SAFE_JOINT_FIX_REQUIRED")
        print("\nAll J2a-J2d candidates failed. Possible next steps:")
        print("  1. Relax pitch gate to 0.12-0.15 rad for low_0p300 only")
        print("  2. Investigate pitch-aware position term")
        print("  3. Consider whether 0.300m is within operational envelope")
        return 1
    else:
        print(f"\n[RESULT] Selected: {selected_candidate}")
        print("CLASSIFICATION: BOUNDARY_RANGE_PASS")
        print(f"\nNext: Proceed to Phase 4+ (Step C, grid, five-variant regression)")
        return 0


if __name__ == "__main__":
    exit(main())
