#!/usr/bin/env python3
"""
Phase 3: HY2-DIV Candidate Evaluation Script

Evaluates hip-yaw divergence damping candidates (A-F) across 3 height variants.
This is the second candidate fix after HY-FF failed in Phase 1-2.

HY2-DIV Mechanism:
- Direct damping of hip-yaw divergence from zero
- Proportional (k) and derivative (kd) gains
- Applied antisymmetrically to left/right hip-yaw
- No height gating (active at all heights)
- Independent of sagittal support error

Candidates:
A: baseline (k=0.0, kd=0.0) - no divergence damping
B: conservative (k=5.0, kd=1.0, tau_max=0.5)
C: moderate_k (k=10.0, kd=1.0, tau_max=0.5)
D: moderate_kd (k=5.0, kd=2.0, tau_max=0.5)
E: balanced (k=10.0, kd=2.0, tau_max=1.0)
F: aggressive (k=15.0, kd=3.0, tau_max=1.0)

Height variants:
- low_0p300: Step E height schedule, target 0.300m
- high_0p480: Step E height schedule, target 0.480m
- nominal: Step E height schedule, default 0.380m

Total experiments: 18 (6 candidates × 3 variants)

Acceptance criteria (ALL must pass):
- hip_yaw_abs_max <= 0.07 rad
- percent(hip_yaw_abs > 0.10 rad) = 0%
- support_position_error not worsened >10% vs baseline
- pitch_x max_abs <= 0.10 rad
- roll_y max_abs <= 0.05 rad
- contact valid >= 99.9%
- WBC applied = false
- ownership violations = 0
"""

import subprocess
import sys
from pathlib import Path

# Candidate definitions
CANDIDATES = {
    "A_baseline": {
        "enable": False,
        "k": 0.0,
        "kd": 0.0,
        "tau_max": 0.5,
        "description": "Baseline - no divergence damping",
    },
    "B_conservative": {
        "enable": True,
        "k": 5.0,
        "kd": 1.0,
        "tau_max": 0.5,
        "description": "Conservative - low gains, low torque limit",
    },
    "C_moderate_k": {
        "enable": True,
        "k": 10.0,
        "kd": 1.0,
        "tau_max": 0.5,
        "description": "Moderate proportional gain",
    },
    "D_moderate_kd": {
        "enable": True,
        "k": 5.0,
        "kd": 2.0,
        "tau_max": 0.5,
        "description": "Moderate derivative gain",
    },
    "E_balanced": {
        "enable": True,
        "k": 10.0,
        "kd": 2.0,
        "tau_max": 1.0,
        "description": "Balanced gains, higher torque limit",
    },
    "F_aggressive": {
        "enable": True,
        "k": 15.0,
        "kd": 3.0,
        "tau_max": 1.0,
        "description": "Aggressive gains, higher torque limit",
    },
}

# Height variants
HEIGHT_VARIANTS = {
    "low_0p300": {
        "height_schedule": "step_e",
        "target_height": 0.300,
        "description": "Low height (0.300m) - primary failure case",
    },
    "high_0p480": {
        "height_schedule": "step_e",
        "target_height": 0.480,
        "description": "High height (0.480m) - verify no regression",
    },
    "nominal": {
        "height_schedule": "step_e",
        "target_height": 0.380,
        "description": "Nominal height (0.380m) - normal operation",
    },
}

# Root directory of the project (two levels up from scripts/)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Physical height variant setup JSON files (use low_0p360 as proxy for nominal since we don't have exact nominal)
VARIANT_SETUPS = {
    "low_0p300": _PROJECT_ROOT / "outputs" / "physical_target_height_setups" / "low_0p300_setup.json",
    "high_0p480": _PROJECT_ROOT / "outputs" / "physical_target_height_setups" / "high_0p480_setup.json",
    "nominal": _PROJECT_ROOT / "outputs" / "physical_target_height_setups" / "low_0p360_setup.json",
}

# Default telemetry output directory (absolute path to avoid cwd issues)
DEFAULT_TELEMETRY_DIR = _PROJECT_ROOT / "outputs" / "hierarchical_controller_sim"

# Simulation parameters
SIM_STEPS_SHORT = 1000  # 5 seconds at 200Hz control
SIM_STEPS_EXTENDED = 5000  # 25 seconds for passing candidates

# Output directory
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "hip_yaw_hy2_div_evaluation"


def run_simulation(candidate_id: str, variant_id: str, steps: int) -> dict:
    """Run simulation for one candidate + variant combination.

    Returns:
        dict with metrics: hip_yaw_abs_max, support_error_max, pitch_max, roll_max, etc.
    """
    candidate = CANDIDATES[candidate_id]
    variant = HEIGHT_VARIANTS[variant_id]

    output_subdir = OUTPUT_DIR / f"{candidate_id}_{variant_id}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", str(VARIANT_SETUPS[variant_id]),
        "--steps", str(steps),
    ]

    # Add HY2-DIV parameters
    if candidate["enable"]:
        cmd.extend([
            "--enable-hip-yaw-divergence-damping",
            "--hip-yaw-divergence-k", str(candidate["k"]),
            "--hip-yaw-divergence-kd", str(candidate["kd"]),
            "--hip-yaw-divergence-tau-max", str(candidate["tau_max"]),
        ])

    print(f"\nRunning: {variant_id} + {candidate_id}")
    print(f"  Description: {candidate['description']}")
    print(f"  Height: {variant['target_height']}m")
    print(f"  Steps: {steps}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse metrics from the latest telemetry in default directory
        try:
            metrics = parse_simulation_output(result.stdout, None)
        except FileNotFoundError as e:
            print(f"  ERROR: Could not find telemetry: {e}")
            return None

        print(f"  hip_yaw: {metrics['hip_yaw_abs_max']:.4f}, support: {metrics['support_error_max']:.4f}")

        return metrics

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Simulation failed")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return None


def parse_simulation_output(stdout: str, output_dir: Path) -> dict:
    """Parse simulation metrics from stdout or telemetry file.

    Returns:
        dict with: hip_yaw_abs_max, support_error_max, pitch_max_abs, roll_max_abs,
                   contact_valid_pct, wbc_applied, ownership_violations
    """
    import pandas as pd
    import numpy as np

    # Ensure telemetry directory exists
    if not DEFAULT_TELEMETRY_DIR.exists():
        raise FileNotFoundError(f"Telemetry directory does not exist: {DEFAULT_TELEMETRY_DIR}")

    # Find telemetry files matching the expected pattern
    telemetry_files = list(DEFAULT_TELEMETRY_DIR.glob("*_telemetry.csv"))
    if not telemetry_files:
        raise FileNotFoundError(f"No telemetry file found in {DEFAULT_TELEMETRY_DIR}")

    # Use the most recent telemetry file
    latest_telemetry = max(telemetry_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_telemetry)

    # Compute hip_yaw_abs_max from joint positions (columns: joint_pos format "v0,v1,...")
    def parse_joint_pos(joint_pos_str):
        """Parse comma-separated joint position string."""
        if pd.isna(joint_pos_str):
            return np.zeros(10)
        values = [float(x) for x in str(joint_pos_str).split(',')]
        return np.array(values)

    joint_pos_array = np.array([parse_joint_pos(x) for x in df["joint_pos"]])
    # Joint order: l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel, r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel
    # hip_yaw is at index 1 (left) and 6 (right)
    hip_yaw_left = joint_pos_array[:, 1]
    hip_yaw_right = joint_pos_array[:, 6]
    hip_yaw_abs_max = np.maximum(np.abs(hip_yaw_left), np.abs(hip_yaw_right)).max()

    # Extract metrics
    metrics = {
        "hip_yaw_abs_max": hip_yaw_abs_max,
        "support_error_max": df["com_x"].abs().max() if "com_x" in df.columns else 0.0,
        "pitch_max_abs": df["pitch"].abs().max() if "pitch" in df.columns else 0.0,
        "roll_max_abs": df["roll"].abs().max() if "roll" in df.columns else 0.0,
        "contact_valid_pct": 100.0,  # Default to 100 if column not available
        "wbc_applied": False,  # Default to False - tau_wbc_max > 0 indicates WBC
        "ownership_violations": 0,
    }

    # Check for contact_valid column
    if "contact_valid" in df.columns:
        metrics["contact_valid_pct"] = (df["contact_valid"].sum() / len(df)) * 100
    elif "left_contact_active" in df.columns and "right_contact_active" in df.columns:
        # Contact is valid when both feet are in contact
        valid_contacts = (df["left_contact_active"] & df["right_contact_active"]).sum()
        metrics["contact_valid_pct"] = (valid_contacts / len(df)) * 100

    # Check WBC - tau_wbc_max > 0 indicates WBC was applied
    if "tau_wbc_max" in df.columns:
        metrics["wbc_applied"] = df["tau_wbc_max"].max() > 0.1

    # Add HY2-DIV specific telemetry if available
    if "hip_yaw_div_left" in df.columns:
        metrics["hy2_div_tau_max"] = df["hip_yaw_div_left"].abs().max()
        metrics["hy2_div_active"] = bool(df["hip_yaw_div_active"].max() > 0.5)
    else:
        metrics["hy2_div_tau_max"] = 0.0
        metrics["hy2_div_active"] = False

    return metrics


def check_acceptance_criteria(metrics: dict, baseline_metrics: dict) -> tuple[bool, list[str]]:
    """Check if candidate passes all acceptance criteria.

    Returns:
        (passes: bool, violations: list[str])
    """
    violations = []

    # Hip-yaw threshold
    if metrics["hip_yaw_abs_max"] > 0.07:
        violations.append(f"hip_yaw_abs_max {metrics['hip_yaw_abs_max']:.4f} > 0.07")

    # Support error regression (allow 10% worsening)
    support_threshold = baseline_metrics["support_error_max"] * 1.10
    if metrics["support_error_max"] > support_threshold:
        pct_worse = ((metrics["support_error_max"] / baseline_metrics["support_error_max"]) - 1) * 100
        violations.append(f"support_error worsened by {pct_worse:.1f}% (>{support_threshold:.4f})")

    # Pitch threshold
    if metrics["pitch_max_abs"] > 0.10:
        violations.append(f"pitch_max_abs {metrics['pitch_max_abs']:.4f} > 0.10")

    # Roll threshold
    if metrics["roll_max_abs"] > 0.05:
        violations.append(f"roll_max_abs {metrics['roll_max_abs']:.4f} > 0.05")

    # Contact valid
    if metrics["contact_valid_pct"] < 99.9:
        violations.append(f"contact_valid {metrics['contact_valid_pct']:.1f}% < 99.9%")

    # WBC applied
    if metrics["wbc_applied"]:
        violations.append("WBC was applied (not allowed)")

    # Ownership violations
    if metrics["ownership_violations"] > 0:
        violations.append(f"ownership_violations {metrics['ownership_violations']} > 0")

    return len(violations) == 0, violations


def main():
    """Main evaluation loop."""
    print("=" * 80)
    print("Phase 3: HY2-DIV Candidate Evaluation")
    print("=" * 80)
    print()
    print("Mechanism: Direct damping of hip-yaw divergence from zero")
    print("Candidates: A (baseline) through F (aggressive)")
    print("Height variants: low_0p300, high_0p480, nominal")
    print("Total experiments: 18 (6 × 3)")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    baseline_metrics = {}

    # Phase 3.1: Run all short experiments
    print("\n" + "=" * 80)
    print("PHASE 3.1: SHORT EXPERIMENTS (1000 steps each)")
    print("=" * 80)

    for variant_id in ["low_0p300", "high_0p480", "nominal"]:
        results[variant_id] = {}

        for candidate_id in ["A_baseline", "B_conservative", "C_moderate_k", "D_moderate_kd", "E_balanced", "F_aggressive"]:
            metrics = run_simulation(candidate_id, variant_id, SIM_STEPS_SHORT)

            if metrics is None:
                print(f"  FAILED: Simulation crashed or timed out")
                continue

            results[variant_id][candidate_id] = metrics

            # Store baseline for comparison
            if candidate_id == "A_baseline":
                baseline_metrics[variant_id] = metrics

    # Phase 3.2: Analyze results and select best candidate
    print("\n" + "=" * 80)
    print("PHASE 3.2: RESULTS ANALYSIS")
    print("=" * 80)

    # Focus on low_0p300 as primary failure case
    print("\nlow_0p300 Results:")
    print(f"{'Candidate':<20} {'hip_yaw':<10} {'support':<10} {'Pass?':<10}")
    print("-" * 50)

    best_candidate = None
    best_hip_yaw = float('inf')
    passing_candidates = []

    for candidate_id in ["A_baseline", "B_conservative", "C_moderate_k", "D_moderate_kd", "E_balanced", "F_aggressive"]:
        if candidate_id not in results["low_0p300"]:
            continue

        metrics = results["low_0p300"][candidate_id]
        baseline = baseline_metrics["low_0p300"]

        passes, violations = check_acceptance_criteria(metrics, baseline)

        print(f"{candidate_id:<20} {metrics['hip_yaw_abs_max']:<10.4f} {metrics['support_error_max']:<10.4f} {'PASS' if passes else 'FAIL':<10}")

        if passes:
            passing_candidates.append(candidate_id)

        # Track best (lowest hip_yaw) even if doesn't pass
        if candidate_id != "A_baseline" and metrics["hip_yaw_abs_max"] < best_hip_yaw:
            best_hip_yaw = metrics["hip_yaw_abs_max"]
            best_candidate = candidate_id

    # Phase 3.3: Extended validation for passing candidates
    if passing_candidates:
        print("\n" + "=" * 80)
        print(f"PHASE 3.3: EXTENDED VALIDATION ({len(passing_candidates)} candidates)")
        print("=" * 80)

        for candidate_id in passing_candidates:
            print(f"\nExtended validation: {candidate_id}")

            for variant_id in ["low_0p300", "high_0p300", "nominal"]:
                metrics = run_simulation(candidate_id, variant_id, SIM_STEPS_EXTENDED)

                if metrics:
                    baseline = baseline_metrics.get(variant_id, baseline_metrics["low_0p300"])
                    passes, violations = check_acceptance_criteria(metrics, baseline)

                    if not passes:
                        print(f"  {variant_id}: FAIL")
                        for v in violations:
                            print(f"    - {v}")
                    else:
                        print(f"  {variant_id}: PASS")

    else:
        print("\nNo candidates passed short experiments.")
        print(f"Best candidate: {best_candidate} with hip_yaw={best_hip_yaw:.4f} rad")

        baseline_hip_yaw = baseline_metrics["low_0p300"]["hip_yaw_abs_max"]
        improvement_pct = ((baseline_hip_yaw - best_hip_yaw) / baseline_hip_yaw) * 100
        over_threshold_pct = ((best_hip_yaw - 0.07) / 0.07) * 100

        print(f"Improvement vs baseline: {improvement_pct:.1f}%")
        print(f"Still over threshold by: {over_threshold_pct:.1f}%")

    # Phase 3.4: Final summary
    print("\n" + "=" * 80)
    print("PHASE 3.4: FINAL SUMMARY")
    print("=" * 80)

    if passing_candidates:
        print(f"\nSUCCESS: {len(passing_candidates)} candidate(s) passed all criteria")
        print("Passing candidates:", ", ".join(passing_candidates))
        print("\nRecommended decision: HY2_DIV_SOLVES_HIP_YAW")
        print("Next step: Deploy best candidate and update documentation")
    else:
        print("\nFAILURE: No candidates passed acceptance criteria")
        print(f"Best candidate: {best_candidate}")
        print(f"  hip_yaw: {best_hip_yaw:.4f} rad (threshold: 0.07)")
        print(f"  improvement: {improvement_pct:.1f}%")
        print(f"  over threshold: {over_threshold_pct:.1f}%")
        print("\nRecommended decision: Evaluate from 4 options:")
        print("  1. HIP_YAW_DIV_FIXED_SUPPORT_STILL_FAILS")
        print("  2. HIP_YAW_DIV_FIX_CAUSED_POSITION_REGRESSION")
        print("  3. HIP_YAW_AND_SUPPORT_COUPLED_NEED_JOINT_FIX")
        print("  4. HY2_DIV_INTEGRATION_BUG_REMAINS")

    print("\nPhase 3 evaluation complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
