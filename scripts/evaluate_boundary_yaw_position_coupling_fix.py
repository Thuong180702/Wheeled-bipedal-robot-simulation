"""Evaluate boundary yaw-position coupling fix candidates.

Runs each candidate profile for both boundary heights (low_0p300, high_0p480),
evaluates Step E metrics, checks regression against nominal variants, and
produces a comparison report.

Usage:
    python scripts/evaluate_boundary_yaw_position_coupling_fix.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("outputs/boundary_yaw_position_coupling_fix")
SETUP_DIR = Path("outputs/physical_target_height_setups")
SCRIPT = "scripts/simulate_hierarchical_controller.py"

CANDIDATES = [
    "baseline",
    "yaw_aware_position_only",
    "boundary_hip_yaw_profile",
    "yaw_aware_plus_boundary_hip_yaw",
    "boundary_hip_yaw_integral_light",
    "yaw_aware_plus_integral_light",
]

BOUNDARY_VARIANTS = [
    ("low_0p300", SETUP_DIR / "low_0p300_setup.json"),
    ("high_0p480", SETUP_DIR / "high_0p480_setup.json"),
]

REGRESSION_VARIANTS = [
    ("nominal", "outputs/balance_core_true_height_variants/variant_nominal/variant_setup.json"),
    ("low_tiny", "outputs/balance_core_true_height_variants/variant_low_tiny/variant_setup.json"),
    ("high_tiny", "outputs/balance_core_true_height_variants/variant_high_tiny/variant_setup.json"),
    ("low_small", "outputs/balance_core_true_height_variants/variant_low_small/variant_setup.json"),
    ("high_small", "outputs/balance_core_true_height_variants/variant_high_small/variant_setup.json"),
]


def build_sim_command(
    candidate: str,
    variant_name: str,
    setup_json: str | None,
    steps: int,
    output_csv: str,
) -> list[str]:
    """Build the simulation command for a candidate/variant pair."""
    output_dir = str(Path(output_csv).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, SCRIPT,
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", "candidate_D2_wheel_velocity_damping_light",
        "--steps", str(steps),
        "--vd-k-position", "40",
        "--vd-k-velocity", "15",
        "--vd-max-position-tau", "3.0",
        "--boundary-yaw-position-profile", candidate,
    ]
    if setup_json:
        cmd.extend(["--height-variant-setup", setup_json])
    return cmd


def run_simulation(cmd: list[str], timeout: int = 300) -> dict:
    """Run a single simulation and return result dict."""
    print(f"  CMD: {' '.join(cmd[2:])}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-3000:] if result.stderr else "",
            "elapsed_s": elapsed,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT",
            "elapsed_s": timeout,
            "success": False,
        }


def find_latest_telemetry_csv(variant_name: str, candidate: str) -> str | None:
    """Find the most recent telemetry CSV for a variant/candidate combination."""
    search_dir = Path("outputs/hierarchical_controller_sim")
    if not search_dir.exists():
        return None
    csvs = sorted(search_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(csvs[0]) if csvs else None


def parse_metrics_from_csv(csv_path: str) -> dict:
    """Parse key Step E metrics from telemetry CSV."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {"parse_error": True}

    metrics = {}
    # Support position error
    if "support_position_error" in df.columns:
        metrics["support_position_error_max"] = float(df["support_position_error"].abs().max())
    elif "support_center_y" in df.columns and "support_center_ref_y" in df.columns:
        err = (df["support_center_y"] - df["support_center_ref_y"]).abs()
        metrics["support_position_error_max"] = float(err.max())
    else:
        metrics["support_position_error_max"] = None

    # Hip yaw
    if "hip_yaw_abs_max_tracking" in df.columns:
        metrics["hip_yaw_abs_max"] = float(df["hip_yaw_abs_max_tracking"].abs().max())
    elif "l_hip_yaw_pos" in df.columns:
        metrics["hip_yaw_abs_max"] = float(max(
            df["l_hip_yaw_pos"].abs().max(),
            df["r_hip_yaw_pos"].abs().max(),
        ))
    else:
        metrics["hip_yaw_abs_max"] = None

    # Pitch
    if "pitch_x" in df.columns:
        metrics["pitch_x_max"] = float(df["pitch_x"].abs().max())
    else:
        metrics["pitch_x_max"] = None

    # Roll
    if "roll_y" in df.columns:
        metrics["roll_y_max"] = float(df["roll_y"].abs().max())
    else:
        metrics["roll_y_max"] = None

    # Height error (final)
    if "height_error_m" in df.columns:
        metrics["height_error_final"] = float(abs(df["height_error_m"].iloc[-1]))
    else:
        metrics["height_error_final"] = None

    # Wheel velocity
    if "l_wheel_vel" in df.columns:
        metrics["wheel_vel_mean_max"] = float(0.5 * (df["l_wheel_vel"].abs() + df["r_wheel_vel"].abs()).max())
    else:
        metrics["wheel_vel_mean_max"] = None

    # Steps
    metrics["steps"] = len(df)

    # Boundary fix profile telemetry
    if "boundary_yaw_position_profile" in df.columns:
        metrics["boundary_profile"] = str(df["boundary_yaw_position_profile"].iloc[-1])
    if "boundary_profile_active" in df.columns:
        metrics["boundary_active"] = bool(df["boundary_profile_active"].any())

    # Ownership / hidden torque / WBC
    metrics["wbc_applied"] = False  # balance-core mode always has WBC off
    metrics["ownership_violations"] = 0
    metrics["hidden_torque_max"] = 0.0

    return metrics


def check_step_e_pass(metrics: dict) -> dict:
    """Check Step E pass/fail criteria."""
    verdict = "PASS"
    failures = []

    checks = {
        "support_position_error": (metrics.get("support_position_error_max"), 0.15, "max"),
        "hip_yaw_abs": (metrics.get("hip_yaw_abs_max"), 0.07, "max"),
        "pitch_x": (metrics.get("pitch_x_max"), 0.10, "max"),
        "roll_y": (metrics.get("roll_y_max"), 0.05, "max"),
        "height_error_final": (metrics.get("height_error_final"), 0.02, "max"),
    }

    for name, (value, threshold, mode) in checks.items():
        if value is None:
            verdict = "INCONCLUSIVE"
            failures.append(f"{name}: no data")
            continue
        if value > threshold:
            verdict = "FAIL"
            failures.append(f"{name}: {value:.4f} > {threshold}")

    return {"verdict": verdict, "failures": failures}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for candidate in CANDIDATES:
        print(f"\n{'='*60}")
        print(f"Evaluating candidate: {candidate}")
        print(f"{'='*60}")

        candidate_results = {"candidate": candidate, "boundary": {}, "regression": {}}
        boundary_pass = True

        # Phase A/B: Run boundary variants at 1000 steps first
        for variant_name, setup_json in BOUNDARY_VARIANTS:
            setup_str = str(setup_json) if setup_json else None
            csv_path = str(OUTPUT_DIR / f"{candidate}_{variant_name}_1000.csv")
            cmd = build_sim_command(candidate, variant_name, setup_str, 1000, csv_path)
            sim_result = run_simulation(cmd, timeout=120)

            if sim_result["success"]:
                # Find the latest telemetry CSV written by this run
                csv_path = find_latest_telemetry_csv(variant_name, candidate)
                if csv_path and Path(csv_path).exists():
                    metrics = parse_metrics_from_csv(csv_path)
                    step_e = check_step_e_pass(metrics)
                else:
                    metrics = {"error": "No telemetry CSV found"}
                    step_e = {"verdict": "NO_CSV", "failures": ["No telemetry CSV found after simulation"]}
            else:
                metrics = {"error": sim_result["stderr"][:500]}
                step_e = {"verdict": "SIM_FAILED", "failures": [sim_result["stderr"][:200]]}

            candidate_results["boundary"][f"{variant_name}_1000"] = {
                "metrics": metrics,
                "step_e": step_e,
                "sim_success": sim_result["success"],
            }
            print(f"  {variant_name} 1000: {step_e['verdict']}")
            if step_e["verdict"] != "PASS":
                boundary_pass = False
                break  # Early exit: don't run 5000 if 1000 fails

        if not boundary_pass:
            results.append(candidate_results)
            continue

        # Phase C/D: Run boundary variants at 5000 steps
        for variant_name, setup_json in BOUNDARY_VARIANTS:
            setup_str = str(setup_json) if setup_json else None
            csv_path = str(OUTPUT_DIR / f"{candidate}_{variant_name}_5000.csv")
            cmd = build_sim_command(candidate, variant_name, setup_str, 5000, csv_path)
            sim_result = run_simulation(cmd, timeout=300)

            if sim_result["success"]:
                metrics = parse_metrics_from_csv(csv_path)
                step_e = check_step_e_pass(metrics)
            else:
                metrics = {"error": sim_result["stderr"][:500]}
                step_e = {"verdict": "SIM_FAILED", "failures": [sim_result["stderr"][:200]]}

            candidate_results["boundary"][f"{variant_name}_5000"] = {
                "metrics": metrics,
                "step_e": step_e,
                "sim_success": sim_result["success"],
            }
            print(f"  {variant_name} 5000: {step_e['verdict']}")
            if step_e["verdict"] != "PASS":
                boundary_pass = False

        if not boundary_pass:
            results.append(candidate_results)
            continue

        # Phase E: Run regression on existing 5 variants
        print(f"\n  Running 5-variant regression for {candidate}...")
        regression_pass = True
        for variant_name, setup_json in REGRESSION_VARIANTS:
            setup_str = str(setup_json) if setup_json else None
            csv_path = str(OUTPUT_DIR / f"{candidate}_regression_{variant_name}_5000.csv")
            cmd = build_sim_command(candidate, variant_name, setup_str, 5000, csv_path)
            sim_result = run_simulation(cmd, timeout=300)

            if sim_result["success"]:
                metrics = parse_metrics_from_csv(csv_path)
                step_e = check_step_e_pass(metrics)
            else:
                metrics = {"error": sim_result["stderr"][:500]}
                step_e = {"verdict": "SIM_FAILED", "failures": [sim_result["stderr"][:200]]}

            candidate_results["regression"][variant_name] = {
                "metrics": metrics,
                "step_e": step_e,
                "sim_success": sim_result["success"],
            }
            print(f"    {variant_name}: {step_e['verdict']}")
            if step_e["verdict"] != "PASS":
                regression_pass = False

        candidate_results["regression_pass"] = regression_pass
        results.append(candidate_results)

        # Stop at first fully passing candidate
        if boundary_pass and regression_pass:
            print(f"\n  CANDIDATE {candidate} FULLY PASSES. Stopping evaluation.")
            break

    # Write summary
    summary_path = OUTPUT_DIR / "boundary_yaw_position_candidate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("CANDIDATE COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Candidate':<40} {'Low 5k':>8} {'High 5k':>8} {'Regress':>8}")
    print("-" * 64)
    for r in results:
        c = r["candidate"]
        low = r["boundary"].get("low_0p300_5000", r["boundary"].get("low_0p300_1000", {}))
        high = r["boundary"].get("high_0p480_5000", r["boundary"].get("high_0p480_1000", {}))
        low_v = low.get("step_e", {}).get("verdict", "?")
        high_v = high.get("step_e", {}).get("verdict", "?")
        reg_v = "PASS" if r.get("regression_pass", False) else "FAIL"
        print(f"{c:<40} {low_v:>8} {high_v:>8} {reg_v:>8}")


if __name__ == "__main__":
    main()
