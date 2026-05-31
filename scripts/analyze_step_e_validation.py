"""Analyze Step E validation results for position hold gating fix.

Compares baseline (before fix) vs fixed controller performance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def analyze_run(telemetry_path: str) -> dict:
    """Analyze a single run's telemetry."""
    df = pd.read_csv(telemetry_path)

    # Support position error analysis
    if "support_position_error_m" in df.columns:
        spe = df["support_position_error_m"].values
        spe_max = float(np.max(np.abs(spe)))
        spe_final = float(spe[-1])
        spe_mean = float(np.mean(spe))
        spe_std = float(np.std(spe))

        # Find transient peak location
        spe_abs = np.abs(spe)
        spe_peak_idx = int(np.argmax(spe_abs))
        spe_peak_step = int(df["step"].values[spe_peak_idx])
        spe_peak_value = float(spe[spe_peak_idx])
    else:
        spe_max = None
        spe_final = None
        spe_mean = None
        spe_std = None
        spe_peak_idx = None
        spe_peak_step = None
        spe_peak_value = None

    # Pitch analysis
    pitch_x = df["pitch_x_rad"].values * 57.3  # to degrees
    pitch_max = float(np.max(np.abs(pitch_x)))
    pitch_final = float(pitch_x[-1])
    pitch_rms = float(np.sqrt(np.mean(pitch_x**2)))

    # Height analysis
    com_z = df["com_z_m"].values
    com_z_min = float(np.min(com_z))
    com_z_final = float(com_z[-1])
    com_z_initial = float(com_z[0])

    # Wheel velocity analysis
    if "wheel_vel_mean_rad_s" in df.columns:
        wheel_vel = df["wheel_vel_mean_rad_s"].values
        wheel_vel_max = float(np.max(np.abs(wheel_vel)))
        wheel_vel_rms = float(np.sqrt(np.mean(wheel_vel**2)))
    else:
        wheel_vel_max = None
        wheel_vel_rms = None

    # Contact state
    if "contact_supervisor_state" in df.columns:
        contact_states = df["contact_supervisor_state"].unique()
        contact_stable = "stable" in contact_states or "STABLE" in contact_states
    else:
        contact_stable = None

    # Termination
    actual_steps = len(df)
    survived = actual_steps >= 5000 if actual_steps > 1000 else actual_steps >= 1000

    return {
        "actual_steps": actual_steps,
        "survived": survived,
        "support_position_error_max_m": spe_max,
        "support_position_error_final_m": spe_final,
        "support_position_error_mean_m": spe_mean,
        "support_position_error_std_m": spe_std,
        "support_position_error_peak_step": spe_peak_step,
        "support_position_error_peak_value_m": spe_peak_value,
        "pitch_max_deg": pitch_max,
        "pitch_final_deg": pitch_final,
        "pitch_rms_deg": pitch_rms,
        "com_z_min_m": com_z_min,
        "com_z_final_m": com_z_final,
        "com_z_initial_m": com_z_initial,
        "wheel_vel_max_rad_s": wheel_vel_max,
        "wheel_vel_rms_rad_s": wheel_vel_rms,
        "contact_stable": contact_stable,
    }


def check_acceptance_gates(metrics: dict) -> dict:
    """Check if metrics pass acceptance gates."""
    spe_max = metrics.get("support_position_error_max_m")
    spe_final = metrics.get("support_position_error_final_m")

    if spe_max is None or spe_final is None:
        return {
            "preferred_pass": False,
            "fallback_pass": False,
            "hard_minimum_pass": False,
            "reason": "Missing support position error metrics",
        }

    # Preferred: ±0.10 m transient, ≤0.05 m final
    preferred_pass = (abs(spe_max) <= 0.10) and (abs(spe_final) <= 0.05)

    # Fallback: ±0.15 m transient, ≤0.10 m final
    fallback_pass = (abs(spe_max) <= 0.15) and (abs(spe_final) <= 0.10)

    # Hard minimum: ≤0.30 m transient, ≤0.10 m final
    hard_minimum_pass = (abs(spe_max) <= 0.30) and (abs(spe_final) <= 0.10)

    if preferred_pass:
        reason = "PREFERRED gate passed"
    elif fallback_pass:
        reason = "FALLBACK gate passed"
    elif hard_minimum_pass:
        reason = "HARD_MINIMUM gate passed"
    else:
        reason = f"Failed all gates: max={spe_max:.3f}m, final={spe_final:.3f}m"

    return {
        "preferred_pass": preferred_pass,
        "fallback_pass": fallback_pass,
        "hard_minimum_pass": hard_minimum_pass,
        "reason": reason,
    }


def main():
    # Find latest telemetry files
    telemetry_dir = Path("outputs/hierarchical_controller_sim")
    telemetry_files = sorted(telemetry_dir.glob("telemetry_*.csv"))

    if len(telemetry_files) < 2:
        print("Need at least 2 telemetry files (1000 and 5000 step runs)")
        return

    # Analyze latest two runs
    run_1000 = telemetry_files[-2]
    run_5000 = telemetry_files[-1]

    print("Analyzing Step E validation runs...")
    print(f"1000-step run: {run_1000.name}")
    print(f"5000-step run: {run_5000.name}")
    print()

    metrics_1000 = analyze_run(str(run_1000))
    metrics_5000 = analyze_run(str(run_5000))

    print("=" * 80)
    print("1000-STEP RUN RESULTS")
    print("=" * 80)
    for key, value in metrics_1000.items():
        print(f"{key}: {value}")
    print()

    print("=" * 80)
    print("5000-STEP RUN RESULTS")
    print("=" * 80)
    for key, value in metrics_5000.items():
        print(f"{key}: {value}")
    print()

    # Check acceptance gates for 5000-step run
    gates_5000 = check_acceptance_gates(metrics_5000)

    print("=" * 80)
    print("ACCEPTANCE GATES (5000-step run)")
    print("=" * 80)
    print(f"Preferred (±0.10m, ≤0.05m final): {gates_5000['preferred_pass']}")
    print(f"Fallback (±0.15m, ≤0.10m final): {gates_5000['fallback_pass']}")
    print(f"Hard minimum (≤0.30m, ≤0.10m final): {gates_5000['hard_minimum_pass']}")
    print(f"Reason: {gates_5000['reason']}")
    print()

    # Save results
    output_dir = Path("outputs/sagittal_position_hold_return")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "run_1000": metrics_1000,
        "run_5000": metrics_5000,
        "acceptance_gates": gates_5000,
    }

    with open(output_dir / "v0_baseline_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir / 'v0_baseline_analysis.json'}")


if __name__ == "__main__":
    main()
