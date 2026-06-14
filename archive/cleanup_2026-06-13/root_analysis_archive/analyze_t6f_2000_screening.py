"""Analyze T6F high_0p480 2000-step screening (Phase 8).

Compare T6F 2000-step vs T5 first 2000 steps.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


def find_latest_telemetry(pattern="telemetry_*.csv"):
    """Find most recent telemetry file."""
    telem_dir = Path("outputs/hierarchical_controller_sim")
    files = sorted(telem_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_t5_reference():
    """Load T5 high_0p480 5000-step reference."""
    t5_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv")
    if not t5_path.exists():
        raise FileNotFoundError(f"T5 reference not found: {t5_path}")

    df = pd.read_csv(t5_path)
    # Extract first 2000 steps
    df_2000 = df.head(2001)  # 0-2000 inclusive
    print(f"Loaded T5 reference: {len(df_2000)} rows (first 2000 steps)")
    return df_2000, str(t5_path)


def get_drift_column(df):
    """Get primary physical drift column."""
    priority = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m"
    ]

    for col in priority:
        if col in df.columns:
            print(f"Using drift column: {col}")
            return col, df[col].values

    raise ValueError("No valid drift column found")


def compute_drift_metrics(drift_values):
    """Compute drift statistics."""
    return {
        "min_e": float(np.min(drift_values)),
        "max_e": float(np.max(drift_values)),
        "max_abs_e": float(np.max(np.abs(drift_values))),
        "p2p": float(np.ptp(drift_values)),
        "mean_e": float(np.mean(drift_values)),
        "mean_abs_e": float(np.mean(np.abs(drift_values))),
        "final_e": float(drift_values[-1]),
        "positive_pct": float(100.0 * np.sum(drift_values > 0) / len(drift_values)),
        "negative_pct": float(100.0 * np.sum(drift_values < 0) / len(drift_values)),
        "zero_crossings": int(np.sum(np.diff(np.sign(drift_values)) != 0))
    }


def compute_band_metrics(drift_values):
    """Compute band exceedance metrics."""
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]
    band_metrics = {}

    for thresh in thresholds:
        outside = np.abs(drift_values) > thresh
        count = int(np.sum(outside))
        pct = float(100.0 * count / len(drift_values))
        band_metrics[f"outside_{thresh:.2f}".replace(".", "p")] = {
            "count": count,
            "pct": pct
        }

        # Directional
        positive = np.sum(drift_values > thresh)
        negative = np.sum(drift_values < -thresh)
        band_metrics[f"above_{thresh:.2f}".replace(".", "p")] = int(positive)
        band_metrics[f"below_{thresh:.2f}".replace(".", "p")] = int(negative)

    return band_metrics


def compute_window_metrics(drift_values, window_size=500):
    """Compute metrics for each 500-step window."""
    n_windows = len(drift_values) // window_size
    windows = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = drift_values[start:end]

        window_metrics = {
            "window": i + 1,
            "start_step": start,
            "end_step": end,
            "max_abs_e": float(np.max(np.abs(window_data))),
            "p2p": float(np.ptp(window_data)),
            "mean_abs_e": float(np.mean(np.abs(window_data))),
            "final_e": float(window_data[-1]),
            "outside_0p08_count": int(np.sum(np.abs(window_data) > 0.08)),
            "outside_0p10_count": int(np.sum(np.abs(window_data) > 0.10)),
            "outside_0p15_count": int(np.sum(np.abs(window_data) > 0.15)),
            "zero_crossings": int(np.sum(np.diff(np.sign(window_data)) != 0))
        }
        windows.append(window_metrics)

    return windows


def analyze_arch_fix_activation(df):
    """Analyze architecture fix activation."""
    if "arch_fix_active" not in df.columns:
        return {"error": "arch_fix_active column not found"}

    active_count = int(df["arch_fix_active"].sum())
    active_pct = float(100.0 * active_count / len(df))

    metrics = {
        "active_count": active_count,
        "active_pct": active_pct
    }

    # Reasons
    if "arch_fix_reason" in df.columns:
        reasons = df[df["arch_fix_active"] == True]["arch_fix_reason"].value_counts().to_dict()
        metrics["reasons"] = reasons

    return metrics


def analyze_torque_transmission(df):
    """Analyze torque transmission and architecture fix impact."""
    metrics = {}

    # Effective max position tau distribution
    if "effective_max_position_tau_after_arch_fix" in df.columns:
        tau_after = df["effective_max_position_tau_after_arch_fix"].values
        unique_caps = np.unique(tau_after)
        metrics["cap_distribution"] = {float(cap): int(np.sum(tau_after == cap)) for cap in unique_caps}
        metrics["max_cap"] = float(np.max(tau_after))
        metrics["steps_above_4nm"] = int(np.sum(tau_after > 4.0))

    # Position torque transmission
    if "tau_position" in df.columns:
        tau_pos = df["tau_position"].values
        abs_tau = np.abs(tau_pos)
        metrics["tau_position_max_abs"] = float(np.max(abs_tau))
        metrics["tau_position_above_4nm"] = int(np.sum(abs_tau > 4.0))

    # Final wheel torque
    if "final_wheel_tau_with_apc" in df.columns:
        tau_wheel = df["final_wheel_tau_with_apc"].values
        metrics["final_wheel_tau_max_abs"] = float(np.max(np.abs(tau_wheel)))

    return metrics


def analyze_stability(df):
    """Analyze stability and structural safety."""
    metrics = {}

    # Survival
    metrics["survived_steps"] = len(df)
    if "terminated" in df.columns:
        terminated = df["terminated"].iloc[-1] if len(df) > 0 else False
        metrics["terminated"] = bool(terminated)
        if terminated and "termination_reason" in df.columns:
            metrics["termination_reason"] = str(df["termination_reason"].iloc[-1])

    # Contact
    if "n_contacts" in df.columns:
        n_contacts = df["n_contacts"].values
        metrics["contact_pct"] = float(100.0 * np.sum(n_contacts > 0) / len(df))
        metrics["double_contact_pct"] = float(100.0 * np.sum(n_contacts == 2) / len(df))

    # Height
    if "com_z" in df.columns:
        com_z = df["com_z"].values
        metrics["com_z_min"] = float(np.min(com_z))
        metrics["com_z_mean"] = float(np.mean(com_z))
        metrics["com_z_max"] = float(np.max(com_z))

    # Attitude
    if "pitch_x" in df.columns:
        pitch = df["pitch_x"].values * 180 / np.pi
        metrics["pitch_min_deg"] = float(np.min(pitch))
        metrics["pitch_max_deg"] = float(np.max(pitch))
        metrics["pitch_rms_deg"] = float(np.sqrt(np.mean(pitch**2)))

    if "roll_y" in df.columns:
        roll = df["roll_y"].values * 180 / np.pi
        metrics["roll_min_deg"] = float(np.min(roll))
        metrics["roll_max_deg"] = float(np.max(roll))
        metrics["roll_rms_deg"] = float(np.sqrt(np.mean(roll**2)))

    # Wheel velocity
    if "wheel_vel_mean_rad_s" in df.columns:
        wheel_vel = np.abs(df["wheel_vel_mean_rad_s"].values)
        metrics["wheel_vel_max"] = float(np.max(wheel_vel))
        metrics["wheel_vel_rms"] = float(np.sqrt(np.mean(wheel_vel**2)))
        metrics["wheel_vel_above_5"] = int(np.sum(wheel_vel > 5.0))
        metrics["wheel_vel_above_6"] = int(np.sum(wheel_vel > 6.0))
        metrics["wheel_vel_above_7"] = int(np.sum(wheel_vel > 7.0))

    return metrics


def classify_phase8(t5_metrics, t6f_metrics, t6f_stability):
    """Classify Phase 8 result."""

    # Check survival (allow 1999 or 2000 steps)
    if t6f_stability["survived_steps"] < 1999:
        return "T6F_2000_FAIL_STABILITY", "T6F fell before completing screening"

    # Check termination
    if t6f_stability.get("terminated", False):
        reason = t6f_stability.get("termination_reason", "unknown")
        return "T6F_2000_FAIL_STABILITY", f"T6F terminated: {reason}"

    # Check WBC/safety violations
    # (Would check here if telemetry had flags)

    # Get drift improvements
    t5_outside_08 = t5_metrics["band"]["outside_0p08"]["count"]
    t6f_outside_08 = t6f_metrics["band"]["outside_0p08"]["count"]
    improvement_08 = t5_outside_08 - t6f_outside_08

    t5_outside_10 = t5_metrics["band"]["outside_0p10"]["count"]
    t6f_outside_10 = t6f_metrics["band"]["outside_0p10"]["count"]
    improvement_10 = t5_outside_10 - t6f_outside_10

    t5_outside_15 = t5_metrics["band"]["outside_0p15"]["count"]
    t6f_outside_15 = t6f_metrics["band"]["outside_0p15"]["count"]
    improvement_15 = t5_outside_15 - t6f_outside_15

    t6f_max_abs = t6f_metrics["drift"]["max_abs_e"]
    t6f_outside_15_pct = t6f_metrics["band"]["outside_0p15"]["pct"]

    # PASS criteria
    pass_outside_08 = improvement_08 > 0
    pass_outside_10 = improvement_10 > 0
    pass_max_abs = t6f_max_abs <= 0.20
    pass_outside_15 = t6f_outside_15_pct <= 5.0

    # Classification
    if pass_outside_08 and pass_outside_10 and pass_max_abs and pass_outside_15:
        return "T6F_2000_PASS_PROCEED_5000", "All criteria met"
    elif (improvement_08 >= 0 and improvement_10 >= 0 and
          pass_max_abs and t6f_outside_15_pct <= 10.0):
        return "T6F_2000_PASS_WITH_MONITORING", "Drift improves or stable, monitoring needed"
    elif improvement_10 < 0 or improvement_15 < -100:
        return "T6F_2000_NOT_BETTER_THAN_T5", f"Drift degraded: 0.10 band={improvement_10:+d}, 0.15 band={improvement_15:+d}"
    elif not pass_max_abs or not pass_outside_15:
        return "T6F_2000_FAIL_BAND_TARGET", f"Band targets exceeded: max|e|={t6f_max_abs:.3f}, outside 0.15={t6f_outside_15_pct:.1f}%"
    else:
        return "T6F_2000_INCONCLUSIVE", "Metrics ambiguous"


def main():
    print("="*80)
    print("Phase 8: T6F high_0p480 2000-step screening")
    print("="*80)

    # Load T5 reference
    print("\nLoading T5 reference (first 2000 steps)...")
    t5_df, t5_path = load_t5_reference()

    # Load T6F
    print("\nLoading T6F 2000-step telemetry...")
    t6f_path = find_latest_telemetry()
    if not t6f_path:
        raise FileNotFoundError("No T6F telemetry found")

    t6f_df = pd.read_csv(t6f_path)
    print(f"T6F: {t6f_path} ({len(t6f_df)} rows)")

    # Verify profile identity
    if "sagittal_schedule_profile" in t6f_df.columns:
        t6f_profile = t6f_df["sagittal_schedule_profile"].iloc[0]
        print(f"T6F profile: {t6f_profile}")
        if "T6F" not in str(t6f_profile):
            print("WARNING: Profile name doesn't contain T6F")


    # Get drift columns
    print("\nExtracting drift metrics...")
    t5_drift_col, t5_drift = get_drift_column(t5_df)
    t6f_drift_col, t6f_drift = get_drift_column(t6f_df)

    if t5_drift_col != t6f_drift_col:
        print(f"WARNING: Drift columns differ: T5={t5_drift_col}, T6F={t6f_drift_col}")

    # Compute metrics
    print("\n" + "="*80)
    print("Phase 8C: Drift comparison")
    print("="*80)

    t5_drift_metrics = compute_drift_metrics(t5_drift)
    t5_band_metrics = compute_band_metrics(t5_drift)
    t6f_drift_metrics = compute_drift_metrics(t6f_drift)
    t6f_band_metrics = compute_band_metrics(t6f_drift)

    print("\nT5 first 2000 steps:")
    print(f"  max |e|: {t5_drift_metrics['max_abs_e']:.4f} m")
    print(f"  P2P: {t5_drift_metrics['p2p']:.4f} m")
    print(f"  mean |e|: {t5_drift_metrics['mean_abs_e']:.4f} m")
    print(f"  outside +-0.08: {t5_band_metrics['outside_0p08']['count']} ({t5_band_metrics['outside_0p08']['pct']:.1f}%)")
    print(f"  outside +-0.10: {t5_band_metrics['outside_0p10']['count']} ({t5_band_metrics['outside_0p10']['pct']:.1f}%)")
    print(f"  outside +-0.15: {t5_band_metrics['outside_0p15']['count']} ({t5_band_metrics['outside_0p15']['pct']:.1f}%)")

    print("\nT6F 2000 steps:")
    print(f"  max |e|: {t6f_drift_metrics['max_abs_e']:.4f} m")
    print(f"  P2P: {t6f_drift_metrics['p2p']:.4f} m")
    print(f"  mean |e|: {t6f_drift_metrics['mean_abs_e']:.4f} m")
    print(f"  outside +-0.08: {t6f_band_metrics['outside_0p08']['count']} ({t6f_band_metrics['outside_0p08']['pct']:.1f}%)")
    print(f"  outside +-0.10: {t6f_band_metrics['outside_0p10']['count']} ({t6f_band_metrics['outside_0p10']['pct']:.1f}%)")
    print(f"  outside +-0.15: {t6f_band_metrics['outside_0p15']['count']} ({t6f_band_metrics['outside_0p15']['pct']:.1f}%)")

    print("\nImprovement:")
    improvement_08 = t5_band_metrics['outside_0p08']['count'] - t6f_band_metrics['outside_0p08']['count']
    improvement_10 = t5_band_metrics['outside_0p10']['count'] - t6f_band_metrics['outside_0p10']['count']
    print(f"  outside +-0.08: {improvement_08:+d} steps")
    print(f"  outside +-0.10: {improvement_10:+d} steps")


    # Window analysis
    print("\n" + "="*80)
    print("Phase 8D: Window analysis")
    print("="*80)

    t5_windows = compute_window_metrics(t5_drift)
    t6f_windows = compute_window_metrics(t6f_drift)

    print("\nT5 windows:")
    for w in t5_windows:
        print(f"  Window {w['window']} ({w['start_step']}-{w['end_step']}): max|e|={w['max_abs_e']:.4f}, outside 0.08={w['outside_0p08_count']}, outside 0.10={w['outside_0p10_count']}")

    print("\nT6F windows:")
    for w in t6f_windows:
        print(f"  Window {w['window']} ({w['start_step']}-{w['end_step']}): max|e|={w['max_abs_e']:.4f}, outside 0.08={w['outside_0p08_count']}, outside 0.10={w['outside_0p10_count']}")

    # Architecture fix analysis
    print("\n" + "="*80)
    print("Phase 8E: Architecture fix and torque analysis")
    print("="*80)

    arch_fix = analyze_arch_fix_activation(t6f_df)
    print(f"\nArchitecture fix active: {arch_fix.get('active_count', 0)} steps ({arch_fix.get('active_pct', 0):.1f}%)")
    if "reasons" in arch_fix:
        print("Activation reasons:")
        for reason, count in arch_fix["reasons"].items():
            print(f"  {reason}: {count}")

    torque = analyze_torque_transmission(t6f_df)
    print(f"\nTorque transmission:")
    if "max_cap" in torque:
        print(f"  Max cap: {torque['max_cap']:.1f} Nm")
        print(f"  Steps above 4.0 Nm: {torque.get('steps_above_4nm', 0)}")
    if "tau_position_above_4nm" in torque:
        print(f"  tau_position above 4.0 Nm: {torque['tau_position_above_4nm']} steps")

    # Stability analysis
    print("\n" + "="*80)
    print("Phase 8F: Stability analysis")
    print("="*80)

    stability = analyze_stability(t6f_df)
    print(f"\nSurvival: {stability['survived_steps']} steps")
    if "terminated" in stability:
        print(f"Terminated: {stability['terminated']}")
        if stability.get("terminated"):
            print(f"Reason: {stability.get('termination_reason', 'unknown')}")

    print(f"\nContact: {stability.get('contact_pct', 0):.1f}%")
    print(f"Height: min={stability.get('com_z_min', 0):.3f}, mean={stability.get('com_z_mean', 0):.3f}, max={stability.get('com_z_max', 0):.3f}")
    print(f"Pitch RMS: {stability.get('pitch_rms_deg', 0):.2f} deg")
    print(f"Roll RMS: {stability.get('roll_rms_deg', 0):.2f} deg")
    print(f"Wheel vel max: {stability.get('wheel_vel_max', 0):.2f} rad/s")
    print(f"Wheel vel >5: {stability.get('wheel_vel_above_5', 0)}")


    # Classification
    print("\n" + "="*80)
    print("Phase 8H: Classification")
    print("="*80)

    t5_full = {"drift": t5_drift_metrics, "band": t5_band_metrics}
    t6f_full = {"drift": t6f_drift_metrics, "band": t6f_band_metrics}

    classification, reason = classify_phase8(t5_full, t6f_full, stability)
    print(f"\nClassification: {classification}")
    print(f"Reason: {reason}")

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main screening result
    result = {
        "classification": classification,
        "reason": reason,
        "date": "2026-06-12",
        "phase": "8_of_11",
        "t5_file": t5_path,
        "t6f_file": str(t6f_path),
        "t5_steps": len(t5_df),
        "t6f_steps": len(t6f_df),
        "drift_column": t6f_drift_col,
        "t5_metrics": t5_full,
        "t6f_metrics": t6f_full,
        "arch_fix": arch_fix,
        "torque": torque,
        "stability": stability
    }

    json_path = output_dir / "t6f_high_0p480_2000_screening.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Window metrics CSV
    window_df = pd.DataFrame(t6f_windows)
    window_csv = output_dir / "t6f_high_0p480_2000_window_metrics.csv"
    window_df.to_csv(window_csv, index=False)
    print(f"Window metrics saved to: {window_csv}")

    # Decision
    decision = {
        "classification": classification,
        "reason": reason,
        "date": "2026-06-12",
        "proceed_to_5000": classification == "T6F_2000_PASS_PROCEED_5000"
    }

    decision_json = output_dir / "t6f_high_0p480_2000_decision.json"
    with open(decision_json, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Decision saved to: {decision_json}")

    print("\n" + "="*80)
    print("Phase 8 complete!")
    print(f"Classification: {classification}")
    print("="*80)

    return classification


if __name__ == "__main__":
    main()

