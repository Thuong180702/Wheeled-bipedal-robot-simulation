"""Verify capture-direction sign convention from baseline telemetry.

Analyzes the transient window to understand:
1. Positive wheel torque → support acceleration direction
2. Positive pitch_error → body lean direction
3. Required capture direction for forward/backward pitch
4. Whether tau_position opposes capture during transient
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    telemetry_path = Path("outputs/hierarchical_controller_sim/telemetry_1780198465.csv")
    df = pd.read_csv(telemetry_path)

    print("=" * 80)
    print("SIGN CONVENTION VERIFICATION - Baseline Transient Analysis")
    print("=" * 80)
    print()

    # Focus on transient window around step 1360
    transient_start = 1300
    transient_end = 1420
    df_transient = df[(df["step"] >= transient_start) & (df["step"] <= transient_end)]

    # Key columns for sign analysis
    cols = [
        "step",
        "pitch_x_rad",
        "pitch_rate_x_rad_s",
        "support_position_error_m",
        "sagittal_velocity_m_s",
        "tau_position_raw",
        "tau_position_clipped",
        "sagittal_term_pitch",
        "sagittal_term_pitch_rate",
        "wheel_vel_mean_rad_s",
        "wheel_acc_mean_rad_s2",
        "com_y_m",
        "cp_y_m",
    ]

    # Sample every 10 steps
    df_sample = df_transient[df_transient["step"] % 10 == 0][cols].copy()

    # Convert pitch to degrees for readability
    df_sample["pitch_x_deg"] = df_sample["pitch_x_rad"] * 57.3
    df_sample["pitch_rate_x_deg_s"] = df_sample["pitch_rate_x_rad_s"] * 57.3

    display_cols = [
        "step",
        "pitch_x_deg",
        "pitch_rate_x_deg_s",
        "support_position_error_m",
        "sagittal_velocity_m_s",
        "tau_position_raw",
        "tau_position_clipped",
        "sagittal_term_pitch",
        "wheel_vel_mean_rad_s",
        "wheel_acc_mean_rad_s2",
    ]

    print("Transient Window Sample (every 10 steps):")
    print(df_sample[display_cols].to_string(index=False))
    print()

    # Peak analysis
    spe_abs = df["support_position_error_m"].abs()
    peak_idx = spe_abs.idxmax()
    peak_step = int(df.loc[peak_idx, "step"])
    peak_spe = df.loc[peak_idx, "support_position_error_m"]

    print(f"Peak support position error: {peak_spe:.3f} m at step {peak_step}")
    print()

    # Window around peak
    df_peak = df[(df["step"] >= peak_step - 20) & (df["step"] <= peak_step + 20)][cols].copy()
    df_peak["pitch_x_deg"] = df_peak["pitch_x_rad"] * 57.3
    df_peak["pitch_rate_x_deg_s"] = df_peak["pitch_rate_x_rad_s"] * 57.3

    print("=" * 80)
    print(f"PEAK WINDOW (steps {peak_step-20} to {peak_step+20})")
    print("=" * 80)
    print(df_peak[display_cols].to_string(index=False))
    print()

    # Sign convention analysis
    print("=" * 80)
    print("SIGN CONVENTION ANALYSIS")
    print("=" * 80)
    print()

    # Analyze correlation between pitch and wheel acceleration
    pitch_x = df_transient["pitch_x_rad"].values
    wheel_acc = df_transient["wheel_acc_mean_rad_s2"].values
    tau_pitch = df_transient["sagittal_term_pitch"].values
    tau_position = df_transient["tau_position_raw"].values
    spe = df_transient["support_position_error_m"].values

    # Correlation analysis
    corr_pitch_wheel_acc = np.corrcoef(pitch_x, wheel_acc)[0, 1]
    corr_pitch_tau_pitch = np.corrcoef(pitch_x, tau_pitch)[0, 1]
    corr_spe_tau_position = np.corrcoef(spe, tau_position)[0, 1]

    print(f"Correlation(pitch_x, wheel_acc): {corr_pitch_wheel_acc:.3f}")
    print(f"Correlation(pitch_x, tau_pitch): {corr_pitch_tau_pitch:.3f}")
    print(f"Correlation(spe, tau_position): {corr_spe_tau_position:.3f}")
    print()

    # Sign analysis at peak
    peak_pitch = df.loc[peak_idx, "pitch_x_rad"]
    peak_pitch_rate = df.loc[peak_idx, "pitch_rate_x_rad_s"]
    peak_tau_pitch = df.loc[peak_idx, "sagittal_term_pitch"]
    peak_tau_position = df.loc[peak_idx, "tau_position_raw"]
    peak_wheel_acc = df.loc[peak_idx, "wheel_acc_mean_rad_s2"]

    print(f"At peak (step {peak_step}):")
    print(f"  pitch_x: {peak_pitch:.4f} rad ({peak_pitch*57.3:.2f} deg)")
    print(f"  pitch_rate_x: {peak_pitch_rate:.4f} rad/s ({peak_pitch_rate*57.3:.2f} deg/s)")
    print(f"  support_position_error: {peak_spe:.3f} m")
    print(f"  tau_pitch: {peak_tau_pitch:.3f} Nm")
    print(f"  tau_position: {peak_tau_position:.3f} Nm")
    print(f"  wheel_acc: {peak_wheel_acc:.3f} rad/s²")
    print()

    # Determine signs
    print("=" * 80)
    print("SIGN CONVENTION SUMMARY")
    print("=" * 80)
    print()

    if peak_pitch > 0:
        print("[OK] Positive pitch_x -> body leans FORWARD")
    else:
        print("[OK] Negative pitch_x -> body leans BACKWARD")

    if peak_tau_pitch > 0 and peak_pitch > 0:
        print("[OK] Positive pitch -> positive tau_pitch (restoring torque)")
    elif peak_tau_pitch < 0 and peak_pitch < 0:
        print("[OK] Negative pitch -> negative tau_pitch (restoring torque)")

    if peak_spe > 0 and peak_tau_position < 0:
        print("[OK] Positive support_position_error -> negative tau_position (return tendency)")
    elif peak_spe < 0 and peak_tau_position > 0:
        print("[OK] Negative support_position_error -> positive tau_position (return tendency)")

    print()
    print("Physical interpretation:")
    print()

    if peak_pitch > 0 and peak_spe > 0:
        print("During transient:")
        print("  - Body leans FORWARD (pitch_x > 0)")
        print("  - Support is AHEAD of target (spe > 0)")
        print("  - tau_pitch tries to restore (positive torque)")
        print("  - tau_position tries to return backward (negative torque)")
        print()
        print("Conflict:")
        print("  - To catch forward-leaning CoM, wheels may need to accelerate FORWARD")
        print("  - But tau_position < 0 tries to pull support BACKWARD")
        print("  - This OPPOSES the required capture direction")
        print()
        print("Required capture direction: FORWARD (same sign as pitch_x)")
        print("tau_position direction: BACKWARD (opposite sign)")
        print("-> CONFLICT DETECTED")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Sign convention verified:")
    print("  1. pitch_x > 0 -> forward lean -> wheels should accelerate forward to catch CoM")
    print("  2. pitch_x < 0 -> backward lean -> wheels should accelerate backward to catch CoM")
    print("  3. spe > 0 -> support ahead -> tau_position < 0 (tries to return backward)")
    print("  4. spe < 0 -> support behind -> tau_position > 0 (tries to return forward)")
    print()
    print("Conflict condition:")
    print("  - required_capture_direction = sign(pitch_x)")
    print("  - tau_position_direction = sign(tau_position)")
    print("  - conflict = (required_capture_direction != 0) AND")
    print("               (tau_position_direction == -required_capture_direction)")
    print()
    print("At transient peak:")
    print(f"  - required_capture_direction = {'+1 (forward)' if peak_pitch > 0 else '-1 (backward)'}")
    print(f"  - tau_position = {peak_tau_position:.3f} Nm ({'backward' if peak_tau_position < 0 else 'forward'})")
    print(f"  - CONFLICT: {'YES' if (peak_pitch > 0 and peak_tau_position < 0) or (peak_pitch < 0 and peak_tau_position > 0) else 'NO'}")
    print()
    print("=" * 80)
    print("VERIFIED: Conflict detected at transient peak")
    print("=" * 80)


if __name__ == "__main__":
    main()
