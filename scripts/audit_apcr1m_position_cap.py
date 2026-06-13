"""APCR1m position torque cap audit - Phase 7.

Analyzes whether the position torque cap (±3 Nm) limits recenter effectiveness.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load APCR1m telemetry
BASE_DIR = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
APCR1M_CSV = BASE_DIR / "apcr1m_low_0p300_1000_full_telemetry" / "telemetry.csv"


def main():
    print("=" * 80)
    print("APCR1m POSITION TORQUE CAP AUDIT")
    print("=" * 80)

    df = pd.read_csv(APCR1M_CSV)
    print(f"Loaded {len(df)} rows")

    results = {}

    # Get tau_position and related signals
    tau_position = df["tau_position"]
    signed_error = df["active_pitch_crossing_signed_error_m"]
    recenter_active = df["apcr1m_recenter_active"]

    # Position cap values
    CAP = 3.0  # Nm

    # 1. Saturation analysis
    print("\n--- TAU_POSITION SATURATION ANALYSIS ---")

    # Check for saturation at limits
    sat_plus = (tau_position >= CAP - 0.01)  # At or near +3
    sat_minus = (tau_position <= -CAP + 0.01)  # At or near -3

    print(f"tau_position range: [{tau_position.min():.4f}, {tau_position.max():.4f}] Nm")
    print(f"Saturated at +{CAP} Nm: {sat_plus.sum()} steps ({sat_plus.mean()*100:.1f}%)")
    print(f"Saturated at -{CAP} Nm: {sat_minus.sum()} steps ({sat_minus.mean()*100:.1f}%)")
    print(f"Either saturated: {(sat_plus | sat_minus).sum()} steps ({(sat_plus | sat_minus).mean()*100:.1f}%)")

    results["saturation"] = {
        "cap_value": CAP,
        "sat_plus_count": int(sat_plus.sum()),
        "sat_plus_pct": float(sat_plus.mean() * 100),
        "sat_minus_count": int(sat_minus.sum()),
        "sat_minus_pct": float(sat_minus.mean() * 100),
        "either_saturated_count": int((sat_plus | sat_minus).sum()),
        "either_saturated_pct": float((sat_plus | sat_minus).mean() * 100),
    }

    # 2. Saturation during RECENTER
    print("\n--- SATURATION DURING RECENTER ---")

    recenter_df = df[recenter_active]
    tau_pos_recenter = recenter_df["tau_position"]

    sat_plus_recenter = (tau_pos_recenter >= CAP - 0.01).sum()
    sat_minus_recenter = (tau_pos_recenter <= -CAP + 0.01).sum()

    print(f"During RECENTER ({recenter_active.sum()} steps):")
    print(f"  Saturated at +{CAP} Nm: {sat_plus_recenter} ({sat_plus_recenter/len(tau_pos_recenter)*100:.1f}%)")
    print(f"  Saturated at -{CAP} Nm: {sat_minus_recenter} ({sat_minus_recenter/len(tau_pos_recenter)*100:.1f}%)")

    results["saturation_during_recenter"] = {
        "recenter_steps": int(recenter_active.sum()),
        "sat_plus_recenter": int(sat_plus_recenter),
        "sat_plus_recenter_pct": float(sat_plus_recenter / len(tau_pos_recenter) * 100),
        "sat_minus_recenter": int(sat_minus_recenter),
        "sat_minus_recenter_pct": float(sat_minus_recenter / len(tau_pos_recenter) * 100),
    }

    # 3. Saturation when |e| > threshold
    print("\n--- SATURATION WHEN |E| > THRESHOLD ---")

    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15]:
        large_error = abs(signed_error) > threshold
        if large_error.sum() > 0:
            tau_pos_large = df.loc[large_error, "tau_position"]
            sat_plus_large = (tau_pos_large >= CAP - 0.01).sum()
            sat_minus_large = (tau_pos_large <= -CAP + 0.01).sum()

            print(f"|e| > {threshold:.2f}m ({large_error.sum()} steps):")
            print(f"  Saturated at +{CAP} Nm: {sat_plus_large} ({sat_plus_large/len(tau_pos_large)*100:.1f}%)")
            print(f"  Saturated at -{CAP} Nm: {sat_minus_large} ({sat_minus_large/len(tau_pos_large)*100:.1f}%)")

            results[f"saturation_e_gt_{threshold:.2f}"] = {
                "steps": int(large_error.sum()),
                "sat_plus_pct": float(sat_plus_large / len(tau_pos_large) * 100),
                "sat_minus_pct": float(sat_minus_large / len(tau_pos_large) * 100),
            }

    # 4. Sign correctness of tau_position
    print("\n--- SIGN CORRECTNESS OF TAU_POSITION ---")

    # tau_position should have opposite sign to error (oppose drift)
    tau_pos_sign = np.sign(tau_position)
    error_sign = np.sign(signed_error)

    # Correct = opposite sign
    correct = (tau_pos_sign * error_sign < 0) | (tau_position == 0)  # Either opposite or zero

    print(f"tau_position has correct sign (opposes drift): {correct.sum()} ({correct.mean()*100:.1f}%)")

    # During RECENTER
    tau_pos_sign_recenter = np.sign(tau_pos_recenter)
    error_sign_recenter = np.sign(signed_error[recenter_active])
    correct_recenter = (tau_pos_sign_recenter * error_sign_recenter < 0) | (tau_pos_recenter == 0)

    print(f"During RECENTER: {correct_recenter.sum()} ({correct_recenter.mean()*100:.1f}%)")

    results["sign_correctness"] = {
        "correct_overall_pct": float(correct.mean() * 100),
        "correct_during_recenter_pct": float(correct_recenter.mean() * 100),
    }

    # 5. Cap correlation with drift growth
    print("\n--- CAP CORRELATION WITH DRIFT GROWTH ---")

    # When tau_position is saturated, is error growing?
    saturated = sat_plus | sat_minus

    # Check error rate before and after saturation
    e_diff = signed_error.diff()

    print(f"When tau_position is saturated:")
    print(f"  mean |e|: {abs(signed_error[saturated]).mean():.4f}m")
    print(f"  mean |e| change per step: {abs(e_diff[saturated]).mean():.6f}m/step")

    print(f"\nWhen tau_position is NOT saturated:")
    print(f"  mean |e|: {abs(signed_error[~saturated]).mean():.4f}m")
    print(f"  mean |e| change per step: {abs(e_diff[~saturated]).mean():.6f}m/step")

    results["drift_correlation"] = {
        "saturated_mean_abs_error": float(abs(signed_error[saturated]).mean()),
        "saturated_mean_abs_error_diff": float(abs(e_diff[saturated]).mean()),
        "not_saturated_mean_abs_error": float(abs(signed_error[~saturated]).mean()),
        "not_saturated_mean_abs_error_diff": float(abs(e_diff[~saturated]).mean()),
    }

    # 6. What would raw tau_position be without cap?
    print("\n--- WHAT WOULD RAW TAU_POSITION BE? ---")

    # Get tau_position_raw if available
    if "tau_position_raw" in df.columns:
        tau_pos_raw = df["tau_position_raw"]

        # Compare raw vs clipped
        clipping_occurred = (tau_pos_raw.abs() > CAP)

        print(f"tau_position_raw range: [{tau_pos_raw.min():.4f}, {tau_pos_raw.max():.4f}] Nm")
        print(f"Would be clipped: {clipping_occurred.sum()} steps ({clipping_occurred.mean()*100:.1f}%)")

        # How much is being clipped?
        clipped_amount = tau_pos_raw - tau_position
        print(f"Mean clipping amount: {clipped_amount.abs().mean():.4f} Nm")
        print(f"Max clipping amount: {clipped_amount.abs().max():.4f} Nm")

        results["raw_vs_clipped"] = {
            "raw_range": [float(tau_pos_raw.min()), float(tau_pos_raw.max())],
            "would_be_clipped_count": int(clipping_occurred.sum()),
            "would_be_clipped_pct": float(clipping_occurred.mean() * 100),
            "mean_clipped_amount": float(clipped_amount.abs().mean()),
            "max_clipped_amount": float(clipped_amount.abs().max()),
        }
    else:
        print("tau_position_raw not available")
        results["raw_vs_clipped"] = {"error": "tau_position_raw not available"}

    # 7. Comparison across profiles
    print("\n--- COMPARISON ACROSS PROFILES ---")

    profiles = {
        "APCR1h": BASE_DIR / "comparison_1000_apcr1h" / "telemetry.csv",
        "APCR1j": BASE_DIR / "comparison_1000_apcr1j" / "telemetry.csv",
        "APCR1k": BASE_DIR / "comparison_1000_apcr1k" / "telemetry.csv",
        "APCR1m": APCR1M_CSV,
    }

    cap_comparison = {}
    for profile, csv_path in profiles.items():
        if csv_path.exists():
            df_p = pd.read_csv(csv_path)
            tau_pos_p = df_p["tau_position"]
            cap_comparison[profile] = {
                "range": [float(tau_pos_p.min()), float(tau_pos_p.max())],
                "abs_mean": float(tau_pos_p.abs().mean()),
                "max_abs": float(tau_pos_p.abs().max()),
            }
            print(f"{profile}: range=[{tau_pos_p.min():.2f}, {tau_pos_p.max():.2f}], abs_mean={tau_pos_p.abs().mean():.2f}")

    results["profile_comparison"] = cap_comparison

    # 8. Classification
    print("\n" + "=" * 80)
    print("POSITION CAP CLASSIFICATION")
    print("=" * 80)

    # Check if cap is a primary issue
    sat_pct = results["saturation"]["either_saturated_pct"]
    sat_recenter_pct = results["saturation_during_recenter"]["sat_plus_recenter_pct"] + \
                      results["saturation_during_recenter"]["sat_minus_recenter_pct"]

    if sat_pct > 50 and sat_recenter_pct > 40:
        classification = "APCR1M_POSITION_CAP_LIMITS_RECENTER"
        reason = f"tau_position saturated {sat_pct:.1f}% overall, {sat_recenter_pct:.1f}% during RECENTER"
    elif sat_pct > 30:
        classification = "APCR1M_POSITION_CAP_PARTIAL"
        reason = f"tau_position saturated {sat_pct:.1f}% overall"
    else:
        classification = "APCR1M_POSITION_CAP_NOT_PRIMARY"
        reason = f"tau_position only saturated {sat_pct:.1f}% overall"

    print(f"\nClassification: {classification}")
    print(f"Reason: {reason}")

    results["classification"] = classification
    results["classification_reason"] = reason

    # Save results
    json_path = BASE_DIR / "apcr1m_position_torque_cap_audit.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    return results


if __name__ == "__main__":
    results = main()
