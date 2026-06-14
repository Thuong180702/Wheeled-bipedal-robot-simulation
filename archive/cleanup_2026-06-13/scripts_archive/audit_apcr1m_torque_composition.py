"""APCR1m torque composition and dominance audit - Phase 6.

Analyzes which torque component dominates and whether wheel velocity damping
is causing the drift problem.
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
    print("APCR1m TORQUE COMPOSITION AND DOMINANCE AUDIT")
    print("=" * 80)

    df = pd.read_csv(APCR1M_CSV)
    print(f"Loaded {len(df)} rows")

    results = {}

    # Get signed error and RECENTER state
    signed_error = df["active_pitch_crossing_signed_error_m"]
    recenter_active = df["apcr1m_recenter_active"]

    # Get torque components
    tau_pitch = df["apcr1m_tau_pitch_after_blend"]  # After blend
    tau_position = df["tau_position"]
    tau_wheel_vel_left = df["tau_wheel_velocity_left"]
    tau_wheel_vel_right = df["tau_wheel_velocity_right"]
    final_tau_with_apc = df["final_wheel_tau_with_apc"]
    final_tau_without_apc = df["final_wheel_tau_without_apc"]

    # Compute APCR tau (difference between with and without)
    apcr_tau = final_tau_with_apc - final_tau_without_apc

    # Add to dataframe for analysis
    df["apcr_tau"] = apcr_tau

    # 1. Component magnitude analysis
    print("\n--- COMPONENT MAGNITUDE ANALYSIS ---")
    components = {
        "tau_pitch": tau_pitch,
        "tau_position": tau_position,
        "tau_wheel_vel_left": tau_wheel_vel_left,
        "tau_wheel_vel_right": tau_wheel_vel_right,
        "apcr_tau": apcr_tau,
        "final_tau_with_apc": final_tau_with_apc,
    }

    for name, series in components.items():
        abs_series = series.abs()
        print(f"{name}:")
        print(f"  range: [{series.min():.2f}, {series.max():.2f}] Nm")
        print(f"  mean abs: {abs_series.mean():.2f} Nm")
        print(f"  max abs: {abs_series.max():.2f} Nm")

    results["component_magnitudes"] = {
        name: {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean_abs": float(series.abs().mean()),
            "max_abs": float(series.abs().max()),
        }
        for name, series in components.items()
    }

    # 2. Dominance analysis - which component is largest per step?
    print("\n--- DOMINANCE ANALYSIS ---")

    abs_components = {
        "tau_pitch": tau_pitch.abs(),
        "tau_position": tau_position.abs(),
        "tau_wheel_vel": ((tau_wheel_vel_left + tau_wheel_vel_right) / 2).abs(),  # Average wheel damping
        "apcr_tau": apcr_tau.abs(),
    }

    # For each step, find dominant component
    dominance = pd.DataFrame(abs_components)
    dominant_component = dominance.idxmax(axis=1)

    print("Dominant component distribution:")
    for comp, count in dominant_component.value_counts().items():
        print(f"  {comp}: {count} steps ({count/len(df)*100:.1f}%)")

    results["dominance"] = dominant_component.value_counts().to_dict()

    # 3. RECENTER-specific analysis
    print("\n--- RECENTER-SPECIFIC ANALYSIS ---")
    recenter_df = df[recenter_active]

    # Sign analysis
    e_sign = np.sign(signed_error[recenter_active])

    # Check torque signs vs error sign
    tau_pitch_sign = np.sign(tau_pitch[recenter_active])
    tau_pos_sign = np.sign(tau_position[recenter_active])
    tau_wheel_sign = np.sign((tau_wheel_vel_left[recenter_active] + tau_wheel_vel_right[recenter_active]) / 2)
    apcr_sign = np.sign(apcr_tau[recenter_active])
    final_sign = np.sign(final_tau_with_apc[recenter_active])

    # Correct = opposite sign to error (opposes drift)
    tau_pitch_correct = (tau_pitch_sign * e_sign < 0)  # Negative product = opposite signs
    tau_pos_correct = (tau_pos_sign * e_sign < 0)
    tau_wheel_correct = (tau_wheel_sign * e_sign < 0)
    apcr_correct = (apcr_sign * e_sign < 0)
    final_correct = (final_sign * e_sign < 0)

    # Fights drift = same sign as error
    tau_pitch_fights = ~tau_pitch_correct
    tau_pos_fights = ~tau_pos_correct
    tau_wheel_fights = ~tau_wheel_correct
    apcr_fights = ~apcr_correct
    final_fights = ~final_correct

    print(f"\nDuring RECENTER ({recenter_active.sum()} steps):")
    print(f"\nTorque direction correctness (opposes drift):")
    print(f"  tau_pitch correct: {tau_pitch_correct.sum()} ({tau_pitch_correct.mean()*100:.1f}%)")
    print(f"  tau_position correct: {tau_pos_correct.sum()} ({tau_pos_correct.mean()*100:.1f}%)")
    print(f"  tau_wheel_vel correct: {tau_wheel_correct.sum()} ({tau_wheel_correct.mean()*100:.1f}%)")
    print(f"  apcr_tau correct: {apcr_correct.sum()} ({apcr_correct.mean()*100:.1f}%)")
    print(f"  final_tau correct: {final_correct.sum()} ({final_correct.mean()*100:.1f}%)")

    print(f"\nTorque fights drift (same sign as error):")
    print(f"  tau_pitch fights: {tau_pitch_fights.sum()} ({tau_pitch_fights.mean()*100:.1f}%)")
    print(f"  tau_position fights: {tau_pos_fights.sum()} ({tau_pos_fights.mean()*100:.1f}%)")
    print(f"  tau_wheel_vel fights: {tau_wheel_fights.sum()} ({tau_wheel_fights.mean()*100:.1f}%)")
    print(f"  apcr_tau fights: {apcr_fights.sum()} ({apcr_fights.mean()*100:.1f}%)")
    print(f"  final_tau fights: {final_fights.sum()} ({final_fights.mean()*100:.1f}%)")

    results["recenter_analysis"] = {
        "tau_pitch_correct_pct": float(tau_pitch_correct.mean() * 100),
        "tau_position_correct_pct": float(tau_pos_correct.mean() * 100),
        "tau_wheel_vel_correct_pct": float(tau_wheel_correct.mean() * 100),
        "apcr_correct_pct": float(apcr_correct.mean() * 100),
        "final_correct_pct": float(final_correct.mean() * 100),
        "tau_pitch_fights_pct": float(tau_pitch_fights.mean() * 100),
        "tau_position_fights_pct": float(tau_pos_fights.mean() * 100),
        "tau_wheel_vel_fights_pct": float(tau_wheel_fights.mean() * 100),
        "apcr_fights_pct": float(apcr_fights.mean() * 100),
        "final_fights_pct": float(final_fights.mean() * 100),
    }

    # 4. Worst drift events - top 50 steps where |e| is largest
    print("\n--- WORST 50 DRIFT EVENTS ---")

    worst_50_idx = abs(signed_error).nlargest(50).index
    worst_df = df.loc[worst_50_idx]

    print("\nTorque composition at max |e|:")
    for col in ["tau_pitch", "tau_position", "tau_wheel_velocity_left", "tau_wheel_velocity_right", "apcr_tau"]:
        print(f"  {col}: mean={worst_df[col].mean():.2f}, abs_mean={worst_df[col].abs().mean():.2f}")

    # Check how often final tau accelerates drift at worst events
    worst_e_sign = np.sign(signed_error[worst_50_idx])
    worst_final_sign = np.sign(final_tau_with_apc[worst_50_idx])
    worst_accelerates = (worst_final_sign * worst_e_sign > 0).sum()

    print(f"\nFinal tau accelerates drift at worst events: {worst_accelerates}/50 ({worst_accelerates/50*100:.1f}%)")

    results["worst_events"] = {
        "torque_pitch_mean": float(worst_df["tau_pitch"].mean()),
        "torque_position_mean": float(worst_df["tau_position"].mean()),
        "torque_wheel_left_mean": float(worst_df["tau_wheel_velocity_left"].mean()),
        "torque_wheel_right_mean": float(worst_df["tau_wheel_velocity_right"].mean()),
        "final_accelerates_drift_count": int(worst_accelerates),
        "final_accelerates_drift_pct": float(worst_accelerates / 50 * 100),
    }

    # 5. Final torque direction violation events
    print("\n--- FINAL TORQUE DIRECTION VIOLATIONS ---")

    # A violation = final tau has same sign as error (accelerates drift)
    violation = (np.sign(final_tau_with_apc) * np.sign(signed_error)) > 0
    print(f"Final tau accelerates drift: {violation.sum()} steps ({violation.mean()*100:.1f}%)")

    # When is this violation occurring?
    violation_steps = df[violation]
    print(f"\nAt violation steps:")
    print(f"  mean |e|: {abs(violation_steps['active_pitch_crossing_signed_error_m']).mean():.4f}m")
    print(f"  mean tau_wheel_vel: {((violation_steps['tau_wheel_velocity_left'] + violation_steps['tau_wheel_velocity_right'])/2).mean():.2f} Nm")
    print(f"  RECENTER active: {violation_steps['apcr1m_recenter_active'].sum()} ({violation_steps['apcr1m_recenter_active'].mean()*100:.1f}%)")

    results["violations"] = {
        "total_violation_steps": int(violation.sum()),
        "violation_pct": float(violation.mean() * 100),
        "violation_mean_abs_error": float(abs(violation_steps['active_pitch_crossing_signed_error_m']).mean()),
        "violation_mean_wheel_vel": float(((violation_steps['tau_wheel_velocity_left'] + violation_steps['tau_wheel_velocity_right'])/2).mean()),
        "violation_recenter_active_pct": float(violation_steps['apcr1m_recenter_active'].mean() * 100),
    }

    # 6. Torque composition at max |e|
    print("\n--- TORQUE COMPOSITION AT MAX |E| ---")
    max_e_idx = abs(signed_error).idxmax()
    max_e_row = df.loc[max_e_idx]

    print(f"At step {max_e_idx}, |e| = {abs(signed_error[max_e_idx]):.4f}m:")
    print(f"  signed error: {signed_error[max_e_idx]:.4f}m")
    print(f"  tau_pitch: {max_e_row['tau_pitch']:.2f} Nm")
    print(f"  tau_position: {max_e_row['tau_position']:.2f} Nm")
    print(f"  tau_wheel_vel_left: {max_e_row['tau_wheel_velocity_left']:.2f} Nm")
    print(f"  tau_wheel_vel_right: {max_e_row['tau_wheel_velocity_right']:.2f} Nm")
    print(f"  apcr_tau: {apcr_tau[max_e_idx]:.2f} Nm")
    print(f"  final_tau_with_apc: {max_e_row['final_wheel_tau_with_apc']:.2f} Nm")
    print(f"  RECENTER: {max_e_row['apcr1m_recenter_active']}")

    results["max_e_composition"] = {
        "step": int(max_e_idx),
        "signed_error": float(signed_error[max_e_idx]),
        "tau_pitch": float(max_e_row['tau_pitch']),
        "tau_position": float(max_e_row['tau_position']),
        "tau_wheel_vel_left": float(max_e_row['tau_wheel_velocity_left']),
        "tau_wheel_vel_right": float(max_e_row['tau_wheel_velocity_right']),
        "apcr_tau": float(apcr_tau[max_e_idx]),
        "final_tau": float(max_e_row['final_wheel_tau_with_apc']),
        "recenter_active": bool(max_e_row['apcr1m_recenter_active']),
    }

    # 7. When |e| > 0.15
    print("\n--- TORQUE COMPOSITION WHEN |E| > 0.15 ---")
    large_error = abs(signed_error) > 0.15
    large_df = df[large_error]

    print(f"When |e| > 0.15m ({large_error.sum()} steps):")
    print(f"  tau_pitch: mean abs = {large_df['tau_pitch'].abs().mean():.2f} Nm")
    print(f"  tau_position: mean abs = {large_df['tau_position'].abs().mean():.2f} Nm")
    print(f"  tau_wheel_vel_left: mean abs = {large_df['tau_wheel_velocity_left'].abs().mean():.2f} Nm")
    print(f"  tau_wheel_vel_right: mean abs = {large_df['tau_wheel_velocity_right'].abs().mean():.2f} Nm")

    # Dominance at large error
    large_dominance = dominance.loc[large_error]
    large_dominant = large_dominance.idxmax(axis=1)
    print(f"\nDominant component when |e| > 0.15:")
    for comp, count in large_dominant.value_counts().items():
        print(f"  {comp}: {count} ({count/len(large_error)*100:.1f}%)")

    results["large_error_composition"] = {
        "steps": int(large_error.sum()),
        "tau_pitch_abs_mean": float(large_df['tau_pitch'].abs().mean()),
        "tau_position_abs_mean": float(large_df['tau_position'].abs().mean()),
        "tau_wheel_vel_left_abs_mean": float(large_df['tau_wheel_velocity_left'].abs().mean()),
        "tau_wheel_vel_right_abs_mean": float(large_df['tau_wheel_velocity_right'].abs().mean()),
    }

    # 8. Classification
    print("\n" + "=" * 80)
    print("TORQUE DOMINANCE CLASSIFICATION")
    print("=" * 80)

    # Determine root cause
    wheel_dominance_pct = (dominant_component == "tau_wheel_vel").mean() * 100
    pitch_dominance_pct = (dominant_component == "tau_pitch").mean() * 100
    final_fights_pct = results["recenter_analysis"]["final_fights_pct"]

    if wheel_dominance_pct > 50:
        root_cause = "wheel_velocity_damping_dominance"
        reason = f"Wheel velocity damping dominates {wheel_dominance_pct:.1f}% of the time"
    elif final_fights_pct > 30:
        root_cause = "final_torque_direction_violations"
        reason = f"Final torque fights drift {final_fights_pct:.1f}% of the time"
    elif results["recenter_analysis"]["tau_pitch_fights_pct"] > 50:
        root_cause = "tau_pitch_still_dominant"
        reason = f"tau_pitch fights drift {results['recenter_analysis']['tau_pitch_fights_pct']:.1f}% of the time"
    else:
        root_cause = "mixed_torque_dominance"
        reason = "No single component clearly dominates"

    classification = f"APCR1M_DRIFT_FROM_{root_cause.upper()}"

    print(f"\nClassification: {classification}")
    print(f"Reason: {reason}")
    print(f"\nKey findings:")
    print(f"  - Wheel vel dominance: {wheel_dominance_pct:.1f}%")
    print(f"  - Pitch dominance: {pitch_dominance_pct:.1f}%")
    print(f"  - Final fights drift: {final_fights_pct:.1f}%")
    print(f"  - tau_pitch fights drift: {results['recenter_analysis']['tau_pitch_fights_pct']:.1f}%")
    print(f"  - tau_wheel fights drift: {results['recenter_analysis']['tau_wheel_vel_fights_pct']:.1f}%")

    results["classification"] = classification
    results["classification_reason"] = reason

    # Save worst events CSV
    worst_events_df = worst_df[[
        "active_pitch_crossing_signed_error_m",
        "tau_pitch", "tau_position",
        "tau_wheel_velocity_left", "tau_wheel_velocity_right",
        "apcr_tau", "final_wheel_tau_with_apc",
        "apcr1m_recenter_active", "apcr1m_pitch_blend_active"
    ]].copy()
    worst_events_df.columns = [
        "signed_error_m", "tau_pitch_Nm", "tau_position_Nm",
        "tau_wheel_vel_left_Nm", "tau_wheel_vel_right_Nm",
        "apcr_tau_Nm", "final_tau_Nm",
        "recenter_active", "blend_active"
    ]

    csv_path = BASE_DIR / "apcr1m_torque_composition_worst_events.csv"
    worst_events_df.to_csv(csv_path, index=False)
    print(f"\nSaved worst events CSV: {csv_path}")

    # Save results JSON
    json_path = BASE_DIR / "apcr1m_torque_composition_dominance_audit.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    return results


if __name__ == "__main__":
    results = main()
