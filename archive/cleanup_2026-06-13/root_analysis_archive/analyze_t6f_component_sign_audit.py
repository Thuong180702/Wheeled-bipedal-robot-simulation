"""Phase 2: T6F Component-Level Sign Audit

Analyzes T5 vs T6F telemetry to identify which component has wrong sign.

Uses existing Phase 8 telemetry:
- T5: first 2000 steps from 5000-step run
- T6F: 2000-step screening run

Computes sign correctness for each torque component:
A. tau_position_before_clip
B. tau_position (after upstream clip)
C. apcr1n_tau_position_after_cap
D. tau_velocity_damping
E. tau_pitch
F. final_wheel_tau_with_apc
G. wheel velocity response

For each component, computes:
- Sign opposes drift %
- Sign opposes drift rate %
- Sign helps convergence %
- Sign fights convergence %
- Correctness by band state
- Correctness when arch_fix_active
- Correctness when |tau| > 4.0, 5.0, 6.0
- Correctness when outside ±0.10, ±0.15
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def load_telemetry():
    """Load T5 and T6F telemetry."""
    t5_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv")
    t5_df = pd.read_csv(t5_path).head(2001)  # First 2000 steps (0-2000 inclusive)

    t6f_path = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_screening/telemetry_t6f_high_0p480_2000.csv")
    t6f_df = pd.read_csv(t6f_path)

    return t5_df, t6f_df


def compute_sign_correctness(df, component_col, drift_col):
    """Compute sign correctness for a component vs drift error.

    Returns dict with percentages and breakdowns.
    """
    if component_col not in df.columns or drift_col not in df.columns:
        return {
            "available": False,
            "reason": f"Missing column: {component_col if component_col not in df.columns else drift_col}"
        }

    component = df[component_col].values
    drift = df[drift_col].values

    # Compute drift rate
    drift_rate = np.diff(drift, prepend=drift[0])

    # Sign correctness: component should oppose drift
    # sign(component) * sign(drift) should be NEGATIVE
    sign_component = np.sign(component)
    sign_drift = np.sign(drift)
    sign_drift_rate = np.sign(drift_rate)

    # Opposes drift: different signs
    opposes_drift = sign_component * sign_drift < 0

    # Opposes drift rate: different signs
    opposes_drift_rate = sign_component * sign_drift_rate < 0

    # Helps convergence: when error is moving toward zero (e * e_dot < 0), component opposes drift
    converging = drift * drift_rate < 0
    helps_when_converging = opposes_drift & converging
    fights_when_converging = ~opposes_drift & converging

    # Moving away: error growing
    moving_away = drift * drift_rate > 0
    helps_when_moving_away = opposes_drift & moving_away
    fights_when_moving_away = ~opposes_drift & moving_away

    # Filter out near-zero values (meaningless sign)
    nonzero_component = np.abs(component) > 1e-6
    nonzero_drift = np.abs(drift) > 1e-6
    valid_mask = nonzero_component & nonzero_drift

    if valid_mask.sum() == 0:
        return {
            "available": True,
            "valid_steps": 0,
            "reason": "All values near zero"
        }

    results = {
        "available": True,
        "total_steps": len(df),
        "valid_steps": int(valid_mask.sum()),
        "opposes_drift_pct": float(opposes_drift[valid_mask].sum() / valid_mask.sum() * 100),
        "opposes_drift_rate_pct": float(opposes_drift_rate[valid_mask].sum() / valid_mask.sum() * 100),
        "helps_when_converging_pct": float(helps_when_converging[valid_mask].sum() / valid_mask.sum() * 100),
        "fights_when_converging_pct": float(fights_when_converging[valid_mask].sum() / valid_mask.sum() * 100),
        "helps_when_moving_away_pct": float(helps_when_moving_away[valid_mask].sum() / valid_mask.sum() * 100),
        "fights_when_moving_away_pct": float(fights_when_moving_away[valid_mask].sum() / valid_mask.sum() * 100),
    }

    # Breakdown by band state (if available)
    if "apcr1nd_tuned_band_state" in df.columns:
        band_state = df["apcr1nd_tuned_band_state"].values
        for band_val in [0, 1, 2, 3, 4]:
            band_name = ["normal", "soft", "desired", "hard", "emergency"][band_val]
            band_mask = (band_state == band_val) & valid_mask
            if band_mask.sum() > 0:
                results[f"opposes_drift_in_{band_name}_pct"] = float(
                    opposes_drift[band_mask].sum() / band_mask.sum() * 100
                )

    # Breakdown by arch_fix_active (if available)
    if "arch_fix_active" in df.columns:
        arch_fix = df["arch_fix_active"].values.astype(bool)
        arch_fix_mask = arch_fix & valid_mask
        arch_fix_inactive_mask = ~arch_fix & valid_mask
        if arch_fix_mask.sum() > 0:
            results["opposes_drift_when_arch_fix_active_pct"] = float(
                opposes_drift[arch_fix_mask].sum() / arch_fix_mask.sum() * 100
            )
        if arch_fix_inactive_mask.sum() > 0:
            results["opposes_drift_when_arch_fix_inactive_pct"] = float(
                opposes_drift[arch_fix_inactive_mask].sum() / arch_fix_inactive_mask.sum() * 100
            )

    # Breakdown by torque magnitude
    abs_component = np.abs(component)
    for threshold in [4.0, 5.0, 6.0]:
        high_mag_mask = (abs_component > threshold) & valid_mask
        if high_mag_mask.sum() > 0:
            results[f"opposes_drift_when_abs_tau_gt_{threshold:.1f}_pct"] = float(
                opposes_drift[high_mag_mask].sum() / high_mag_mask.sum() * 100
            )
            results[f"steps_with_abs_tau_gt_{threshold:.1f}"] = int(high_mag_mask.sum())

    # Breakdown by drift magnitude
    abs_drift = np.abs(drift)
    for threshold_m in [0.10, 0.15]:
        outside_mask = (abs_drift > threshold_m) & valid_mask
        if outside_mask.sum() > 0:
            results[f"opposes_drift_when_outside_{threshold_m:.2f}m_pct"] = float(
                opposes_drift[outside_mask].sum() / outside_mask.sum() * 100
            )
            results[f"steps_outside_{threshold_m:.2f}m"] = int(outside_mask.sum())

    return results


def analyze_component_signs(df, profile_name, drift_col):
    """Analyze all components for a profile."""
    components = {
        "tau_position_before_clip": "tau_position_before_clip",
        "tau_position": "tau_position",
        "apcr1n_tau_position_after_cap": "apcr1n_tau_position_after_cap",
        "tau_velocity_damping_mean": None,  # Computed below
        "tau_pitch": "tau_pitch",
        "final_wheel_tau_with_apc": "final_wheel_tau_with_apc",
    }

    # Compute mean wheel velocity damping
    if "tau_wheel_velocity_left" in df.columns and "tau_wheel_velocity_right" in df.columns:
        df["tau_velocity_damping_mean"] = (df["tau_wheel_velocity_left"] + df["tau_wheel_velocity_right"]) / 2.0
        components["tau_velocity_damping_mean"] = "tau_velocity_damping_mean"

    results = {}
    for component_name, component_col in components.items():
        if component_col is None:
            continue
        print(f"  Analyzing {profile_name} {component_name}...")
        results[component_name] = compute_sign_correctness(df, component_col, drift_col)

    return results


def identify_first_wrong_component(t5_results, t6f_results):
    """Identify which component first shows wrong sign in T6F."""
    components_order = [
        "tau_position_before_clip",
        "tau_position",
        "apcr1n_tau_position_after_cap",
        "tau_velocity_damping_mean",
        "tau_pitch",
        "final_wheel_tau_with_apc",
    ]

    first_wrong = None
    for comp in components_order:
        if comp not in t6f_results or not t6f_results[comp].get("available", False):
            continue

        t6f_opposes = t6f_results[comp].get("opposes_drift_pct", 100.0)

        # If T6F opposes drift < 60%, this component is wrong
        if t6f_opposes < 60.0:
            first_wrong = comp
            break

    return first_wrong


def classify_root_cause(first_wrong_component, t6f_results):
    """Classify the root cause based on which component first shows wrong sign."""
    if first_wrong_component is None:
        return "SIGN_BUG_INCONCLUSIVE", "No component shows clear sign error"

    if first_wrong_component == "tau_position_before_clip":
        return "SIGN_BUG_IN_POSITION_TORQUE", "tau_position sign wrong before any clipping"
    elif first_wrong_component == "tau_position":
        return "SIGN_BUG_IN_UPSTREAM_CLIP_COMPOSITION", "Sign flips during clipping or scheduling"
    elif first_wrong_component == "apcr1n_tau_position_after_cap":
        return "SIGN_BUG_IN_TUNED_CAP_COMPOSITION", "Sign flips when applying APCR1n/arch_fix raised cap"
    elif first_wrong_component == "tau_velocity_damping_mean":
        return "SIGN_BUG_IN_DAMPING_TERM", "Damping term has wrong sign"
    elif first_wrong_component == "tau_pitch":
        return "SIGN_BUG_IN_PITCH_TERM_DOMINANCE", "Pitch term dominates and has wrong sign"
    elif first_wrong_component == "final_wheel_tau_with_apc":
        # Check if earlier components are correct
        earlier_correct = True
        for comp in ["tau_position_before_clip", "tau_position", "apcr1n_tau_position_after_cap"]:
            if comp in t6f_results and t6f_results[comp].get("available", False):
                if t6f_results[comp].get("opposes_drift_pct", 100.0) < 60.0:
                    earlier_correct = False
                    break

        if earlier_correct:
            return "SIGN_BUG_IN_FINAL_WHEEL_COMPOSITION", "Sign flips in final composition with APC"
        else:
            return "SIGN_BUG_MIXED_OR_INCONCLUSIVE", "Multiple components show sign errors"

    return "SIGN_BUG_MIXED_OR_INCONCLUSIVE", "Could not determine single root cause"


def main():
    print("=" * 80)
    print("Phase 2: T6F Component-Level Sign Audit")
    print("=" * 80)
    print()

    print("Loading telemetry...")
    t5_df, t6f_df = load_telemetry()
    print(f"  T5: {len(t5_df)} steps")
    print(f"  T6F: {len(t6f_df)} steps")
    print()

    # Determine drift column
    drift_col_candidates = [
        "active_pitch_crossing_signed_error_m",
        "sagittal_position_error_m",
        "support_position_error_m",
        "hip_yaw_comp_support_error_m",
    ]

    drift_col = None
    for col in drift_col_candidates:
        if col in t5_df.columns and col in t6f_df.columns:
            drift_col = col
            print(f"Using drift column: {drift_col}")
            break

    if drift_col is None:
        print("ERROR: No valid drift column found!")
        return

    print()
    print("Analyzing T5 components...")
    t5_results = analyze_component_signs(t5_df, "T5", drift_col)

    print()
    print("Analyzing T6F components...")
    t6f_results = analyze_component_signs(t6f_df, "T6F", drift_col)

    print()
    print("=" * 80)
    print("COMPONENT SIGN CORRECTNESS COMPARISON")
    print("=" * 80)
    print()

    components_order = [
        "tau_position_before_clip",
        "tau_position",
        "apcr1n_tau_position_after_cap",
        "tau_velocity_damping_mean",
        "tau_pitch",
        "final_wheel_tau_with_apc",
    ]

    comparison_table = []
    for comp in components_order:
        t5_avail = comp in t5_results and t5_results[comp].get("available", False)
        t6f_avail = comp in t6f_results and t6f_results[comp].get("available", False)

        if not (t5_avail and t6f_avail):
            continue

        t5_opposes = t5_results[comp].get("opposes_drift_pct", 0.0)
        t6f_opposes = t6f_results[comp].get("opposes_drift_pct", 0.0)
        delta = t6f_opposes - t5_opposes

        verdict = "OK" if t6f_opposes >= 70.0 else ("WARN" if t6f_opposes >= 55.0 else "FAIL")

        comparison_table.append({
            "component": comp,
            "t5_opposes_pct": t5_opposes,
            "t6f_opposes_pct": t6f_opposes,
            "delta_pct": delta,
            "verdict": verdict,
        })

        print(f"{comp:35s} T5: {t5_opposes:5.1f}%  T6F: {t6f_opposes:5.1f}%  Delta: {delta:+6.1f}%  [{verdict}]")

    print()
    print("=" * 80)
    print("ROOT CAUSE IDENTIFICATION")
    print("=" * 80)
    print()

    first_wrong = identify_first_wrong_component(t5_results, t6f_results)
    classification, reason = classify_root_cause(first_wrong, t6f_results)

    print(f"First component with wrong sign: {first_wrong if first_wrong else 'NONE'}")
    print(f"Classification: {classification}")
    print(f"Reason: {reason}")
    print()

    if first_wrong:
        print("Detailed breakdown for first wrong component:")
        comp_results = t6f_results[first_wrong]
        print(f"  Opposes drift: {comp_results.get('opposes_drift_pct', 0):.1f}%")
        print(f"  Opposes drift rate: {comp_results.get('opposes_drift_rate_pct', 0):.1f}%")
        print(f"  Helps when converging: {comp_results.get('helps_when_converging_pct', 0):.1f}%")
        print(f"  Fights when converging: {comp_results.get('fights_when_converging_pct', 0):.1f}%")
        print(f"  Helps when moving away: {comp_results.get('helps_when_moving_away_pct', 0):.1f}%")
        print(f"  Fights when moving away: {comp_results.get('fights_when_moving_away_pct', 0):.1f}%")
        print()

        # Print breakdown by conditions
        for key, value in comp_results.items():
            if key.endswith("_pct") and "when_" in key:
                if "arch_fix" in key or "abs_tau" in key or "outside" in key:
                    print(f"  {key}: {value:.1f}%")

    # Save results
    output_dir = Path("outputs/step_e_extreme_support_fix_eval/active_pitch_crossing")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "date": "2026-06-12",
        "drift_column": drift_col,
        "t5_component_results": t5_results,
        "t6f_component_results": t6f_results,
        "comparison_table": comparison_table,
        "first_wrong_component": first_wrong,
        "classification": classification,
        "reason": reason,
    }

    json_path = output_dir / "t6f_component_sign_audit.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Results saved to: {json_path}")

    # Create CSV summary
    csv_path = output_dir / "t6f_component_sign_audit.csv"
    pd.DataFrame(comparison_table).to_csv(csv_path, index=False)
    print(f"CSV summary saved to: {csv_path}")

    print()
    print("=" * 80)
    print("Phase 2: Component-level sign audit COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
