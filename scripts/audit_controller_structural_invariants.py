#!/usr/bin/env python3
"""Audit structural invariants in controller system.

This script audits structural invariants:
1. WBC: raw diagnostic vs applied contribution
2. Hidden torque
3. Torque ownership violations
4. Legacy path status
5. Torque composition consistency
"""

import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def safe_float(val, default=0.0):
    """Safely convert a value to float."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return default
    return default


def load_telemetry_csv(variant_name):
    """Load a telemetry CSV file using proper CSV parser."""
    path = Path(f"outputs/step_e_best_current_profile_5000_eval/{variant_name}_5000_telemetry.csv")
    if not path.exists():
        return None

    data = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric values to float
            cleaned_row = {}
            for key, value in row.items():
                key = key.strip()
                # Skip array columns (contain commas)
                if ',' in value:
                    cleaned_row[key] = value
                else:
                    try:
                        cleaned_row[key] = float(value)
                    except ValueError:
                        cleaned_row[key] = value
            data.append(cleaned_row)

    return data


def audit_wbc_status(telemetry, variant_name):
    """Audit WBC status: diagnostic vs applied."""

    if not telemetry or len(telemetry) == 0:
        return None

    # Key columns for WBC analysis
    wbc_columns = [
        "tau_wbc_norm",
        "tau_wbc_max",
        "tau_wbc_applied",
        "tau_total_norm",
        "hidden_torque_norm",
        "tau_inverse_dynamics_norm",
    ]

    # Get column availability
    first_row = telemetry[0]
    available_cols = [c for c in wbc_columns if c in first_row]

    # Compute statistics with safe conversion
    wbc_norms = [safe_float(row.get("tau_wbc_norm", 0)) for row in telemetry if "tau_wbc_norm" in row]
    hidden_torques = [safe_float(row.get("hidden_torque_norm", 0)) for row in telemetry if "hidden_torque_norm" in row]
    tau_totals = [safe_float(row.get("tau_total_norm", 0)) for row in telemetry if "tau_total_norm" in row]

    # Check for applied WBC (look at per-joint torque)
    tau_wbc_per_joint = [row.get("tau_wbc_per_joint") for row in telemetry if "tau_wbc_per_joint" in row]
    tau_total_per_joint = [row.get("tau_total_per_joint") for row in telemetry if "tau_total_per_joint" in row]

    # Check if WBC norm > 0 and tau_total_norm > 0
    wbc_max = max(wbc_norms) if wbc_norms else 0
    hidden_max = max(hidden_torques) if hidden_torques else 0
    tau_total_max = max(tau_totals) if tau_totals else 0

    # WBC applied check: Look at tau_wbc_scaled_per_joint vs tau_wbc_per_joint
    # If they're different, WBC was scaled/clipped
    # If tau_wbc_per_joint != 0 and tau_total_per_joint != 0, WBC was applied

    # Classification
    if wbc_max > 1.0 and hidden_max == 0:
        classification = "WBC_DIAGNOSTIC_ONLY"
        explanation = f"WBC norm max={wbc_max:.2f} Nm but hidden_torque_norm max={hidden_max:.2f} Nm indicates WBC not applied to joints"
    elif hidden_max > 0:
        classification = "HIDDEN_TORQUE_PRESENT"
        explanation = f"Hidden torque detected: max={hidden_max:.2f} Nm"
    elif wbc_max > tau_total_max * 0.5:
        classification = "WBC_ACTUALLY_APPLIED"
        explanation = f"WBC norm ({wbc_max:.2f}) is significant portion of tau_total ({tau_total_max:.2f})"
    else:
        classification = "WBC_STATUS_AMBIGUOUS"
        explanation = f"WBC max={wbc_max:.2f}, tau_total max={tau_total_max:.2f}"

    return {
        "variant": variant_name,
        "classification": classification,
        "explanation": explanation,
        "wbc_norm_max": float(wbc_max),
        "wbc_norm_mean": float(np.mean(wbc_norms)) if wbc_norms else 0,
        "hidden_torque_max": float(hidden_max),
        "hidden_torque_mean": float(np.mean(hidden_torques)) if hidden_torques else 0,
        "tau_total_max": float(tau_total_max),
        "tau_total_mean": float(np.mean(tau_totals)) if tau_totals else 0,
        "available_columns": available_cols,
    }


def audit_torque_ownership(telemetry, variant_name):
    """Audit torque ownership and violations."""

    if not telemetry or len(telemetry) == 0:
        return None

    first_row = telemetry[0]

    # Check ownership columns
    ownership_violations = [safe_float(row.get("ownership_violation_count", 0)) for row in telemetry]
    active_torque_owners = [row.get("active_torque_owner_per_joint") for row in telemetry if "active_torque_owner_per_joint" in row]

    violation_max = max(ownership_violations) if ownership_violations else 0
    violation_mean = np.mean(ownership_violations) if ownership_violations else 0

    # Check for saturation
    torque_saturations = [safe_float(row.get("torque_saturation_mask_per_joint", 0)) for row in telemetry if "torque_saturation_mask_per_joint" in row]

    return {
        "variant": variant_name,
        "ownership_violation_max": float(violation_max),
        "ownership_violation_mean": float(violation_mean),
        "torque_saturation_events": len([s for s in torque_saturations if s > 0]) if torque_saturations else 0,
        "classification": "CLEAN" if violation_max == 0 else "OWNERSHIP_VIOLATION",
    }


def audit_legacy_path(telemetry, variant_name):
    """Audit legacy torque path status."""

    if not telemetry or len(telemetry) == 0:
        return None

    # Check for legacy torque columns
    legacy_columns = [
        "tau_legacy_wheel_balance_norm",
        "tau_legacy_hip_roll_centering_norm",
    ]

    first_row = telemetry[0]
    available_legacy = [c for c in legacy_columns if c in first_row]

    legacy_norms = {}
    for col in available_legacy:
        values = [safe_float(row.get(col, 0)) for row in telemetry]
        if values:
            legacy_norms[col] = {
                "max": float(max(values)),
                "mean": float(np.mean(values)),
            }

    # Check controller_mode column
    controller_mode = telemetry[0].get("controller_mode", "unknown")
    control_mode = telemetry[0].get("control_mode", "unknown")

    return {
        "variant": variant_name,
        "controller_mode": controller_mode,
        "control_mode": control_mode,
        "legacy_columns_available": available_legacy,
        "legacy_norms": legacy_norms,
        "legacy_path_active": any(n.get("max", 0) > 0.1 for n in legacy_norms.values()),
    }


def audit_torque_composition(telemetry, variant_name):
    """Audit torque composition consistency."""

    if not telemetry or len(telemetry) == 0:
        return None

    # Check for per-joint torque columns
    per_joint_columns = [
        "tau_wbc_per_joint",
        "tau_wbc_scaled_per_joint",
        "tau_hip_roll_centering_per_joint",
        "tau_posture_per_joint",
        "tau_leg_position_per_joint",
        "tau_wheel_balance_per_joint",
        "tau_static_feedforward_per_joint",
        "tau_total_per_joint",
        "tau_total_raw_per_joint",
        "tau_total_clipped_per_joint",
        "tau_smooth_per_joint",
        "tau_shape_posture_per_joint",
        "tau_support_feedforward_per_joint",
        "tau_sagittal_wheel_balance_per_joint",
        "tau_lateral_roll_balance_per_joint",
        "tau_final_per_joint",
    ]

    first_row = telemetry[0]
    available_per_joint = [c for c in per_joint_columns if c in first_row]

    # Check clipping
    torque_saturation_mask = [safe_float(row.get("torque_saturation_mask_per_joint", 0)) for row in telemetry]
    torque_rate_saturation = [safe_float(row.get("torque_rate_saturation_mask_per_joint", 0)) for row in telemetry]

    saturation_events = len([s for s in torque_saturation_mask if s > 0]) if torque_saturation_mask else 0
    rate_limit_events = len([s for s in torque_rate_saturation if s > 0]) if torque_rate_saturation else 0

    return {
        "variant": variant_name,
        "per_joint_columns_available": available_per_joint,
        "torque_saturation_events": saturation_events,
        "torque_rate_limit_events": rate_limit_events,
    }


def audit_structural_invariants():
    """Main audit function."""

    variants = ["low_0p300", "nominal", "high_0p480"]
    results = {
        "wbc_audit": {},
        "ownership_audit": {},
        "legacy_path_audit": {},
        "torque_composition_audit": {},
        "overall_classification": "UNKNOWN",
    }

    all_classifications = []

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Auditing {variant}")
        print(f"{'='*60}")

        telemetry = load_telemetry_csv(variant)
        if telemetry is None:
            print(f"  No telemetry found for {variant}")
            continue

        print(f"  Loaded {len(telemetry)} rows")

        # WBC Status
        wbc_result = audit_wbc_status(telemetry, variant)
        if wbc_result:
            results["wbc_audit"][variant] = wbc_result
            all_classifications.append(wbc_result["classification"])
            print(f"  WBC Status: {wbc_result['classification']}")
            print(f"    - WBC norm max: {wbc_result['wbc_norm_max']:.4f} Nm")
            print(f"    - Hidden torque max: {wbc_result['hidden_torque_max']:.4f} Nm")
            print(f"    - tau_total max: {wbc_result['tau_total_max']:.4f} Nm")

        # Ownership
        ownership_result = audit_torque_ownership(telemetry, variant)
        if ownership_result:
            results["ownership_audit"][variant] = ownership_result
            print(f"  Ownership: {ownership_result['classification']}")
            print(f"    - Violation max: {ownership_result['ownership_violation_max']:.0f}")
            print(f"    - Saturation events: {ownership_result['torque_saturation_events']}")

        # Legacy path
        legacy_result = audit_legacy_path(telemetry, variant)
        if legacy_result:
            results["legacy_path_audit"][variant] = legacy_result
            print(f"  Legacy Path: {'ACTIVE' if legacy_result['legacy_path_active'] else 'INACTIVE'}")
            print(f"    - controller_mode: {legacy_result['controller_mode']}")
            print(f"    - control_mode: {legacy_result['control_mode']}")

        # Torque composition
        composition_result = audit_torque_composition(telemetry, variant)
        if composition_result:
            results["torque_composition_audit"][variant] = composition_result
            print(f"  Torque Composition:")
            print(f"    - Per-joint columns: {len(composition_result['per_joint_columns_available'])}")
            print(f"    - Saturation events: {composition_result['torque_saturation_events']}")

    # Overall classification
    if "WBC_ACTUALLY_APPLIED" in all_classifications:
        results["overall_classification"] = "STRUCTURAL_INVARIANT_VIOLATION"
    elif "HIDDEN_TORQUE_PRESENT" in all_classifications:
        results["overall_classification"] = "HIDDEN_TORQUE_ISSUE"
    elif "WBC_DIAGNOSTIC_ONLY" in all_classifications or "CLEAN" in all_classifications:
        results["overall_classification"] = "STRUCTURAL_INVARIANTS_CLEAN"
    else:
        results["overall_classification"] = "STRUCTURAL_STATUS_AMBIGUOUS"

    return results


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 3: STRUCTURAL INVARIANTS AUDIT")
    print("=" * 80)

    results = audit_structural_invariants()

    # Save results
    output_dir = Path("outputs/controller_system_root_cause_audit/structural_invariants")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "structural_invariant_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Create CSV summary
    csv_lines = ["variant,classification,wbc_norm_max,hidden_torque_max,tau_total_max,ownership_violations,saturation_events"]

    for variant in ["low_0p300", "nominal", "high_0p480"]:
        wbc = results["wbc_audit"].get(variant, {})
        own = results["ownership_audit"].get(variant, {})
        comp = results["torque_composition_audit"].get(variant, {})

        csv_lines.append(
            f"{variant},{wbc.get('classification', 'UNKNOWN')},"
            f"{wbc.get('wbc_norm_max', 0):.4f},"
            f"{wbc.get('hidden_torque_max', 0):.4f},"
            f"{wbc.get('tau_total_max', 0):.4f},"
            f"{own.get('ownership_violation_max', 0):.0f},"
            f"{comp.get('torque_saturation_events', 0)}"
        )

    with open(output_dir / "wbc_application_audit.csv", "w") as f:
        f.write("\n".join(csv_lines))

    print(f"\n{'='*80}")
    print(f"OVERALL CLASSIFICATION: {results['overall_classification']}")
    print(f"{'='*80}")

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
